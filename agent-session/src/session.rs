//! AgentSession: orchestration actor owning journal, event bus, queues, compaction, retry.
//! Single actor (mpsc) serializes all state mutation; tools run in JoinSet outside it.

use crate::bus::{Event, EventBus, HookAction, Hooks};
use crate::context::{ContextBuffer, TokenBudget};
use crate::entry::Entry;
use crate::error::SessionError;
use crate::journal::Journal;
use agent_ai::model::Model;
use agent_ai::provider::ToolSpec;

use agent_core::loop_::{estimate_tokens, estimate_message_tokens, run_loop, LoopConfig};
use agent_core::messages::{ContentBlock, Message, MessageContent, Role};
use agent_core::state::AgentState;
use agent_core::tools::ToolRegistry;
use agent_core::Cancelled;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct AgentSession {
    pub journal: Journal,
    pub state: AgentState,
    pub bus: EventBus,
    pub hooks: Hooks,
    pub tokens: TokenBudget,
    pub context: ContextBuffer,
    /// steering queue (delivered after current tool batch, before next LLM call)
    pub steering: mpsc::UnboundedSender<String>,
    /// follow-up queue (delivered only when agent settles)
    pub follow_up: mpsc::UnboundedSender<String>,
    pub compaction_settings: crate::compaction::CompactionSettings,
    pub phase: crate::Phase,
    pub provider: Option<Arc<dyn agent_ai::provider::ChatProvider>>,
    pub client: agent_ai::Client,
    pub tool_registry: ToolRegistry,
    pub model: Option<Model>,
    pub system_prompt: String,
    pub cancel: Cancelled,
    pub _receivers: mpsc::UnboundedReceiver<String>,
}

impl AgentSession {
    /// Open or create a session journal at path. In-memory when None.
    pub fn open(path: Option<&Path>) -> Result<Self, SessionError> {
        let journal = match path {
            Some(p) => Journal::open(p)?,
            None => Journal::default(),
        };
        let (steering_tx, steering_rx) = mpsc::unbounded_channel();
        let (follow_tx, _follow_rx) = mpsc::unbounded_channel();
        Ok(Self {
            journal,
            state: AgentState::new(),
            bus: EventBus::new(1024),
            hooks: Hooks::default(),
            tokens: TokenBudget::default(),
            context: ContextBuffer::new(),
            steering: steering_tx,
            follow_up: follow_tx,
            compaction_settings: crate::compaction::CompactionSettings::default(),
            phase: crate::Phase::Idle,
            provider: None,
            client: agent_ai::Client::new(),
            tool_registry: ToolRegistry::default(),
            model: None,
            system_prompt: String::new(),
            cancel: Cancelled::new(),
            _receivers: steering_rx,
        })
    }

    /// Configure the provider (call after open, before prompt).
    pub fn set_provider(&mut self, provider: Arc<dyn agent_ai::provider::ChatProvider>) {
        self.provider = Some(provider);
    }

    /// Set the model to use.
    pub fn set_model(&mut self, model: Model) {
        self.model = Some(model);
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    /// Register a tool.
    pub fn register_tool(&self, tool: Arc<dyn agent_core::tools::Tool>) {
        self.tool_registry.register(tool);
    }

    /// Fired when compaction or branch switch invalidates the incremental context.
    pub fn invalidate_context(&mut self) {
        self.context.invalidate();
    }

    /// Queue state publication for frontends.
    pub fn publish_queues(&self, steering_len: usize, follow_up_len: usize) {
        self.bus.emit(crate::bus::Event::QueueUpdate {
            steering: steering_len,
            follow_up: follow_up_len,
        });
    }

    /// Run one prompt end-to-end: builds context, calls agent loop, writes results to journal.
    pub async fn prompt(&mut self, text: String) -> Result<String, SessionError> {
        // 1. Append user message to journal
        let user_id = uuid_str();
        let user_entry = Entry::Message {
            id: user_id.clone(),
            parent_id: self.journal.leaf.clone(),
            timestamp: now_iso(),
            message: Message {
                id: user_id.clone(),
                role: Role::User,
                content: MessageContent::Text(text.clone()),
                usage: None,
                stop_reason: None,
                timestamp: now_ts(),
                provider: None,
                model: None,
            },
        };
        self.journal.append(user_entry);
        self.bus.emit(Event::TurnStart {
            index: self.journal.entries.len() as u32,
        });

        // 2. Build messages for the agent loop
        let _provider = self
            .provider
            .as_ref()
            .ok_or_else(|| SessionError::Core(agent_core::CoreError::Tool("未配置服务商".into())))?;
        let model = self
            .model
            .clone()
            .ok_or_else(|| SessionError::Core(agent_core::CoreError::Tool("未配置模型".into())))?;

        // Collect messages from journal entries
        let messages: Vec<Message> = self
            .journal
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::Message { message, .. } => Some(message.clone()),
                _ => None,
            })
            .collect();

        // Build tool specs from registry
        let tool_specs: Vec<ToolSpec> = self
            .tool_registry
            .describe_all()
            .iter()
            .map(|desc| {
                // Parse "name: description" format from describe_all
                let parts: Vec<&str> = desc.splitn(2, ':').collect();
                let name = parts[0].trim().to_string();
                let description = parts.get(1).map(|s| s.trim().to_string()).unwrap_or_default();
                ToolSpec {
                    name,
                    description,
                    input_schema: json!({
                        "type": "object",
                        "properties": {}
                    }),
                }
            })
            .collect();

        // 3. Emit event (with hook dispatch)
        let hook_action = self.hooks.run_handlers(&Event::AgentStart).await;
        if hook_action == HookAction::Blocked {
            tracing::info!("agent_start 被钩子拦截");
            return Ok(String::new());
        }
        self.bus.emit(Event::AgentStart);

        // 4. Call agent loop with overflow recovery
        self.phase = crate::Phase::Streaming;
        self.cancel = Cancelled::new();

        // Clone provider Arc to avoid borrow issues in the loop
        let provider_arc = self.provider.clone().unwrap();

        // Run tool_call interceptors: register a wrapper tool registry that applies hooks
        let hook_registry = if self.has_tool_hooks().await {
            let intercepted = self.create_intercepted_registry().await;
            Some(intercepted)
        } else {
            None
        };
        let effective_registry = hook_registry.as_ref().unwrap_or(&self.tool_registry).clone();

        // pi parity: overflow → compact → retry. Max 1 retry to prevent infinite loop.
        let mut current_messages = messages.clone();
        let mut compacted = false;
        let loop_result = loop {
            match run_loop(
                provider_arc.as_ref(),
                &self.client,
                &effective_registry,
                &self.cancel,
                LoopConfig {
                    model: &model,
                    system: &self.system_prompt,
                    messages: &current_messages,
                    tools: &tool_specs,
                    max_tokens: model.max_tokens,
                    thinking: self.state.thinking_level,
                    max_turns: 10,
                    context_window: model.context_window as u64,
                    reserve_tokens: self.compaction_settings.reserve_tokens,
                },
            )
            .await
            {
                Ok(result) => break Ok(result),
                Err(agent_core::CoreError::ContextOverflow { used, window, reserve }) => {
                    if compacted {
                        // Already compacted once, don't retry again
                        tracing::warn!(
                            "压缩后仍然溢出 (used={}, window={}, reserve={}), 放弃重试",
                            used, window, reserve
                        );
                        break Err(SessionError::Core(agent_core::CoreError::ContextOverflow { used, window, reserve }));
                    }
                    tracing::info!(
                        "上下文溢出 ({} token), 触发压缩后重试",
                        used
                    );
                    // Trigger compaction
                    match self.compact_context(&current_messages, &model, provider_arc.as_ref()).await {
                        Ok(new_messages) => {
                            current_messages = new_messages;
                            compacted = true;
                            self.invalidate_context();
                            self.bus.emit(Event::CompactionStart {
                                reason: crate::bus::CompactionReason::Overflow,
                            });
                            continue; // retry with compacted context
                        }
                        Err(e) => {
                            tracing::error!("压缩失败: {e}");
                            break Err(e);
                        }
                    }
                }
                Err(e) => break Err(SessionError::Core(e)),
            }
        }?;

        // 5. Write results to journal
        let mut final_text = String::new();
        for msg in &loop_result.messages {
            if msg.role == Role::Assistant {
                // Extract text from assistant message
                if let MessageContent::Assistant(blocks) = &msg.content {
                    for block in blocks {
                        if let ContentBlock::Text(t) = block {
                            final_text.push_str(t);
                        }
                    }
                }
                let entry = Entry::Message {
                    id: msg.id.clone(),
                    parent_id: self.journal.leaf.clone(),
                    timestamp: now_iso(),
                    message: msg.clone(),
                };
                self.journal.append(entry);
            }
        }

        // 6. Update usage tracking
        self.tokens.note_usage(loop_result.usage);
        self.state.push_message(Message {
            id: uuid_str(),
            role: Role::User,
            content: MessageContent::Text(text),
            usage: None,
            stop_reason: None,
            timestamp: now_ts(),
            provider: None,
            model: None,
        });

        // 7. Flush journal
        self.journal.flush().await?;

        // 8. Emit end events (with hook dispatch)
        self.bus.emit(Event::TurnEnd {
            index: self.journal.entries.len() as u32,
        });
        self.hooks.run_handlers(&Event::AgentSettled).await;
        self.bus.emit(Event::AgentSettled);
        self.phase = crate::Phase::Idle;

        Ok(final_text)
    }

    // ─── Compaction (auto-compress on overflow) ───────────────────

    /// Compact context when overflow detected. pi parity:
    /// - keepRecentTokens from the tail are preserved uncompressed
    /// - older messages are summarized via LLM
    /// - summary is written as a Compaction entry in the journal
    /// - prefix (system + tools) is NOT touched → prompt cache stays warm
    async fn compact_context(
        &mut self,
        messages: &[Message],
        model: &Model,
        provider: &dyn agent_ai::provider::ChatProvider,
    ) -> Result<Vec<Message>, SessionError> {
        // Count recent tokens to determine the cut point
        let keep_tokens = self.compaction_settings.keep_recent_tokens;
        let mut tail_tokens: u64 = 0;
        let mut cut_idx = messages.len();
        for (i, msg) in messages.iter().enumerate().rev() {
            let msg_tokens = estimate_message_tokens(msg);
            if tail_tokens + msg_tokens > keep_tokens {
                cut_idx = i + 1;
                break;
            }
            tail_tokens += msg_tokens;
        }
        // Don't cut in the middle of a tool call/result pair
        while cut_idx > 0 {
            match &messages[cut_idx - 1].role {
                Role::User | Role::Assistant => break,
                Role::ToolResult => cut_idx -= 1,
            }
        }
        if cut_idx == 0 || cut_idx >= messages.len() {
            // Nothing to compact
            return Ok(messages.to_vec());
        }

        let (to_summarize, retained) = messages.split_at(cut_idx);
        let tokens_before = estimate_tokens(&self.system_prompt)
            + to_summarize.iter().map(estimate_message_tokens).sum::<u64>();

        // Serialize older messages for summarization
        let serialized = serialize_for_summary(to_summarize);

        // Call LLM for summary (pi: independent request, no cache write)
        let summary_prompt = format!(
            "请简洁地总结以下对话历史。\
             保留关键事实、决策、文件路径和后续对话需要的上下文。\
             只输出摘要，不要前言。\n\n{serialized}"
        );
        let req = agent_ai::provider::ProviderRequest {
            model: model.clone(),
            system: "你是一个对话总结器。请简洁地总结对话历史，\
                      保留关键事实、决策、文件路径和重要上下文。"
                .to_string(),
            messages: vec![agent_ai::provider::ChatMessage {
                role: "user".into(),
                parts: vec![agent_ai::provider::Part::Text {
                    text: summary_prompt,
                }],
            }],
            thinking: agent_ai::model::ThinkingLevel::Off,
            max_tokens: 2048,
            tools: Vec::new(),
        };
        let resp = provider.chat(&self.client, &req).await.map_err(SessionError::Ai)?;
        let summary = match resp {
            agent_ai::provider::ProviderResponse::Stream(mut sr) => {
                let mut text = String::new();
                while let Some(ev) = sr.next().await {
                    match ev.map_err(SessionError::Ai)? {
                        agent_ai::stream::StreamEvent::TextDelta { delta } => text.push_str(&delta),
                        agent_ai::stream::StreamEvent::Done { .. } => break,
                        _ => {}
                    }
                }
                text
            }
            agent_ai::provider::ProviderResponse::Done { text, .. } => text,
        };

        // Write Compaction entry to journal (self-contained checkpoint)
        let compaction_id = uuid_str();
        let compaction_entry = Entry::Compaction {
            id: compaction_id,
            parent_id: self.journal.leaf.clone(),
            timestamp: now_iso(),
            summary: summary.clone(),
            first_kept_entry_id: retained.first().map(|m| m.id.clone()),
            tokens_before,
            retained_tail: None,
            details: None,
        };
        self.journal.append(compaction_entry);

        // Build new message list: [summary as assistant message] + [retained messages]
        let summary_msg = Message {
            id: uuid_str(),
            role: Role::Assistant,
            content: MessageContent::Text(format!("[会话摘要] {summary}")),
            usage: None,
            stop_reason: None,
            timestamp: now_ts(),
            provider: None,
            model: None,
        };
        let mut new_messages = vec![summary_msg];
        new_messages.extend_from_slice(retained);

        tracing::info!(
            "压缩完成: {} 条消息 → 摘要 + {} 条保留 (tokens_before={})",
            cut_idx,
            retained.len(),
            tokens_before
        );

        Ok(new_messages)
    }

    // ─── M5 Plugin API ───────────────────────────────────────────

    /// Run all handlers for a given event.
    pub async fn run_hooks(&self, ev: &Event) -> HookAction {
        self.hooks.run_handlers(ev).await
    }

    /// Check if any tool_call interceptors are registered.
    async fn has_tool_hooks(&self) -> bool {
        // Simple check: if we have any interceptors, create a wrapped registry
        let test_args = serde_json::json!({});
        self.hooks
            .run_tool_interceptors("__probe__", &test_args)
            .await
            .is_some()
    }

    /// Create a ToolRegistry that wraps the original with hook interceptors.
    /// Each tool execution passes through tool_call interceptors first.
    async fn create_intercepted_registry(&self) -> ToolRegistry {
        let hooks = self.hooks.clone();
        let original = self.tool_registry.clone();

        // Create a wrapper tool for each registered tool
        let registry = ToolRegistry::default();
        for name in original.names() {
            if let Some(tool) = original.get(&name) {
                let hooks = hooks.clone();
                let tool = tool.clone();
                let wrapper = agent_core::tools::FnTool::new(
                    Box::leak(name.clone().into_boxed_str()),
                    Box::leak(tool.describe().into_boxed_str()),
                    move |args, cancel| {
                        let hooks = hooks.clone();
                        let tool = tool.clone();
                        Box::pin(async move {
                            // Run interceptors
                            let intercepted_args = hooks
                                .run_tool_interceptors(&args.name, &args.arguments)
                                .await;
                            let new_args = match intercepted_args {
                                Some(a) => agent_core::tools::ToolArgs {
                                    call_id: args.call_id,
                                    name: args.name,
                                    arguments: a,
                                },
                                None => {
                                    return Err(agent_core::CoreError::Tool(
                                        format!("工具 '{}' 被钩子拦截", args.name),
                                    ));
                                }
                            };
                            tool.execute(&new_args, &cancel, None).await
                        })
                    },
                );
                registry.register(Arc::new(wrapper));
            }
        }
        registry
    }
}

/// Simple UUID-like string for message IDs.
fn uuid_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = t.as_nanos();
    let r: u32 = rand_u32();
    format!(
        "{:016x}-{:04x}-{:04x}-{:04x}-{:012x}",
        nanos as u64,
        ((nanos >> 64) as u16) as u32,
        (r & 0x0FFF) | 0x4000,
        ((r >> 12) & 0x3FFF) | 0x8000,
        r as u64,
    )
}

fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = t.as_secs();
    format!("{secs}")
}

fn now_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn rand_u32() -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u64(std::process::id() as u64);
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    h.finish() as u32
}

/// Serialize messages to text for LLM summarization (for compaction).
fn serialize_for_summary(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let role_str = match msg.role {
            agent_core::messages::Role::User => "用户",
            agent_core::messages::Role::Assistant => "助手",
            agent_core::messages::Role::ToolResult => "工具结果",
        };
        let content = match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Assistant(blocks) => {
                let mut s = String::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text(t) => s.push_str(t),
                        ContentBlock::Thinking(t) => s.push_str(&format!("[思考: {t}]")),
                        ContentBlock::ToolCall { name, arguments, .. } => {
                            s.push_str(&format!("[工具调用: {name}({arguments})]"));
                        }
                    }
                }
                s
            }
            MessageContent::Image { mime, data } => {
                format!("[图片: {mime}, {} 字节]", data.len())
            }
        };
        parts.push(format!("[{role_str}]: {content}"));
    }
    parts.join("\n\n")
}
