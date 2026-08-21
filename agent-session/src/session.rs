//! AgentSession: orchestration actor owning journal, event bus, queues, compaction, retry.
//! Single actor (mpsc) serializes all state mutation; tools run in JoinSet outside it.

use crate::bus::{Event, EventBus, HookAction, Hooks};
use crate::context::{ContextBuffer, TokenBudget};
use crate::entry::Entry;
use crate::error::SessionError;
use crate::journal::Journal;
use agent_ai::model::Model;
use agent_ai::provider::ToolSpec;

use agent_core::loop_::{run_loop, LoopConfig};
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
        let provider = self
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

        // 4. Call agent loop
        self.phase = crate::Phase::Streaming;
        self.cancel = Cancelled::new();

        // Run tool_call interceptors: register a wrapper tool registry that applies hooks
        let hook_registry = if self.has_tool_hooks().await {
            let intercepted = self.create_intercepted_registry().await;
            Some(intercepted)
        } else {
            None
        };
        let effective_registry = hook_registry.as_ref().unwrap_or(&self.tool_registry);

        let loop_result = run_loop(
            provider.as_ref(),
            &self.client,
            effective_registry,
            &self.cancel,
            LoopConfig {
                model: &model,
                system: &self.system_prompt,
                messages: &messages,
                tools: &tool_specs,
                max_tokens: model.max_tokens,
                thinking: self.state.thinking_level,
                max_turns: 10,
            },
        )
        .await
        .map_err(SessionError::Core)?;

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
