//! Agent loop: streaming LLM call → tool execution → result feedback → next turn.
//!
//! The loop is owned by agent-session; this module is the pure computation engine
//! that does not touch session files or event buses. It returns structured results
//! that the session layer maps into journal entries and events.

use crate::cancel::Cancelled;
use crate::error::CoreError;
use crate::messages::{ContentBlock, Message, MessageContent, Role};
use crate::tools::{ToolArgs, ToolOutput, ToolRegistry};
use agent_ai::model::{Model, Usage};
use agent_ai::provider::{
    ChatMessage, ChatProvider, Part, ProviderRequest, ProviderResponse, ToolSpec,
};
use agent_ai::stream::{StopReason, StreamEvent};
use serde_json::Value;


/// Result of one complete agent loop run (all turns until a non-tool-use stop).
#[derive(Debug)]
pub struct LoopResult {
    /// All new messages generated during this run (assistant + tool results).
    pub messages: Vec<Message>,
    /// Accumulated usage across all turns.
    pub usage: Usage,
    /// Final stop reason from the last LLM response.
    pub stop_reason: StopReason,
}

/// Tracks a single in-flight tool call accumulated from streaming deltas.
struct PendingToolCall {
    id: String,
    name: String,
    args_buf: String,
}

/// Config for one loop execution.
pub struct LoopConfig<'a> {
    pub model: &'a Model,
    pub system: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
    pub max_tokens: usize,
    pub thinking: agent_ai::model::ThinkingLevel,
    /// Maximum number of tool-use turns before forced stop (prevents infinite loops).
    pub max_turns: u32,
    /// Context window size (from model). 0 = no overflow check.
    pub context_window: u64,
    /// Reserve tokens below the window that triggers compaction.
    pub reserve_tokens: u64,
}

/// Rough token estimation: ~4 chars per token for English, ~2 for CJK.
pub fn estimate_tokens(text: &str) -> u64 {
    let chars = text.len() as u64;
    // Rough heuristic: 4 chars per token (works for English-heavy text)
    chars / 4
}

/// Estimate tokens for a single message.
pub fn estimate_message_tokens(msg: &Message) -> u64 {
    let content_chars: usize = match &msg.content {
        MessageContent::Text(t) => t.len(),
        MessageContent::Assistant(blocks) => blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text(t) => t.len(),
                ContentBlock::Thinking(t) => t.len(),
                ContentBlock::ToolCall { arguments, .. } => arguments.len(),
            })
            .sum(),
        MessageContent::ToolResult { content, .. } => content.len(),
        MessageContent::Image { .. } => 100,
    };
    (content_chars as u64) / 4 + 4 // +4 for role/formatting overhead
}

/// Run the agent loop: stream from provider, execute tools, loop until non-tool stop.
///
/// `provider` is used to call the LLM; `registry` to execute tools.
/// Returns `LoopResult` with all generated messages, or `CoreError` on failure.
pub async fn run_loop(
    provider: &dyn ChatProvider,
    client: &agent_ai::Client,
    registry: &ToolRegistry,
    cancel: &Cancelled,
    config: LoopConfig<'_>,
) -> Result<LoopResult, CoreError> {
    let mut messages: Vec<Message> = config.messages.to_vec();
    let mut accumulated_usage = Usage::default();
    let mut turn_count = 0u32;

    loop {
        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        turn_count += 1;
        if turn_count > config.max_turns {
            tracing::warn!(
                "智能体循环达到最大轮次 ({}), 停止",
                config.max_turns
            );
            return Ok(LoopResult {
                messages,
                usage: accumulated_usage,
                stop_reason: StopReason::Length,
            });
        }

        // 1. Estimate current context tokens and check for overflow
        if config.context_window > 0 {
            let system_tokens = estimate_tokens(config.system);
            let tool_tokens: u64 = config.tools.iter().map(|t| {
                estimate_tokens(&t.name) + estimate_tokens(&t.description)
            }).sum();
            let msg_tokens: u64 = messages.iter().map(estimate_message_tokens).sum();
            let total = system_tokens + tool_tokens + msg_tokens;
            if total > config.context_window.saturating_sub(config.reserve_tokens) {
                tracing::warn!(
                    "上下文溢出: {} token > {} (窗口 {}, 保留 {})",
                    total, config.context_window - config.reserve_tokens,
                    config.context_window, config.reserve_tokens
                );
                return Err(CoreError::ContextOverflow {
                    used: total,
                    window: config.context_window,
                    reserve: config.reserve_tokens,
                });
            }
        }

        // 2. Build ProviderRequest from messages
        let provider_msgs = convert_messages_for_provider(&messages);
        let req = ProviderRequest {
            model: config.model.clone(),
            system: config.system.to_string(),
            messages: provider_msgs,
            thinking: config.thinking,
            max_tokens: config.max_tokens,
            tools: config.tools.to_vec(),
        };

        // 3. Call provider
        let resp = provider.chat(client, &req).await?;

        // 4. Stream response, accumulate text + tool calls
        let (stream_text, stream_thinking, tool_calls, final_usage, stop_reason) =
            match resp {
                ProviderResponse::Stream(mut sr) => {
                    let mut text = String::new();
                    let mut thinking = String::new();
                    let mut usage = Usage::default();
                    let mut pending_tools: Vec<PendingToolCall> = Vec::new();
                    let mut active_tool_idx: Option<usize> = None;
                    let mut sr_result = StopReason::Stop;

                    while let Some(ev) = sr.next().await {
                        if cancel.is_cancelled() {
                            return Err(CoreError::Cancelled);
                        }
                        match ev? {
                            StreamEvent::TextDelta { delta } => text.push_str(&delta),
                            StreamEvent::ThinkingDelta { delta } => thinking.push_str(&delta),
                            StreamEvent::ToolCallStarted { id, name } => {
                                pending_tools.push(PendingToolCall {
                                    id,
                                    name,
                                    args_buf: String::new(),
                                });
                                active_tool_idx = Some(pending_tools.len() - 1);
                            }
                            StreamEvent::ToolCallArgsDelta { id: _, delta } => {
                                if let Some(idx) = active_tool_idx {
                                    if idx < pending_tools.len() {
                                        pending_tools[idx].args_buf.push_str(&delta);
                                    }
                                }
                                // Also try to match by id for multi-tool streams
                                else if !pending_tools.is_empty() {
                                    let last = pending_tools.len() - 1;
                                    pending_tools[last].args_buf.push_str(&delta);
                                }
                            }
                            StreamEvent::ToolCallDone { id: _ } => {
                                active_tool_idx = None;
                            }
                            StreamEvent::Usage { usage: u } => usage = u,
                            StreamEvent::Done { stop_reason: sr } => {
                                sr_result = sr;
                                break;
                            }
                            _ => {}
                        }
                    }

                    (
                        text,
                        thinking,
                        pending_tools,
                        usage,
                        sr_result,
                    )
                }
                ProviderResponse::Done {
                    text,
                    thinking,
                    usage,
                } => (text, thinking, Vec::new(), usage, StopReason::Stop),
            };

        // 5. Accumulate usage
        accumulated_usage.accumulate(&final_usage);

        // 6. Build assistant message content blocks
        let mut blocks: Vec<ContentBlock> = Vec::new();
        if !stream_thinking.is_empty() {
            blocks.push(ContentBlock::Thinking(stream_thinking));
        }
        if !stream_text.is_empty() {
            blocks.push(ContentBlock::Text(stream_text));
        }
        for tc in &tool_calls {
            blocks.push(ContentBlock::ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.args_buf.clone(),
            });
        }

        let assistant_id = uuid_str();
        let assistant_msg = Message {
            id: assistant_id.clone(),
            role: Role::Assistant,
            content: MessageContent::Assistant(blocks),
            usage: Some(final_usage),
            stop_reason: Some(stop_reason),
            timestamp: now_ts(),
            provider: Some(provider.id().to_string()),
            model: Some(config.model.id.clone()),
        };
        messages.push(assistant_msg);

        // 7. If not tool_use, we're done
        if stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
            return Ok(LoopResult {
                messages,
                usage: accumulated_usage,
                stop_reason,
            });
        }

        // 8. Execute tools in parallel
        let tool_args: Vec<ToolArgs> = tool_calls
            .iter()
            .map(|tc| {
                let parsed: Value =
                    serde_json::from_str(&tc.args_buf).unwrap_or(Value::Null);
                ToolArgs {
                    call_id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: parsed,
                }
            })
            .collect();

        let results = registry.run_all(&tool_args, cancel).await;

        // 9. Add tool results as ToolResult messages (one per tool call, source order)
        for (tc, result) in tool_calls.iter().zip(results) {
            let output = match result {
                Ok(o) => o,
                Err(e) => ToolOutput {
                    content: format!("[error] {e}"),
                    full_output_path: None,
                    is_error: true,
                },
            };

            let result_msg = Message {
                id: uuid_str(),
                role: Role::ToolResult,
                content: MessageContent::ToolResult {
                    tool_call_id: tc.id.clone(),
                    content: output.content,
                    is_error: output.is_error,
                },
                usage: None,
                stop_reason: None,
                timestamp: now_ts(),
                provider: None,
                model: None,
            };
            messages.push(result_msg);
        }

        // Loop back for next LLM turn
    }
}

/// Convert internal messages to provider-neutral ChatMessages.
fn convert_messages_for_provider(messages: &[Message]) -> Vec<ChatMessage> {
    let mut result = Vec::new();
    for msg in messages {
        let (role, parts) = match (&msg.role, &msg.content) {
            (Role::User, MessageContent::Text(text)) => {
                ("user".to_string(), vec![Part::Text { text: text.clone() }])
            }
            (Role::Assistant, MessageContent::Assistant(blocks)) => {
                let mut parts = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text(t) => parts.push(Part::Text { text: t.clone() }),
                        ContentBlock::Thinking(t) => {
                            parts.push(Part::Thinking { thinking: t.clone() })
                        }
                        ContentBlock::ToolCall { id, name, arguments } => {
                            parts.push(Part::ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            })
                        }
                    }
                }
                ("assistant".to_string(), parts)
            }
            (Role::ToolResult, MessageContent::ToolResult { tool_call_id, content, is_error }) => {
                ("tool".to_string(), vec![Part::ToolResult {
                    tool_call_id: tool_call_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                }])
            }
            // Fallback: map any other combination to text with the appropriate role
            (_, MessageContent::Text(text)) => {
                let r = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::ToolResult => "tool",
                };
                (r.to_string(), vec![Part::Text { text: text.clone() }])
            }
            (_, MessageContent::Assistant(blocks)) => {
                let mut parts = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text(t) => parts.push(Part::Text { text: t.clone() }),
                        ContentBlock::Thinking(t) => {
                            parts.push(Part::Thinking { thinking: t.clone() })
                        }
                        ContentBlock::ToolCall { id, name, arguments } => {
                            parts.push(Part::ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            })
                        }
                    }
                }
                let r = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::ToolResult => "tool",
                };
                (r.to_string(), parts)
            }
            (_, MessageContent::ToolResult { tool_call_id, content, is_error }) => {
                let r = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::ToolResult => "tool",
                };
                (r.to_string(), vec![Part::ToolResult {
                    tool_call_id: tool_call_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                }])
            }
            (_, MessageContent::Image { mime, data }) => {
                ("user".to_string(), vec![Part::Text {
                    text: format!("[image: {mime}, {} bytes]", data.len()),
                }])
            }
        };

        result.push(ChatMessage { role, parts });
    }
    result
}

/// Simple UUID-like string for message IDs.
fn uuid_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = t.as_nanos();
    let r: u32 = rand_u32();
    format!("{:016x}-{:04x}-{:04x}-{:04x}-{:012x}",
        nanos as u64,
        ((nanos >> 64) as u16) as u32,
        (r & 0x0FFF) | 0x4000,
        ((r >> 12) & 0x3FFF) | 0x8000,
        r as u64,
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ai::error::AiError;
    use agent_ai::stream::StreamReader;

    /// A mock provider that returns a fixed text response.
    struct MockProvider {
        response_text: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for MockProvider {
        fn id(&self) -> &str {
            "mock"
        }

        fn supports_model(&self, _model: &Model) -> bool {
            true
        }

        async fn chat(
            &self,
            _client: &agent_ai::Client,
            _req: &ProviderRequest,
        ) -> Result<ProviderResponse, AiError> {
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            let text = self.response_text.clone();
            tokio::spawn(async move {
                let _ = tx
                    .send(Ok(StreamEvent::TextDelta {
                        delta: text,
                    }))
                    .await;
                let _ = tx
                    .send(Ok(StreamEvent::Usage {
                        usage: Usage {
                            input: 10,
                            output: 5,
                            cache_read: 0,
                            cache_write: 0,
                            total: 15,
                            cost: 0.0,
                        },
                    }))
                    .await;
                let _ = tx
                    .send(Ok(StreamEvent::Done {
                        stop_reason: StopReason::Stop,
                    }))
                    .await;
            });
            Ok(ProviderResponse::Stream(StreamReader::new(rx)))
        }
    }

    #[tokio::test]
    async fn simple_text_response() {
        let provider = MockProvider {
            response_text: "Hello, world!".to_string(),
        };
        let client = agent_ai::Client::new();
        let registry = ToolRegistry::default();
        let cancel = Cancelled::new();
        let model = Model {
            provider: "mock".into(),
            id: "test".into(),
            context_window: 10000,
            max_tokens: 100,
        };

        let result = run_loop(
            &provider,
            &client,
            &registry,
            &cancel,
            LoopConfig {
                model: &model,
                system: "You are a test",
                messages: &[],
                tools: &[],
                max_tokens: 100,
                thinking: agent_ai::model::ThinkingLevel::Off,
                max_turns: 5,
                context_window: 10000,
                reserve_tokens: 1000,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.usage.input, 10);
        assert_eq!(result.usage.output, 5);
    }
}
