//! Compaction (pi parity). Threshold + reserve; keepRecentTokens cut point; split-turn merge.
//!
//! When context_tokens > context_window - reserve, older messages are summarized
//! into a single compaction entry. Recent messages (keepRecentTokens) are preserved.

use crate::bus::CompactionReason;
use agent_ai::model::Model;
use agent_ai::provider::{ChatMessage, Part, ProviderRequest};

use agent_core::messages::{ContentBlock, Message, MessageContent};


#[derive(Debug, Clone, Copy)]
pub struct CompactionSettings {
    pub enabled: bool,
    /// context_window - reserve triggers compaction
    pub reserve_tokens: u64,
    /// newest tokens kept uncompressed
    pub keep_recent_tokens: u64,
    /// tool result serialization cap for summaries
    pub serialize_tool_result_max_chars: usize,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            serialize_tool_result_max_chars: 2_000,
        }
    }
}

impl CompactionSettings {
    pub fn triggered(&self, context_tokens: u64, context_window: u64) -> Option<CompactionReason> {
        if !self.enabled {
            return None;
        }
        if context_tokens > context_window.saturating_sub(self.reserve_tokens) {
            return Some(CompactionReason::Threshold);
        }
        None
    }
}

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// The summary text produced by the LLM.
    pub summary: String,
    /// Messages to keep (recent tail that was not summarized).
    pub retained_messages: Vec<Message>,
    /// Estimated tokens before compaction.
    pub tokens_before: u64,
}

/// Summarize older messages via LLM call, preserving recent tail.
///
/// `messages` - full message list; older ones will be summarized.
/// `keep_count` - number of recent messages to preserve (by count, not tokens).
/// `model` - model to use for summarization.
/// `provider` - provider client.
pub async fn compact_messages(
    messages: &[Message],
    keep_count: usize,
    model: &Model,
    provider: &dyn agent_ai::provider::ChatProvider,
    client: &agent_ai::Client,
) -> Result<CompactionResult, agent_core::CoreError> {
    if messages.len() <= keep_count {
        // Nothing to compact
        return Ok(CompactionResult {
            summary: String::new(),
            retained_messages: messages.to_vec(),
            tokens_before: estimate_tokens(messages),
        });
    }

    // Split: older messages to summarize, recent tail to keep
    let (to_summarize, retained) = messages.split_at(messages.len() - keep_count);

    // Serialize older messages to text for summarization
    let serialized = serialize_for_summary(to_summarize);

    // Call LLM for summary
    let summary_prompt = format!(
        "请简洁地总结以下对话历史。\
         保留关键事实、决策、文件路径和后续对话需要的上下文。\
         只输出摘要，不要前言。\n\n{serialized}"
    );

    let req = ProviderRequest {
        model: model.clone(),
        system: "你是一个对话总结器。请简洁地总结对话历史，\
                  保留关键事实、决策、文件路径和重要上下文。"
            .to_string(),
        messages: vec![ChatMessage {
            role: "user".into(),
            parts: vec![Part::Text {
                text: summary_prompt,
            }],
        }],
        thinking: agent_ai::model::ThinkingLevel::Off,
        max_tokens: 2048,
        tools: Vec::new(),
    };

    let resp = provider.chat(client, &req).await?;

    let summary = match resp {
        agent_ai::provider::ProviderResponse::Stream(mut sr) => {
            let mut text = String::new();
            while let Some(ev) = sr.next().await {
                match ev? {
                    agent_ai::stream::StreamEvent::TextDelta { delta } => text.push_str(&delta),
                    agent_ai::stream::StreamEvent::Done { .. } => break,
                    _ => {}
                }
            }
            text
        }
        agent_ai::provider::ProviderResponse::Done { text, .. } => text,
    };

    Ok(CompactionResult {
        summary,
        retained_messages: retained.to_vec(),
        tokens_before: estimate_tokens(messages),
    })
}

/// Serialize messages to text for LLM summarization.
fn serialize_for_summary(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let role_str = match msg.role {
            agent_core::messages::Role::User => "User",
            agent_core::messages::Role::Assistant => "Assistant",
            agent_core::messages::Role::ToolResult => "ToolResult",
        };
        let content = match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Assistant(blocks) => {
                let mut s = String::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text(t) => s.push_str(t),
                        ContentBlock::Thinking(t) => s.push_str(&format!("[thinking: {t}]")),
                        ContentBlock::ToolCall { name, arguments, .. } => {
                            s.push_str(&format!("[tool call: {name}({arguments})]"));
                        }
                    }
                }
                s
            }
            MessageContent::ToolResult { content, is_error, tool_call_id } => {
                let prefix = if *is_error { "[ERROR] " } else { "" };
                // Truncate long tool results for summary
                let truncated = if content.len() > 2000 {
                    format!("{}...", &content[..2000])
                } else {
                    content.clone()
                };
                format!("{prefix}[tool_result for {tool_call_id}]: {truncated}")
            }
            MessageContent::Image { mime, data } => {
                format!("[image: {mime}, {} bytes]", data.len())
            }
        };
        parts.push(format!("[{role_str}]: {content}"));
    }
    parts.join("\n\n")
}

/// Rough token estimation (4 chars ≈ 1 token).
fn estimate_tokens(messages: &[Message]) -> u64 {
    let chars: usize = messages
        .iter()
        .map(|m| match &m.content {
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
        })
        .sum();
    (chars as u64) / 4
}
