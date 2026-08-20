//! Conversation messages (provider-neutral, zero-copy friendly).
//! agent-session maps these to/from session entries.

use agent_ai::model::Usage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    ToolResult,
}

/// A single message. Content is boxed/sliced, never copied per token.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: MessageContent,
    pub usage: Option<Usage>,
    pub stop_reason: Option<agent_ai::stream::StopReason>,
    pub timestamp: u64,
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text(String),
    /// text, thinking, tool calls interleaved (assistant output)
    Assistant(Vec<ContentBlock>),
    /// image data carried by reference; decoded only when rendered
    Image {
        mime: String,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(String),
    Thinking(String),
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

pub type UserMessage = Message;
pub type AssistantMessage = Message;
pub type ToolResultMessage = Message;
