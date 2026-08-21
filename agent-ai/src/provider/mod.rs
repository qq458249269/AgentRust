//! Provider abstraction. Two adapters first: Anthropic SSE + OpenAI-compatible SSE.

pub mod anthropic;
pub use anthropic::AnthropicProvider;

use crate::error::AiError;
use crate::model::Model;
use crate::stream::{StreamEvent, StreamReader};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;

/// A tool exposed to the model (Anthropic `tools` / OpenAI `tools`).
#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// A message part in the provider-neutral request body.
/// agent-session converts its own message types into these.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone)]
pub struct ProviderRequest {
    pub model: Model,
    pub system: String,
    pub messages: Vec<ChatMessage>,
    pub thinking: crate::model::ThinkingLevel,
    /// max output tokens (provider `max_tokens`)
    pub max_tokens: usize,
    /// tools exposed to the model
    pub tools: Vec<ToolSpec>,
}

impl ProviderRequest {
    pub fn new(model: Model, system: String, messages: Vec<ChatMessage>) -> Self {
        Self {
            model,
            system,
            messages,
            thinking: crate::model::ThinkingLevel::Medium,
            max_tokens: 4096,
            tools: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum ProviderResponse {
    /// Streaming response. Reader yields deltas; must be consumed to completion for usage.
    Stream(StreamReader),
    /// One-shot (non-streaming providers). M1 used only as fallback.
    Done {
        text: String,
        thinking: String,
        usage: crate::model::Usage,
    },
}

/// Consumer side of a stream.
pub type StreamSender = mpsc::Sender<Result<StreamEvent, AiError>>;
pub type StreamReceiver = mpsc::Receiver<Result<StreamEvent, AiError>>;

#[async_trait]
pub trait ChatProvider: Send + Sync {
    fn id(&self) -> &str;
    fn supports_model(&self, model: &Model) -> bool;

    /// Fire the request. Transport-specific deserialization happens in stream.rs.
    /// `api_key` is resolved per-call (runtime override > stored > env).
    async fn chat(
        &self,
        client: &crate::Client,
        req: &ProviderRequest,
        api_key: &str,
    ) -> Result<ProviderResponse, AiError>;

    /// Stub for M3 task: usage extraction after stream completion.
    async fn extract_usage(&self, _tail: &[StreamEvent]) -> crate::model::Usage {
        crate::model::Usage::default()
    }
}

pub struct ProviderClient {
    providers: Vec<Box<dyn ChatProvider>>,
}

impl ProviderClient {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn register(&mut self, p: Box<dyn ChatProvider>) {
        self.providers.push(p);
    }

    pub fn provider_for(&self, model: &Model) -> Option<&dyn ChatProvider> {
        self.providers
            .iter()
            .find(|p| p.supports_model(model))
            .map(|p| p.as_ref())
    }
}

impl Default for ProviderClient {
    fn default() -> Self {
        Self::new()
    }
}
