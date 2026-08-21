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

    /// Fire the request. API key + base URL are bound to the provider at setup time
    /// (see `ProviderKind::setup`); nothing transport-specific leaks into the call.
    async fn chat(
        &self,
        client: &crate::Client,
        req: &ProviderRequest,
    ) -> Result<ProviderResponse, AiError>;

    /// Stub for M3 task: usage extraction after stream completion.
    async fn extract_usage(&self, _tail: &[StreamEvent]) -> crate::model::Usage {
        crate::model::Usage::default()
    }
}

/// Registry of configured providers. Configure with `setup(kind, key, base_url)`.
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

    /// Replace (or add) the provider of `kind`. Re-running with a different
    /// key/base_url swaps the config in place — kind is the stable identity.
    pub fn setup(&mut self, kind: ProviderKind, api_key: Option<String>, base_url: Option<String>) {
        let id = kind.id();
        self.providers.retain(|p| p.id() != id);
        self.providers.push(kind.build(api_key, base_url));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_parse_and_id() {
        assert_eq!(
            ProviderKind::parse("anthropic"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(ProviderKind::parse("Claude"), Some(ProviderKind::Anthropic));
        assert_eq!(ProviderKind::parse("openai"), None);
        assert_eq!(ProviderKind::Anthropic.id(), "anthropic");
    }

    /// key resolution chain: override wins over env var
    #[test]
    fn resolve_key_prefers_override() {
        std::env::set_var("ANTHROPIC_API_KEY", "env-key");
        assert_eq!(
            ProviderKind::Anthropic
                .resolve_key(Some("cli-key"))
                .as_deref(),
            Some("cli-key")
        );
        assert_eq!(
            ProviderKind::Anthropic.resolve_key(Some("")).as_deref(),
            Some("env-key")
        );
    }

    /// setup() by kind swaps the provider in place, retaining kind as identity
    #[test]
    fn setup_swaps_by_kind() {
        let mut c = ProviderClient::new();
        assert!(c
            .provider_for(&Model {
                provider: "anthropic".into(),
                id: "m".into(),
                context_window: 1,
                max_tokens: 1,
            })
            .is_none());
        c.setup(ProviderKind::Anthropic, Some("k1".into()), None);
        assert!(c
            .provider_for(&Model {
                provider: "anthropic".into(),
                id: "m".into(),
                context_window: 1,
                max_tokens: 1,
            })
            .is_some());
        // re-setup keeps exactly one anthropic provider
        c.setup(ProviderKind::Anthropic, Some("k2".into()), None);
        assert_eq!(c.providers.len(), 1);
        assert_eq!(c.providers[0].id(), "anthropic");
    }
}

/// Provider selection: pick the kind first, then fill in url + key.
/// CLAUDE.md-friendly glossary: kind=type, key=api key, base_url=url.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
}

impl ProviderKind {
    pub fn parse(s: &str) -> Option<ProviderKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Some(ProviderKind::Anthropic),
            _ => None,
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
        }
    }

    pub fn default_base_url(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "https://api.anthropic.com/v1/messages",
        }
    }

    /// env var holding the key for this kind
    pub fn env_key_var(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    /// Key resolution chain: explicit override -> auth.json -> env var.
    pub fn resolve_key(&self, override_key: Option<&str>) -> Option<String> {
        if let Some(k) = override_key.filter(|k| !k.is_empty()) {
            return Some(k.to_string());
        }
        if let Some(k) = self.auth_entry::<String>("api_key") {
            return Some(k);
        }
        std::env::var(self.env_key_var()).ok()
    }

    /// Base URL resolution: explicit override -> auth.json -> provider default.
    pub fn resolve_base_url(&self, override_url: Option<&str>) -> String {
        if let Some(u) = override_url.filter(|u| !u.is_empty()) {
            return u.to_string();
        }
        if let Some(u) = self.auth_entry::<String>("base_url") {
            return u;
        }
        self.default_base_url().to_string()
    }

    /// Build a fully-configured provider instance from this kind + resolved creds.
    pub fn build(
        &self,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Box<dyn ChatProvider> {
        match self {
            ProviderKind::Anthropic => Box::new(AnthropicProvider::new(
                self.resolve_base_url(base_url.as_deref()),
                api_key.unwrap_or_default(),
            )),
        }
    }

    /// auth.json entry for this kind: `{ "<kind>": { "api_key": ..., "base_url": ... } }`.
    /// File location: $AGENTRUST_AUTH or ~/.agentrust/auth.json.
    fn auth_entry<T: serde::de::DeserializeOwned>(&self, field: &str) -> Option<T> {
        let path = std::env::var_os("AGENTRUST_AUTH")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
                Some(
                    std::path::Path::new(&home)
                        .join(".agentrust")
                        .join("auth.json"),
                )
            })?;
        let root: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let v = root.get(self.id())?.get(field)?;
        // String field in auth.json is a plain value (no JSON quotes); everything else
        // (arrays/objects) is deserialized as a JSON literal.
        if let Some(s) = v.as_str() {
            return serde_json::from_value(Value::String(s.to_string())).ok();
        }
        serde_json::from_value(v.clone()).ok()
    }
}
