//! Provider abstraction. Two adapters first: Anthropic SSE + OpenAI-compatible SSE.

pub mod anthropic;
pub use anthropic::AnthropicProvider;

pub mod openai;
pub use openai::{OpenAiChatProvider, OpenAiResponsesProvider, PendingProvider};

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
            Some(ProviderKind {
                vendor: "anthropic",
                api: ApiVariant::Messages,
            })
        );
        assert_eq!(
            ProviderKind::parse("Claude"),
            Some(ProviderKind {
                vendor: "anthropic",
                api: ApiVariant::Messages,
            })
        );
        assert_eq!(
            ProviderKind::parse("openai chat"),
            Some(ProviderKind {
                vendor: "openai",
                api: ApiVariant::Chat,
            })
        );
        assert_eq!(
            ProviderKind::parse("openai chat-resp"),
            Some(ProviderKind {
                vendor: "openai",
                api: ApiVariant::Responses,
            })
        );
        assert_eq!(
            ProviderKind::parse("openai responses").unwrap().api,
            ApiVariant::Responses
        );
        assert_eq!(
            ProviderKind::parse("deepseek chat").unwrap().vendor,
            "deepseek"
        );
        assert_eq!(ProviderKind::parse("bogus"), None);
        assert_eq!(ProviderKind::parse("openai chat extra"), None);
        assert_eq!(
            ProviderKind::parse("anthropic").unwrap().display(),
            "anthropic messages"
        );
        assert_eq!(
            ProviderKind::parse("openai").unwrap().display(),
            "openai chat"
        );
    }

    /// key resolution chain: override wins over env var
    #[test]
    fn resolve_key_prefers_override() {
        std::env::set_var("ANTHROPIC_API_KEY", "env-key");
        let k = ProviderKind::parse("anthropic").unwrap();
        assert_eq!(k.resolve_key(Some("cli-key")).as_deref(), Some("cli-key"));
        assert_eq!(k.resolve_key(Some("")).as_deref(), Some("env-key"));
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
        let kind = ProviderKind::parse("anthropic").unwrap();
        c.setup(kind, Some("k1".into()), None);
        assert!(c
            .provider_for(&Model {
                provider: "anthropic".into(),
                id: "m".into(),
                context_window: 1,
                max_tokens: 1,
            })
            .is_some());
        // re-setup keeps exactly one anthropic provider
        c.setup(kind, Some("k2".into()), None);
        assert_eq!(c.providers.len(), 1);
        assert_eq!(c.providers[0].id(), "anthropic");
    }
}

/// API surface shape of a provider. `openai chat` vs `openai responses` differ enough
/// (endpoint, stream schema, auth header) that the variant is part of the identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiVariant {
    /// OpenAI-compatible chat completions (`/chat/completions`, SSE `choices[].delta`)
    Chat,
    /// OpenAI Responses API (`/responses`, SSE `response.output_text.delta`)
    Responses,
    /// Anthropic Messages API (`/v1/messages`, SSE `content_block_delta`)
    Messages,
}

/// Location of the local settings/credentials file: $AGENTRUST_AUTH or ~/.agentrust/auth.json.
pub fn auth_file_path() -> Option<std::path::PathBuf> {
    std::env::var_os("AGENTRUST_AUTH")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
            Some(
                std::path::Path::new(&home)
                    .join(".agentrust")
                    .join("auth.json"),
            )
        })
}

/// Read the raw auth.json as a JSON object (empty object if absent/invalid).
pub fn read_auth_json() -> Value {
    auth_file_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null)
}

/// Merge `patch` into the cached auth.json and write it back (creates dirs as needed).
/// Returns the file path written, or None if no home dir is resolvable.
pub fn write_auth_json(
    patch: &Value,
) -> std::result::Result<Option<std::path::PathBuf>, crate::AiError> {
    use std::io::Write;
    let Some(path) = auth_file_path() else {
        return Ok(None);
    };
    let mut root = read_auth_json();
    if !root.is_object() {
        root = Value::Object(Default::default());
    }
    let obj = root.as_object_mut().expect("object");
    for (k, v) in patch.as_object().expect("patch must be object") {
        obj.insert(k.clone(), v.clone());
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut f = std::fs::File::create(&path)?;
    f.write_all(serde_json::to_string_pretty(&root)?.as_bytes())?;
    Ok(Some(path))
}

/// Provider selection: pick the vendor + api variant first, then fill in url + key.
/// CLAUDE.md-friendly glossary: kind=type, key=api key, base_url=url.
/// Strings look like `anthropic messages`, `openai chat`, `openai responses`,
/// `deepseek chat`. The variant defaults per vendor when omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderKind {
    pub vendor: &'static str,
    pub api: ApiVariant,
}

impl ProviderKind {
    pub fn parse(s: &str) -> Option<ProviderKind> {
        let mut it = s.split_whitespace();
        let vendor = it.next()?.to_ascii_lowercase();
        let api = it.next().map(|a| a.to_ascii_lowercase());
        if it.next().is_some() {
            return None; // too many words
        }
        match vendor.as_str() {
            "anthropic" | "claude" => match api.as_deref() {
                None | Some("messages") => Some(ProviderKind {
                    vendor: "anthropic",
                    api: ApiVariant::Messages,
                }),
                _ => None,
            },
            "openai" => match api.as_deref() {
                None | Some("chat") | Some("chat-completions") | Some("chatcompletions") => {
                    Some(ProviderKind {
                        vendor: "openai",
                        api: ApiVariant::Chat,
                    })
                }
                Some("responses") | Some("resp") | Some("chat-resp") | Some("chatresp") => {
                    Some(ProviderKind {
                        vendor: "openai",
                        api: ApiVariant::Responses,
                    })
                }
                _ => None,
            },
            "deepseek" => match api.as_deref() {
                None | Some("chat") | Some("chat-completions") | Some("chatcompletions") => {
                    Some(ProviderKind {
                        vendor: "deepseek",
                        api: ApiVariant::Chat,
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// stable vendor id (matches `Model.provider` and auth.json section)
    pub fn id(&self) -> &'static str {
        self.vendor
    }

    /// base URL prefix for this kind (scheme + host + /v1 when the vendor uses it);
    /// the per-API path is appended by [`Self::join_api_path`].
    pub fn default_base_url(&self) -> &'static str {
        match (self.vendor, self.api) {
            ("openai", ApiVariant::Chat) | ("openai", ApiVariant::Responses) => {
                "https://api.openai.com/v1"
            }
            ("deepseek", ApiVariant::Chat) => "https://api.deepseek.com/v1",
            ("anthropic", ApiVariant::Messages) => "https://api.anthropic.com/v1",
            (v, _) => unreachable!("unknown vendor {v}"),
        }
    }

    /// API path suffix for this kind, e.g. `/chat/completions`.
    pub fn api_path(&self) -> &'static str {
        match (self.vendor, self.api) {
            ("openai", ApiVariant::Chat) | ("deepseek", ApiVariant::Chat) => "/chat/completions",
            ("openai", ApiVariant::Responses) => "/responses",
            ("anthropic", ApiVariant::Messages) => "/messages",
            (v, _) => unreachable!("unknown vendor {v}"),
        }
    }

    /// Join a base URL (`https://api.example.com/v1`) with the API path.
    /// Configuration may already include the full path (`…/v1/chat/completions`);
    /// we then leave it untouched.
    pub fn join_api_path(&self, base: &str) -> String {
        let path = self.api_path();
        if base.ends_with(path) {
            return base.to_string();
        }
        let trimmed = base.trim_end_matches('/');
        if trimmed.is_empty() {
            return self.default_base_url().to_string();
        }
        format!("{trimmed}{path}")
    }

    /// human form, e.g. `openai chat`
    pub fn display(&self) -> String {
        match self.api {
            ApiVariant::Messages => format!("{} messages", self.vendor),
            ApiVariant::Responses => format!("{} responses", self.vendor),
            ApiVariant::Chat => format!("{} chat", self.vendor),
        }
    }

    /// env var holding the key for this kind
    pub fn env_key_var(&self) -> &'static str {
        match self.vendor {
            "anthropic" => "ANTHROPIC_API_KEY",
            "openai" => "OPENAI_API_KEY",
            "deepseek" => "DEEPSEEK_API_KEY",
            v => unreachable!("unknown vendor {v}"),
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
        let url = self.resolve_base_url(base_url.as_deref());
        let url = self.join_api_path(&url);
        let key = api_key.unwrap_or_default();
        match (self.vendor, self.api) {
            ("anthropic", ApiVariant::Messages) => Box::new(AnthropicProvider::new(url, key)),
            ("openai", ApiVariant::Chat) | ("deepseek", ApiVariant::Chat) => {
                Box::new(OpenAiChatProvider::new(url, key))
            }
            (v, a) => {
                // vendor+api registered but not yet implemented; return a stub that errors
                Box::new(PendingProvider { vendor: v, api: a })
            }
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

#[cfg(test)]
mod auth_tests {
    use super::*;

    fn tmp_auth() -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("agentrust_auth_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn write_then_read_roundtrip() {
        let p = tmp_auth();
        std::env::set_var("AGENTRUST_AUTH", &p);
        let patch = serde_json::json!({
            "provider": "openai chat",
            "openai": {"api_key": "sk-test", "base_url": "http://127.0.0.1:9"},
            "default_model": "gpt-4o-mini"
        });
        let written = write_auth_json(&patch).unwrap();
        assert_eq!(written.as_deref(), Some(p.as_path()));

        let kind = ProviderKind::parse("openai chat").unwrap();
        assert_eq!(kind.resolve_key(None).as_deref(), Some("sk-test"));
        assert_eq!(
            kind.resolve_base_url(None),
            "http://127.0.0.1:9".to_string()
        );

        let root = read_auth_json();
        assert_eq!(root["default_model"], "gpt-4o-mini");
        let _ = std::fs::remove_file(&p);
        std::env::remove_var("AGENTRUST_AUTH");
    }

    #[test]
    fn join_api_path_handles_prefix_and_full() {
        let chat = ProviderKind::parse("openai chat").unwrap();
        // prefix only -> append path
        assert_eq!(
            chat.join_api_path("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        // full path already present -> unchanged
        assert_eq!(
            chat.join_api_path("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        // trailing slash tolerated
        assert_eq!(
            chat.join_api_path("http://127.0.0.1:18081/v1/"),
            "http://127.0.0.1:18081/v1/chat/completions"
        );
        let messages = ProviderKind::parse("anthropic").unwrap();
        assert_eq!(
            messages.join_api_path("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        let responses = ProviderKind::parse("openai responses").unwrap();
        assert_eq!(
            responses.join_api_path("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
    }
}
