//! Anthropic Messages API adapter (SSE streaming).
//!
//! Maps the Anthropic SSE event stream into provider-neutral `StreamEvent`s.
//! Usage aggregation: `message_start` gives input tokens; `message_delta` gives output
//! tokens + stop_reason; we emit a single `Usage` at `message_stop`.

use crate::error::AiError;
use crate::model::{Model, Usage};
use crate::provider::{ChatProvider, ProviderRequest, ProviderResponse, StreamSender};
use crate::stream::{spawn_sse_producer, StopReason, StreamEvent, StreamReader};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct AnthropicProvider {
    pub base_url: String,
    pub version: String,
    api_key: String,
}

impl AnthropicProvider {
    /// Kind-first construction: pick `Anthropic` then supply url + key.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            version: "2023-06-01".to_string(),
            api_key: api_key.into(),
        }
    }

    /// Credentials bound at setup time; never logged or serialized (field is private).
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new("https://api.anthropic.com/v1/messages", "")
    }
}

/// Request body (subset of fields we send).
#[derive(serde::Serialize)]
struct Body {
    model: String,
    max_tokens: usize,
    system: String,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
}

// ---- Anthropic SSE event JSON shapes ----

#[derive(Deserialize)]
struct EventEnvelope {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(flatten)]
    rest: Value,
}

#[async_trait]
impl ChatProvider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    fn supports_model(&self, model: &Model) -> bool {
        model.provider == "anthropic"
    }

    async fn chat(
        &self,
        client: &crate::Client,
        req: &ProviderRequest,
    ) -> Result<ProviderResponse, AiError> {
        if self.api_key.is_empty() {
            return Err(AiError::Other(
                "未配置 API 密钥；请通过 --api-key、auth.json 或环境变量设置".into(),
            ));
        }
        let messages: Vec<Value> = req.messages.iter().map(conv_message).collect();
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();

        let thinking = match req.thinking {
            crate::model::ThinkingLevel::Off => None,
            _ => Some(json!({"type": "enabled", "budget_tokens": 2048})),
        };

        let body = Body {
            model: req.model.id.clone(),
            max_tokens: req.max_tokens,
            system: req.system.clone(),
            messages,
            stream: true,
            tools,
            thinking,
        };

        let resp = client
            .inner()
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&body)?)
            .send()
            .await?;

        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp.text().await.unwrap_or_default();
            return Err(if status == 429 {
                AiError::RateLimited(None)
            } else {
                AiError::Provider { status, body }
            });
        }

        // Spawn the SSE parse+map task feeding the returned StreamReader.
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        tokio::spawn(async move {
            run_sse(resp, tx).await;
        });

        Ok(ProviderResponse::Stream(StreamReader::new(rx)))
    }
}

/// Convert a provider-neutral ChatMessage into an Anthropic content array.
fn conv_message(m: &crate::provider::ChatMessage) -> Value {
    let mut content: Vec<Value> = Vec::new();
    for part in &m.parts {
        match part {
            crate::provider::Part::Text { text } => {
                content.push(json!({"type": "text", "text": text}));
            }
            crate::provider::Part::Thinking { thinking } => {
                content.push(json!({"type": "thinking", "thinking": thinking}));
            }
            crate::provider::Part::ToolCall {
                id,
                name,
                arguments,
            } => {
                let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": args,
                }));
            }
            crate::provider::Part::ToolResult {
                tool_call_id,
                content: text,
                is_error,
            } => {
                content.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": text,
                    "is_error": is_error,
                }));
            }
        }
    }
    json!({"role": m.role, "content": content})
}

/// Drain a reqwest bytes stream, split into SSE `data:` lines, map to StreamEvents.
async fn run_sse(stream: reqwest::Response, tx: StreamSender) {
    let mut raw_rx = spawn_sse_producer(stream);

    let mut parser = AnthropicParser::default();
    while let Some(payload) = raw_rx.recv().await {
        match payload {
            Ok(line) => {
                tracing::trace!("sse raw line");
                if let Err(e) = parser.feed(&line, &tx).await {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
}

/// Aggregates Anthropic event JSON into StreamEvents.
#[derive(Default)]
struct AnthropicParser {
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    last_stop: StopReason,
}

impl AnthropicParser {
    async fn feed(&mut self, line: &str, tx: &StreamSender) -> Result<(), AiError> {
        if line.trim().is_empty() {
            return Ok(());
        }
        let ev: EventEnvelope = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => return Ok(()), // keep-alive / non-JSON
        };
        match ev.event_type.as_str() {
            "message_start" => {
                if let Some(usage) = ev.rest.get("message").and_then(|m| m.get("usage")) {
                    self.input_tokens = usage
                        .get("input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    self.cache_read = usage
                        .get("cache_read_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    self.cache_write = usage
                        .get("cache_creation_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
            }
            "content_block_start" => {
                if let Some(cb) = ev.rest.get("content_block") {
                    if cb.get("type").and_then(Value::as_str) == Some("tool_use") {
                        tx.send(Ok(StreamEvent::ToolCallStarted {
                            id: cb
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            name: cb
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        }))
                        .await
                        .map_err(|_| AiError::Other("流已关闭".into()))?;
                    }
                }
            }
            "content_block_delta" => {
                if let Some(delta) = ev.rest.get("delta") {
                    match delta.get("type").and_then(Value::as_str) {
                        Some("text_delta") => {
                            let t = delta
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            tx.send(Ok(StreamEvent::TextDelta { delta: t }))
                                .await
                                .map_err(|_| AiError::Other("流已关闭".into()))?;
                        }
                        Some("thinking_delta") => {
                            let t = delta
                                .get("thinking")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            tx.send(Ok(StreamEvent::ThinkingDelta { delta: t }))
                                .await
                                .map_err(|_| AiError::Other("流已关闭".into()))?;
                        }
                        Some("input_json_delta") => {
                            let id = ev
                                .rest
                                .get("index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0)
                                .to_string();
                            let d = delta
                                .get("partial_json")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            tx.send(Ok(StreamEvent::ToolCallArgsDelta { id, delta: d }))
                                .await
                                .map_err(|_| AiError::Other("流已关闭".into()))?;
                        }
                        _ => {}
                    }
                }
            }
            "message_delta" => {
                if let Some(usage) = ev.rest.get("usage") {
                    self.output_tokens = usage
                        .get("output_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
                let stop = ev
                    .rest
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.last_stop = match stop {
                    "max_tokens" => StopReason::Length,
                    "tool_use" => StopReason::ToolUse,
                    "stop_sequence" | "end_turn" => StopReason::Stop,
                    _ => StopReason::Stop,
                };
            }
            "message_stop" => {
                let usage = Usage {
                    input: self.input_tokens,
                    output: self.output_tokens,
                    cache_read: self.cache_read,
                    cache_write: self.cache_write,
                    total: self.input_tokens
                        + self.output_tokens
                        + self.cache_read
                        + self.cache_write,
                    cost: 0.0,
                };
                tx.send(Ok(StreamEvent::Usage { usage }))
                    .await
                    .map_err(|_| AiError::Other("流已关闭".into()))?;
                let sr = self.last_stop;
                tx.send(Ok(StreamEvent::Done { stop_reason: sr }))
                    .await
                    .map_err(|_| AiError::Other("流已关闭".into()))?;
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// maps the full Anthropic event sequence (as emitted by tests/anthropic_mock.py)
    /// into TextDelta / Usage / Done with correct aggregation
    #[tokio::test]
    async fn maps_anthropic_event_sequence() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut p = AnthropicParser::default();
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":0}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" World"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        for e in events {
            p.feed(e, &tx).await.unwrap();
        }
        drop(tx);

        let mut got = Vec::new();
        while let Some(ev) = rx.recv().await {
            got.push(ev.unwrap());
        }
        assert!(
            matches!(got[0], StreamEvent::TextDelta { ref delta } if delta == "Hello"),
            "{got:?}"
        );
        assert!(matches!(got[1], StreamEvent::TextDelta { ref delta } if delta == " World"));
        match &got[2] {
            StreamEvent::Usage { usage } => assert_eq!(
                (usage.input, usage.output, usage.cache_write, usage.total),
                (10, 5, 2, 17)
            ),
            other => panic!("expected Usage, got {other:?}"),
        }
        assert!(
            matches!(
                got[3],
                StreamEvent::Done {
                    stop_reason: StopReason::Stop
                }
            ),
            "{got:?}"
        );
    }

    /// tool_use blocks produce ToolCallStart/ArgsDelta/Done
    #[tokio::test]
    async fn maps_tool_use_events() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut p = AnthropicParser::default();
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"ls\"}"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        for e in events {
            p.feed(e, &tx).await.unwrap();
        }
        drop(tx);
        let mut got = Vec::new();
        while let Some(ev) = rx.recv().await {
            got.push(ev.unwrap());
        }
        assert!(
            matches!(got[0], StreamEvent::ToolCallStarted { ref name, .. } if name == "bash"),
            "{got:?}"
        );
        assert!(
            matches!(got[1], StreamEvent::ToolCallArgsDelta { .. }),
            "{got:?}"
        );
        assert!(
            matches!(got[2], StreamEvent::Usage { usage } if usage.output == 3),
            "{got:?}"
        );
        assert!(matches!(
            got[3],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse
            }
        ));
    }
}
