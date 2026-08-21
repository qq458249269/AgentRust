//! OpenAI-compatible adapters.
//!
//! Two API surface variants share the codebase:
//! - `Chat` (`/chat/completions`): SSE events carry `choices[].delta`
//! - `Responses` (`/responses`): SSE events carry `response.output_text.delta`
//!
//! Both authenticate with `Authorization: Bearer <key>` and stream tokens the same way.
//! `deepseek chat` is the same Chat adapter against a different base URL.

use crate::error::AiError;
use crate::model::Usage;
use crate::provider::{ApiVariant, ChatProvider, ProviderRequest, ProviderResponse, StreamSender};
use crate::stream::{spawn_sse_producer, StopReason, StreamEvent, StreamReader};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct OpenAiChatProvider {
    pub base_url: String,
    api_key: String,
}

impl OpenAiChatProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }
}

pub struct OpenAiResponsesProvider {
    pub base_url: String,
    /// reserved until the Responses adapter is implemented
    #[allow(dead_code)]
    api_key: String,
}

impl OpenAiResponsesProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }
}

/// Registered vendor+api that is not implemented yet; fails loudly at call time.
pub struct PendingProvider {
    pub vendor: &'static str,
    pub api: ApiVariant,
}

#[async_trait]
impl ChatProvider for OpenAiChatProvider {
    fn id(&self) -> &str {
        // same vendor, so id() matches vendor alone; setup() dedupes on vendor
        "openai"
    }

    fn supports_model(&self, model: &Model) -> bool {
        model.provider == "openai" || model.provider == "deepseek"
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
        let body = chat_body(req)?;
        let resp = client
            .inner()
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let (tx, rx) = tokio::sync::mpsc::channel(256);
        tokio::spawn(async move {
            tracing::info!("OpenAI SSE 任务启动");
            let mut raw_rx = spawn_sse_producer(resp);
            let mut parser = ChatCompletionsParser::default();
            let mut line_count = 0u32;
            while let Some(payload) = raw_rx.recv().await {
                match payload {
                    Ok(line) => {
                        line_count += 1;
                        let preview = if line.len() > 200 { &line[..200] } else { &line };
                        tracing::info!("OpenAI SSE raw #{line_count}: {preview}");
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
            tracing::info!("OpenAI SSE raw lines 总计: {line_count}");
        });

        Ok(ProviderResponse::Stream(StreamReader::new(rx)))
    }
}

#[async_trait]
impl ChatProvider for OpenAiResponsesProvider {
    fn id(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &Model) -> bool {
        model.provider == "openai"
    }

    async fn chat(
        &self,
        _client: &crate::Client,
        _req: &ProviderRequest,
    ) -> Result<ProviderResponse, AiError> {
        Err(AiError::Other(
            "OpenAI Responses API 尚未实现；请使用 'openai chat'".into(),
        ))
    }
}

#[async_trait]
impl ChatProvider for PendingProvider {
    fn id(&self) -> &str {
        self.vendor
    }

    fn supports_model(&self, model: &Model) -> bool {
        model.provider == self.vendor
    }

    async fn chat(
        &self,
        _client: &crate::Client,
        _req: &ProviderRequest,
    ) -> Result<ProviderResponse, AiError> {
        Err(AiError::Other(format!(
            "服务商 '{} {}' 已注册但尚未实现",
            self.vendor,
            match self.api {
                ApiVariant::Chat => "chat",
                ApiVariant::Responses => "responses",
                ApiVariant::Messages => "messages",
            }
        )))
    }
}

use crate::model::Model;

/// Build the `/chat/completions` request body from a neutral request.
fn chat_body(req: &ProviderRequest) -> Result<Value, AiError> {
    let mut messages: Vec<Value> = Vec::new();
    for m in &req.messages {
        let mut content: Vec<String> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut tool_call_id: Option<&str> = None;
        let mut is_tool_result = false;
        let mut is_error = false;
        for part in &m.parts {
            match part {
                crate::provider::Part::Text { text }
                | crate::provider::Part::Thinking { thinking: text } => {
                    content.push(text.clone());
                }
                crate::provider::Part::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments },
                    }));
                }
                crate::provider::Part::ToolResult {
                    tool_call_id: tid,
                    content: text,
                    is_error: err,
                } => {
                    tool_call_id = Some(tid);
                    is_tool_result = true;
                    is_error = *err;
                    content.push(text.clone());
                }
            }
        }
        if is_tool_result {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call_id.unwrap_or(""),
                "content": if is_error { format!("[error]\n{}", content.join("\n")) } else { content.join("\n") },
            }));
        } else if !tool_calls.is_empty() {
            messages.push(json!({
                "role": "assistant",
                "content": content.join("\n"),
                "tool_calls": tool_calls,
            }));
        } else {
            messages.push(json!({ "role": m.role, "content": content.join("\n") }));
        }
    }

    let tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.input_schema },
            })
        })
        .collect();

    let mut body = json!({
        "model": req.model.id,
        "messages": messages,
        "stream": true,
        "max_tokens": req.max_tokens,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    Ok(body)
}

/// Aggregates chat.completion.chunk events into StreamEvents.
#[derive(Default)]
struct ChatCompletionsParser {
    input: u64,
    output: u64,
    total: u64,
    stop: StopReason,
    saw_stop: bool,
    /// accumulate tool call fragments per index: (id, name, args)
    tool_calls: Vec<(String, String, String)>,
    current_idx: Option<usize>,
}

#[derive(serde::Deserialize)]
struct Chunk {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(serde::Deserialize)]
struct Choice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolDelta>>,
}

#[derive(serde::Deserialize)]
struct ToolDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FnDelta>,
}

#[derive(serde::Deserialize)]
struct FnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

impl ChatCompletionsParser {
    async fn feed(&mut self, line: &str, tx: &StreamSender) -> Result<(), AiError> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        if line == "[DONE]" {
            if !self.saw_stop {
                let sr = self.stop;
                tx.send(Ok(StreamEvent::Done { stop_reason: sr }))
                    .await
                    .map_err(|_| AiError::Other("流已关闭".into()))?;
            }
            return Ok(());
        }
        let chunk: Chunk = match serde_json::from_str(line) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!("OpenAI chunk 解析失败: {e}");
                return Ok(());
            }
        };
        tracing::debug!("OpenAI chunk: choices={}, finish_reason={:?}", chunk.choices.len(), chunk.choices.first().and_then(|c| c.finish_reason.as_deref()));

        if let Some(u) = chunk.usage {
            self.input = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
            self.output = u
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.total = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(0);
        }

        for choice in &chunk.choices {
            let has_content = choice.delta.content.as_ref().map(|c| !c.is_empty()).unwrap_or(false);
            tracing::debug!("OpenAI choice: has_content={has_content}, finish_reason={:?}", choice.finish_reason);
            if let Some(text) = &choice.delta.content {
                if !text.is_empty() {
                    tracing::info!("OpenAI 发送 TextDelta: {} 字节", text.len());
                    tx.send(Ok(StreamEvent::TextDelta {
                        delta: text.clone(),
                    }))
                    .await
                    .map_err(|_| AiError::Other("流已关闭".into()))?;
                }
            }
            if let Some(calls) = &choice.delta.tool_calls {
                for tc in calls {
                    let idx = tc.index;
                    while self.tool_calls.len() <= idx {
                        self.tool_calls
                            .push((String::new(), String::new(), String::new()));
                    }
                    let slot = &mut self.tool_calls[idx];
                    if let Some(id) = &tc.id {
                        slot.0 = id.clone();
                    }
                    if let Some(f) = &tc.function {
                        if let Some(name) = &f.name {
                            slot.1 = name.clone();
                        }
                        if let Some(args_delta) = &f.arguments {
                            if args_delta.is_empty() {
                                continue; // first empty args fragment carries no data
                            }
                            if self.current_idx != Some(idx) {
                                // new args segment: emit start for this call
                                tx.send(Ok(StreamEvent::ToolCallStarted {
                                    id: slot.0.clone(),
                                    name: slot.1.clone(),
                                }))
                                .await
                                .map_err(|_| AiError::Other("流已关闭".into()))?;
                                self.current_idx = Some(idx);
                            }
                            tx.send(Ok(StreamEvent::ToolCallArgsDelta {
                                id: slot.0.clone(),
                                delta: args_delta.clone(),
                            }))
                            .await
                            .map_err(|_| AiError::Other("流已关闭".into()))?;
                        }
                    }
                }
            }
            if let Some(reason) = &choice.finish_reason {
                self.stop = match reason.as_str() {
                    "length" => StopReason::Length,
                    "tool_calls" => StopReason::ToolUse,
                    "content_filter" | "stop" => StopReason::Stop,
                    _ => StopReason::Stop,
                };
                self.saw_stop = true;
                // emit Usage before Done; usage may be in the same chunk
                if self.input + self.output + self.total > 0 {
                    let usage = Usage {
                        input: self.input,
                        output: self.output,
                        cache_read: 0,
                        cache_write: 0,
                        total: self.total,
                        cost: 0.0,
                    };
                    tracing::info!("OpenAI 发送 Usage: in={} out={}", usage.input, usage.output);
                    tx.send(Ok(StreamEvent::Usage { usage }))
                        .await
                        .map_err(|_| AiError::Other("流已关闭".into()))?;
                }
                let sr = self.stop;
                tracing::info!("OpenAI 发送 Done: {sr:?}");
                tx.send(Ok(StreamEvent::Done { stop_reason: sr }))
                    .await
                    .map_err(|_| AiError::Other("流已关闭".into()))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn maps_chat_completion_chunks() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut p = ChatCompletionsParser::default();
        let lines = [
            r#"{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#,
            "data: [DONE]",
            "",
        ];
        for l in lines {
            p.feed(l, &tx).await.unwrap();
        }
        drop(tx);
        let mut got = Vec::new();
        while let Some(ev) = rx.recv().await {
            got.push(ev.unwrap());
        }
        assert!(matches!(got[0], StreamEvent::TextDelta { ref delta } if delta == "Hello"));
        assert!(matches!(got[1], StreamEvent::TextDelta { ref delta } if delta == " world"));
        assert!(matches!(
            got[2],
            StreamEvent::Usage { usage } if usage.input == 4 && usage.output == 2 && usage.total == 6
        ));
        assert!(matches!(
            got[3],
            StreamEvent::Done {
                stop_reason: StopReason::Stop
            }
        ));
        assert_eq!(got.len(), 4, "{got:?}");
    }

    #[tokio::test]
    async fn maps_tool_calls() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut p = ChatCompletionsParser::default();
        let lines = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":null}]}"#,
            "data: [DONE]",
        ];
        for l in lines {
            p.feed(l, &tx).await.unwrap();
        }
        drop(tx);
        let mut got = Vec::new();
        while let Some(ev) = rx.recv().await {
            got.push(ev.unwrap());
        }
        assert!(matches!(got[0], StreamEvent::ToolCallStarted { ref name, .. } if name == "bash"));
        assert!(
            matches!(got[1], StreamEvent::ToolCallArgsDelta { ref delta, .. } if delta == "{\"cmd\":")
        );
        assert!(matches!(got[2], StreamEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(
            got[3],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse
            }
        ));
    }
}
