//! RPC mode: strict LF JSONL framing over stdin/stdout.
//!
//! Protocol:
//!   Client → Agent: `{ "id": "1", "type": "prompt", "message": "hello" }`
//!   Agent → Client: `{ "id": "1", "type": "response", "command": "prompt", "success": true }`
//!   Agent → Client: `{ "type": "event", "event": { "type": "delta", "delta": "..." } }`
//!   Agent → Client: `{ "type": "event", "event": { "type": "turn_end" } }`

use crate::{client, CommonArgs};
use agent_ai::model::{Model, ThinkingLevel};
use agent_ai::provider::{ChatMessage, Part, ProviderClient, ProviderKind, ProviderRequest};
use agent_ai::stream::StreamEvent;
use agent_session::AgentSession;
use serde_json::{json, Value};
use std::io::BufRead;
use tokio::io::{AsyncWriteExt, BufWriter};

pub async fn run(_session: AgentSession, cli: &CommonArgs) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(tokio::io::stdout());

    // Resolve provider config
    let root = agent_ai::provider::read_auth_json();
    let kind_s = if cli.provider.is_empty() {
        root.get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("openai chat")
    } else {
        &cli.provider
    };
    let kind = match ProviderKind::parse(kind_s) {
        Some(k) => k,
        None => {
            send_error(&mut stdout, None, "未配置有效的服务商").await?;
            return Ok(());
        }
    };

    let api_key = kind.resolve_key(cli.api_key.as_deref()).unwrap_or_default();
    let model_id = cli
        .model
        .clone()
        .or_else(|| {
            root.get("default_model")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| match kind.id() {
            "anthropic" => "claude-sonnet-4-5".into(),
            "openai" => "gpt-4o-mini".into(),
            _ => "deepseek-chat".into(),
        });

    let model = Model {
        provider: kind.id().to_string(),
        id: model_id,
        context_window: 200_000,
        max_tokens: 4096,
    };

    let mut provider_client = ProviderClient::new();
    provider_client.setup(kind, Some(api_key), cli.base_url.clone());

    let provider = match provider_client.provider_for(&model) {
        Some(p) => p,
        None => {
            send_error(&mut stdout, None, &format!("没有可用的服务商: {}", model.id)).await?;
            return Ok(());
        }
    };

    // Track conversation history
    let mut messages: Vec<ChatMessage> = Vec::new();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let cmd: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                send_error(&mut stdout, None, &format!("无效的 JSON: {e}")).await?;
                continue;
            }
        };

        let cmd_id = cmd.get("id").cloned();
        let cmd_type = cmd.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match cmd_type {
            "prompt" => {
                let text = cmd
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Send acceptance response
                let resp = json!({
                    "id": cmd_id,
                    "type": "response",
                    "command": "prompt",
                    "success": true,
                });
                stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;

                // Build messages
                messages.push(ChatMessage {
                    role: "user".into(),
                    parts: vec![Part::Text { text }],
                });

                // Call provider
                let req = ProviderRequest {
                    model: model.clone(),
                    system: String::new(),
                    messages: messages.clone(),
                    thinking: ThinkingLevel::Off,
                    max_tokens: 4096,
                    tools: Vec::new(),
                };

                match provider.chat(client(), &req).await {
                    Ok(resp) => {
                        match resp {
                            agent_ai::provider::ProviderResponse::Stream(mut sr) => {
                                let mut text_buf = String::new();
                                let mut usage = None;
                                while let Some(ev) = sr.next().await {
                                    match ev {
                                        Ok(StreamEvent::TextDelta { delta }) => {
                                            text_buf.push_str(&delta);
                                            // Send delta event
                                            let ev = json!({
                                                "type": "event",
                                                "event": {
                                                    "type": "delta",
                                                    "delta": delta,
                                                }
                                            });
                                            stdout.write_all(
                                                serde_json::to_string(&ev)?.as_bytes(),
                                            ).await?;
                                            stdout.write_all(b"\n").await?;
                                        }
                                        Ok(StreamEvent::Usage { usage: u }) => {
                                            usage = Some(u);
                                        }
                                        Ok(StreamEvent::Done { stop_reason }) => {
                                            let ev = json!({
                                                "type": "event",
                                                "event": {
                                                    "type": "turn_end",
                                                    "stop_reason": format!("{stop_reason:?}")
                                                }
                                            });
                                            stdout.write_all(
                                                serde_json::to_string(&ev)?.as_bytes(),
                                            ).await?;
                                            stdout.write_all(b"\n").await?;
                                        }
                                        _ => {}
                                    }
                                }
                                // Add assistant response to history
                                messages.push(ChatMessage {
                                    role: "assistant".into(),
                                    parts: vec![Part::Text { text: text_buf }],
                                });
                                if let Some(u) = usage {
                                    let ev = json!({
                                        "type": "event",
                                        "event": {
                                            "type": "usage",
                                            "input": u.input,
                                            "output": u.output,
                                        }
                                    });
                                    stdout.write_all(
                                        serde_json::to_string(&ev)?.as_bytes(),
                                    ).await?;
                                    stdout.write_all(b"\n").await?;
                                }
                            }
                            agent_ai::provider::ProviderResponse::Done { text, usage: _, .. } => {
                                messages.push(ChatMessage {
                                    role: "assistant".into(),
                                    parts: vec![Part::Text { text }],
                                });
                                let ev = json!({
                                    "type": "event",
                                    "event": {
                                        "type": "turn_end",
                                        "stop_reason": "stop"
                                    }
                                });
                                stdout.write_all(serde_json::to_string(&ev)?.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                            }
                        }
                    }
                    Err(e) => {
                        send_error(&mut stdout, cmd_id.as_ref(), &format!("{e}")).await?;
                    }
                }
                stdout.flush().await?;
            }
            "abort" => {
                let resp = json!({
                    "id": cmd_id,
                    "type": "response",
                    "command": "abort",
                    "success": true,
                });
                stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            "clear" => {
                messages.clear();
                let resp = json!({
                    "id": cmd_id,
                    "type": "response",
                    "command": "clear",
                    "success": true,
                });
                stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            "ping" => {
                let resp = json!({
                    "id": cmd_id,
                    "type": "response",
                    "command": "ping",
                    "success": true,
                    "message": "pong"
                });
                stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            other => {
                tracing::warn!("未处理的 RPC 命令: {other}");
                send_error(&mut stdout, cmd_id.as_ref(), &format!("未知命令: {other}"))
                    .await?;
            }
        }
    }

    Ok(())
}

async fn send_error(
    stdout: &mut BufWriter<tokio::io::Stdout>,
    id: Option<&Value>,
    message: &str,
) -> anyhow::Result<()> {
    let resp = json!({
        "id": id,
        "type": "response",
        "success": false,
        "error": message,
    });
    stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}
