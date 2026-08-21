//! JSON mode: forward all session events to stdout as JSON lines.
//!
//! Each event is serialized as a single JSON line (no framing beyond LF).
//! Subscribe to the event bus and stream to stdout.

use crate::{client, CommonArgs};
use agent_ai::model::{Model, ThinkingLevel};
use agent_ai::provider::{ChatMessage, Part, ProviderClient, ProviderKind, ProviderRequest};
use agent_ai::stream::StreamEvent;
use agent_session::AgentSession;
use serde_json::json;

pub async fn run(_session: AgentSession, cli: &CommonArgs) -> anyhow::Result<()> {
    // Read provider config from auth.json
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
            tracing::error!("未配置有效的服务商；请使用 --provider 或 /settings 设置");
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
            tracing::error!("没有为模型 {} 注册服务商", model.id);
            return Ok(());
        }
    };

    // Emit a session_start event
    let start_ev = json!({
        "type": "session_start",
        "model": model.id,
        "provider": kind.display(),
    });
    println!("{start_ev}");

    // Read prompts from stdin (one per line) and process them
    use std::io::BufRead;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        // Emit turn_start
        let turn_ev = json!({ "type": "turn_start", "message": trimmed });
        println!("{turn_ev}");

        // Call provider
        let req = ProviderRequest {
            model: model.clone(),
            system: String::new(),
            messages: vec![ChatMessage {
                role: "user".into(),
                parts: vec![Part::Text { text: trimmed }],
            }],
            thinking: ThinkingLevel::Off,
            max_tokens: 4096,
            tools: Vec::new(),
        };

        match provider.chat(client(), &req).await {
            Ok(resp) => match resp {
                agent_ai::provider::ProviderResponse::Stream(mut sr) => {
                    let mut text_buf = String::new();
                    let mut usage = None;
                    while let Some(ev) = sr.next().await {
                        match ev {
                            Ok(StreamEvent::TextDelta { delta }) => {
                                let ev = json!({
                                    "type": "delta",
                                    "delta": delta,
                                });
                                println!("{ev}");
                                text_buf.push_str(&delta);
                            }
                            Ok(StreamEvent::Usage { usage: u }) => {
                                usage = Some(u);
                            }
                            Ok(StreamEvent::Done { stop_reason }) => {
                                let ev = json!({
                                    "type": "turn_end",
                                    "stop_reason": format!("{stop_reason:?}"),
                                });
                                println!("{ev}");
                            }
                            Err(e) => {
                                let ev = json!({
                                    "type": "error",
                                    "message": format!("{e}"),
                                });
                                println!("{ev}");
                            }
                            _ => {}
                        }
                    }
                    if let Some(u) = usage {
                        let ev = json!({
                            "type": "usage",
                            "input": u.input,
                            "output": u.output,
                            "cache_read": u.cache_read,
                            "cache_write": u.cache_write,
                            "total": u.total,
                        });
                        println!("{ev}");
                    }
                }
                agent_ai::provider::ProviderResponse::Done { text, usage, .. } => {
                    let ev = json!({
                        "type": "delta",
                        "delta": text,
                    });
                    println!("{ev}");
                    let ev = json!({
                        "type": "usage",
                        "input": usage.input,
                        "output": usage.output,
                        "cache_read": usage.cache_read,
                        "cache_write": usage.cache_write,
                        "total": usage.total,
                    });
                    println!("{ev}");
                    let ev = json!({ "type": "turn_end", "stop_reason": "stop" });
                    println!("{ev}");
                }
            },
            Err(e) => {
                let ev = json!({
                    "type": "error",
                    "message": format!("{e}"),
                });
                println!("{ev}");
            }
        }
    }

    let end_ev = json!({ "type": "session_end" });
    println!("{end_ev}");
    Ok(())
}
