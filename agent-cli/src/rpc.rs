//! RPC mode: strict LF JSONL framing over stdin/stdout.
//!
//! Protocol:
//!   Client → Agent: `{ "id": "1", "type": "prompt", "message": "hello" }`
//!   Agent → Client: `{ "id": "1", "type": "response", "command": "prompt", "success": true }`
//!   Agent → Client: `{ "type": "event", "event": { "type": "delta", "delta": "..." } }`
//!   Agent → Client: `{ "type": "event", "event": { "type": "turn_end" } }`
//!
//! Now uses AgentSession for full agent loop with tool support (bash, read/write/edit, grep, etc.).

use crate::CommonArgs;
use agent_session::AgentSession;
use serde_json::{json, Value};
use std::io::BufRead;
use tokio::io::{AsyncWriteExt, BufWriter};

pub async fn run(mut session: AgentSession, _cli: &CommonArgs) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(tokio::io::stdout());

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

                // Use AgentSession::prompt() for full agent loop with tools
                match session.prompt(text).await {
                    Ok(response_text) => {
                        // Emit the full response as a delta event (clients can display it)
                        if !response_text.is_empty() {
                            let ev = json!({
                                "type": "event",
                                "event": {
                                    "type": "delta",
                                    "delta": response_text,
                                }
                            });
                            stdout.write_all(serde_json::to_string(&ev)?.as_bytes()).await?;
                            stdout.write_all(b"\n").await?;
                        }
                        // Emit turn_end
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
                // Re-create session to clear state
                session = crate::setup_session(_cli)?;
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
