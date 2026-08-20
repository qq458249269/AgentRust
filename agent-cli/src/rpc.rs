//! RPC mode: strict LF JSONL framing over stdin/stdout. Command responses + async events.
//! Full protocol in M4; here: read commands, echo accept, wire prompt/abort to session.

use crate::Cli;
use agent_session::AgentSession;
use serde_json::{json, Value};
use std::io::BufRead;
use tokio::io::{AsyncWriteExt, BufWriter};

pub async fn run(_session: AgentSession, _cli: &Cli) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(tokio::io::stdout());

    for line in stdin.lock().lines() {
        let line = line?;
        let cmd: Value = serde_json::from_str(&line)?;
        let ty = cmd.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match ty {
            "prompt" => {
                let msg = cmd.get("message").cloned().unwrap_or(json!(""));
                stdout
                    .write_all(
                        serde_json::to_string(&json!({
                            "type": "response",
                            "command": "prompt",
                            "success": true,
                            "id": cmd.get("id"),
                            "message": msg
                        }))?
                        .as_bytes(),
                    )
                    .await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            "abort" => {
                stdout
                    .write_all(br#"{"type":"response","command":"abort","success":true}"#)
                    .await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            other => {
                tracing::warn!("unhandled rpc command: {other}");
            }
        }
    }
    Ok(())
}
