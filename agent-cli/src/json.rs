//! JSON mode: forward all session events to stdout as JSON lines.
//!
//! Each event is serialized as a single JSON line (no framing beyond LF).
//! Now uses AgentSession for full agent loop with tool support.

use crate::CommonArgs;
use agent_session::AgentSession;
use serde_json::json;

pub async fn run(mut session: AgentSession, _cli: &CommonArgs) -> anyhow::Result<()> {
    // Emit a session_start event
    let model_str = session
        .model
        .as_ref()
        .map(|m| m.id.as_str())
        .unwrap_or("unknown");
    let provider_str = session
        .model
        .as_ref()
        .map(|m| m.provider.as_str())
        .unwrap_or("unknown");
    let start_ev = json!({
        "type": "session_start",
        "model": model_str,
        "provider": provider_str,
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

        // Use AgentSession::prompt() for full agent loop with tools
        match session.prompt(trimmed).await {
            Ok(response_text) => {
                // Emit the response as a delta event
                if !response_text.is_empty() {
                    let ev = json!({
                        "type": "delta",
                        "delta": response_text,
                    });
                    println!("{ev}");
                }
                let ev = json!({
                    "type": "turn_end",
                    "stop_reason": "stop",
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
        }
    }

    let end_ev = json!({ "type": "session_end" });
    println!("{end_ev}");
    Ok(())
}
