//! Print mode: send prompts through AgentSession, print response, exit.
//! Now uses AgentSession for full agent loop with tool support.

use crate::{CommonArgs, PrintArgs};
use agent_session::AgentSession;

pub async fn run(mut session: AgentSession, _cli: &CommonArgs, args: &PrintArgs) -> anyhow::Result<()> {
    // Use AgentSession::prompt() for each prompt (supports multi-turn with tools)
    for prompt in &args.prompts {
        if prompt.is_empty() {
            continue;
        }
        match session.prompt(prompt.clone()).await {
            Ok(response_text) => {
                if !response_text.is_empty() {
                    println!("{response_text}");
                }
            }
            Err(e) => {
                eprintln!("错误: {e}");
            }
        }
    }
    Ok(())
}
