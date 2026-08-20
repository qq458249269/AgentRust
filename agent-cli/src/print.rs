//! Print mode: send prompts, print final text, exit.

use crate::Cli;
use agent_session::AgentSession;

pub async fn run(_session: AgentSession, _cli: &Cli) -> anyhow::Result<()> {
    tracing::warn!("print mode lands in M4");
    Ok(())
}
