//! JSON mode: forward event bus to stdout as JSON lines.

use crate::Cli;
use agent_session::AgentSession;

pub async fn run(_session: AgentSession, _cli: &Cli) -> anyhow::Result<()> {
    tracing::warn!("json mode lands in M4");
    Ok(())
}
