//! JSON mode: forward event bus to stdout as JSON lines.

use crate::CommonArgs;
use agent_session::AgentSession;

pub async fn run(_session: AgentSession, _cli: &CommonArgs) -> anyhow::Result<()> {
    tracing::warn!("json mode lands in M4");
    Ok(())
}
