//! agent-session: session JSONL tree, context building, compaction, event bus,
//! steer/followUp queues, orchestration (mirrors pi-coding-agent core).

pub mod bus;
pub mod compaction;
pub mod context;
pub mod entry;
pub mod error;
pub mod journal;
pub mod session;

pub use error::SessionError;
pub use session::AgentSession;

/// Orchestrated run phases (per session actor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Streaming,
    RunningTools,
    Compacting,
    Retrying,
}
