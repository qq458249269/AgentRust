//! agent-core: LLM loop, tool executor, cancellation. No session/IO concepts (mirrors pi-agent-core).

pub mod cancel;
pub mod error;
pub mod loop_; // loop_ because `loop` is a keyword
pub mod messages;
pub mod state;
pub mod tools;

pub use cancel::Cancelled;
pub use error::CoreError;
pub use loop_::{run_loop, LoopConfig, LoopResult};
pub use messages::{AssistantMessage, Message, Role, ToolResultMessage, UserMessage};
pub use state::AgentState;
pub use tools::{Tool, ToolArgs, ToolExecution, ToolOutput, ToolRegistry};

/// Core agent loop lives here (M2). Explicit states; no implicit flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Streaming,
    RunningTools,
    Compacting,
}
