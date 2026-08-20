use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum CoreError {
    #[error("ai error: {0}")]
    Ai(#[from] agent_ai::AiError),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("invalid tool call {name}: {message}")]
    InvalidToolCall { name: String, message: String },
    #[error("aborted")]
    Cancelled,
    #[error("context overflow, compaction required")]
    ContextOverflow { needed: usize, available: usize },
}
