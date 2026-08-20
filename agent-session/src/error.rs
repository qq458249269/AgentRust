use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("core error: {0}")]
    Core(#[from] agent_core::CoreError),
    #[error("corrupt session file at line {line}: {message}")]
    Corrupt { line: usize, message: String },
}
