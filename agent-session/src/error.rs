use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON 错误: {0}")]
    Json(#[from] serde_json::Error),
    #[error("核心错误: {0}")]
    Core(#[from] agent_core::CoreError),
    #[error("会话文件损坏，第 {line} 行: {message}")]
    Corrupt { line: usize, message: String },
}
