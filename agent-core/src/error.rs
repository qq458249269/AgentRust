use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum CoreError {
    #[error("AI 错误: {0}")]
    Ai(#[from] agent_ai::AiError),
    #[error("工具错误: {0}")]
    Tool(String),
    #[error("无效的工具调用 {name}: {message}")]
    InvalidToolCall { name: String, message: String },
    #[error("已取消")]
    Cancelled,
    #[error("上下文溢出，需要压缩")]
    ContextOverflow { needed: usize, available: usize },
}
