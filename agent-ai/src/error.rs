use thiserror::Error;

/// Cloneable error surface. Underlying errors are flattened to string for
/// cheap Clone in parallel tool result batching; full detail lives in logs.
#[derive(Debug, Clone, Error)]
pub enum AiError {
    #[error("HTTP 错误: {0}")]
    Http(String),
    #[error("流错误: {0}")]
    Stream(String),
    #[error("服务商错误 {status}: {body}")]
    Provider { status: u16, body: String },
    #[error("请求频率限制 (429)，重试等待 {0:?}秒")]
    RateLimited(Option<u64>),
    #[error("序列化错误: {0}")]
    Json(String),
    #[error("{0}")]
    Other(String),
}

impl From<reqwest::Error> for AiError {
    fn from(e: reqwest::Error) -> Self {
        AiError::Http(e.to_string())
    }
}

impl From<std::io::Error> for AiError {
    fn from(e: std::io::Error) -> Self {
        AiError::Stream(e.to_string())
    }
}

impl From<serde_json::Error> for AiError {
    fn from(e: serde_json::Error) -> Self {
        AiError::Json(e.to_string())
    }
}

/// Status-code handling lives here; provider adapters call this after `execute()`.
impl From<&reqwest::Response> for AiError {
    fn from(r: &reqwest::Response) -> Self {
        let status = r.status().as_u16();
        if status == 429 {
            let retry_after = r
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            return AiError::RateLimited(retry_after);
        }
        AiError::Provider {
            status,
            body: format!("{status} {}", r.status().canonical_reason().unwrap_or("")),
        }
    }
}
