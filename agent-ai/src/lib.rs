//! agent-ai: provider adaptation, model catalog, credentials, streaming, usage accounting.
//!
//! Mirrors `@earendil-works/pi-ai`. No session/agent concepts here.

pub mod creds;
pub mod error;
pub mod model;
pub mod provider;
pub mod stream;

pub use error::AiError;
pub use model::{Model, ThinkingLevel, Usage};
pub use provider::{ChatProvider, ProviderClient, ProviderRequest, ProviderResponse};
pub use stream::{StreamEvent, StreamReader};

/// Shared reqwest::Client (HTTP/2 pooling). Owned here so callers do not build per-request clients.
pub struct Client {
    inner: reqwest::Client,
}

impl Client {
    pub fn new() -> Self {
        Self {
            inner: reqwest::Client::builder()
                .pool_max_idle_per_host(8)
                .connect_timeout(std::time::Duration::from_secs(15))
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .expect("reqwest client build"),
        }
    }

    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}
