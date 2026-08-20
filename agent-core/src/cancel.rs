//! Cancellation: piped CancellationToken per turn. Esc cancels the turn, not the queues.

use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct Cancelled(CancellationToken);

impl Cancelled {
    pub fn new() -> Self {
        Self(CancellationToken::new())
    }

    /// Used by trigger tasks: Esc handler, RPC abort command.
    pub fn cancel(&self) {
        self.0.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    /// awaitable: tool executors and provider reads poll this.
    pub fn token(&self) -> &CancellationToken {
        &self.0
    }
}

impl Default for Cancelled {
    fn default() -> Self {
        Self::new()
    }
}
