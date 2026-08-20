//! Context building: incremental buffer + token estimation + prompt-cache-friendly ordering.
//!
//! Performance contract (DESIGN.md §4): ordinary turns only append; full rebuild happens
//! at session open, after compaction, and after branch switch.

use crate::bus::EventBus;
use agent_ai::model::Usage;

/// Per-message token estimate cache + calibration against last provider usage.
#[derive(Debug, Default)]
pub struct TokenBudget {
    /// cached estimate per message id
    estimates: std::collections::HashMap<String, u64>,
    /// calibration: sum of estimates for messages known present in last usage
    last_usage: Option<Usage>,
}

impl TokenBudget {
    pub fn estimate_of(&self, id: &str) -> Option<u64> {
        self.estimates.get(id).copied()
    }

    /// pi parity (getContextUsage): prefer last assistant usage, estimate the tail.
    pub fn context_tokens(&self, _tail: &[agent_core::messages::Message]) -> u64 {
        self.last_usage.as_ref().map(|u| u.input).unwrap_or(0)
    }

    pub fn note_usage(&mut self, u: Usage) {
        self.last_usage = Some(u);
    }
}

/// A rebuilt context snapshot handed to the provider request builder.
pub struct ContextSnapshot {
    pub system_prompt: String,
    pub messages: Vec<agent_core::messages::Message>,
    /// generation counter: increments on compression/branch switch
    pub generation: u64,
}

/// Placeholder for M3: incremental append path keeps a live buffer keyed by generation.
pub struct ContextBuffer {
    pub generation: u64,
    _played: Vec<String>,
}

impl ContextBuffer {
    pub fn new() -> Self {
        Self {
            generation: 0,
            _played: Vec::new(),
        }
    }

    pub fn invalidate(&mut self) {
        self.generation += 1;
    }
}

impl Default for ContextBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Event bus ref kept here so frontends share generation info; removed when unused.
#[allow(dead_code)]
pub struct ContextService {
    pub bus: EventBus,
}
