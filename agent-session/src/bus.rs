//! Event bus + event types. Journal is authoritative; events are projections.
//! pi parity: AgentSession.subscribe and extension pi.on() share one bus.

use agent_core::tools::ToolOutput;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    MessageStart {
        message: MessageMeta,
    },
    /// streaming: text/thinking deltas
    MessageUpdate {
        message_id: String,
        delta: String,
        is_thinking: bool,
    },
    MessageEnd {
        message: MessageMeta,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        partial: String,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolOutput,
        is_error: bool,
    },
    TurnStart {
        index: u32,
    },
    TurnEnd {
        index: u32,
    },
    AgentStart,
    AgentEnd,
    AgentSettled,
    CompactionStart {
        reason: CompactionReason,
    },
    CompactionEnd {
        entry_id: String,
        tokens_before: u64,
    },
    QueueUpdate {
        steering: usize,
        follow_up: usize,
    },
    ModelSelect {
        provider: String,
        model_id: String,
    },
}

/// lightweight, serialization-safe message fingerprint (full content stays in journal)
#[derive(Debug, Clone, Serialize)]
pub struct MessageMeta {
    pub id: String,
    pub role: String,
    pub usage_total: Option<u64>,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn emit(&self, ev: Event) {
        // slow subscribers lag; they can re-snapshot from the journal
        let _ = self.tx.send(ev);
    }
}

/// Extension hooks: registration table evaluated in order per event (M5).
#[derive(Default)]
#[allow(dead_code)] // populated in M5
pub struct Hooks {
    inner: Arc<
        std::collections::HashMap<
            &'static str,
            Vec<Arc<dyn Fn(Event) -> HookAction + Send + Sync>>,
        >,
    >,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookAction {
    Continue,
    Blocked,
    Handled,
}
