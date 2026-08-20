//! AgentSession: orchestration actor owning journal, event bus, queues, compaction, retry.
//! Single actor (mpsc) serializes all state mutation; tools run in JoinSet outside it.

use crate::bus::{EventBus, HookAction, Hooks};
use crate::context::{ContextBuffer, TokenBudget};
use crate::error::SessionError;
use crate::journal::Journal;
use agent_core::state::AgentState;
use std::path::Path;
use tokio::sync::mpsc;

pub struct AgentSession {
    pub journal: Journal,
    pub state: AgentState,
    pub bus: EventBus,
    pub hooks: Hooks,
    pub tokens: TokenBudget,
    pub context: ContextBuffer,
    /// steering queue (delivered after current tool batch, before next LLM call)
    pub steering: mpsc::UnboundedSender<String>,
    /// follow-up queue (delivered only when agent settles)
    pub follow_up: mpsc::UnboundedSender<String>,
    pub compaction_settings: crate::compaction::CompactionSettings,
    pub _phase: crate::Phase,
    pub _receivers: mpsc::UnboundedReceiver<String>, // placeholder, M4
}

impl AgentSession {
    /// Open or create a session journal at path. In-memory when None.
    pub fn open(path: Option<&Path>) -> Result<Self, SessionError> {
        let journal = match path {
            Some(p) => Journal::open(p)?,
            None => Journal::default(),
        };
        let (steering_tx, steering_rx) = mpsc::unbounded_channel();
        let (follow_tx, _follow_rx) = mpsc::unbounded_channel();
        Ok(Self {
            journal,
            state: AgentState::new(),
            bus: EventBus::new(1024),
            hooks: Hooks::default(),
            tokens: TokenBudget::default(),
            context: ContextBuffer::new(),
            steering: steering_tx,
            follow_up: follow_tx,
            compaction_settings: crate::compaction::CompactionSettings::default(),
            _phase: crate::Phase::Idle,
            _receivers: steering_rx,
        })
    }

    /// Fired when compaction or branch switch invalidates the incremental context.
    pub fn invalidate_context(&mut self) {
        self.context.invalidate();
    }

    /// Queue state publication for frontends.
    pub fn publish_queues(&self, steering_len: usize, follow_up_len: usize) {
        self.bus.emit(crate::bus::Event::QueueUpdate {
            steering: steering_len,
            follow_up: follow_up_len,
        });
    }

    /// Run one prompt end-to-end (M4; wires phase machine to agent-core loop).
    pub async fn prompt(&mut self, _text: String) -> Result<(), SessionError> {
        tracing::info!("prompt accepted; loop lands in M4");
        // TODO(M4): push user message to journal, drive agent loop, steer/followUp delivery,
        // overflow compaction + retry. Phase transitions emit event bus messages.
        Ok(())
    }

    /// Extension hook dispatch (M5): run handlers in registration order.
    pub fn run_hooks(&self, _ev: &crate::bus::Event) -> HookAction {
        HookAction::Continue
    }
}

/// Note: allocates one unused receiver kept for M4 queue wiring; removed then.
#[allow(dead_code)]
struct Placeholder;
