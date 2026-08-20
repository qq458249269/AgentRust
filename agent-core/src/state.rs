//! Agent state: messages, model, thinking, system prompt. Copied-on-write via Arc.

use crate::messages::Message;
use agent_ai::model::{Model, ThinkingLevel};

#[derive(Debug, Clone)]
pub struct AgentState {
    pub messages: Arc<Vec<Message>>,
    pub model: Option<Model>,
    pub thinking_level: ThinkingLevel,
    pub system_prompt: String,
}

// add Arc import after the struct; kept separate to make the single allocation obvious
use std::sync::Arc;

impl AgentState {
    pub fn new() -> Self {
        Self {
            messages: Arc::new(Vec::new()),
            model: None,
            thinking_level: ThinkingLevel::Medium,
            system_prompt: String::new(),
        }
    }

    /// Push via copy-on-write: cheap when no other reference holds the Vec.
    pub fn push_message(&mut self, m: Message) {
        Arc::make_mut(&mut self.messages).push(m);
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}
