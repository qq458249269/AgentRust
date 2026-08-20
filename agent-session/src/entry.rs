//! Session entries (mirror pi session-format v3). Journal = JSONL of these.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Entry {
    SessionHeader {
        version: u32,
        id: String,
        timestamp: String,
        cwd: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_session: Option<String>,
    },
    Message {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        #[serde(flatten)]
        message: agent_core::messages::Message,
    },
    Compaction {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
        /// self-contained checkpoint: entries before it are ignored when present
        retained_tail: Option<serde_json::Value>,
        details: Option<CompactionDetails>,
    },
    BranchSummary {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        from_id: String,
        summary: String,
    },
    Custom {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        data: serde_json::Value,
    },
    Label {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        target_id: String,
        label: String,
    },
    ModelChange {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        model_id: String,
    },
    ThinkingLevelChange {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        thinking_level: String,
    },
    SessionInfo {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        name: String,
    },
}

/// cumulative file tracking across compactions (pi parity)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionDetails {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

impl Entry {
    pub fn id(&self) -> Option<&str> {
        match self {
            Entry::SessionHeader { .. } => None,
            Entry::Message { id, .. }
            | Entry::Compaction { id, .. }
            | Entry::BranchSummary { id, .. }
            | Entry::Custom { id, .. }
            | Entry::Label { id, .. }
            | Entry::ModelChange { id, .. }
            | Entry::ThinkingLevelChange { id, .. }
            | Entry::SessionInfo { id, .. } => Some(id),
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Entry::SessionHeader { .. } => None,
            Entry::Message { parent_id, .. }
            | Entry::Compaction { parent_id, .. }
            | Entry::BranchSummary { parent_id, .. }
            | Entry::Custom { parent_id, .. }
            | Entry::Label { parent_id, .. }
            | Entry::ModelChange { parent_id, .. }
            | Entry::ThinkingLevelChange { parent_id, .. }
            | Entry::SessionInfo { parent_id, .. } => parent_id.as_deref(),
        }
    }
}
