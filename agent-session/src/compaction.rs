//! Compaction (pi parity). Threshold + reserve; keepRecentTokens cut point; split-turn merge.
//! Full implementation is M3; here: settings + trigger math.

use crate::bus::CompactionReason;

#[derive(Debug, Clone, Copy)]
pub struct CompactionSettings {
    pub enabled: bool,
    /// context_window - reserve triggers compaction
    pub reserve_tokens: u64,
    /// newest tokens kept uncompressed
    pub keep_recent_tokens: u64,
    /// tool result serialization cap for summaries
    pub serialize_tool_result_max_chars: usize,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            serialize_tool_result_max_chars: 2_000,
        }
    }
}

impl CompactionSettings {
    pub fn triggered(&self, context_tokens: u64, context_window: u64) -> Option<CompactionReason> {
        if !self.enabled {
            return None;
        }
        if context_tokens > context_window.saturating_sub(self.reserve_tokens) {
            return Some(CompactionReason::Threshold);
        }
        None
    }
}
