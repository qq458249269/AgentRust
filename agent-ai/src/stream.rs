//! SSE / delta parsing into a provider-neutral event stream.
//!
//! Performance contract: parse in ONE task, zero-copy Bytes slices, no per-token alloc.

use crate::error::AiError;
use crate::model::{ThinkingLevel, Usage};
use bytes::Bytes;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta {
        delta: String,
    },
    ThinkingDelta {
        delta: String,
    },
    ToolCallStarted {
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        id: String,
        delta: String,
    },
    ToolCallDone {
        id: String,
    },
    /// Final usage from the provider (Anthropic message_stop / OpenAI usage chunk).
    Usage {
        usage: Usage,
    },
    Partial {
        thinking: Option<ThinkingLevel>,
    },
    /// Terminal event; stream then closes.
    Done {
        stop_reason: StopReason,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

/// Raw line source; adapters (SSE chunk splitter) implement this.
#[derive(Debug)]
pub struct StreamReader {
    rx: mpsc::Receiver<Result<StreamEvent, AiError>>,
}

impl StreamReader {
    pub fn new(rx: mpsc::Receiver<Result<StreamEvent, AiError>>) -> Self {
        Self { rx }
    }

    pub async fn next(&mut self) -> Option<Result<StreamEvent, AiError>> {
        self.rx.recv().await
    }

    /// Drain and aggregate (non-streaming fallback).
    pub async fn collect(self) -> (String, String, Usage) {
        let mut text = String::new();
        let mut thinking = String::new();
        let mut usage = Usage::default();
        let mut sr = self;
        while let Some(ev) = sr.next().await {
            match ev {
                Ok(StreamEvent::TextDelta { delta }) => text.push_str(&delta),
                Ok(StreamEvent::ThinkingDelta { delta }) => thinking.push_str(&delta),
                Ok(StreamEvent::Usage { usage: u }) => usage = u,
                _ => {}
            }
        }
        (text, thinking, usage)
    }
}

pub type StreamSender = mpsc::Sender<Result<StreamEvent, AiError>>;

/// Split an SSE byte stream into data lines. Users: Anthropic + OpenAI adapters.
/// Implemented later in M1 against reqwest's BytesStream.
pub struct SseSplitter;

impl SseSplitter {
    pub fn new(_input: Bytes, _tx: StreamSender) -> Self {
        Self
    }
}
