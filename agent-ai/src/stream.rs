//! SSE / delta parsing into a provider-neutral event stream.
//!
//! Performance contract: parse in ONE task, no per-token alloc. Incoming reqwest bytes
//! are appended to a working buffer; only `data:` lines are forwarded as raw strings,
//! which the provider adapter maps into `StreamEvent`.

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

impl Default for StopReason {
    fn default() -> Self {
        StopReason::Stop
    }
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

/// Split an SSE byte stream into `data:` lines, emitting raw payload strings.
///
/// SSE framing: events separated by blank lines; each field `name: value`. We only care
/// about `data:` lines. A `data:` payload may span multiple `data:` lines concatenated
/// with `\n`; Anthropic emits one JSON per event so each `data:` line is one payload.
pub struct SseSplitter {
    buf: Vec<u8>,
    /// pending data payload accumulated across multiple `data:` lines
    pending: Vec<u8>,
    out: mpsc::UnboundedSender<Result<String, AiError>>,
}

impl SseSplitter {
    pub fn new(out: mpsc::UnboundedSender<Result<String, AiError>>) -> Self {
        Self {
            buf: Vec::new(),
            pending: Vec::new(),
            out,
        }
    }

    /// Append incoming bytes, split on `\n`, dispatch complete lines. Incomplete tail
    /// stays buffered until more bytes arrive or `finish()` is called.
    pub fn push(&mut self, chunk: Bytes) -> Result<(), AiError> {
        self.buf.extend_from_slice(&chunk);
        // take ownership to break the borrow of self.buf while dispatching
        let buf = std::mem::take(&mut self.buf);
        let mut last = 0;
        for i in 0..=buf.len() {
            if i < buf.len() && buf[i] == b'\n' {
                self.dispatch_line(&buf[last..i])?;
                last = i + 1;
            }
        }
        self.buf.extend_from_slice(&buf[last..]);
        Ok(())
    }

    /// Flush any buffered tail (stream end).
    pub fn finish(&mut self) -> Result<(), AiError> {
        let tail = std::mem::take(&mut self.buf);
        if !tail.is_empty() {
            self.dispatch_line(&tail)?;
        }
        if !self.pending.is_empty() {
            self.emit_pending()?;
        }
        Ok(())
    }

    fn dispatch_line(&mut self, line: &[u8]) -> Result<(), AiError> {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            // blank line terminates the current event (if any pending payload)
            if !self.pending.is_empty() {
                self.emit_pending()?;
            }
        } else if line.starts_with(b"data:") {
            let data = &line[5..];
            let data = data.strip_prefix(b" ").unwrap_or(data);
            // Empty `data:` line also terminates a multi-line payload.
            if data.is_empty() && !self.pending.is_empty() {
                self.emit_pending()?;
            } else if !data.is_empty() {
                if !self.pending.is_empty() {
                    self.pending.push(b'\n');
                }
                self.pending.extend_from_slice(data);
            }
        }
        Ok(())
    }

    fn emit_pending(&mut self) -> Result<(), AiError> {
        let payload = std::mem::take(&mut self.pending);
        let s = String::from_utf8_lossy(&payload).into_owned();
        self.out
            .send(Ok(s))
            .map_err(|_| AiError::Other("stream consumer closed".into()))
    }

    /// Send an error downstream (used when the HTTP layer fails mid-stream).
    pub fn out_send_err(&self, e: AiError) {
        let _ = self.out.send(Err(e));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a full SSE document in awkwardly small chunks; every data payload must
    /// come out exactly once, in order, regardless of where line boundaries fall.
    #[test]
    fn splitter_handles_chunk_boundaries() {
        let sse = "event: message_start\ndata: {\"type\":\"a\"}\n\n".repeat(3);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut s = SseSplitter::new(tx);

        // chunk sizes [1,3,7,2] cycling
        let mut pos = 0usize;
        for size in [1usize, 3, 7, 2].into_iter().cycle() {
            if pos >= sse.len() {
                break;
            }
            let end = (pos + size).min(sse.len());
            s.push(Bytes::copy_from_slice(sse[pos..end].as_bytes()))
                .expect("push");
            pos = end;
        }
        if pos < sse.len() {
            s.push(Bytes::copy_from_slice(sse[pos..].as_bytes()))
                .expect("push");
        }
        s.finish().expect("finish");
        drop(s);

        let mut got = Vec::new();
        while let Ok(item) = rx.try_recv() {
            got.push(item.expect("payload"));
        }
        assert_eq!(got.len(), 3, "{got:?}");
        assert!(got.iter().all(|l| l == &"{\"type\":\"a\"}"));
    }

    /// Multi-line data payloads are joined with a single \n.
    #[test]
    fn splitter_joins_multiline_data() {
        let sse = "data: line1\ndata: line2\n\ndata: done\n\n";
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut s = SseSplitter::new(tx);
        s.push(Bytes::from_static(sse.as_bytes())).unwrap();
        s.finish().unwrap();
        drop(s);
        let mut got = Vec::new();
        while let Ok(item) = rx.try_recv() {
            got.push(item.unwrap());
        }
        assert_eq!(got, vec!["line1\nline2".to_string(), "done".to_string()]);
    }
}
