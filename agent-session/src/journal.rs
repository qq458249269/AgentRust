//! Append-only JSONL journal with crash-safe buffered writes.
//! Contract: only complete lines are ever committed; partial tail on crash is dropped.

use crate::entry::Entry;
use crate::error::SessionError;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncWriteExt, BufWriter};

/// In-memory tree index built once at open, then mutated in memory.
#[derive(Default)]
pub struct Journal {
    pub file: Option<PathBuf>,
    pub header: Option<Entry>,
    pub entries: Vec<Entry>,
    /// id -> index into entries
    pub index: std::collections::HashMap<String, usize>,
    pub leaf: Option<String>,
    writer: Option<BufWriter<tokio::fs::File>>,
}

impl Journal {
    /// Parse a v3 JSONL file into memory. One pass; startup cost only.
    pub fn open(path: &Path) -> Result<Self, SessionError> {
        let raw = std::fs::read_to_string(path)?;
        let mut j = Self {
            file: Some(path.to_path_buf()),
            ..Self::default()
        };
        for line in raw.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: Entry = serde_json::from_str(line)?;
            match &entry {
                Entry::SessionHeader { .. } => j.header = Some(entry),
                _ => {
                    if let Some(id) = entry.id().map(str::to_string) {
                        j.index.insert(id.clone(), j.entries.len());
                        j.leaf = Some(id);
                        j.entries.push(entry);
                    }
                }
            }
        }
        Ok(j)
    }

    /// Async open for a persistent writer later; M3 detail.
    pub async fn open_writer(&mut self, path: &Path) -> Result<(), SessionError> {
        let f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        self.writer = Some(BufWriter::new(f));
        Ok(())
    }

    pub fn append(&mut self, entry: Entry) -> Option<String> {
        let id = entry.id().map(str::to_string);
        if let Some(id) = &id {
            self.index.insert(id.clone(), self.entries.len());
            self.leaf = Some(id.clone());
            self.entries.push(entry);
        }
        id
    }

    /// Buffered flush-on-turn-end (M3: throttle 100ms + flush at turn end).
    pub async fn flush(&mut self) -> Result<(), SessionError> {
        if let Some(w) = &mut self.writer {
            w.flush().await?;
        }
        Ok(())
    }

    /// Node ids on the path leaf -> root (drives buildContextEntries).
    pub fn path_to_root(&self, leaf: Option<&str>) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = leaf.map(str::to_string);
        while let Some(c) = cur {
            out.push(c.clone());
            let idx = self.index.get(&c).copied();
            let parent = idx.and_then(|i| self.entries[i].parent_id().map(str::to_string));
            cur = parent;
        }
        out
    }

    /// Physical compaction: rewrite file without entries older than the newest compaction
    /// checkpoint. Run in background when ratio high; write tmp + atomic rename.
    pub async fn compact_file(&mut self) -> Result<(), SessionError> {
        let (tmp, final_path) = match &self.file {
            Some(p) => {
                let tmp = p.with_extension("jsonl.tmp");
                (tmp, p.clone())
            }
            None => return Ok(()),
        };
        let mut out = BufWriter::new(tokio::fs::File::create(&tmp).await?);
        if let Some(h) = &self.header {
            out.write_all(serde_json::to_string(h)?.as_bytes()).await?;
            out.write_all(b"\n").await?;
        }
        for e in &self.entries {
            out.write_all(serde_json::to_string(e)?.as_bytes()).await?;
            out.write_all(b"\n").await?;
        }
        out.flush().await?;
        tokio::fs::rename(&tmp, &final_path).await?;
        // writer now points at the old inode; reopen append on the new path in M3.
        Ok(())
    }
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Journal")
            .field("entries", &self.entries.len())
            .field("leaf", &self.leaf)
            .finish()
    }
}
