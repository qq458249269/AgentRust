//! Built-in tools: bash, read_file, write_file, edit_file, list_dir, glob, grep.
//!
//! Search/glob use native Rust crates (regex, globset) for performance.

use crate::cancel::Cancelled;
use crate::error::CoreError;
use crate::tools::{Tool, ToolArgs, ToolOutput};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

// ─── Bash ──────────────────────────────────────────────────────────

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn describe(&self) -> String {
        "bash: Execute a bash command and return stdout/stderr output".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let cmd = args
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'command' 参数".into()))?;

        tracing::info!(工具 = "bash", 命令 = cmd, "执行 bash 命令");

        let mut builder = std::process::Command::new("bash");
        builder.arg("-c").arg(cmd);

        // Inherit cwd if available
        if let Some(cwd) = args.arguments.get("cwd").and_then(Value::as_str) {
            builder.current_dir(cwd);
        }

        // Run in a blocking thread to avoid blocking the async runtime
        let output = tokio::task::spawn_blocking(move || {
            builder.output()
        })
        .await
        .map_err(|e| CoreError::Tool(format!("spawn_blocking 失败: {e}")))?
        .map_err(|e| CoreError::Tool(format!("执行失败: {e}")))?;

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let is_error = !output.status.success();

        let mut content = String::new();
        if !stdout.is_empty() {
            content.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !content.is_empty() {
                content.push_str("\n--- stderr ---\n");
            }
            content.push_str(&stderr);
        }
        if content.is_empty() {
            content = format!("(exit code: {})", output.status.code().unwrap_or(-1));
        }

        // Truncate very long output
        let max_chars = 50_000;
        if content.len() > max_chars {
            content.truncate(max_chars);
            content.push_str("\n... (output truncated)");
        }

        Ok(ToolOutput {
            content,
            full_output_path: None,
            is_error,
        })
    }
}

// ─── Read File ─────────────────────────────────────────────────────

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn describe(&self) -> String {
        "read_file: Read the contents of a file at the given path".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let path = args
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'path' 参数".into()))?;

        tracing::info!(工具 = "read_file", 路径 = path, "读取文件");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| CoreError::Tool(format!("读取失败: {e}")))?;

        // Truncate very long files
        let max_chars = 100_000;
        let truncated;
        let display = if content.len() > max_chars {
            truncated = format!("{}...\n(文件过长，已截断，原始大小: {} 字节)", &content[..max_chars], content.len());
            &truncated
        } else {
            &content
        };

        Ok(ToolOutput {
            content: display.clone(),
            full_output_path: None,
            is_error: false,
        })
    }
}

// ─── Write File ────────────────────────────────────────────────────

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn describe(&self) -> String {
        "write_file: Write content to a file at the given path".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let path = args
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'path' 参数".into()))?;
        let content = args
            .arguments
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'content' 参数".into()))?;

        tracing::info!(工具 = "write_file", 路径 = path, 字节 = content.len(), "写入文件");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| CoreError::Tool(format!("创建目录失败: {e}")))?;
        }

        tokio::fs::write(path, content)
            .await
            .map_err(|e| CoreError::Tool(format!("写入失败: {e}")))?;

        Ok(ToolOutput {
            content: format!("已写入 {} 字节到 {}", content.len(), path),
            full_output_path: None,
            is_error: false,
        })
    }
}

// ─── Edit File ────────────────────────────────────────────────────
// Supports two modes:
//   1. Single: old_string + new_string
//   2. Batch:  changes = [{old_string, new_string}, ...] — applies all at once
// Uses context-aware matching: when exact match fails, finds the block
// using surrounding lines as anchors.

pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn describe(&self) -> String {
        "edit_file: Replace text in a file. Supports batch mode via 'changes' array. Context-aware matching handles line shifts.".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let path = args
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'path' 参数".into()))?;

        tracing::info!(工具 = "edit_file", 路径 = path, "编辑文件");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| CoreError::Tool(format!("读取失败: {e}")))?;

        // ── Detect mode: batch (changes array) vs single (old_string + new_string) ──
        let is_batch = args.arguments.get("changes")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

        if is_batch {
            execute_batch_edit(path, &content, args, cancel).await
        } else {
            execute_single_edit(path, &content, args, cancel).await
        }
    }
}

// ─── Batch Mode ────────────────────────────────────────────────────

async fn execute_batch_edit(
    path: &str,
    content: &str,
    args: &ToolArgs,
    cancel: &Cancelled,
) -> Result<ToolOutput, CoreError> {
    let changes = args.arguments.get("changes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CoreError::Tool("'changes' 必须是数组".into()))?;

    let mut current = content.to_string();
    let mut results: Vec<String> = Vec::new();
    let mut file_lines: Vec<String> = current.lines().map(String::from).collect();

    for (i, change) in changes.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let old_str = change.get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::Tool(format!("changes[{i}] 缺少 'old_string'")))?;
        let new_str = change.get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::Tool(format!("changes[{i}] 缺少 'new_string'")))?;

        if old_str == new_str {
            results.push(format!("#{i}: 跳过 (old == new)"));
            continue;
        }

        // Try exact match first on current content
        let exact_count = current.matches(old_str).count();
        if exact_count == 1 {
            current = current.replacen(old_str, new_str, 1);
            file_lines = current.lines().map(String::from).collect();
            results.push(format!("#{i}: 精确替换成功"));
            continue;
        }
        if exact_count > 1 {
            results.push(format!("#{i}: 找到 {exact_count} 处精确匹配，跳过 (需 allow_multiple)"));
            continue;
        }

        // Exact match failed → context-aware fuzzy match
        let old_lines: Vec<&str> = old_str.lines().collect();
        if old_lines.is_empty() {
            results.push(format!("#{i}: 跳过 (空 old_string)"));
            continue;
        }

        let file_refs: Vec<&str> = file_lines.iter().map(|s| s.as_str()).collect();
        let best = find_best_context_match(&file_refs, &old_lines);

        if let Some((start, end, score)) = best {
            let matched_block = file_lines[start..end].join("\n");
            current = current.replacen(&matched_block, new_str, 1);
            file_lines = current.lines().map(String::from).collect();
            results.push(format!("#{i}: 模糊替换成功 (相似度 {}%, L{}-L{})", score, start + 1, end));
        } else {
            // Show first line of old_str for debugging
            let first = old_lines[0].trim();
            let candidates: Vec<String> = file_refs.iter().enumerate()
                .filter(|(_, l)| l.trim().contains(first))
                .take(3)
                .map(|(j, l)| format!("L{}: {}", j + 1, l.trim()))
                .collect();
            let hint = if candidates.is_empty() {
                String::new()
            } else {
                format!("\n  候选: {}", candidates.join(" | "))
            };
            results.push(format!("#{i}: 未找到匹配{hint}"));
        }
    }

    // Write if any changes were applied
    let changed = current != content;
    if changed {
        tokio::fs::write(path, &current)
            .await
            .map_err(|e| CoreError::Tool(format!("写入失败: {e}")))?;
    }

    Ok(ToolOutput {
        content: format!("批量编辑 {} ({} 项):\n{}", path, changes.len(), results.join("\n")),
        full_output_path: None,
        is_error: false,
    })
}

// ─── Single Mode ───────────────────────────────────────────────────

async fn execute_single_edit(
    path: &str,
    content: &str,
    args: &ToolArgs,
    _cancel: &Cancelled,
) -> Result<ToolOutput, CoreError> {
    let old_string = args
        .arguments
        .get("old_string")
        .and_then(Value::as_str)
        .ok_or_else(|| CoreError::Tool("缺少 'old_string' 参数".into()))?;
    let new_string = args
        .arguments
        .get("new_string")
        .and_then(Value::as_str)
        .ok_or_else(|| CoreError::Tool("缺少 'new_string' 参数".into()))?;
    let allow_multiple = args
        .arguments
        .get("allow_multiple")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── Strategy 1: Exact match ──
    let exact_count = content.matches(old_string).count();
    if exact_count > 0 {
        if exact_count > 1 && !allow_multiple {
            let lines: Vec<(usize, &str)> = content.lines().enumerate()
                .filter(|(_, l)| l.contains(old_string))
                .collect();
            let hint: Vec<String> = lines.iter().take(5)
                .map(|(i, l)| format!("  L{}: {}", i + 1, l.trim()))
                .collect();
            return Err(CoreError::Tool(format!(
                "在 {path} 中找到 {exact_count} 处匹配:\n{}\n请使用 allow_multiple: true 或提供更精确的匹配",
                hint.join("\n")
            )));
        }
        return apply_single_edit(path, content, old_string, new_string, allow_multiple, exact_count).await;
    }

    // ── Strategy 2: Context-aware fuzzy match ──
    let old_lines: Vec<&str> = old_string.lines().collect();
    if old_lines.is_empty() {
        return Err(CoreError::Tool(format!("在 {path} 中未找到匹配的字符串 (empty old_string)")));
    }

    let file_lines: Vec<&str> = content.lines().collect();
    let best = find_best_context_match(&file_lines, &old_lines);

    if let Some((match_start, match_end, score)) = best {
        let matched_block: Vec<&str> = file_lines[match_start..match_end].to_vec();
        let matched_str = matched_block.join("\n");
        let new_content = content.replacen(&matched_str, new_string, 1);

        tokio::fs::write(path, &new_content)
            .await
            .map_err(|e| CoreError::Tool(format!("写入失败: {e}")))?;

        let line_info = format!("\n模糊匹配成功 (相似度 {}%): L{}-L{}", score, match_start + 1, match_end);
        Ok(ToolOutput {
            content: format!("已替换 1 处 in {path}{line_info}"),
            full_output_path: None,
            is_error: false,
        })
    } else {
        let old_first_line = old_lines[0].trim();
        let candidates: Vec<String> = file_lines.iter().enumerate()
            .filter(|(_, l)| l.trim().contains(old_first_line))
            .take(5)
            .map(|(i, l)| format!("  L{}: {}", i + 1, l.trim()))
            .collect();

        let mut err_msg = format!("在 {path} 中未找到匹配的字符串");
        if !candidates.is_empty() {
            err_msg.push_str(&format!("\n\n包含首行文本的候选位置:\n{}", candidates.join("\n")));
        }
        err_msg.push_str("\n\n提示: 请重新读取文件并使用当前内容中的文本。或使用 changes 数组批量编辑。");
        Err(CoreError::Tool(err_msg))
    }
}

/// Apply a single edit to the content string.
async fn apply_single_edit(
    path: &str,
    content: &str,
    old_string: &str,
    new_string: &str,
    allow_multiple: bool,
    count: usize,
) -> Result<ToolOutput, CoreError> {
    let new_content = if allow_multiple {
        content.replace(old_string, new_string)
    } else {
        let idx = content.find(old_string).unwrap();
        let mut s = content[..idx].to_string();
        s.push_str(new_string);
        s.push_str(&content[idx + old_string.len()..]);
        s
    };

    tokio::fs::write(path, &new_content)
        .await
        .map_err(|e| CoreError::Tool(format!("写入失败: {e}")))?;

    let matched_lines: Vec<usize> = new_content.lines().enumerate()
        .filter(|(_, l)| l.contains(new_string))
        .map(|(i, _)| i + 1)
        .take(10)
        .collect();
    let replacements = if allow_multiple { count } else { 1 };
    let line_info = if matched_lines.is_empty() {
        String::new()
    } else {
        format!("\n匹配行: L{}", matched_lines.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", L"))
    };
    Ok(ToolOutput {
        content: format!("已替换 {replacements} 处 in {path}{line_info}"),
        full_output_path: None,
        is_error: false,
    })
}

/// Find the best matching block in file_lines using context-aware scoring.
/// Returns (start_line, end_line, score_pct) or None.
fn find_best_context_match(
    file_lines: &[&str],
    old_lines: &[&str],
) -> Option<(usize, usize, u32)> {
    if old_lines.is_empty() || file_lines.is_empty() {
        return None;
    }

    let target_first = old_lines[0].trim();
    let target_len = old_lines.len();

    // Quick pass: find candidates by first line match
    let candidates: Vec<usize> = file_lines.iter().enumerate()
        .filter(|(_, l)| l.trim() == target_first)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let mut best: Option<(usize, usize, u32)> = None;

    for &start in &candidates {
        let end = start + target_len;
        if end > file_lines.len() {
            continue;
        }

        let block = &file_lines[start..end];
        let score = context_score(old_lines, block);

        // Require at least 60% similarity for a fuzzy match
        if score >= 60 {
            match &best {
                None => best = Some((start, end, score)),
                Some((_, _, best_score)) if score > *best_score => {
                    best = Some((start, end, score));
                }
                _ => {}
            }
        }
    }

    best
}

/// Score how similar two line blocks are (0-100).
/// Uses line-by-line comparison with fuzzy tolerance.
fn context_score(target: &[&str], candidate: &[&str]) -> u32 {
    if target.is_empty() {
        return 100;
    }
    let total = target.len() as u32;
    let mut matched = 0u32;
    for (t, c) in target.iter().zip(candidate.iter()) {
        let t_trim = t.trim();
        let c_trim = c.trim();
        if t_trim == c_trim {
            matched += 100; // exact match
        } else if t_trim.contains(c_trim) || c_trim.contains(t_trim) {
            matched += 70; // substring match
        } else {
            // Levenshtein-like: check if non-space chars are similar
            let t_chars: Vec<char> = t_trim.chars().filter(|c| !c.is_whitespace()).collect();
            let c_chars: Vec<char> = c_trim.chars().filter(|c| !c.is_whitespace()).collect();
            if t_chars == c_chars {
                matched += 90; // same content, different whitespace
            } else {
                // Count common chars
                let common = t_chars.iter().filter(|c| c_chars.contains(c)).count();
                let max_len = t_chars.len().max(c_chars.len());
                if max_len > 0 {
                    matched += (common as u32 * 50) / max_len as u32;
                }
            }
        }
    }
    matched / total
}

// ─── List Directory ────────────────────────────────────────────────

pub struct ListDirectoryTool;

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn describe(&self) -> String {
        "list_directory: List files and directories at the given path".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let path = args
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".");

        tracing::info!(工具 = "list_directory", 路径 = path, "列出目录");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let mut entries = tokio::fs::read_dir(path)
            .await
            .map_err(|e| CoreError::Tool(format!("读取目录失败: {e}")))?;

        let mut files = Vec::new();
        let mut dirs = Vec::new();

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| CoreError::Tool(format!("读取条目失败: {e}")))? 
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let metadata = entry.metadata().await
                .map_err(|e| CoreError::Tool(format!("读取元数据失败: {e}")))?;
            
            if cancel.is_cancelled() {
                return Err(CoreError::Cancelled);
            }
            
            if metadata.is_dir() {
                dirs.push(name);
            } else {
                let size = metadata.len();
                files.push(format!("{name} ({size} bytes)"));
            }
        }

        dirs.sort();
        files.sort();

        let mut output = String::new();
        if !dirs.is_empty() {
            output.push_str("Directories:\n");
            for d in &dirs {
                output.push_str(&format!("  {d}/\n"));
            }
        }
        if !files.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str("Files:\n");
            for f in &files {
                output.push_str(&format!("  {f}\n"));
            }
        }
        if output.is_empty() {
            output = "(empty directory)".to_string();
        }

        Ok(ToolOutput {
            content: output,
            full_output_path: None,
            is_error: false,
        })
    }
}

// ─── Glob / Find ──────────────────────────────────────────────────
// Uses `globset` crate (same engine as ripgrep) for full glob support:
// **, *, ?, [abc], {a,b}, brace expansion.

pub struct GlobFindTool;

#[async_trait]
impl Tool for GlobFindTool {
    fn name(&self) -> &str {
        "glob_find"
    }

    fn describe(&self) -> String {
        "glob_find: Find files matching a glob pattern (e.g. '**/*.rs', 'src/**/*.{ts,js}')".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let pattern = args
            .arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'pattern' 参数".into()))?;
        let cwd = args
            .arguments
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or(".");

        tracing::info!(工具 = "glob_find", 模式 = pattern, "搜索文件");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        // Compile globset matcher (supports **, *, ?, [abc], {a,b})
        let compiled = globset::GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| CoreError::Tool(format!("无效的 glob 模式: {e}")))?
            .compile_matcher();

        let root = std::path::Path::new(cwd);
        let mut matches = Vec::new();

        // Skip dirs: .git, node_modules, target, dist, __pycache__, .next, .venv
        const SKIP_DIRS: &[&str] = &[
            ".git", "node_modules", "target", "dist", "__pycache__",
            ".next", ".venv", "vendor", ".svn", ".hg",
        ];

        fn walk(
            dir: &std::path::Path,
            compiled: &globset::GlobMatcher,
            matches: &mut Vec<String>,
            cancel: &Cancelled,
        ) -> Result<(), CoreError> {
            if cancel.is_cancelled() {
                return Err(CoreError::Cancelled);
            }
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(_) => return Ok(()),
            };
            for entry in entries.flatten() {
                if cancel.is_cancelled() {
                    return Err(CoreError::Cancelled);
                }
                let path = entry.path();
                let metadata = match entry.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if metadata.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if SKIP_DIRS.contains(&name.as_str()) {
                        continue;
                    }
                    let _ = walk(&path, compiled, matches, cancel);
                } else if compiled.is_match(&path) {
                    matches.push(path.to_string_lossy().to_string());
                }
            }
            Ok(())
        }

        walk(root, &compiled, &mut matches, cancel)?;
        matches.sort();

        let output = if matches.is_empty() {
            "No matches found".to_string()
        } else {
            let shown = &matches[..matches.len().min(200)];
            format!("Found {} matches:\n{}", matches.len(), shown.join("\n"))
        };

        Ok(ToolOutput {
            content: output,
            full_output_path: None,
            is_error: false,
        })
    }
}

// ─── Grep / Search ────────────────────────────────────────────────
// Native Rust regex search — no subprocess, cross-platform.
// Uses `regex` crate (same engine as ripgrep) for fast pattern matching.

pub struct GrepSearchTool;

#[async_trait]
impl Tool for GrepSearchTool {
    fn name(&self) -> &str {
        "grep_search"
    }

    fn describe(&self) -> String {
        "grep_search: Search file contents with regex. Returns file:line:content matches.".to_string()
    }

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        let pattern = args
            .arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::Tool("缺少 'pattern' 参数".into()))?;
        let path = args
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".");
        let context_lines = args
            .arguments
            .get("context_lines")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let case_insensitive = args
            .arguments
            .get("case_insensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let max_results = args
            .arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(100) as usize;

        tracing::info!(工具 = "grep_search", 模式 = pattern, 路径 = path, "搜索内容");

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        // Compile regex
        let mut flags = regex::RegexBuilder::new(pattern);
        flags.case_insensitive(case_insensitive);
        let re = flags
            .build()
            .map_err(|e| CoreError::Tool(format!("无效的正则表达式: {e}")))?;

        let root = std::path::Path::new(path).to_path_buf();
        let re = Arc::new(re);
        let cancel = cancel.clone();

        // Run search in blocking thread (file I/O + regex is CPU-bound)
        let result = tokio::task::spawn_blocking(move || {
            grep_search_inner(&root, &re, context_lines, max_results, &cancel)
        })
        .await
        .map_err(|e| CoreError::Tool(format!("spawn_blocking 失败: {e}")))??;

        Ok(result)
    }
}

fn grep_search_inner(
    root: &std::path::Path,
    re: &regex::Regex,
    context_lines: usize,
    max_results: usize,
    cancel: &Cancelled,
) -> Result<ToolOutput, CoreError> {
    const SKIP_DIRS: &[&str] = &[
        ".git", "node_modules", "target", "dist", "__pycache__",
        ".next", ".venv", "vendor", ".svn", ".hg",
    ];

    // Binary file detection: skip files with null bytes in first 8KB
    fn is_likely_binary(path: &std::path::Path) -> bool {
        let Ok(bytes) = std::fs::read(path) else {
            return true;
        };
        let check = &bytes[..bytes.len().min(8192)];
        check.contains(&0)
    }

    let mut results: Vec<String> = Vec::new();
    let mut files_searched = 0u32;
    let mut total_matches = 0u32;

    #[allow(clippy::too_many_arguments)]
    fn walk(
        dir: &std::path::Path,
        re: &regex::Regex,
        context_lines: usize,
        results: &mut Vec<String>,
        files_searched: &mut u32,
        total_matches: &mut u32,
        max_results: usize,
        cancel: &Cancelled,
    ) -> Result<(), CoreError> {
        if cancel.is_cancelled() || results.len() >= max_results {
            return Ok(());
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };
        for entry in entries.flatten() {
            if cancel.is_cancelled() || results.len() >= max_results {
                return Ok(());
            }
            let path = entry.path();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if metadata.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if SKIP_DIRS.contains(&name.as_str()) {
                    continue;
                }
                let _ = walk(&path, re, context_lines, results, files_searched, total_matches, max_results, cancel);
            } else if metadata.len() > 0 && metadata.len() < 2_000_000 && !is_likely_binary(&path) {
                *files_searched += 1;
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let lines: Vec<&str> = content.lines().collect();
                    for (idx, line) in lines.iter().enumerate() {
                        if re.is_match(line) {
                            *total_matches += 1;
                            if results.len() >= max_results {
                                break;
                            }
                            // Format: file:line: content
                            let line_num = idx + 1;
                            let display_path = path.to_string_lossy().to_string();
                            let line_content = line.to_string();
                            results.push(format!("{display_path}:{line_num}: {line_content}"));

                            // Add context lines if requested
                            if context_lines > 0 {
                                let start = idx.saturating_sub(context_lines);
                                let end = (idx + context_lines + 1).min(lines.len());
                                for (ci, ctx_line_text) in lines[start..end].iter().enumerate() {
                                    let actual_idx = start + ci;
                                    if actual_idx == idx {
                                        continue; // skip the match line itself (already shown)
                                    }
                                    let marker = if actual_idx < idx { "-" } else { "+" };
                                    let ln = actual_idx + 1;
                                    let mut ctx_line = display_path.clone();
                                    ctx_line.push(':');
                                    ctx_line.push_str(&ln.to_string());
                                    ctx_line.push_str(": ");
                                    ctx_line.push_str(marker);
                                    ctx_line.push(' ');
                                    ctx_line.push_str(ctx_line_text);
                                    results.push(ctx_line);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    walk(root, re, context_lines, &mut results, &mut files_searched, &mut total_matches, max_results, cancel)?;

    let output = if results.is_empty() {
        format!("No matches found (searched {files_searched} files)")
    } else {
        let truncated = if total_matches > max_results as u32 {
            format!("\n... (showing {} of {} matches)", max_results, total_matches)
        } else {
            String::new()
        };
        format!("Found {total_matches} matches in {files_searched} files:\n{}{truncated}", results.join("\n"))
    };

    // Truncate if too long
    let max_chars = 50_000;
    let output = if output.len() > max_chars {
        format!("{}\n... (output truncated at {} chars)", &output[..max_chars], max_chars)
    } else {
        output
    };

    Ok(ToolOutput {
        content: output,
        full_output_path: None,
        is_error: false,
    })
}

/// Register all built-in tools into the given registry.
pub fn register_builtins(registry: &crate::tools::ToolRegistry) {
    registry.register(Arc::new(BashTool));
    registry.register(Arc::new(ReadFileTool));
    registry.register(Arc::new(WriteFileTool));
    registry.register(Arc::new(EditFileTool));
    registry.register(Arc::new(ListDirectoryTool));
    registry.register(Arc::new(GlobFindTool));
    registry.register(Arc::new(GrepSearchTool));
}

/// Convert ToolSpec list from registry describe_all into provider-neutral format.
pub fn tool_specs_from_registry(registry: &crate::tools::ToolRegistry) -> Vec<agent_ai::provider::ToolSpec> {
    registry
        .describe_all()
        .iter()
        .map(|desc| {
            let parts: Vec<&str> = desc.splitn(2, ':').collect();
            let name = parts[0].trim().to_string();
            let description = parts.get(1).map(|s| s.trim().to_string()).unwrap_or_default();

            let input_schema = tool_input_schema(&name);

            agent_ai::provider::ToolSpec {
                name,
                description,
                input_schema,
            }
        })
        .collect()
}

/// Generate a system prompt with tool instructions.
pub fn system_prompt_with_tools(base_prompt: &str, registry: &crate::tools::ToolRegistry) -> String {
    let tool_descriptions = registry.describe_all();
    if tool_descriptions.is_empty() {
        return base_prompt.to_string();
    }

    let mut prompt = base_prompt.to_string();
    prompt.push_str("\n\n## Available Tools\n\n");
    prompt.push_str("You have access to the following tools:\n\n");

    for desc in &tool_descriptions {
        let parts: Vec<&str> = desc.splitn(2, ':').collect();
        let name = parts[0].trim();
        let description = parts.get(1).map(|s| s.trim()).unwrap_or("");
        prompt.push_str(&format!("### {name}\n{description}\n\n"));
    }

    prompt.push_str("## Tool Usage Guidelines\n\n");
    prompt.push_str("- Use tools to read, write, and edit files\n");
    prompt.push_str("- Use bash for system commands and operations\n");
    prompt.push_str("- Always verify file operations before and after\n");
    prompt.push_str("- For large outputs, tools will truncate and save full output to files\n\n");

    prompt
}

/// Generate JSON Schema for tool input parameters.
pub fn tool_input_schema(name: &str) -> serde_json::Value {
    match name {
        "bash" => serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional)"
                }
            },
            "required": ["command"]
        }),
        "read_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }),
        "write_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }),
        "edit_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "[Single mode] Exact text to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "[Single mode] Replacement text"
                },
                "allow_multiple": {
                    "type": "boolean",
                    "description": "[Single mode] Allow replacing multiple occurrences (default false)"
                },
                "changes": {
                    "type": "array",
                    "description": "[Batch mode] Array of {old_string, new_string} pairs. Apply all edits at once. Preferred over single mode when editing multiple locations.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": { "type": "string", "description": "Text to find" },
                            "new_string": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["old_string", "new_string"]
                    }
                }
            },
            "required": ["path"]
        }),
        "list_directory" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)"
                }
            }
        }),
        "glob_find" => serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.ts')"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory to search from (default: current)"
                }
            },
            "required": ["pattern"]
        }),
        "grep_search" => serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g. 'fn main', 'TODO|FIXME', '\\berror\\b')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: current)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines around matches (default: 0)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)"
                }
            },
            "required": ["pattern"]
        }),
        _ => serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }
}
