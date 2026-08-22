//! Built-in tools: bash, read_file, write_file.
//!
//! These provide the minimum viable tool set for an agent loop.

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

/// Register all built-in tools into the given registry.
pub fn register_builtins(registry: &crate::tools::ToolRegistry) {
    registry.register(Arc::new(BashTool));
    registry.register(Arc::new(ReadFileTool));
    registry.register(Arc::new(WriteFileTool));
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

            // Generate input schema based on tool name
            let input_schema = match name.as_str() {
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
                _ => serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            };

            agent_ai::provider::ToolSpec {
                name,
                description,
                input_schema,
            }
        })
        .collect()
}
