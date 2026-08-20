//! Tool contract + registry + parallel executor.
//!
//! Tool execution semantics (pi): preflight serial (order + interception), execution
//! concurrent via JoinSet, results reordered to source order on commit.

use crate::cancel::Cancelled;
use crate::error::CoreError;
use dashmap::DashMap;
use serde_json::Value;
use std::sync::Arc;
use tokio::task::JoinSet;

#[derive(Debug, Clone)]
pub struct ToolArgs {
    pub call_id: String,
    pub name: String,
    /// parsed JSON arguments; validated before execution
    pub arguments: Value,
}

/// Executor trait implementable by builtin (bash/read/write/edit/grep/find/ls) and plugins.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn describe(&self) -> String; // one-line snippet for system prompt

    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError>;
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolOutput {
    /// truncated, context-bound content
    pub content: String,
    /// full output path when truncated (mirror BashExecutionMessage.fullOutputPath)
    pub full_output_path: Option<String>,
    pub is_error: bool,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: DashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&self, t: Arc<dyn Tool>) {
        self.tools.insert(t.name().to_string(), t);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).map(|g| g.clone())
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|e| e.key().clone()).collect()
    }

    pub fn describe_all(&self) -> Vec<String> {
        self.tools.iter().map(|e| e.value().describe()).collect()
    }

    /// Parallel execution. Preflight already done by caller (order preserved there).
    pub async fn run_all(
        &self,
        calls: &[ToolArgs],
        cancel: &Cancelled,
    ) -> Vec<Result<ToolOutput, CoreError>> {
        let mut set: JoinSet<(usize, Result<ToolOutput, CoreError>)> = JoinSet::new();
        for (i, args) in calls.iter().enumerate() {
            let tools = self.clone();
            let args = args.clone();
            let cancel = cancel.clone();
            set.spawn(async move {
                let name = args.name.clone();
                let tool = match tools.get(&name) {
                    Some(t) => t,
                    None => {
                        return (i, Err(CoreError::Tool(format!("unknown tool {name}"))));
                    }
                };
                (i, tool.execute(&args, &cancel, None).await)
            });
        }
        let mut by_idx: Vec<Option<Result<ToolOutput, CoreError>>> = vec![None; calls.len()];
        while let Some(res) = set.join_next().await {
            match res {
                Ok((i, r)) => by_idx[i] = Some(r),
                Err(e) => {
                    let i = by_idx.iter().position(|x| x.is_none()).unwrap_or(0);
                    by_idx[i] = Some(Err(CoreError::Tool(format!("task panicked: {e}"))));
                }
            }
        }
        by_idx
            .into_iter()
            .map(|o| o.unwrap_or_else(|| Err(CoreError::Tool("task did not report".into()))))
            .collect()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

/// A generic function tool: name + describe + async fn. Cheap plugin path, M5.
pub struct FnTool {
    name: &'static str,
    description: &'static str,
    f: Arc<dyn ToolFn>,
}

/// Boxed async fn taking owned args; avoids lifetime/dyn pitfalls of borrowed futures.
trait ToolFn: Send + Sync {
    fn call(
        &self,
        args: ToolArgs,
        cancel: Cancelled,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send>>;
}

impl<F> ToolFn for F
where
    F: Fn(
            ToolArgs,
            Cancelled,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send>,
        > + Send
        + Sync,
{
    fn call(
        &self,
        args: ToolArgs,
        cancel: Cancelled,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send>>
    {
        self(args, cancel)
    }
}

impl FnTool {
    pub fn new(
        name: &'static str,
        description: &'static str,
        f: impl Fn(
                ToolArgs,
                Cancelled,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send>,
            > + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            name,
            description,
            f: Arc::new(f),
        }
    }
}

#[async_trait::async_trait]
impl Tool for FnTool {
    fn name(&self) -> &str {
        self.name
    }
    fn describe(&self) -> String {
        self.description.to_string()
    }
    async fn execute(
        &self,
        args: &ToolArgs,
        cancel: &Cancelled,
        _on_update: Option<Arc<dyn Fn(ToolOutput) + Send + Sync>>,
    ) -> Result<ToolOutput, CoreError> {
        self.f.call(args.clone(), cancel.clone()).await
    }
}

pub type ToolExecution = ToolArgs;
