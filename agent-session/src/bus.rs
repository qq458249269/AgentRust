//! Event bus + event types + plugin hook system (M5).
//!
//! Journal is authoritative; events are projections.
//! pi parity: AgentSession.subscribe and extension pi.on() share one bus.
//!
//! M5 additions:
//! - `Hooks`: per-event handler list, evaluated in registration order
//! - `ToolCallHook`: intercept/modify tool calls before execution
//! - `ToolRegistry`: external tools registered at runtime
//! - `reload()`: rebuild handler table (drops in-memory state, documented)

use agent_core::tools::ToolOutput;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

// ─── Event Types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    MessageStart {
        message: MessageMeta,
    },
    /// streaming: text/thinking deltas
    MessageUpdate {
        message_id: String,
        delta: String,
        is_thinking: bool,
    },
    MessageEnd {
        message: MessageMeta,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        partial: String,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolOutput,
        is_error: bool,
    },
    TurnStart {
        index: u32,
    },
    TurnEnd {
        index: u32,
    },
    AgentStart,
    AgentEnd,
    AgentSettled,
    CompactionStart {
        reason: CompactionReason,
    },
    CompactionEnd {
        entry_id: String,
        tokens_before: u64,
    },
    QueueUpdate {
        steering: usize,
        follow_up: usize,
    },
    ModelSelect {
        provider: String,
        model_id: String,
    },
}

/// lightweight, serialization-safe message fingerprint (full content stays in journal)
#[derive(Debug, Clone, Serialize)]
pub struct MessageMeta {
    pub id: String,
    pub role: String,
    pub usage_total: Option<u64>,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

// ─── Event Bus ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn emit(&self, ev: Event) {
        // slow subscribers lag; they can re-snapshot from the journal
        let _ = self.tx.send(ev);
    }
}

// ─── Hook System (M5) ───────────────────────────────────────────────

/// Action returned by a hook handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookAction {
    /// Continue processing (next handler / default behavior).
    Continue,
    /// Block the event (tool call will not execute, message will not be sent).
    Blocked,
    /// Event was fully handled by this handler (skip default behavior).
    Handled,
}

/// Event name string used for hook registration (matches Event variant names).
pub type EventName = &'static str;

/// A hook handler: given an event, return an action.
pub type EventHandler = Arc<dyn Fn(&Event) -> HookAction + Send + Sync>;

/// A tool-call interceptor: given tool name + arguments JSON, return modified args or block.
/// Returns `None` to block, `Some(args)` to proceed (possibly modified).
pub type ToolCallInterceptor =
    Arc<dyn Fn(&str, &serde_json::Value) -> Option<serde_json::Value> + Send + Sync>;

/// Extension plugin hook table: per-event handler lists evaluated in registration order.
///
/// # Usage
/// ```ignore
/// let mut hooks = Hooks::new();
/// hooks.on("tool_execution_start", Arc::new(|ev| {
///     tracing::info!("tool starting");
///     HookAction::Continue
/// }));
/// ```
#[derive(Clone)]
pub struct Hooks {
    /// Per-event handler lists.
    handlers: Arc<Mutex<HashMap<EventName, Vec<EventHandler>>>>,
    /// Tool-call interceptors (run before tool execution, can modify/block args).
    tool_interceptors: Arc<Mutex<Vec<ToolCallInterceptor>>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Hooks {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(HashMap::new())),
            tool_interceptors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a handler for a named event. Handlers run in registration order.
    pub async fn on(&self, event_name: EventName, handler: EventHandler) {
        let mut map = self.handlers.lock().await;
        map.entry(event_name).or_default().push(handler);
    }

    /// Register a tool-call interceptor. Runs before tool execution.
    /// Return `Some(modified_args)` to proceed, `None` to block.
    pub async fn on_tool_call(&self, interceptor: ToolCallInterceptor) {
        let mut list = self.tool_interceptors.lock().await;
        list.push(interceptor);
    }

    /// Run all handlers for a given event. Returns the final action.
    /// If any handler returns `Blocked`, the event is blocked.
    /// If any handler returns `Handled`, default behavior is skipped.
    pub async fn run_handlers(&self, event: &Event) -> HookAction {
        let map = self.handlers.lock().await;
        let event_name = event_name_of(event);
        if let Some(handlers) = map.get(event_name) {
            for handler in handlers {
                let action = handler(event);
                match action {
                    HookAction::Blocked => return HookAction::Blocked,
                    HookAction::Handled => return HookAction::Handled,
                    HookAction::Continue => {}
                }
            }
        }
        HookAction::Continue
    }

    /// Run all tool-call interceptors on the given tool call.
    /// Returns `None` to block, `Some(args)` to proceed.
    pub async fn run_tool_interceptors(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<serde_json::Value> {
        let list = self.tool_interceptors.lock().await;
        let mut current = args.clone();
        for interceptor in list.iter() {
            match interceptor(tool_name, &current) {
                Some(modified) => current = modified,
                None => return None, // blocked
            }
        }
        Some(current)
    }

    /// Hot-reload: drop all handlers and interceptors. Call `/reload` to rebuild.
    /// Documented: drops in-memory state, external tools re-register after.
    pub async fn reload(&self) {
        let mut map = self.handlers.lock().await;
        map.clear();
        let mut list = self.tool_interceptors.lock().await;
        list.clear();
        tracing::info!("钩子已重新加载（所有处理器已清空）");
    }

    /// Check if any handlers are registered for a given event.
    pub async fn has_handlers(&self, event_name: EventName) -> bool {
        let map = self.handlers.lock().await;
        map.get(event_name).is_some_and(|v| !v.is_empty())
    }
}

/// Map an Event variant to its string name for hook lookup.
fn event_name_of(ev: &Event) -> EventName {
    match ev {
        Event::MessageStart { .. } => "message_start",
        Event::MessageUpdate { .. } => "message_update",
        Event::MessageEnd { .. } => "message_end",
        Event::ToolExecutionStart { .. } => "tool_execution_start",
        Event::ToolExecutionUpdate { .. } => "tool_execution_update",
        Event::ToolExecutionEnd { .. } => "tool_execution_end",
        Event::TurnStart { .. } => "turn_start",
        Event::TurnEnd { .. } => "turn_end",
        Event::AgentStart => "agent_start",
        Event::AgentEnd => "agent_end",
        Event::AgentSettled => "agent_settled",
        Event::CompactionStart { .. } => "compaction_start",
        Event::CompactionEnd { .. } => "compaction_end",
        Event::QueueUpdate { .. } => "queue_update",
        Event::ModelSelect { .. } => "model_select",
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn hook_handler_runs_on_event() {
        let hooks = Hooks::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        hooks
            .on(
                "agent_start",
                Arc::new(move |_ev| {
                    c.fetch_add(1, Ordering::SeqCst);
                    HookAction::Continue
                }),
            )
            .await;

        let action = hooks.run_handlers(&Event::AgentStart).await;
        assert_eq!(action, HookAction::Continue);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn blocked_event_stops_chain() {
        let hooks = Hooks::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();
        hooks
            .on(
                "turn_start",
                Arc::new(move |_ev| {
                    c1.fetch_add(1, Ordering::SeqCst);
                    HookAction::Blocked
                }),
            )
            .await;
        hooks
            .on(
                "turn_start",
                Arc::new(move |_ev| {
                    c2.fetch_add(1, Ordering::SeqCst);
                    HookAction::Continue
                }),
            )
            .await;

        let action = hooks.run_handlers(&Event::TurnStart { index: 0 }).await;
        assert_eq!(action, HookAction::Blocked);
        // second handler should NOT have run
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tool_call_interceptor_modifies_args() {
        let hooks = Hooks::new();
        hooks
            .on_tool_call(Arc::new(|name, args| {
                if name == "bash" {
                    // inject a wrapper
                    let mut modified = args.clone();
                    if let Some(obj) = modified.as_object_mut() {
                        obj.insert(
                            "wrapped".to_string(),
                            serde_json::Value::Bool(true),
                        );
                    }
                    Some(modified)
                } else {
                    Some(args.clone())
                }
            }))
            .await;

        let args = serde_json::json!({"cmd": "ls"});
        let result = hooks.run_tool_interceptors("bash", &args).await;
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result["cmd"], "ls");
        assert_eq!(result["wrapped"], true);
    }

    #[tokio::test]
    async fn tool_call_interceptor_blocks() {
        let hooks = Hooks::new();
        hooks
            .on_tool_call(Arc::new(|_name, _args| {
                None // block all tool calls
            }))
            .await;

        let args = serde_json::json!({"cmd": "rm -rf /"});
        let result = hooks.run_tool_interceptors("bash", &args).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn reload_clears_all_handlers() {
        let hooks = Hooks::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        hooks
            .on(
                "agent_start",
                Arc::new(move |_ev| {
                    c.fetch_add(1, Ordering::SeqCst);
                    HookAction::Continue
                }),
            )
            .await;

        hooks.reload().await;

        let action = hooks.run_handlers(&Event::AgentStart).await;
        assert_eq!(action, HookAction::Continue);
        assert_eq!(counter.load(Ordering::SeqCst), 0); // handler was cleared
    }

    #[tokio::test]
    async fn event_name_mapping() {
        assert_eq!(event_name_of(&Event::AgentStart), "agent_start");
        assert_eq!(
            event_name_of(&Event::ToolExecutionStart {
                tool_call_id: "x".into(),
                tool_name: "bash".into(),
            }),
            "tool_execution_start"
        );
        assert_eq!(
            event_name_of(&Event::TurnStart { index: 1 }),
            "turn_start"
        );
    }
}
