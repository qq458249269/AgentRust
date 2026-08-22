//! agent: rpc / json / print frontends over the same session runtime (mirooring pi modes).

mod json;
mod print;
mod rpc;
mod tui;

use agent_ai::model::Model;
use agent_ai::provider::{ProviderClient, ProviderKind};
use agent_core::builtins;
use agent_session::AgentSession;
use clap::{Args, Parser, Subcommand};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "agent",
    version,
    about = "高性能编程智能体"
)]
struct Cli {
    /// run mode; empty = interactive TUI (double-click friendly)
    #[command(subcommand)]
    mode: Option<Mode>,

    /// 会话的工作目录
    #[arg(long, global = true, default_value = ".")]
    cwd: String,

    /// 无头模式的服务商类型覆盖（TUI 使用 auth.json）
    #[arg(long, global = true, default_value = "")]
    provider: String,

    /// 无头模式的模型 ID 覆盖
    #[arg(long, global = true)]
    model: Option<String>,

    /// 服务商基础 URL 覆盖（例如本地模拟服务器）
    #[arg(long, global = true)]
    base_url: Option<String>,

    /// API 密钥覆盖；否则使用 auth.json 然后环境变量
    #[arg(long, global = true)]
    api_key: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// interactive TUI (default)
    Tui,
    /// RPC over stdin/stdout (JSONL, LF-only framing)
    Rpc,
    /// all events as JSON lines to stdout
    Json,
    /// print response(s) and exit; reads piped stdin into first prompt
    Print(PrintArgs),
}

#[derive(Args, Debug)]
struct PrintArgs {
    /// prompt text for print mode (repeatable: follow-up messages)
    #[arg(long = "prompt", action = clap::ArgAction::Append)]
    prompts: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_ansi(false)
        .init();

    let cli = Cli::parse();
    let cwd = cli.cwd.clone();
    let provider = cli.provider.clone();
    let model = cli.model.clone();
    let base_url = cli.base_url.clone();
    let api_key = cli.api_key.clone();

    let common = crate::CommonArgs {
        cwd,
        provider,
        model,
        base_url,
        api_key,
    };

    // Create and configure AgentSession with provider, model, tools, system prompt
    let session = setup_session(&common)?;

    match cli.mode.unwrap_or(Mode::Tui) {
        Mode::Tui => tui::run(session, &common).await?,
        Mode::Rpc => rpc::run(session, &common).await?,
        Mode::Json => json::run(session, &common).await?,
        Mode::Print(args) => print::run(session, &common, &args).await?,
    }
    Ok(())
}

/// Configure an AgentSession with provider, model, built-in tools, and system prompt.
pub fn setup_session(cli: &CommonArgs) -> anyhow::Result<AgentSession> {
    let mut session = AgentSession::open(None)?;

    // Resolve provider config
    let root = agent_ai::provider::read_auth_json();
    let kind_s = if cli.provider.is_empty() {
        root.get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("openai chat")
    } else {
        &cli.provider
    };
    let kind = match ProviderKind::parse(kind_s) {
        Some(k) => k,
        None => {
            anyhow::bail!("未配置有效的服务商: {kind_s}");
        }
    };

    let api_key = kind.resolve_key(cli.api_key.as_deref()).unwrap_or_default();
    let model_id = cli
        .model
        .clone()
        .or_else(|| {
            root.get("default_model")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| match kind.id() {
            "anthropic" => "claude-sonnet-4-5".into(),
            "openai" => "gpt-4o-mini".into(),
            _ => "deepseek-chat".into(),
        });

    let model = Model {
        provider: kind.id().to_string(),
        id: model_id,
        context_window: 200_000,
        max_tokens: 4096,
    };

    // Build provider and set it on session
    let mut pc = ProviderClient::new();
    pc.setup(kind, Some(api_key), cli.base_url.clone());
    // Build provider using kind.build() for Arc<dyn ChatProvider>
    let url = kind.resolve_base_url(cli.base_url.as_deref());
    let url = kind.join_api_path(&url);
    let key = cli.api_key.clone().unwrap_or_else(|| {
        kind.resolve_key(None).unwrap_or_default()
    });
    let provider_arc: Arc<dyn agent_ai::provider::ChatProvider> =
        Arc::from(kind.build(Some(key), Some(url)));
    session.set_provider(provider_arc);
    session.set_model(model);

    // Register built-in tools
    builtins::register_builtins(&session.tool_registry);

    // Set system prompt with tool instructions
    let system = builtins::system_prompt_with_tools(
        "你是一个有工具能力的智能编程助手。\n\
你可以使用以下工具来帮助用户完成任务：\n\
- 执行 bash 命令\n- 读取、写入、编辑文件\n- 列出目录内容\n- 搜索文件 (glob)\n- 搜索文件内容 (grep)\n\
请积极使用工具来完成编程任务，而不是只给出建议。",
        &session.tool_registry,
    );
    session.set_system_prompt(system);

    Ok(session)
}

/// Shared HTTP client (HTTP/2 pooling). Owned once at the process level.
pub fn client() -> &'static agent_ai::Client {
    use std::sync::OnceLock;
    static CLIENT: OnceLock<agent_ai::Client> = OnceLock::new();
    CLIENT.get_or_init(agent_ai::Client::new)
}

/// Fields shared across subcommands (moved out of `cli` before dispatching).
#[derive(Clone, Debug)]
pub struct CommonArgs {
    pub cwd: String,
    pub provider: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}
