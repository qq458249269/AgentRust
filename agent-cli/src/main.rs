//! agent: rpc / json / print frontends over the same session runtime (mirooring pi modes).

mod json;
mod print;
mod rpc;
mod tui;

use clap::{Args, Parser, Subcommand};

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
    let session = agent_session::AgentSession::open(None)?;
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

    match cli.mode.unwrap_or(Mode::Tui) {
        Mode::Tui => tui::run(session, &common).await?,
        Mode::Rpc => rpc::run(session, &common).await?,
        Mode::Json => json::run(session, &common).await?,
        Mode::Print(args) => print::run(session, &common, &args).await?,
    }
    Ok(())
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
