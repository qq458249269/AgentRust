//! agent: rpc / json / print frontends over the same session runtime (mirooring pi modes).

mod json;
mod print;
mod rpc;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "agent",
    version,
    about = "high-performance coding agent (DESIGN.md)"
)]
struct Cli {
    /// run mode; rpc is the embeddable default
    #[command(subcommand)]
    mode: Mode,

    /// working directory for the session
    #[arg(long, global = true, default_value = ".")]
    cwd: String,

    /// provider: anthropic | openai | deepseek ...
    #[arg(long, global = true, default_value = "anthropic")]
    provider: String,

    /// model id pattern, e.g. claude-sonnet-4-5, or provider/id
    #[arg(long, global = true)]
    model: Option<String>,

    /// override provider base URL (e.g. local mock server); default per provider
    #[arg(long, global = true)]
    base_url: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Mode {
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

    let common = crate::CommonArgs {
        cwd,
        provider,
        model,
        base_url,
    };

    match cli.mode {
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
}
