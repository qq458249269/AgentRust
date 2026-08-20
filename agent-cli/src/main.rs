//! agent: rpc / json / print frontends over the same session runtime (mirooring pi modes).

mod json;
mod print;
mod rpc;

use clap::{Parser, Subcommand};

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
    #[arg(long, default_value = ".")]
    cwd: String,

    /// provider: anthropic | openai | deepseek ...
    #[arg(long, default_value = "anthropic")]
    provider: String,

    /// model id pattern, e.g. claude-sonnet-4-5, or provider/id
    #[arg(long)]
    model: Option<String>,

    /// prompt text for print mode (repeatable: follow-up messages)
    #[arg(long = "prompt", short = 'p', action = clap::ArgAction::Append)]
    prompts: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// RPC over stdin/stdout (JSONL, LF-only framing)
    Rpc,
    /// all events as JSON lines to stdout
    Json,
    /// print response(s) and exit; reads piped stdin into first prompt
    Print,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();
    let session = agent_session::AgentSession::open(None)?;

    match cli.mode {
        Mode::Rpc => rpc::run(session, &cli).await?,
        Mode::Json => json::run(session, &cli).await?,
        Mode::Print => print::run(session, &cli).await?,
    }
    Ok(())
}
