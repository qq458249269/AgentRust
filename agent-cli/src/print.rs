//! Print mode: send prompts, stream-print text, print usage accounting, exit.

use crate::{CommonArgs, PrintArgs};
use agent_ai::model::{Model, ThinkingLevel};
use agent_ai::provider::{ChatMessage, Part, ProviderClient, ProviderKind, ProviderRequest};
use agent_ai::stream::StreamEvent;
use agent_session::AgentSession;

pub async fn run(_session: AgentSession, cli: &CommonArgs, args: &PrintArgs) -> anyhow::Result<()> {
    // kind first, then url + key (each: override -> auth.json -> env/default)
    let kind = ProviderKind::parse(&cli.provider)
        .ok_or_else(|| anyhow::anyhow!("unknown provider kind: {}", cli.provider))?;
    let model_id = cli
        .model
        .clone()
        .unwrap_or_else(|| "claude-sonnet-4-5".to_string());
    let model = Model {
        provider: kind.id().to_string(),
        id: model_id,
        context_window: 200_000,
        max_tokens: 4096,
    };

    let api_key = kind.resolve_key(cli.api_key.as_deref());
    let mut client = ProviderClient::new();
    client.setup(kind, api_key, cli.base_url.clone());

    let p = client
        .provider_for(&model)
        .ok_or_else(|| anyhow::anyhow!("no provider for model {}", model.id))?;

    let first = args.prompts.first().cloned().unwrap_or_default();
    let req = ProviderRequest {
        model,
        system: String::new(),
        messages: vec![ChatMessage {
            role: "user".into(),
            parts: vec![Part::Text { text: first }],
        }],
        thinking: ThinkingLevel::Off,
        max_tokens: 1024,
        tools: Vec::new(),
    };

    use anyhow::Context;
    let resp = p
        .chat(&crate::client(), &req)
        .await
        .context("provider chat")?;

    match resp {
        agent_ai::provider::ProviderResponse::Stream(mut sr) => {
            let mut usage = None;
            while let Some(ev) = sr.next().await {
                match ev? {
                    StreamEvent::TextDelta { delta } => print!("{delta}"),
                    StreamEvent::Usage { usage: u } => usage = Some(u),
                    StreamEvent::Done { .. } => break,
                    _ => {}
                }
            }
            println!();
            if let Some(u) = usage {
                eprintln!(
                    "usage: in={} out={} cache_read={} cache_write={} total={}",
                    u.input, u.output, u.cache_read, u.cache_write, u.total
                );
            }
        }
        agent_ai::provider::ProviderResponse::Done { text, usage, .. } => {
            println!("{text}");
            eprintln!(
                "usage: in={} out={} cache_read={} cache_write={} total={}",
                usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total
            );
        }
    }
    Ok(())
}
