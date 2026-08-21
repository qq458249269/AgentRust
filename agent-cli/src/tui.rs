//! Interactive TUI: chat view + `/settings` sub-view (vendor/api/url/key/model form).
//! All configuration lives in auth.json (no built-in model catalog — configure as you go).

use crate::{client, CommonArgs};
use agent_ai::model::{Model, ThinkingLevel};
use agent_ai::provider::{
    read_auth_json, write_auth_json, ChatMessage, Part, ProviderClient, ProviderKind,
    ProviderRequest,
};
use agent_session::AgentSession;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::sync::mpsc;
use std::time::Duration;

pub async fn run(_session: AgentSession, _cli: &CommonArgs) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let res = run_loop(&mut terminal).await;
    ratatui::restore();
    res
}

/// One chat message shown in the transcript.
struct ChatItem {
    role: String,
    text: String,
}

impl ChatItem {
    fn new(role: &str, text: &str) -> Self {
        Self {
            role: role.to_string(),
            text: text.to_string(),
        }
    }
}

/// Settings form field (kind, base_url, api_key, model id).
struct FormField {
    label: &'static str,
    value: String,
    hint: String,
}

const FIELD_LABELS: [(&str, &str); 4] = [
    (
        "kind",
        "anthropic messages | openai chat | openai responses | deepseek chat",
    ),
    (
        "base url",
        "leave empty for provider default; e.g. https://api.anthropic.com/v1/messages",
    ),
    ("api key", "leave empty to fall back to env var"),
    (
        "model id",
        "e.g. claude-sonnet-4-5, gpt-4o-mini, deepseek-chat",
    ),
];

struct ChatApp {
    mode: Mode,
    history: Vec<ChatItem>,
    input: String,
    busy: bool,
    stream_tx: Option<mpsc::Sender<StreamMsg>>,
    stream_rx: Option<mpsc::Receiver<StreamMsg>>,
    status: String,
    form: Vec<FormField>,
    form_active: usize,
    streaming_item: Option<usize>,
    usage_str: String,
}

enum Mode {
    Chat,
    Settings,
}

enum StreamMsg {
    Delta(String),
    Done(String), // stop reason
    Err(String),
}

impl ChatApp {
    fn new() -> Self {
        let form: Vec<FormField> = FIELD_LABELS
            .into_iter()
            .map(|(label, hint)| FormField {
                label,
                value: String::new(),
                hint: hint.to_string(),
            })
            .collect();
        Self {
            mode: Mode::Chat,
            history: vec![ChatItem::new(
                "system",
                "AgentRust — type /settings to configure provider, /help for commands",
            )],
            input: String::new(),
            busy: false,
            stream_tx: None,
            stream_rx: None,
            status: String::new(),
            form,
            form_active: 0,
            streaming_item: None,
            usage_str: String::new(),
        }
    }

    /// Load current config into the settings form from auth.json.
    fn load_form(&mut self) {
        let root = read_auth_json();
        self.form[0].value = root
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("openai chat")
            .to_string();
        let vendor = self.form[0]
            .value
            .split_whitespace()
            .next()
            .unwrap_or("openai");
        let sec = root.get(vendor);
        self.form[1].value = sec
            .and_then(|s| s.get("base_url"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.form[2].value = sec
            .and_then(|s| s.get("api_key"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.form[3].value = root
            .get("default_model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
    }

    /// Persist the current form to auth.json. Returns an error string on validation failure.
    fn save_form(&mut self) -> Result<(), String> {
        let kind = ProviderKind::parse(&self.form[0].value)
            .ok_or_else(|| format!("unknown kind: '{}'", self.form[0].value))?;
        let mut patch = serde_json::Map::new();
        patch.insert("provider".into(), serde_json::json!(kind.display()));
        let mut sec = serde_json::Map::new();
        if !self.form[1].value.is_empty() {
            sec.insert("base_url".into(), serde_json::json!(self.form[1].value));
        }
        if !self.form[2].value.is_empty() {
            sec.insert("api_key".into(), serde_json::json!(self.form[2].value));
        }
        patch.insert(kind.id().into(), serde_json::Value::Object(sec));
        if !self.form[3].value.is_empty() {
            patch.insert(
                "default_model".into(),
                serde_json::json!(self.form[3].value),
            );
        }
        let merged = serde_json::Value::Object(patch);
        match write_auth_json(&merged) {
            Ok(Some(path)) => {
                self.status = format!("saved → {}", path.display());
                Ok(())
            }
            Ok(None) => Err("no HOME/USERPROFILE to write auth.json".into()),
            Err(e) => Err(format!("write failed: {e}")),
        }
    }

    fn send(&mut self, text: String) {
        if self.busy {
            return;
        }
        self.history.push(ChatItem::new("user", &text));
        self.input.clear();
        let (tx, rx) = mpsc::channel::<StreamMsg>();
        self.stream_tx = Some(tx.clone());
        self.stream_rx = Some(rx);
        self.busy = true;
        self.status = "streaming…".to_string();
        let item_idx = self.history.len();
        self.history.push(ChatItem::new("assistant", ""));
        self.streaming_item = Some(item_idx);

        tokio::spawn(async move {
            let root = read_auth_json();
            let kind_s = root
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or("openai chat");
            let Some(kind) = ProviderKind::parse(kind_s) else {
                let _ = tx.send(StreamMsg::Err(
                    "no valid provider configured; run /settings".into(),
                ));
                return;
            };
            let sec = root.get(kind.id());
            let api_key = sec
                .and_then(|s| s.get("api_key"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| kind.resolve_key(None))
                .unwrap_or_default();
            let model_id = root
                .get("default_model")
                .and_then(|v| v.as_str())
                .unwrap_or(match kind.id() {
                    "anthropic" => "claude-sonnet-4-5",
                    "openai" => "gpt-4o-mini",
                    _ => "deepseek-chat",
                })
                .to_string();

            let model = Model {
                provider: kind.id().to_string(),
                id: model_id,
                context_window: 200_000,
                max_tokens: 4096,
            };
            let mut pc = ProviderClient::new();
            pc.setup(kind, Some(api_key), None);
            let provider = match pc.provider_for(&model) {
                Some(p) => p,
                None => {
                    let _ = tx.send(StreamMsg::Err(format!("no provider for {}", model.id)));
                    return;
                }
            };
            let req = ProviderRequest {
                model,
                system: String::new(),
                messages: vec![ChatMessage {
                    role: "user".into(),
                    parts: vec![Part::Text { text }],
                }],
                thinking: ThinkingLevel::Off,
                max_tokens: 4096,
                tools: Vec::new(),
            };
            let resp = match provider.chat(client(), &req).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(StreamMsg::Err(format!("{e}")));
                    return;
                }
            };
            match resp {
                agent_ai::provider::ProviderResponse::Stream(mut sr) => {
                    while let Some(ev) = sr.next().await {
                        match ev {
                            Ok(agent_ai::stream::StreamEvent::TextDelta { delta }) => {
                                if tx.send(StreamMsg::Delta(delta)).is_err() {
                                    return;
                                }
                            }
                            Ok(agent_ai::stream::StreamEvent::Usage { usage }) => {
                                let _ = tx.send(StreamMsg::Done(format!(
                                    "usage: in={} out={} cache_read={} cache_write={} total={}",
                                    usage.input,
                                    usage.output,
                                    usage.cache_read,
                                    usage.cache_write,
                                    usage.total
                                )));
                            }
                            Ok(agent_ai::stream::StreamEvent::Done { stop_reason }) => {
                                let _ = tx.send(StreamMsg::Done(format!("stop: {stop_reason:?}")));
                            }
                            Err(e) => {
                                let _ = tx.send(StreamMsg::Err(format!("{e}")));
                            }
                            _ => {}
                        }
                    }
                }
                agent_ai::provider::ProviderResponse::Done { text, usage, .. } => {
                    let _ = tx.send(StreamMsg::Delta(text));
                    let _ = tx.send(StreamMsg::Done(format!(
                        "usage: in={} out={}",
                        usage.input, usage.output
                    )));
                }
            }
        });
    }
}

async fn run_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
) -> anyhow::Result<()> {
    let mut app = ChatApp::new();
    app.load_form();

    loop {
        // drain stream messages: Usage first (carried as Done), final stop Done finishes the row.
        if let Some(rx) = &app.stream_rx {
            let mut finished = false;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    StreamMsg::Delta(d) => {
                        if let Some(idx) = app.streaming_item {
                            app.history[idx].text.push_str(&d);
                        }
                    }
                    // payload is the usage summary when emitted by the Usage event,
                    // or the stop reason when emitted by the final Done event.
                    StreamMsg::Done(s) => {
                        if s.starts_with("usage:") {
                            app.usage_str = s;
                        } else {
                            app.busy = false;
                            app.streaming_item = None;
                            app.status = format!(
                                "{}",
                                if app.usage_str.is_empty() {
                                    s
                                } else {
                                    app.usage_str.clone()
                                }
                            );
                            app.usage_str = String::new();
                        }
                    }
                    StreamMsg::Err(e) => {
                        app.status = format!("error: {e}");
                        app.history
                            .push(ChatItem::new("system", &format!("error: {e}")));
                        app.busy = false;
                        app.stream_rx = None;
                        app.stream_tx = None;
                        app.streaming_item = None;
                        finished = true;
                        break;
                    }
                }
            }
            if app.stream_rx.is_some() && !app.busy && app.streaming_item.is_none() {
                app.stream_rx = None;
                app.stream_tx = None;
            }
            if finished {
                continue;
            }
        }

        terminal.draw(|f| draw(f, &mut app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    if !handle_key(&mut app, k.code, k.modifiers) {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

fn draw(f: &mut Frame, app: &mut ChatApp) {
    match app.mode {
        Mode::Settings => draw_settings(f, app),
        Mode::Chat => draw_chat(f, app),
    }
}

fn draw_chat(f: &mut Frame, app: &ChatApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(f.area());

    // history
    let lines: Vec<Line> = app
        .history
        .iter()
        .flat_map(|item| {
            let role_color = match item.role.as_str() {
                "user" => Color::Cyan,
                "assistant" => Color::LightGreen,
                "system" => Color::DarkGray,
                _ => Color::White,
            };
            let item_lines: Vec<Line> = item
                .text
                .lines()
                .map(|l| {
                    Line::from(vec![
                        Span::styled(
                            format!("{} ", item.role),
                            Style::default().fg(role_color).add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(l),
                    ])
                })
                .collect();
            item_lines
        })
        .collect();
    let title = if app.busy {
        " AgentRust (streaming…) "
    } else {
        " AgentRust "
    };
    let hist = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    f.render_widget(hist, chunks[0]);

    // input
    let input = Paragraph::new(app.input.as_str())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" input (/settings /help /exit) "),
        );
    let input_area = chunks[1];
    f.render_widget(input, input_area);
    f.set_cursor_position((input_area.x + app.input.len() as u16 + 1, input_area.y + 1));

    // status
    let cfg = format!(
        "{} · model: {}",
        app.form[0].value,
        if app.form[3].value.is_empty() {
            "(not set; run /settings)"
        } else {
            &app.form[3].value
        }
    );
    let status = Paragraph::new(format!("{}  |  {}", cfg, app.status))
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(status, chunks[2]);

    // cursor on available message? keep simple
    let _ = app;
}

fn draw_settings(f: &mut Frame, app: &ChatApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(4),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(f.area());

    let mut lines: Vec<Line> = Vec::new();
    for (i, field) in app.form.iter().enumerate() {
        let label_style = if i == app.form_active {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        let masked = field.label == "api key" && !field.value.is_empty();
        let shown = if masked {
            "••••••".to_string()
        } else {
            field.value.clone()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {}:", field.label), label_style),
            Span::raw(" "),
            Span::styled(shown, Style::default().fg(Color::White)),
        ]));
        lines.push(Line::from(Span::styled(
            format!("    ↳ {}", field.hint),
            Style::default().fg(Color::DarkGray),
        )));
    }

    let form = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" /settings — provider "),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(form, chunks[0]);

    let help = Paragraph::new(
        " ↑/↓ or Tab select field · type to edit · Enter save & return · Esc cancel ",
    )
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[1]);

    // edit bar
    let status_bar = if app.form_active == 2 {
        Paragraph::new(" api key stored locally in auth.json (plaintext); chmod recommended on shared machines ")
            .style(Style::default().fg(Color::Yellow))
    } else {
        Paragraph::new(app.status.as_str())
    };
    f.render_widget(status_bar, chunks[2]);
}

/// Returns false to quit the loop.
fn handle_key(app: &mut ChatApp, code: KeyCode, mods: KeyModifiers) -> bool {
    match app.mode {
        Mode::Settings => handle_settings_key(app, code),
        Mode::Chat => handle_chat_key(app, code, mods),
    }
}

fn handle_settings_key(app: &mut ChatApp, code: KeyCode) -> bool {
    match code {
        KeyCode::Esc => {
            app.mode = Mode::Chat;
            app.status = "settings discarded".to_string();
            true
        }
        KeyCode::Up => {
            app.form_active = app.form_active.saturating_sub(1);
            true
        }
        KeyCode::Down => {
            app.form_active = (app.form_active + 1).min(app.form.len() - 1);
            true
        }
        KeyCode::Tab => {
            app.form_active = (app.form_active + 1) % app.form.len();
            true
        }
        KeyCode::Enter => {
            match app.save_form() {
                Ok(()) => {
                    app.mode = Mode::Chat;
                    app.status = "settings saved".to_string();
                }
                Err(e) => app.status = e,
            }
            true
        }
        KeyCode::Backspace => {
            app.form[app.form_active].value.pop();
            true
        }
        KeyCode::Char(c) => {
            app.form[app.form_active].value.push(c);
            true
        }
        _ => true,
    }
}

fn handle_chat_key(app: &mut ChatApp, code: KeyCode, mods: KeyModifiers) -> bool {
    if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
        return false; // Ctrl+C quit
    }
    match code {
        KeyCode::Enter => {
            let text = app.input.trim().to_string();
            if text.is_empty() {
                return true;
            }
            if text == "/exit" || text == "/quit" {
                return false;
            }
            match text.as_str() {
                "/settings" => {
                    app.load_form();
                    app.mode = Mode::Settings;
                    app.input.clear();
                    app.status = String::new();
                }
                "/help" => {
                    app.history.push(ChatItem::new(
                        "system",
                        "/settings — configure provider · /exit — quit · anything else — send to model",
                    ));
                    app.input.clear();
                }
                _ => app.send(text),
            }
            true
        }
        KeyCode::Backspace => {
            app.input.pop();
            true
        }
        KeyCode::Char(c) => {
            if c == '/' && app.input.is_empty() {
                app.input.push(c);
            } else {
                app.input.push(c);
            }
            true
        }
        _ => true,
    }
}
