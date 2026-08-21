//! Interactive TUI: chat view + `/settings` sub-view with M6 enhancements.
//!
//! M6 features:
//! - Input history (↑/↓ recall previous inputs)
//! - Scroll support (PgUp/PgDn to scroll chat history)
//! - Streaming animation (spinner during generation)
//! - Better text rendering (long lines, code block hints)
//! - Keyboard shortcut hints
//! - Status bar with message count + token stats
//! - Tab command completion

use crate::{client, CommonArgs};
use agent_ai::model::{Model, ThinkingLevel};
use agent_ai::provider::{
    read_auth_json, write_auth_json, ChatMessage, Part, ProviderClient, ProviderKind,
    ProviderRequest,
};
use agent_session::AgentSession;
use chrono::Local;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use tokio::sync::mpsc as tokio_mpsc;
use std::time::{Duration, Instant};

pub async fn run(_session: AgentSession, _cli: &CommonArgs) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let res = run_loop(&mut terminal).await;
    ratatui::restore();
    res
}

// ─── Data Types ─────────────────────────────────────────────────────

/// One chat message shown in the transcript.
struct ChatItem {
    role: String,
    text: String,
    /// 发送/接收时间（HH:MM:SS）
    time: String,
}

impl ChatItem {
    fn new(role: &str, text: &str) -> Self {
        Self {
            role: role.to_string(),
            text: text.to_string(),
            time: Local::now().format("%H:%M:%S").to_string(),
        }
    }
}

/// Settings form field.
struct FormField {
    label: &'static str,
    value: String,
    hint: String,
}

const KIND_OPTIONS: [&str; 4] = [
    "anthropic messages",
    "openai chat",
    "openai responses",
    "deepseek chat",
];

const FORM_LABELS: [(&str, &str); 4] = [
    ("类型", "← → 切换类型"),
    (
        "接口地址",
        "填到前缀即可，例如 https://api.openai.com/v1；留空用默认",
    ),
    ("API 密钥", "留空则尝试环境变量"),
    (
        "模型 ID",
        "例如 claude-sonnet-4-5, gpt-4o-mini, deepseek-chat",
    ),
];

/// 斜杠命令: (触发, 说明).
const COMMANDS: [(&str, &str); 6] = [
    ("/设置", "配置接口（类型/地址/密钥/模型）"),
    ("/帮助", "查看帮助"),
    ("/清空", "清空会话记录"),
    ("/统计", "显示统计信息"),
    ("/模型", "显示当前模型"),
    ("/退出", "退出"),
];

enum Mode {
    Chat,
    Settings,
}

enum StreamMsg {
    Delta(String),
    Done(String),
    Err(String),
}

// ─── App State ──────────────────────────────────────────────────────

struct ChatApp {
    mode: Mode,
    history: Vec<ChatItem>,
    input: String,
    /// Input history for ↑/↓ recall
    input_history: Vec<String>,
    input_history_idx: Option<usize>,
    busy: bool,
    stream_tx: Option<tokio_mpsc::Sender<StreamMsg>>,
    stream_rx: Option<tokio_mpsc::Receiver<StreamMsg>>,
    status: String,
    form: Vec<FormField>,
    form_active: usize,
    streaming_item: Option<usize>,
    usage_str: String,
    /// when Some(selected_index), the `/` command menu is open
    cmd_menu: Option<usize>,
    /// Scroll offset for chat history (0 = bottom, positive = scrolled up)
    scroll_offset: u16,
    /// Spinner frame index for streaming animation
    spinner_frame: usize,
    /// Last spinner update time
    spinner_tick: Instant,
    /// Total tokens used in this session
    total_input_tokens: u64,
    total_output_tokens: u64,

}

impl ChatApp {
    fn new() -> Self {
        let form: Vec<FormField> = FORM_LABELS
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
                "AgentRust · 输入 / 打开命令菜单，其他内容发送给模型",
            )],
            input: String::new(),
            input_history: Vec::new(),
            input_history_idx: None,
            busy: false,
            stream_tx: None,
            stream_rx: None,
            status: String::new(),
            form,
            form_active: 0,
            streaming_item: None,
            usage_str: String::new(),
            cmd_menu: None,
            scroll_offset: 0,
            spinner_frame: 0,
            spinner_tick: Instant::now(),
            total_input_tokens: 0,
            total_output_tokens: 0,

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

    /// Persist the current form to auth.json.
    fn save_form(&mut self) -> Result<(), String> {
        let kind = ProviderKind::parse(&self.form[0].value)
            .ok_or_else(|| format!("未知类型: '{}'", self.form[0].value))?;
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
                self.status = format!("已保存 → {}", path.display());
                Ok(())
            }
            Ok(None) => Err("没有 HOME/USERPROFILE 目录，无法写入 auth.json".into()),
            Err(e) => Err(format!("写入失败：{e}")),
        }
    }

    /// Push input to history for ↑/↓ recall.
    fn push_input_history(&mut self) {
        if !self.input.trim().is_empty() {
            self.input_history.push(self.input.clone());
        }
        self.input_history_idx = None;
    }

    /// Recall previous input from history (↑ key).
    fn recall_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let idx = match self.input_history_idx {
            Some(i) if i > 0 => i - 1,
            Some(i) => i,
            None => self.input_history.len() - 1,
        };
        self.input_history_idx = Some(idx);
        self.input = self.input_history[idx].clone();
    }

    /// Recall next input from history (↓ key).
    fn recall_next(&mut self) {
        if let Some(idx) = self.input_history_idx {
            if idx + 1 < self.input_history.len() {
                self.input_history_idx = Some(idx + 1);
                self.input = self.input_history[idx + 1].clone();
            } else {
                self.input_history_idx = None;
                self.input.clear();
            }
        }
    }

    /// Tab-complete: find matching command and fill it in.
    fn tab_complete(&mut self) {
        let input = self.input.trim_start();
        if input.is_empty() || !input.starts_with('/') {
            return;
        }
        let matches: Vec<&str> = COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(input))
            .map(|(cmd, _)| *cmd)
            .collect();
        if matches.len() == 1 {
            self.input = format!("{} ", matches[0]);
        } else if matches.len() > 1 {
            // Show available matches
            let list: Vec<&str> = matches.to_vec();
            self.history.push(ChatItem::new(
                "system",
                &format!("可补全: {}", list.join(", ")),
            ));
        }
    }

    fn send(&mut self, text: String) {
        if self.busy {
            self.status = "正在生成中，请稍候…".to_string();
            return;
        }
        self.push_input_history();
        self.history.push(ChatItem::new("user", &text));
        self.input.clear();
        let (tx, rx) = tokio_mpsc::channel::<StreamMsg>(256);
        self.stream_tx = Some(tx.clone());
        self.stream_rx = Some(rx);
        self.busy = true;
        self.status = "生成中…".to_string();
        self.scroll_offset = 0; // auto-scroll to bottom
        let item_idx = self.history.len();
        self.history.push(ChatItem::new("assistant", ""));
        self.streaming_item = Some(item_idx);

        tokio::spawn(async move {
            let root = read_auth_json();
            tracing::info!("TUI 发送任务启动");
            let kind_s = root
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or("openai chat");
            let Some(kind) = ProviderKind::parse(kind_s) else {
                let _ = tx.send(StreamMsg::Err(
                    "没有配置有效的 provider；请执行 /settings 设置".into(),
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
                    let _ = tx.send(StreamMsg::Err(format!("没有可用的 provider：{}", model.id)));
                    return;
                }
            };
            tracing::info!("请求: provider={}", kind_s);
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
                Ok(r) => {
                    tracing::info!("收到 HTTP 响应");
                    r
                }
                Err(e) => {
                    tracing::error!("请求失败: {e}");
                    let _ = tx.send(StreamMsg::Err(format!("{e}")));
                    return;
                }
            };
            match resp {
                agent_ai::provider::ProviderResponse::Stream(mut sr) => {
                    tracing::info!("开始接收流式响应");
                    while let Some(ev) = sr.next().await {
                        match ev {
                            Ok(agent_ai::stream::StreamEvent::TextDelta { delta }) => {
                                tracing::trace!("收到 Delta: {} 字节", delta.len());
                                let _ = tx.send(StreamMsg::Delta(delta));
                            }
                            Ok(agent_ai::stream::StreamEvent::Usage { usage }) => {
                                tracing::info!("收到 Usage: in={} out={}", usage.input, usage.output);
                                let _ = tx.send(StreamMsg::Done(format!(
                                    "用量：输入={} 输出={} 缓存读={} 缓存写={} 总计={}",
                                    usage.input,
                                    usage.output,
                                    usage.cache_read,
                                    usage.cache_write,
                                    usage.total
                                )));
                            }
                            Ok(agent_ai::stream::StreamEvent::Done { stop_reason }) => {
                                tracing::info!("收到 Done: {stop_reason:?}");
                                let _ = tx.send(StreamMsg::Done(format!("结束：{stop_reason:?}")));
                            }
                            Err(e) => {
                                tracing::error!("流错误: {e}");
                                let _ = tx.send(StreamMsg::Err(format!("{e}")));
                            }
                            _ => {}
                        }
                    }
                    tracing::info!("流式响应接收完毕");
                }
                agent_ai::provider::ProviderResponse::Done { text, usage, .. } => {
                    tracing::info!("收到一次性响应: {} 字节", text.len());
                    let _ = tx.send(StreamMsg::Delta(text));
                    let _ = tx.send(StreamMsg::Done(format!(
                        "用量：输入={} 输出={}",
                        usage.input, usage.output
                    )));
                }
            }
            tracing::info!("TUI 发送任务结束");
        });
    }
}

// ─── Spinner Animation ──────────────────────────────────────────────

const SPINNER_FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

fn spinner_char(frame: usize) -> &'static str {
    SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]
}

// ─── Main Loop ──────────────────────────────────────────────────────

async fn run_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
) -> anyhow::Result<()> {
    let mut app = ChatApp::new();
    app.load_form();

    loop {
        // drain stream messages: collect first to avoid borrow conflicts
        let mut msgs = Vec::new();
        if let Some(rx) = &mut app.stream_rx {
            while let Ok(msg) = rx.try_recv() {
                msgs.push(msg);
            }
        }
        let mut finished = false;
        for msg in msgs {
            match msg {
                StreamMsg::Delta(d) => {
                    if let Some(idx) = app.streaming_item {
                        app.history[idx].text.push_str(&d);
                    }
                }
                StreamMsg::Done(s) => {
                    if s.starts_with("用量：") {
                        app.usage_str = s.clone();
                        for part in s.split_whitespace() {
                            if let Some(val) = part.strip_prefix("输入=") {
                                if let Ok(n) = val.parse::<u64>() {
                                    app.total_input_tokens += n;
                                }
                            }
                            if let Some(val) = part.strip_prefix("输出=") {
                                if let Ok(n) = val.parse::<u64>() {
                                    app.total_output_tokens += n;
                                }
                            }
                        }
                    } else {
                        // 结束事件：重置 busy 状态
                        tracing::info!("TUI 处理 Done: {s}");
                        app.busy = false;
                        if let Some(idx) = app.streaming_item {
                            app.history[idx].time =
                                Local::now().format("%H:%M:%S").to_string();
                        }
                        app.streaming_item = None;
                        app.status = if app.usage_str.is_empty() {
                            s
                        } else {
                            app.usage_str.clone()
                        };
                        app.usage_str = String::new();
                    }
                }
                StreamMsg::Err(e) => {
                    app.status = format!("错误：{e}");
                    app.history
                        .push(ChatItem::new("system", &format!("错误：{e}")));
                    app.busy = false;
                    app.stream_rx = None;
                    app.stream_tx = None;
                    app.streaming_item = None;
                    finished = true;
                }
            }
        }
        if finished {
            app.stream_rx = None;
            app.stream_tx = None;
            continue;
        }
        if app.stream_rx.is_some() && !app.busy && app.streaming_item.is_none() {
            app.stream_rx = None;
            app.stream_tx = None;
        }

        // Update spinner
        if app.busy && app.spinner_tick.elapsed() > Duration::from_millis(80) {
            app.spinner_frame = app.spinner_frame.wrapping_add(1);
            app.spinner_tick = Instant::now();
        }

        terminal.draw(|f| draw(f, &mut app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press
                    && !handle_key(&mut app, k.code, k.modifiers)
                {
                    break;
                }
            }
        }
    }
    Ok(())
}

// ─── Drawing ────────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &mut ChatApp) {
    match app.mode {
        Mode::Settings => draw_settings(f, app),
        Mode::Chat => draw_chat(f, app),
    }
}

fn draw_chat(f: &mut Frame, app: &mut ChatApp) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // chat history
            Constraint::Length(3), // input
            Constraint::Length(1), // status bar
        ])
        .split(area);

    // ── Chat History ──
    let mut lines: Vec<Line> = Vec::new();
    for item in &app.history {
        let (role_color, role_label) = match item.role.as_str() {
            "user" => (Color::Cyan, "你"),
            "assistant" => (Color::LightGreen, "助手"),
            "system" => (Color::DarkGray, "系统"),
            _ => (Color::White, item.role.as_str()),
        };

        // Split text into lines and add role prefix to first line
        let text_lines: Vec<&str> = item.text.lines().collect();
        if text_lines.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {role_label}  "),
                    Style::default()
                        .fg(role_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(""),
            ]));
        } else {
            for (i, line_text) in text_lines.iter().enumerate() {
                let mut spans = Vec::new();
                if i == 0 {
                    spans.push(Span::styled(
                        format!("  {role_label}  "),
                        Style::default()
                            .fg(role_color)
                            .add_modifier(Modifier::BOLD),
                    ));
                    spans.push(Span::styled(
                        format!("[{}] ", item.time),
                        Style::default().fg(Color::DarkGray),
                    ));
                } else {
                    // Continuation lines: align with text start
                    spans.push(Span::raw("        "));
                }

                // Highlight code blocks
                if line_text.starts_with("```") {
                    spans.push(Span::styled(
                        line_text.to_string(),
                        Style::default().fg(Color::DarkGray),
                    ));
                } else if line_text.starts_with('`') && line_text.ends_with('`') {
                    // inline code
                    spans.push(Span::styled(
                        line_text.to_string(),
                        Style::default().fg(Color::Yellow),
                    ));
                } else {
                    spans.push(Span::raw(line_text.to_string()));
                }
                lines.push(Line::from(spans));
            }
        }
        // Add blank line between messages
        lines.push(Line::from(""));
    }

    let title = if app.busy {
        format!(" AgentRust {} ", spinner_char(app.spinner_frame))
    } else {
        " AgentRust ".to_string()
    };

    let hist = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    f.render_widget(hist, chunks[0]);

    // ── Input Box ──
    let input_block_title = if app.busy {
        " 输入（生成中…按 Esc 取消） "
    } else {
        " 输入（/ 打开菜单 · Tab 补全 · ↑↓ 历史） "
    };
    let input = Paragraph::new(app.input.as_str())
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(input_block_title));
    let input_area = chunks[1];
    f.render_widget(input, input_area);
    // Place cursor at end of input
    let cursor_x = input_area.x + 1 + app.input.len() as u16;
    let cursor_y = input_area.y + 1;
    f.set_cursor_position((cursor_x.min(input_area.right() - 1), cursor_y));

    // ── Command Menu Overlay ──
    if app.cmd_menu.is_some() {
        let menu_h = COMMANDS.len() as u16 + 2;
        let menu_area = Rect {
            x: input_area.x + 1,
            y: input_area.y.saturating_sub(menu_h),
            width: input_area.width.saturating_sub(2).min(60),
            height: menu_h,
        };
        let menu_lines: Vec<Line> = COMMANDS
            .iter()
            .enumerate()
            .map(|(i, (cmd, desc))| {
                let selected = Some(i) == app.cmd_menu;
                if selected {
                    Line::from(vec![
                        Span::styled(
                            format!("▸ {cmd}"),
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("  {desc}"),
                            Style::default().fg(Color::Black).bg(Color::Cyan),
                        ),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(format!("  {cmd}"), Style::default().fg(Color::White)),
                        Span::styled(
                            format!("  {desc}"),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ])
                }
            })
            .collect();
        let menu = Paragraph::new(menu_lines)
            .block(Block::default().borders(Borders::ALL).title(" 命令 "))
            .style(Style::default().bg(Color::Reset));
        f.render_widget(menu, menu_area);
    }

    // ── Status Bar ──
    let model_str = if app.form[3].value.is_empty() {
        "（未设置）".to_string()
    } else {
        app.form[3].value.clone()
    };
    let msg_count = app.history.len();
    let stats = if app.total_input_tokens + app.total_output_tokens > 0 {
        format!(
            "tok: ↓{} ↑{}",
            app.total_input_tokens, app.total_output_tokens
        )
    } else {
        String::new()
    };
    let status_text = format!(
        "{} · {} · msgs:{} {}  │  {}",
        app.form[0].value, model_str, msg_count, stats, app.status
    );
    let status = Paragraph::new(status_text).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status, chunks[2]);
}

fn draw_settings(f: &mut Frame, app: &ChatApp) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(4),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

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
        let shown: String = if i == 0 {
            let idx = KIND_OPTIONS
                .iter()
                .position(|o| *o == field.value)
                .unwrap_or(0);
            format!("⟨ {} ⟩  （← → 切换）", KIND_OPTIONS[idx])
        } else if field.label == "API 密钥" && !field.value.is_empty() {
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
                .title(" /settings · 配置 "),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(form, chunks[0]);

    let help = Paragraph::new(
        " ↑/↓ 或 Tab 选择字段 · ← → 切换类型 · 输入编辑 · Enter 保存并返回 · Esc 取消 ",
    )
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[1]);

    let status_bar = if app.form_active == 2 {
        Paragraph::new(" API 密钥保存在本地 auth.json（明文）；共享机器建议限制文件权限 ")
            .style(Style::default().fg(Color::Yellow))
    } else {
        Paragraph::new(app.status.as_str())
    };
    f.render_widget(status_bar, chunks[2]);
}

// ─── Key Handling ───────────────────────────────────────────────────

fn cycle_kind(current: &str, right: bool) -> &'static str {
    let cur = KIND_OPTIONS.iter().position(|o| *o == current).unwrap_or(0);
    KIND_OPTIONS[if right {
        (cur + 1) % KIND_OPTIONS.len()
    } else {
        (cur + KIND_OPTIONS.len() - 1) % KIND_OPTIONS.len()
    }]
}

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
            app.status = "已放弃修改".to_string();
        }
        KeyCode::Up => {
            app.form_active = app.form_active.saturating_sub(1);
        }
        KeyCode::Down => {
            app.form_active = (app.form_active + 1).min(app.form.len() - 1);
        }
        KeyCode::Tab => {
            app.form_active = (app.form_active + 1) % app.form.len();
        }
        KeyCode::Left | KeyCode::Right
            if app.form_active == 0 => {
                app.form[0].value =
                    cycle_kind(&app.form[0].value, code == KeyCode::Right).to_string();
            }
        KeyCode::Enter => {
            if let Err(e) = app.save_form() {
                app.status = e;
            } else {
                app.mode = Mode::Chat;
                app.status = "设置已保存".to_string();
            }
        }
        KeyCode::Backspace
            if app.form_active != 0 => {
                app.form[app.form_active].value.pop();
            }
        KeyCode::Char(c)
            if app.form_active != 0 => {
                app.form[app.form_active].value.push(c);
            }
        _ => {}
    }
    true
}

fn handle_chat_key(app: &mut ChatApp, code: KeyCode, mods: KeyModifiers) -> bool {
    // Ctrl+C: quit
    if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
        return false;
    }

    // Ctrl+L: clear screen
    if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('l') {
        app.history.clear();
        app.history.push(ChatItem::new(
            "system",
            "AgentRust · 屏幕已清空",
        ));
        return true;
    }

    // Esc: cancel streaming or close command menu
    if code == KeyCode::Esc {
        if app.cmd_menu.is_some() {
            app.cmd_menu = None;
            app.input.clear();
            return true;
        }
        if app.busy {
            // Cancel streaming (best effort)
            app.busy = false;
            app.streaming_item = None;
            app.stream_rx = None;
            app.stream_tx = None;
            app.status = "已取消".to_string();
            return true;
        }
    }

    // PgUp/PgDn: scroll history
    if code == KeyCode::PageUp {
        app.scroll_offset = app.scroll_offset.saturating_add(5);
        return true;
    }
    if code == KeyCode::PageDown {
        app.scroll_offset = app.scroll_offset.saturating_sub(5);
        return true;
    }

    // Command menu active
    if app.cmd_menu.is_some() && app.input == "/" {
        match code {
            KeyCode::Up => {
                let cur = app.cmd_menu.unwrap_or(0);
                app.cmd_menu = Some((cur + COMMANDS.len() - 1) % COMMANDS.len());
            }
            KeyCode::Down => {
                let cur = app.cmd_menu.unwrap_or(0);
                app.cmd_menu = Some((cur + 1) % COMMANDS.len());
            }
            KeyCode::Enter => {
                let cmd = COMMANDS[app.cmd_menu.unwrap_or(0)].0;
                return run_command(app, cmd);
            }
            KeyCode::Esc => {
                app.cmd_menu = None;
                app.input.clear();
            }
            KeyCode::Backspace => {
                app.cmd_menu = None;
                app.input.clear();
            }
            KeyCode::Tab => {
                // Tab: accept current selection
                let cmd = COMMANDS[app.cmd_menu.unwrap_or(0)].0;
                return run_command(app, cmd);
            }
            KeyCode::Char('/') => {
                app.input.push('/');
            }
            _ => {}
        }
        return true;
    }

    match code {
        KeyCode::Enter => {
            let text = app.input.trim().to_string();
            if text.is_empty() {
                return true;
            }
            if text == "/退出" || text == "/quit" {
                return false;
            }
            match text.as_str() {
                "/设置" => {
                    app.load_form();
                    app.mode = Mode::Settings;
                    app.input.clear();
                    app.status = String::new();
                }
                "/帮助" => {
                    app.history.push(ChatItem::new(
                        "system",
                        "快捷键：\n  ↑/↓    翻阅输入历史\n  Tab    命令补全\n  PgUp/Dn 滚动聊天记录\n  Esc    取消生成\n  Ctrl+C  退出\n  Ctrl+L  清屏\n\n命令：\n  /设置    配置\n  /清空    清空记录\n  /统计    统计信息\n  /模型    当前模型\n  /帮助    帮助\n  /退出    退出",
                    ));
                    app.input.clear();
                }
                "/清空" => {
                    app.history.clear();
                    app.history.push(ChatItem::new(
                        "system",
                        "AgentRust · 记录已清空",
                    ));
                    app.total_input_tokens = 0;
                    app.total_output_tokens = 0;
                    app.input.clear();
                }
                "/统计" => {
                    let msg_count = app.history.len();
                    let user_msgs = app
                        .history
                        .iter()
                        .filter(|m| m.role == "user")
                        .count();
                    let asst_msgs = app
                        .history
                        .iter()
                        .filter(|m| m.role == "assistant")
                        .count();
                    app.history.push(ChatItem::new(
                        "system",
                        &format!(
                            "消息数: {} (用户: {}, 助手: {})\nToken: 输入 {} / 输出 {}",
                            msg_count, user_msgs, asst_msgs,
                            app.total_input_tokens, app.total_output_tokens
                        ),
                    ));
                    app.input.clear();
                }
                "/模型" => {
                    let model = if app.form[3].value.is_empty() {
                        "（未设置）".to_string()
                    } else {
                        app.form[3].value.clone()
                    };
                    app.history.push(ChatItem::new(
                        "system",
                        &format!("服务商: {}\n模型: {}", app.form[0].value, model),
                    ));
                    app.input.clear();
                }
                _ => app.send(text),
            }
        }
        KeyCode::Up => {
            app.recall_prev();
        }
        KeyCode::Down => {
            app.recall_next();
        }
        KeyCode::Tab => {
            app.tab_complete();
        }
        KeyCode::Backspace => {
            app.input.pop();
            if app.input.is_empty() {
                app.cmd_menu = None;
            }
        }
        KeyCode::Char(c) => {
            if c == '/' && app.input.is_empty() {
                app.input.push(c);
                app.cmd_menu = Some(0);
            } else if !app.input.starts_with('/') || c != '/' {
                app.input.push(c);
            }
        }
        _ => {}
    }
    true
}

fn run_command(app: &mut ChatApp, cmd: &str) -> bool {
    app.cmd_menu = None;
    app.input.clear();
    match cmd {
        "/设置" => {
            app.load_form();
            app.mode = Mode::Settings;
            app.status = String::new();
        }
        "/帮助" => {
            app.history.push(ChatItem::new(
                "system",
                "快捷键：\n  ↑/↓    翻阅输入历史\n  Tab    命令补全\n  PgUp/Dn 滚动聊天记录\n  Esc    取消生成\n  Ctrl+C  退出\n  Ctrl+L  清屏\n\n命令：\n  /设置    配置\n  /清空    清空记录\n  /统计    统计信息\n  /模型    当前模型\n  /帮助    帮助\n  /退出    退出",
            ));
        }
        "/清空" => {
            app.history.clear();
            app.history
                .push(ChatItem::new("system", "AgentRust · 记录已清空"));
            app.total_input_tokens = 0;
            app.total_output_tokens = 0;
        }
        "/统计" => {
            let msg_count = app.history.len();
            let user_msgs = app.history.iter().filter(|m| m.role == "user").count();
            let asst_msgs = app
                .history
                .iter()
                .filter(|m| m.role == "assistant")
                .count();
            app.history.push(ChatItem::new(
                "system",
                &format!(
                    "消息数: {} (用户: {}, 助手: {})\nToken: 输入 {} / 输出 {}",
                    msg_count, user_msgs, asst_msgs,
                    app.total_input_tokens, app.total_output_tokens
                ),
            ));
        }
        "/模型" => {
            let model = if app.form[3].value.is_empty() {
                "（未设置）".to_string()
            } else {
                app.form[3].value.clone()
            };
            app.history.push(ChatItem::new(
                "system",
                &format!("服务商: {}\n模型: {}", app.form[0].value, model),
            ));
        }
        "/退出" => return false,
        _ => {}
    }
    true
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_kind_wraps_forward() {
        assert_eq!(cycle_kind("anthropic messages", true), "openai chat");
        assert_eq!(cycle_kind("openai chat", true), "openai responses");
        assert_eq!(cycle_kind("openai responses", true), "deepseek chat");
        assert_eq!(cycle_kind("deepseek chat", true), "anthropic messages");
    }

    #[test]
    fn cycle_kind_wraps_backward() {
        assert_eq!(cycle_kind("anthropic messages", false), "deepseek chat");
        assert_eq!(cycle_kind("deepseek chat", false), "openai responses");
    }

    #[test]
    fn cycle_kind_unknown_falls_to_first() {
        assert_eq!(cycle_kind("bogus", true), "openai chat");
        assert_eq!(cycle_kind("bogus", false), "deepseek chat");
    }

    #[test]
    fn run_command_dispatches() {
        let mut app = ChatApp::new();
        assert!(run_command(&mut app, "/设置"));
        assert!(matches!(app.mode, Mode::Settings));
        app.mode = Mode::Chat;
        assert!(run_command(&mut app, "/清空"));
        assert_eq!(app.history.len(), 1);
        assert!(!run_command(&mut app, "/退出"));
    }

    #[test]
    fn input_history_recall() {
        let mut app = ChatApp::new();
        app.input_history = vec!["hello".into(), "world".into(), "foo".into()];
        app.recall_prev();
        assert_eq!(app.input, "foo");
        app.recall_prev();
        assert_eq!(app.input, "world");
        app.recall_next();
        assert_eq!(app.input, "foo");
        app.recall_next();
        assert!(app.input.is_empty());
    }

    #[test]
    fn spinner_cycles() {
        assert_eq!(spinner_char(0), "⠋");
        assert_eq!(spinner_char(1), "⠙");
        assert_eq!(spinner_char(8), "⠋"); // wraps
    }

    #[test]
    fn tab_complete_single_match() {
        let mut app = ChatApp::new();
        app.input = "/设".to_string();
        app.tab_complete();
        assert_eq!(app.input, "/设置 ");
    }

    #[test]
    fn tab_complete_no_match() {
        let mut app = ChatApp::new();
        app.input = "/xyz".to_string();
        app.tab_complete();
        assert_eq!(app.input, "/xyz"); // unchanged
    }
}
