//! TUI ассистента: шапка с состоянием репозитория, лента ответов, поле ввода.
//! Команды выполняются в фоне по одной; прогресс — в строке состояния.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame, Terminal,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use llm_core::{McpStatus, ToolCallRecord};

use crate::context::{env_or, short, Assistant};
use crate::help;
use crate::input::InputLine;
use crate::markdown;
use crate::review::{self, Review, Target};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const COMMANDS: &str = "Команды:
  <вопрос> или /help <вопрос>   — вопрос о проекте (README + docs/ через RAG, git через MCP); /help без вопроса — обзор
  /review [base]                — ревью текущей ветки с незакоммиченным против base (по умолчанию main)
  /review <base>..<head>        — ревью ветки head против base (только закоммиченное)
  /review --wip                 — ревью только незакоммиченного
  /save [файл]                  — сохранить последнее ревью в Markdown (по умолчанию review.md)
  /diff [цель]                  — какие файлы изменены и дифф (цель — как у /review), без модели
  /status                       — git status
  /files [путь]                 — структура проекта или каталога
  /log [n]                      — последние коммиты
  /clear — очистить ленту · /? — эта справка · /quit или Esc — выход
Ввод: ←/→ — символ · Alt+←/→ или Ctrl+←/→ — слово · Home/End (Ctrl+A/E) — начало/конец строки ·
      Alt+Backspace (Ctrl+W) — удалить слово · Ctrl+U/K — удалить до начала/конца · ↑/↓ — история
Лента: PgUp/PgDn — прокрутка · Ctrl+Home/Ctrl+End — в начало/конец · Ctrl+L — очистить
Вызовы видны в ленте плашками « MCP » (git) и « RAG » (поиск по документации и коду): что, кто вызвал, результат,
а пока команда идёт — в строке состояния внизу.";

/// Команды TUI и подсказки к ним — для подсветки ввода и строки состояния.
const COMMAND_HINTS: &[(&str, &str)] = &[
    ("help", "/help <вопрос> — вопрос о проекте: поиск в документации и коде (RAG) и git (MCP); без вопроса — обзор"),
    ("review", "/review [base | base..head | --wip] — ревью ветки против base (main), двух веток или незакоммиченного"),
    ("save", "/save [файл] — сохранить последнее ревью (review.md; .json — в JSON)"),
    ("diff", "/diff [base | base..head | --wip] — изменённые файлы и дифф, без модели"),
    ("status", "/status — git status: что изменено и не закоммичено"),
    ("files", "/files [путь] — структура проекта или каталога"),
    ("log", "/log [n] — последние коммиты"),
    ("clear", "/clear — очистить ленту"),
    ("quit", "/quit — выход"),
    ("exit", "/exit — выход"),
    ("q", "/q — выход"),
    ("?", "/? — список команд"),
    ("commands", "/commands — список команд"),
];

/// Команда в начале строки: длина `/команды` в символах и подсказка к ней
/// (`None` — такой команды нет). Не команда — `None`.
fn command_at(text: &str) -> Option<(usize, Option<&'static str>)> {
    let rest = text.strip_prefix('/')?;
    let name: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
    let hint = COMMAND_HINTS.iter().find(|(n, _)| *n == name).map(|(_, h)| *h);
    Some((1 + name.chars().count(), hint))
}

/// Известная команда — голубым жирным, неизвестная — красным.
fn command_style(known: bool) -> Style {
    let color = if known { Color::LightCyan } else { Color::LightRed };
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

/// `text` кусками: символы `cmd` (диапазон символов) — стилем команды, остальное — `rest`.
fn highlight(text: &str, cmd: std::ops::Range<usize>, cmd_style: Style, rest: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let (a, b) = (cmd.start.min(chars.len()), cmd.end.min(chars.len()));
    [(0..a, rest), (a..b, cmd_style), (b..chars.len(), rest)]
        .into_iter()
        .filter(|(r, _)| !r.is_empty())
        .map(|(r, style)| Span::styled(chars[r].iter().collect::<String>(), style))
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Info,
    User,
    Context,
    /// Вызов MCP-инструмента: первая строка — плашка, дальше — результат.
    Mcp,
    McpError,
    /// Вызов поиска по индексам (search_docs/search_code) — та же плашка, но « RAG ».
    Rag,
    RagError,
    /// Ответ модели или ревью — Markdown.
    Answer,
    /// Вывод git-команд (/status, /diff, …) — как есть, у диффа цвет строк.
    Output,
    Error,
}

struct Entry {
    kind: Kind,
    text: String,
}

enum AppEvent {
    Progress(String),
    Done(Vec<Entry>),
    Review(Result<Review, String>),
    Git(String),
}

struct App {
    assistant: Arc<Assistant>,
    entries: Vec<Entry>,
    input: InputLine,
    history: Vec<String>,
    history_pos: Option<usize>,
    /// Что сейчас выполняется; `None` — ничего.
    busy: Option<String>,
    git: String,
    /// Прокрутка от низа ленты, в строках.
    scroll_back: usize,
    spinner: usize,
    last_review: Option<Review>,
}

pub async fn run(assistant: Assistant) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = event_loop(&mut terminal, Arc::new(assistant)).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableBracketedPaste, LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

async fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, assistant: Arc<Assistant>) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut app = App {
        assistant: assistant.clone(),
        // Лента пустая — справка по /?, подсказки к командам — в строке состояния.
        entries: Vec::new(),
        input: InputLine::default(),
        history: Vec::new(),
        history_pos: None,
        busy: None,
        git: "подключаю MCP git…".into(),
        scroll_back: 0,
        spinner: 0,
        last_review: None,
    };
    refresh_git(&assistant, &tx);

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        terminal.draw(|f| draw(f, &app))?;
        tokio::select! {
            _ = tick.tick() => {
                if app.busy.is_some() {
                    app.spinner = (app.spinner + 1) % SPINNER.len();
                }
            }
            Some(ev) = rx.recv() => match ev {
                AppEvent::Progress(msg) => {
                    if app.busy.is_some() {
                        app.busy = Some(msg);
                    }
                }
                AppEvent::Done(entries) => {
                    app.entries.extend(entries);
                    app.busy = None;
                    app.scroll_back = 0;
                    refresh_git(&assistant, &tx);
                }
                AppEvent::Review(result) => {
                    match result {
                        Ok(review) => {
                            app.entries.push(mcp_summary_entry(&review.tool_calls, "вызывал код ревью"));
                            for note in &review.notes {
                                app.entries.push(Entry { kind: Kind::Context, text: format!("⚠ {note}") });
                            }
                            app.entries.push(Entry { kind: Kind::Answer, text: review.to_markdown() });
                            app.entries.push(Entry { kind: Kind::Info, text: "/save [файл] — сохранить это ревью".into() });
                            app.last_review = Some(review);
                        }
                        Err(err) => app.entries.push(Entry { kind: Kind::Error, text: err }),
                    }
                    app.busy = None;
                    app.scroll_back = 0;
                    refresh_git(&assistant, &tx);
                }
                AppEvent::Git(line) => app.git = line,
            },
            Some(Ok(event)) = events.next() => {
                if handle_event(&mut app, event, &tx) {
                    return Ok(());
                }
            }
        }
    }
}

/// `true` — выйти.
fn handle_event(app: &mut App, event: Event, tx: &mpsc::UnboundedSender<AppEvent>) -> bool {
    match event {
        Event::Paste(text) => app.input.insert_str(&text),
        Event::Key(key) if key.kind == KeyEventKind::Press => {
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            let alt = key.modifiers.contains(KeyModifiers::ALT);
            let word = ctrl || alt;
            match key.code {
            KeyCode::Esc => return true,
            KeyCode::Char('c') if ctrl => return true,
            KeyCode::Char('l') if ctrl => app.entries.clear(),
            KeyCode::Char('a') if ctrl => app.input.home(),
            KeyCode::Char('e') if ctrl => app.input.end(),
            KeyCode::Char('u') if ctrl => app.input.kill_to_start(),
            KeyCode::Char('k') if ctrl => app.input.kill_to_end(),
            KeyCode::Char('w') if ctrl => app.input.delete_word_back(),
            // Alt+←/→ в Terminal.app и iTerm2 приходят как Esc+b / Esc+f.
            KeyCode::Char('b') if alt => app.input.word_left(),
            KeyCode::Char('f') if alt => app.input.word_right(),
            KeyCode::Char('d') if alt => app.input.delete_word_forward(),
            KeyCode::Char(c) if !ctrl => app.input.insert(c),
            KeyCode::Backspace if word => app.input.delete_word_back(),
            KeyCode::Backspace => app.input.backspace(),
            KeyCode::Delete => app.input.delete(),
            KeyCode::Left if word => app.input.word_left(),
            KeyCode::Right if word => app.input.word_right(),
            KeyCode::Left => app.input.left(),
            KeyCode::Right => app.input.right(),
            KeyCode::Home if ctrl => app.scroll_back = usize::MAX / 2,
            KeyCode::End if ctrl => app.scroll_back = 0,
            KeyCode::Home => app.input.home(),
            KeyCode::End => app.input.end(),
            KeyCode::PageUp => app.scroll_back += 10,
            KeyCode::PageDown => app.scroll_back = app.scroll_back.saturating_sub(10),
            KeyCode::Up if !app.history.is_empty() => {
                let pos = app.history_pos.map(|p| p.saturating_sub(1)).unwrap_or(app.history.len() - 1);
                app.history_pos = Some(pos);
                app.input.set(&app.history[pos]);
            }
            KeyCode::Down => match app.history_pos {
                Some(p) if p + 1 < app.history.len() => {
                    app.history_pos = Some(p + 1);
                    app.input.set(&app.history[p + 1]);
                }
                _ => {
                    app.history_pos = None;
                    app.input.clear();
                }
            },
            KeyCode::Enter => return submit(app, tx),
            _ => {}
            }
        }
        _ => {}
    }
    false
}

/// `true` — выйти.
fn submit(app: &mut App, tx: &mpsc::UnboundedSender<AppEvent>) -> bool {
    let input = app.input.text().trim().to_string();
    if input.is_empty() {
        return false;
    }
    app.input.clear();
    app.history_pos = None;
    if app.history.last() != Some(&input) {
        app.history.push(input.clone());
    }
    app.scroll_back = 0;
    let (cmd, rest) = match input.strip_prefix('/') {
        Some(c) => {
            let (cmd, rest) = c.split_once(char::is_whitespace).unwrap_or((c, ""));
            (cmd.to_string(), rest.trim().to_string())
        }
        None => ("help".to_string(), input.clone()),
    };
    match cmd.as_str() {
        "quit" | "exit" | "q" => return true,
        "clear" => {
            app.entries.clear();
            return false;
        }
        "?" | "commands" => {
            app.entries.push(Entry { kind: Kind::Info, text: COMMANDS.to_string() });
            return false;
        }
        "save" => {
            app.entries.push(save_review(app.last_review.as_ref(), &rest));
            return false;
        }
        _ => {}
    }
    if app.busy.is_some() {
        app.entries.push(Entry { kind: Kind::Error, text: "Ещё выполняется предыдущая команда — дождитесь её.".into() });
        return false;
    }
    app.entries.push(Entry { kind: Kind::User, text: input.clone() });
    let a = app.assistant.clone();
    let tx = tx.clone();
    let default_base = env_or("ASSISTANT_REVIEW_BASE", review::DEFAULT_BASE);
    match cmd.as_str() {
        "help" => {
            app.busy = Some("ищу в документации и спрашиваю модель…".into());
            tokio::spawn(async move {
                let progress_tx = tx.clone();
                let progress = move |msg: String| {
                    let _ = progress_tx.send(AppEvent::Progress(msg));
                };
                let entries = match help::ask(&a, &rest, &progress).await {
                    Ok(reply) => {
                        let mut out = vec![Entry { kind: Kind::Context, text: reply.context_lines().join("\n") }];
                        for (i, call) in reply.tool_calls.iter().enumerate() {
                            // Первым идёт git_branch, который код вызывает сам до вопроса модели.
                            let who = if i == 0 && call.tool == "git_branch" { "вызвал код" } else { "вызвала модель" };
                            out.push(mcp_entry(call, who));
                        }
                        out.push(Entry { kind: Kind::Answer, text: with_usage(reply.text, reply.usage) });
                        out
                    }
                    Err(err) => vec![Entry { kind: Kind::Error, text: format!("{err:#}") }],
                };
                let _ = tx.send(AppEvent::Done(entries));
            });
        }
        "review" => {
            let target = match Target::parse(&rest, &default_base) {
                Ok(t) => t,
                Err(err) => {
                    app.entries.push(Entry { kind: Kind::Error, text: format!("{err:#}\n{}", review::USAGE) });
                    return false;
                }
            };
            app.busy = Some(format!("ревью: {}…", target.label()));
            tokio::spawn(async move {
                let progress_tx = tx.clone();
                let progress = move |msg: String| {
                    let _ = progress_tx.send(AppEvent::Progress(msg));
                };
                let result = review::run(&a, &target, &progress).await.map_err(|e| format!("{e:#}"));
                let _ = tx.send(AppEvent::Review(result));
            });
        }
        "diff" | "status" | "files" | "log" => {
            let call = match git_command(&cmd, &rest, &default_base) {
                Ok(call) => call,
                Err(err) => {
                    app.entries.pop();
                    app.entries.push(Entry { kind: Kind::Error, text: err });
                    return false;
                }
            };
            app.busy = Some(format!("MCP {} · {}…", a.git_server, call.0));
            let who = format!("вызвала команда /{cmd}");
            tokio::spawn(async move {
                let mut log = Vec::new();
                let result = run_git(&a, call.0, call.1, &mut log).await;
                let mut entries: Vec<Entry> = log.iter().map(|c| mcp_entry(c, &who)).collect();
                entries.push(match result {
                    Ok(text) => Entry { kind: Kind::Output, text },
                    Err(err) => Entry { kind: Kind::Error, text: err },
                });
                let _ = tx.send(AppEvent::Done(entries));
            });
        }
        other => {
            app.entries.pop();
            app.entries.push(Entry { kind: Kind::Error, text: format!("Неизвестная команда /{other}. /? — список команд.") });
        }
    }
    false
}

/// Плашка вызова MCP: `⚙ MCP git › read_file({"path":…}) ✓ вызвала модель`
/// и ниже — начало результата.
fn mcp_entry(call: &ToolCallRecord, who: &str) -> Entry {
    let server = call.server.as_deref().unwrap_or("?");
    let args = match call.arguments.trim() {
        "" | "{}" => String::new(),
        a => short(a, 100),
    };
    let rag = server == help::RAG_SERVER;
    let (kind, mark) = match (rag, call.is_error) {
        (false, false) => (Kind::Mcp, "✓"),
        (false, true) => (Kind::McpError, "✗ ошибка"),
        (true, false) => (Kind::Rag, "✓"),
        (true, true) => (Kind::RagError, "✗ ошибка"),
    };
    Entry {
        kind,
        text: format!("{server} › {}({args})  {mark} · {who}\n  ↳ {}", call.tool, call.result_preview(180)),
    }
}

/// Вызовы MCP одной плашкой — у ревью их десятки: `git_diff ×25, read_file ×24`.
fn mcp_summary_entry(calls: &[ToolCallRecord], who: &str) -> Entry {
    let mut counts: Vec<(String, String, usize, usize)> = Vec::new();
    for c in calls {
        let server = c.server.clone().unwrap_or_default();
        match counts.iter_mut().find(|(s, t, ..)| *s == server && *t == c.tool) {
            Some(entry) => {
                entry.2 += 1;
                entry.3 += c.is_error as usize;
            }
            None => counts.push((server, c.tool.clone(), 1, c.is_error as usize)),
        }
    }
    let errors: usize = counts.iter().map(|c| c.3).sum();
    let server = counts.first().map(|c| c.0.clone()).unwrap_or_else(|| "?".into());
    let list: Vec<String> = counts
        .iter()
        .map(|(_, tool, n, err)| if *err > 0 { format!("{tool} ×{n} (ошибок {err})") } else { format!("{tool} ×{n}") })
        .collect();
    let mark = if errors > 0 { format!("✗ ошибок {errors}") } else { "✓".into() };
    Entry {
        kind: if errors > 0 { Kind::McpError } else { Kind::Mcp },
        text: format!("{server} › вызовов {}  {mark} · {who}\n  ↳ {}", calls.len(), list.join(", ")),
    }
}

fn refresh_git(a: &Arc<Assistant>, tx: &mpsc::UnboundedSender<AppEvent>) {
    let (a, tx) = (a.clone(), tx.clone());
    tokio::spawn(async move {
        let line = a.git_summary().await.unwrap_or_else(|e| format!("git недоступен: {e}"));
        let _ = tx.send(AppEvent::Git(line));
    });
}

fn with_usage(text: String, usage: Option<llm_core::Usage>) -> String {
    match usage {
        Some(u) => format!("{text}\n\n[токены: {} (запрос {}, ответ {})]", u.total_tokens, u.prompt_tokens, u.completion_tokens),
        None => text,
    }
}

fn save_review(review: Option<&Review>, path: &str) -> Entry {
    let Some(review) = review else {
        return Entry { kind: Kind::Error, text: "Ревью ещё не было — сначала /review.".into() };
    };
    let path = if path.is_empty() { "review.md" } else { path };
    let text = if path.ends_with(".json") { review.to_json() } else { review.to_markdown() };
    match std::fs::write(path, text) {
        Ok(()) => Entry { kind: Kind::Info, text: format!("Ревью сохранено в {path}") },
        Err(err) => Entry { kind: Kind::Error, text: format!("Не удалось записать {path}: {err}") },
    }
}

/// Команда TUI → инструмент git и его доводы.
fn git_command(cmd: &str, rest: &str, default_base: &str) -> Result<(&'static str, Value), String> {
    Ok(match cmd {
        "status" => ("git_status", json!({})),
        "files" if rest.is_empty() => ("list_files", json!({ "depth": 2 })),
        "files" => ("list_files", json!({ "path": rest, "depth": 2 })),
        "log" => {
            let n: usize = if rest.is_empty() { 10 } else { rest.parse().map_err(|_| format!("/log [n] — число, а не «{rest}»"))? };
            ("git_log", json!({ "n": n }))
        }
        _ => {
            let mut args = match Target::parse(rest, default_base).map_err(|e| e.to_string())? {
                Target::Branch { base } => json!({ "base": base }),
                Target::Range { base, head } => json!({ "base": base, "head": head }),
                Target::Wip => json!({ "base": "HEAD" }),
            };
            args["max_chars"] = json!(12_000);
            ("git_diff", args)
        }
    })
}

async fn run_git(a: &Assistant, tool: &str, args: Value, log: &mut Vec<ToolCallRecord>) -> Result<String, String> {
    a.ensure_git().await?;
    let text = a.git(tool, args, log).await?;
    let v: Value = serde_json::from_str(&text).map_err(|_| text.clone())?;
    Ok(format_git(tool, &v))
}

/// Результат инструмента git — для чтения человеком.
fn format_git(tool: &str, v: &Value) -> String {
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
    let n = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    let arr = |v: &Value, k: &str| v.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    let mut out = String::new();
    match tool {
        "git_status" => {
            out.push_str(&format!(
                "git status · в индексе {} · изменено {} · не отслеживается {} · конфликтов {}\n",
                n(v, "staged"),
                n(v, "unstaged"),
                n(v, "untracked"),
                n(v, "conflicts")
            ));
            for f in arr(v, "files") {
                out.push_str(&format!("  {}{} {}  ({})\n", s(&f, "index"), s(&f, "worktree"), s(&f, "path"), s(&f, "state")));
            }
            if v.get("clean").and_then(Value::as_bool) == Some(true) {
                out.push_str("  рабочая копия чистая\n");
            }
        }
        "list_files" => {
            let tree = arr(v, "tree");
            if tree.is_empty() {
                for f in arr(v, "files") {
                    out.push_str(&format!("  {}\n", f.as_str().unwrap_or_default()));
                }
            } else {
                for d in tree {
                    let files = n(&d, "files");
                    let count = if files > 1 { format!("  ({files} файлов)") } else { String::new() };
                    out.push_str(&format!("  {}{count}\n", s(&d, "path")));
                }
            }
            out.push_str(&format!("всего файлов: {}", n(v, "total")));
            if v.get("truncated").and_then(Value::as_bool) == Some(true) {
                out.push_str(" (список обрезан)");
            }
        }
        "git_log" => {
            for c in arr(v, "commits") {
                out.push_str(&format!("  {} {}  · {} · {}\n", s(&c, "hash"), s(&c, "subject"), s(&c, "author"), s(&c, "date")));
            }
        }
        _ => {
            out.push_str(&format!("{} · +{} −{}\n", s(v, "compared"), n(v, "added"), n(v, "deleted")));
            for f in arr(v, "files") {
                let stat = match (f.get("added").and_then(Value::as_u64), f.get("deleted").and_then(Value::as_u64)) {
                    (Some(a), Some(d)) => format!("+{a} −{d}"),
                    _ => "двоичный".into(),
                };
                out.push_str(&format!("  {}  {stat}\n", s(&f, "path")));
            }
            for f in arr(v, "untracked") {
                out.push_str(&format!("  {}  (новый, не в git)\n", f.as_str().unwrap_or_default()));
            }
            let patch = s(v, "patch");
            if !patch.is_empty() {
                out.push('\n');
                out.push_str(&patch);
            }
        }
    }
    out.trim_end().to_string()
}

// ---------------------------------------------------------------------------
// Отрисовка

fn draw(frame: &mut Frame, app: &App) {
    let [header, body, input, status] =
        Layout::vertical([Constraint::Length(3), Constraint::Min(3), Constraint::Length(3), Constraint::Length(1)])
            .areas(frame.area());

    let a = &app.assistant;
    let (mcp_text, mcp_color) = match a.mcp.server(&a.git_server).map(|s| (s.status, s.tools.len())) {
        Some((McpStatus::Connected, tools)) => (format!("● MCP {}: подключён, инструментов {tools}", a.git_server), Color::Green),
        Some((McpStatus::Connecting, _)) => (format!("◌ MCP {}: подключается", a.git_server), Color::Yellow),
        Some((McpStatus::Failed { .. }, _)) => (format!("✗ MCP {}: ошибка подключения", a.git_server), Color::Red),
        Some((McpStatus::Disabled, _)) => (format!("○ MCP {}: выключен", a.git_server), Color::DarkGray),
        Some((McpStatus::Idle, _)) => (format!("○ MCP {}: не подключён", a.git_server), Color::DarkGray),
        None => (format!("✗ MCP {}: нет в конфигурации", a.git_server), Color::Red),
    };
    let head = Paragraph::new(Line::from(vec![
        Span::styled("✦ code-assistant", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  "),
        Span::styled(mcp_text, Style::default().fg(mcp_color)),
        Span::raw("  ·  "),
        Span::styled(app.git.clone(), Style::default().fg(Color::Cyan)),
        Span::raw("  ·  модель "),
        Span::styled(app.assistant.client.model().to_string(), Style::default().fg(Color::Cyan)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
    frame.render_widget(head, header);

    let width = body.width.saturating_sub(2).max(10) as usize;
    let lines = render_entries(&app.entries, width);
    let height = body.height.saturating_sub(2) as usize;
    let max_scroll = lines.len().saturating_sub(height);
    let top = max_scroll.saturating_sub(app.scroll_back.min(max_scroll));
    let title = if app.scroll_back > 0 && max_scroll > 0 {
        format!(" Лента — {}/{} (End — в конец) ", top + height.min(lines.len()), lines.len())
    } else {
        " Лента — PgUp/PgDn прокрутка ".to_string()
    };
    let feed = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(title))
        .scroll((top.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(feed, body);

    // Длинный ввод — видна его последняя часть.
    let room = input.width.saturating_sub(5) as usize;
    let (shown, col) = app.input.view(room);
    let (start, _) = app.input.window(room);
    let text = app.input.text();
    let mut spans = vec![Span::raw("› ")];
    match command_at(&text) {
        // Команда (или её видимая часть) — своим цветом.
        Some((len, hint)) => spans.extend(highlight(
            &shown,
            0..len.saturating_sub(start),
            command_style(hint.is_some()),
            Style::default(),
        )),
        None => spans.push(Span::raw(shown)),
    }
    let prompt = Paragraph::new(Line::from(spans)).block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" вопрос или /команда (/? — список) "),
    );
    frame.render_widget(prompt, input);
    frame.set_cursor_position((input.x + 3 + col as u16, input.y + 1));

    let status_line = match (&app.busy, command_at(&text)) {
        (Some(msg), _) => Line::from(Span::styled(format!(" {} {msg}", SPINNER[app.spinner]), Style::default().fg(Color::Yellow))),
        // Набирается команда — подсказка к ней.
        (None, Some((_, Some(hint)))) => Line::from(Span::styled(format!(" {hint}"), Style::default().fg(Color::LightCyan))),
        (None, Some((len, None))) if len > 1 => Line::from(Span::styled(
            format!(" нет команды {} — /? список команд", text.chars().take(len).collect::<String>()),
            Style::default().fg(Color::LightRed),
        )),
        _ => Line::from(Span::styled(
            " Enter — отправить · /review — ревью ветки · /help — вопрос о проекте · Esc — выход",
            Style::default().fg(Color::DarkGray),
        )),
    };
    frame.render_widget(Paragraph::new(status_line), status);
}

/// Строка плашки MCP: первая — ` MCP ` на цветном фоне и `сервер › инструмент(…)`,
/// остальные (результат) — приглушённые, с отступом под плашку.
fn mcp_lines(kind: Kind, first: bool, raw: &str, width: usize) -> Vec<Line<'static>> {
    let (badge, bg) = match kind {
        Kind::Rag => (" RAG ", Color::Blue),
        Kind::RagError => (" RAG ", Color::Red),
        Kind::McpError => (" MCP ", Color::Red),
        _ => (" MCP ", Color::Magenta),
    };
    let indent = " ".repeat(badge.chars().count() + 1);
    let style = if first {
        Style::default().fg(bg).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let text_width = width.saturating_sub(indent.chars().count()).max(10);
    textwrap::wrap(raw.trim_start(), text_width)
        .into_iter()
        .enumerate()
        .map(|(j, part)| {
            let lead = if first && j == 0 {
                // Белый жирный на цветном фоне — читается и в светлой, и в тёмной теме.
                Span::styled(badge, Style::default().fg(Color::White).bg(bg).add_modifier(Modifier::BOLD))
            } else {
                Span::raw(" ".repeat(badge.chars().count()))
            };
            Line::from(vec![lead, Span::raw(" "), Span::styled(part.into_owned(), style)])
        })
        .collect()
}

fn render_entries(entries: &[Entry], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in entries {
        match entry.kind {
            Kind::Answer => lines.extend(markdown::render(&entry.text, width)),
            Kind::Mcp | Kind::McpError | Kind::Rag | Kind::RagError => {
                for (i, raw) in entry.text.lines().enumerate() {
                    lines.extend(mcp_lines(entry.kind, i == 0, raw, width));
                }
            }
            kind => {
                let base = match kind {
                    Kind::Info => Style::default().fg(Color::DarkGray),
                    Kind::User => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                    Kind::Context => Style::default().fg(Color::Blue),
                    Kind::Error => Style::default().fg(Color::Red),
                    _ => Style::default(),
                };
                for (i, raw) in entry.text.lines().enumerate() {
                    let text = if kind == Kind::User && i == 0 { format!("› {raw}") } else { raw.to_string() };
                    let style = if kind == Kind::Output { diff_style(raw, base) } else { base };
                    if text.is_empty() {
                        lines.push(Line::default());
                        continue;
                    }
                    for (j, part) in textwrap::wrap(&text, width).into_iter().enumerate() {
                        // Команда в строке ввода пользователя — тем же цветом, что при наборе.
                        let command = (kind == Kind::User && i == 0 && j == 0).then(|| command_at(raw)).flatten();
                        match command {
                            Some((len, hint)) => {
                                lines.push(Line::from(highlight(&part, 2..2 + len, command_style(hint.is_some()), style)))
                            }
                            None => lines.push(Line::from(Span::styled(part.into_owned(), style))),
                        }
                    }
                }
            }
        }
        lines.push(Line::default());
    }
    lines
}

/// Цвет строки диффа в выводе /diff.
fn diff_style(line: &str, base: Style) -> Style {
    if line.starts_with("+++") || line.starts_with("---") || line.starts_with("diff --git") {
        Style::default().add_modifier(Modifier::BOLD)
    } else if line.starts_with('+') {
        Style::default().fg(Color::Green)
    } else if line.starts_with('-') {
        Style::default().fg(Color::Red)
    } else if line.starts_with("@@") {
        Style::default().fg(Color::Cyan)
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_commands_map_to_tools() {
        assert_eq!(git_command("status", "", "main").unwrap().0, "git_status");
        assert_eq!(git_command("files", "core", "main").unwrap().1, json!({ "path": "core", "depth": 2 }));
        assert_eq!(git_command("log", "5", "main").unwrap().1, json!({ "n": 5 }));
        assert!(git_command("log", "пять", "main").is_err());
        let (tool, args) = git_command("diff", "main..feature", "main").unwrap();
        assert_eq!(tool, "git_diff");
        assert_eq!(args, json!({ "base": "main", "head": "feature", "max_chars": 12_000 }));
    }

    fn call(tool: &str, args: &str, is_error: bool) -> ToolCallRecord {
        ToolCallRecord {
            server: Some("git".into()),
            tool: tool.into(),
            arguments: args.into(),
            result: "{\"branch\": \"day-30\"}".into(),
            is_error,
        }
    }

    #[test]
    fn mcp_calls_are_marked() {
        let e = mcp_entry(&call("git_branch", "{}", false), "вызвал код");
        assert!(e.kind == Kind::Mcp);
        assert_eq!(e.text, "git › git_branch()  ✓ · вызвал код\n  ↳ {\"branch\": \"day-30\"}");
        let e = mcp_entry(&call("read_file", r#"{"path":"x"}"#, true), "вызвала модель");
        assert!(e.kind == Kind::McpError);
        assert!(e.text.starts_with(r#"git › read_file({"path":"x"})  ✗ ошибка · вызвала модель"#));

        let calls = [call("git_diff", "{}", false), call("read_file", "{}", false), call("read_file", "{}", true)];
        let e = mcp_summary_entry(&calls, "вызывал код ревью");
        assert_eq!(e.text, "git › вызовов 3  ✗ ошибок 1 · вызывал код ревью\n  ↳ git_diff ×1, read_file ×2 (ошибок 1)");

        let mut rag = call("search_docs", r#"{"query":"RAG"}"#, false);
        rag.server = Some(help::RAG_SERVER.into());
        let e = mcp_entry(&rag, "вызвала модель");
        assert!(e.kind == Kind::Rag);
        assert_eq!(mcp_lines(Kind::Rag, true, "rag › search_docs()", 80)[0].spans[0].content, " RAG ");

        let lines = mcp_lines(Kind::Mcp, true, "git › git_branch()", 80);
        assert_eq!(lines[0].spans[0].content, " MCP ");
        assert_eq!(lines[0].spans[2].content, "git › git_branch()");
    }

    #[test]
    fn commands_are_recognized_and_highlighted() {
        assert_eq!(command_at("/review main..dev").map(|(n, h)| (n, h.is_some())), Some((7, true)));
        assert_eq!(command_at("/help").map(|(n, h)| (n, h.is_some())), Some((5, true)));
        assert_eq!(command_at("/?").map(|(n, h)| (n, h.is_some())), Some((2, true)));
        assert_eq!(command_at("/revew x").map(|(n, h)| (n, h.is_some())), Some((6, false)));
        assert_eq!(command_at("где лежит код?"), None);

        let spans = highlight("› /help где", 2..7, command_style(true), Style::default());
        let parts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(parts, vec!["› ", "/help", " где"]);
        assert_eq!(spans[1].style.fg, Some(Color::LightCyan));
        // Команда частично прокручена за левый край — подсвечена видимая часть.
        let spans = highlight("iew main", 0..3, command_style(true), Style::default());
        assert_eq!(spans[0].content, "iew");
    }

    #[test]
    fn formats_diff_stat() {
        let v = json!({ "compared": "рабочая копия ↔ main", "added": 3, "deleted": 1,
            "files": [{ "path": "a.rs", "added": 3, "deleted": 1 }, { "path": "logo.png", "added": null, "deleted": null }],
            "untracked": ["new.rs"], "patch": "" });
        assert_eq!(
            format_git("git_diff", &v),
            "рабочая копия ↔ main · +3 −1\n  a.rs  +3 −1\n  logo.png  двоичный\n  new.rs  (новый, не в git)"
        );
    }
}
