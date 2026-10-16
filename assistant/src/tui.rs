//! TUI ассистента: шапка с состоянием репозитория, лента ответов, поле ввода.
//! Команды выполняются в фоне по одной; прогресс — в строке состояния.
//! Лента, строка ввода и Markdown — общие с другими ассистентами (`assistant-tui`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use assistant_tui::feed::{tool_entry, tool_summary_entry, with_usage};
use assistant_tui::terminal::{self, Term};
use assistant_tui::{Action, Entry, Feed, Hints, Kind, Prompt};
use crossterm::event::{Event, EventStream};
use futures_util::StreamExt;
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use llm_core::{McpStatus, ToolCallRecord};

use crate::context::{env_or, Assistant};
use crate::help;
use crate::review::{self, Review, Target};

const COMMANDS: &str = "Команды:
  <вопрос> или /help <вопрос>   — вопрос о проекте (README + docs/ через RAG, git через MCP); /help без вопроса — обзор
  /review [base]                — ревью текущей ветки с незакоммиченным против base (по умолчанию main)
  /review <base>..<head>        — ревью ветки head против base (только закоммиченное)
  /review --wip                 — ревью только незакоммиченного
  /review [цель] <путь…>        — ревью только этих файлов и каталогов (/review --wip assistant/src/tui.rs)
  /save [файл]                  — сохранить последнее ревью в Markdown (по умолчанию review.md)
  /diff [цель]                  — какие файлы изменены и дифф (цель — как у /review), без модели
  /status                       — git status
  /files [путь]                 — структура проекта или каталога
  /log [n]                      — последние коммиты
  /calls [off|brief|debug]      — вызовы MCP в ленте: скрыть, кратко (без доводов) или для отладки (доводы целиком)
  /clear — очистить ленту · /? — эта справка · /quit или Esc — выход
Ввод: ←/→ — символ · Alt+←/→ или Ctrl+←/→ — слово · Home/End (Ctrl+A/E) — начало/конец строки ·
      Alt+Backspace (Ctrl+W) — удалить слово · Ctrl+U/K — удалить до начала/конца · ↑/↓ — история
Лента: колесо мыши или PgUp/PgDn — прокрутка · Ctrl+Home/Ctrl+End — в начало/конец · Ctrl+L — очистить
      Ctrl+O — вызовы: кратко → отладка → скрыты (выделить текст мышью — с Option в iTerm2, Fn в Terminal.app)
Вызовы видны в ленте плашками « MCP » (git) и « RAG » (поиск по документации и коду): кто вызвал и результат,
а пока команда идёт — в строке состояния внизу.";

/// Команды TUI и подсказки к ним — для подсветки ввода и строки состояния.
const COMMAND_HINTS: Hints = &[
    ("help", "/help <вопрос> — вопрос о проекте: поиск в документации и коде (RAG) и git (MCP); без вопроса — обзор"),
    ("review", "/review [base | base..head | --wip] [путь…] — ревью ветки против base (main), двух веток или незакоммиченного; пути сужают ревью"),
    ("save", "/save [файл] — сохранить последнее ревью (review.md; .json — в JSON)"),
    ("diff", "/diff [base | base..head | --wip] — изменённые файлы и дифф, без модели"),
    ("status", "/status — git status: что изменено и не закоммичено"),
    ("files", "/files [путь] — структура проекта или каталога"),
    ("log", "/log [n] — последние коммиты"),
    ("calls", "/calls [off | brief | debug] — вызовы MCP: скрыть, кратко или с доводами целиком (Ctrl+O — по кругу)"),
    ("clear", "/clear — очистить ленту"),
    ("quit", "/quit — выход"),
    ("exit", "/exit — выход"),
    ("q", "/q — выход"),
    ("?", "/? — список команд"),
    ("commands", "/commands — список команд"),
];

enum AppEvent {
    Progress(String),
    Done(Vec<Entry>),
    Review(Result<Review, String>),
    Git(String),
}

struct App {
    assistant: Arc<Assistant>,
    feed: Feed,
    prompt: Prompt,
    /// Что сейчас выполняется; `None` — ничего.
    busy: Option<String>,
    git: String,
    spinner: usize,
    last_review: Option<Review>,
}

pub async fn run(assistant: Assistant) -> Result<()> {
    let mut terminal = terminal::enter()?;
    let result = event_loop(&mut terminal, Arc::new(assistant)).await;
    terminal::leave(&mut terminal)?;
    result
}

async fn event_loop(terminal: &mut Term, assistant: Arc<Assistant>) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut app = App {
        assistant: assistant.clone(),
        // Лента пустая — справка по /?, подсказки к командам — в строке состояния.
        feed: Feed { view: llm_core::CallsView::from_env(), ..Feed::default() },
        prompt: Prompt::new(COMMAND_HINTS),
        busy: None,
        git: "подключаю MCP git…".into(),
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
                    app.spinner += 1;
                }
            }
            Some(ev) = rx.recv() => match ev {
                AppEvent::Progress(msg) => {
                    if app.busy.is_some() {
                        app.busy = Some(msg);
                    }
                }
                AppEvent::Done(entries) => {
                    app.feed.entries.extend(entries);
                    app.busy = None;
                    app.feed.scroll_back = 0;
                    refresh_git(&assistant, &tx);
                }
                AppEvent::Review(result) => {
                    match result {
                        Ok(review) => {
                            app.feed.entries.push(tool_summary_entry(&review.tool_calls, ""));
                            for note in &review.notes {
                                app.feed.push(Kind::Context, format!("⚠ {note}"));
                            }
                            app.feed.push(Kind::Answer, review.to_markdown());
                            app.feed.push(Kind::Info, "/save [файл] — сохранить это ревью");
                            app.last_review = Some(review);
                        }
                        Err(err) => app.feed.push(Kind::Error, err),
                    }
                    app.busy = None;
                    app.feed.scroll_back = 0;
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
    match app.prompt.handle(event, &mut app.feed) {
        Action::Quit => true,
        Action::Submit(input) => submit(app, input, tx),
        Action::None => false,
    }
}

/// `true` — выйти.
fn submit(app: &mut App, input: String, tx: &mpsc::UnboundedSender<AppEvent>) -> bool {
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
            app.feed.clear();
            return false;
        }
        "calls" => {
            let view = if rest.is_empty() { Some(app.feed.view.next()) } else { llm_core::CallsView::parse(&rest) };
            match view {
                Some(view) => {
                    app.feed.view = view;
                    app.feed.push(Kind::Info, format!("Вызовы инструментов: {}.", view.label()));
                }
                None => app.feed.push(Kind::Error, "/calls off | brief | debug — скрыть, кратко или для отладки."),
            }
            return false;
        }
        "?" | "commands" => {
            app.feed.push(Kind::Info, COMMANDS);
            return false;
        }
        "save" => {
            app.feed.entries.push(save_review(app.last_review.as_ref(), &rest));
            return false;
        }
        _ => {}
    }
    if app.busy.is_some() {
        app.feed.push(Kind::Error, "Ещё выполняется предыдущая команда — дождитесь её.");
        return false;
    }
    app.feed.push(Kind::User, input.clone());
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
                        let mut out = vec![Entry::new(Kind::Context, reply.context_lines().join("\n"))];
                        for call in &reply.tool_calls {
                            out.push(tool_entry(call, ""));
                        }
                        out.push(Entry::new(Kind::Answer, with_usage(reply.text, reply.usage)));
                        out
                    }
                    Err(err) => vec![Entry::new(Kind::Error, format!("{err:#}"))],
                };
                let _ = tx.send(AppEvent::Done(entries));
            });
        }
        "review" => {
            let words: Vec<&str> = rest.split_whitespace().collect();
            let is_path = |p: &str| std::path::Path::new(p).exists();
            let (target, paths) = match review::parse_args(&words, &default_base, is_path) {
                Ok(parsed) => parsed,
                Err(err) => {
                    app.feed.push(Kind::Error, format!("{err:#}\n{}", review::USAGE));
                    return false;
                }
            };
            let scope = if paths.is_empty() { String::new() } else { format!(" · {}", paths.join(", ")) };
            app.busy = Some(format!("ревью: {}{scope}…", target.label()));
            tokio::spawn(async move {
                let progress_tx = tx.clone();
                let progress = move |msg: String| {
                    let _ = progress_tx.send(AppEvent::Progress(msg));
                };
                let result = review::run(&a, &target, &paths, &progress).await.map_err(|e| format!("{e:#}"));
                let _ = tx.send(AppEvent::Review(result));
            });
        }
        "diff" | "status" | "files" | "log" => {
            let call = match git_command(&cmd, &rest, &default_base) {
                Ok(call) => call,
                Err(err) => {
                    app.feed.entries.pop();
                    app.feed.push(Kind::Error, err);
                    return false;
                }
            };
            app.busy = Some(format!("MCP {} · {}…", a.git_server, call.0));
            let who = format!("вызвала команда /{cmd}");
            tokio::spawn(async move {
                let mut log = Vec::new();
                let result = run_git(&a, call.0, call.1, &mut log).await;
                let mut entries: Vec<Entry> = log.iter().map(|c| tool_entry(c, &who)).collect();
                entries.push(match result {
                    Ok(text) => Entry::new(Kind::Output, text),
                    Err(err) => Entry::new(Kind::Error, err),
                });
                let _ = tx.send(AppEvent::Done(entries));
            });
        }
        other => {
            app.feed.entries.pop();
            app.feed.push(Kind::Error, format!("Неизвестная команда /{other}. /? — список команд."));
        }
    }
    false
}

fn refresh_git(a: &Arc<Assistant>, tx: &mpsc::UnboundedSender<AppEvent>) {
    let (a, tx) = (a.clone(), tx.clone());
    tokio::spawn(async move {
        let line = a.git_summary().await.unwrap_or_else(|e| format!("git недоступен: {e}"));
        let _ = tx.send(AppEvent::Git(line));
    });
}

fn save_review(review: Option<&Review>, path: &str) -> Entry {
    let Some(review) = review else {
        return Entry::new(Kind::Error, "Ревью ещё не было — сначала /review.");
    };
    let path = if path.is_empty() { "review.md" } else { path };
    let text = if path.ends_with(".json") { review.to_json() } else { review.to_markdown() };
    match std::fs::write(path, text) {
        Ok(()) => Entry::new(Kind::Info, format!("Ревью сохранено в {path}")),
        Err(err) => Entry::new(Kind::Error, format!("Не удалось записать {path}: {err}")),
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

    app.feed.draw(frame, body, COMMAND_HINTS);
    app.prompt.draw(frame, input, "вопрос или /команда (/? — список)");
    let status_line = app.prompt.status_line(
        app.busy.as_deref(),
        app.spinner,
        "Enter — отправить · /review — ревью ветки · /help — вопрос о проекте · Esc — выход",
    );
    frame.render_widget(Paragraph::new(status_line), status);
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
