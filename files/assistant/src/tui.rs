//! TUI ассистента файлов: шапка (серверы MCP, git, ожидающие изменения),
//! лента, поле ввода. Задача выполняется в фоне; вызовы инструментов
//! появляются в ленте по мере того, как модель их делает. Лента, ввод и
//! Markdown — общие с code-assistant (`assistant-tui`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use assistant_tui::feed::{tool_entry, with_usage};
use assistant_tui::terminal::{self, Term};
use assistant_tui::{Action, Entry, Feed, Hints, Kind, Prompt};
use crossterm::event::EventStream;
use futures_util::StreamExt;
use llm_core::McpStatus;
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::runs;
use crate::scenarios;
use crate::session::{Pending, Session};
use crate::task::{self, Outcome, Step};

const COMMANDS: &str = "Команды:
  <цель словами> или /task <цель>   — задача: ассистент сам ищет, читает и правит файлы
                                     («найди все места, где используется McpManager»,
                                      «обнови таблицу настроек в docs/developer-assistant.md по коду»)
  /scenario [имя]                    — готовые цели из files/scenarios (без имени — список)
  /diff                              — ожидающие изменения: файлы и дифф
  /apply [путь…]                     — записать изменения на диск (все или только эти файлы)
  /discard [путь…]                   — отказаться от изменений
  /files [путь]                      — структура проекта или каталога
  /clear — очистить ленту · /? — эта справка · /quit или Esc — выход
Изменения модели не попадают на диск сами: они копятся, пока вы не скажете /apply.
Каждая задача сохраняется в files/runs/<id>/ — отчёт, changes.patch (git apply), вызовы.
Ввод: ↑/↓ — история · Alt+←/→ — по словам · Ctrl+U/K — удалить до начала/конца
Лента: PgUp/PgDn — прокрутка · Ctrl+Home/Ctrl+End — в начало/конец · Ctrl+L — очистить";

const COMMAND_HINTS: Hints = &[
    ("task", "/task <цель> — задача словами: ассистент сам находит, читает и правит нужные файлы"),
    ("scenario", "/scenario [имя] — готовая цель из files/scenarios; без имени — список"),
    ("diff", "/diff — ожидающие изменения: какие файлы и дифф"),
    ("apply", "/apply [путь…] — записать ожидающие изменения на диск"),
    ("discard", "/discard [путь…] — отказаться от ожидающих изменений"),
    ("files", "/files [путь] — структура проекта или каталога"),
    ("clear", "/clear — очистить ленту"),
    ("quit", "/quit — выход"),
    ("exit", "/exit — выход"),
    ("q", "/q — выход"),
    ("?", "/? — список команд"),
    ("commands", "/commands — список команд"),
];

enum AppEvent {
    Progress(String),
    Entry(Entry),
    Done(Vec<Entry>),
    Finished(Result<(Outcome, PathBuf), String>),
    Pending(Pending),
    Git(String),
}

struct App {
    session: Arc<Session>,
    feed: Feed,
    prompt: Prompt,
    busy: Option<String>,
    spinner: usize,
    git: String,
    pending: Pending,
    /// Каталог последнего запуска — в его run.json отмечается /apply.
    last_run: Option<PathBuf>,
}

pub async fn run(session: Session) -> Result<()> {
    let mut terminal = terminal::enter()?;
    let result = event_loop(&mut terminal, Arc::new(session)).await;
    terminal::leave(&mut terminal)?;
    result
}

async fn event_loop(terminal: &mut Term, session: Arc<Session>) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut app = App {
        session: session.clone(),
        feed: Feed::default(),
        prompt: Prompt::new(COMMAND_HINTS),
        busy: None,
        spinner: 0,
        git: "подключаю MCP git…".into(),
        pending: Pending::default(),
        last_run: None,
    };
    app.feed.push(
        Kind::Info,
        "Поставьте цель словами — ассистент сам найдёт и прочитает нужные файлы, а правки покажет диффом.\n\
         /scenario — готовые цели · /? — все команды",
    );
    refresh(&session, &tx);

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
                AppEvent::Entry(entry) => {
                    app.feed.entries.push(entry);
                }
                AppEvent::Done(entries) => {
                    app.feed.entries.extend(entries);
                    app.busy = None;
                    app.feed.scroll_back = 0;
                    refresh(&session, &tx);
                }
                AppEvent::Finished(result) => {
                    match result {
                        Ok((outcome, dir)) => {
                            app.feed.entries.extend(outcome_entries(&outcome, &dir));
                            app.last_run = Some(dir);
                        }
                        Err(err) => app.feed.push(Kind::Error, err),
                    }
                    app.busy = None;
                    app.feed.scroll_back = 0;
                    refresh(&session, &tx);
                }
                AppEvent::Pending(p) => app.pending = p,
                AppEvent::Git(line) => app.git = line,
            },
            Some(Ok(event)) = events.next() => {
                match app.prompt.handle(event, &mut app.feed) {
                    Action::Quit => return Ok(()),
                    Action::Submit(input) => {
                        if submit(&mut app, input, &tx) {
                            return Ok(());
                        }
                    }
                    Action::None => {}
                }
            }
        }
    }
}

/// Строки ленты после задачи: замечания, отчёт, изменения.
fn outcome_entries(o: &Outcome, dir: &std::path::Path) -> Vec<Entry> {
    let mut out = Vec::new();
    for note in &o.notes {
        out.push(Entry::new(Kind::Context, format!("⚠ {note}")));
    }
    out.push(Entry::new(Kind::Answer, with_usage(o.report.clone(), o.usage)));
    let touched = o.touched_files();
    let mut info = format!(
        "шагов {} · вызовов {} · файлов прочитано/изменено {} · модель {} · t={}",
        o.rounds,
        o.tool_calls.len(),
        touched.len(),
        o.model,
        o.temperature
    );
    info.push_str(&format!("\nзапуск сохранён: {}", dir.display()));
    if o.pending.is_empty() {
        info.push_str("\nизменений в файлах нет");
        out.push(Entry::new(Kind::Context, info));
    } else {
        out.push(Entry::new(Kind::Context, info));
        out.push(Entry::new(
            Kind::Output,
            format!("Ожидают записи — {}:\n{}\n\n{}", o.pending.summary(), o.pending.listing(), o.pending.patch.trim_end()),
        ));
        out.push(Entry::new(
            Kind::Info,
            format!(
                "На диск не записано. /apply — записать, /discard — отказаться, или позже: git apply {}",
                dir.join("changes.patch").display()
            ),
        ));
    }
    out
}

/// `true` — выйти.
fn submit(app: &mut App, input: String, tx: &mpsc::UnboundedSender<AppEvent>) -> bool {
    let (cmd, rest) = match input.strip_prefix('/') {
        Some(c) => {
            let (cmd, rest) = c.split_once(char::is_whitespace).unwrap_or((c, ""));
            (cmd.to_string(), rest.trim().to_string())
        }
        None => ("task".to_string(), input.clone()),
    };
    match cmd.as_str() {
        "quit" | "exit" | "q" => return true,
        "clear" => {
            app.feed.clear();
            return false;
        }
        "?" | "commands" => {
            app.feed.push(Kind::Info, COMMANDS);
            return false;
        }
        _ => {}
    }
    if app.busy.is_some() {
        app.feed.push(Kind::Error, "Ещё выполняется предыдущая команда — дождитесь её.");
        return false;
    }
    app.feed.push(Kind::User, input.clone());
    let s = app.session.clone();
    let tx = tx.clone();
    let paths: Vec<String> = rest.split_whitespace().map(str::to_string).collect();
    match cmd.as_str() {
        "task" if rest.is_empty() => app.feed.push(Kind::Error, "/task <цель> — что сделать с файлами проекта."),
        "task" => start_task(app, rest, None, tx),
        "scenario" if rest.is_empty() => match scenarios::list() {
            Ok(all) => app.feed.push(Kind::Info, format!("Сценарии — /scenario <имя>:\n{}", scenarios::listing(&all))),
            Err(err) => app.feed.push(Kind::Error, format!("{err:#}")),
        },
        "scenario" => match scenarios::find(&rest) {
            Ok(sc) => {
                app.feed.push(Kind::Context, format!("Сценарий «{}»: {}", sc.title, sc.goal));
                start_task(app, sc.goal, Some(sc.name), tx);
            }
            Err(err) => app.feed.push(Kind::Error, format!("{err:#}")),
        },
        "diff" => {
            app.busy = Some("собираю дифф…".into());
            tokio::spawn(async move {
                let entry = match s.pending(&mut Vec::new()).await {
                    Ok(p) if p.is_empty() => Entry::new(Kind::Info, "Ожидающих изменений нет."),
                    Ok(p) => Entry::new(Kind::Output, format!("{}:\n{}\n\n{}", p.summary(), p.listing(), p.patch.trim_end())),
                    Err(err) => Entry::new(Kind::Error, err),
                };
                let _ = tx.send(AppEvent::Done(vec![entry]));
            });
        }
        "apply" => {
            app.busy = Some("записываю изменения на диск…".into());
            let run_dir = app.last_run.clone();
            tokio::spawn(async move {
                let mut log = Vec::new();
                let mut entries = Vec::new();
                match s.apply(&paths, &mut log).await {
                    Ok(applied) => {
                        entries.extend(log.iter().map(|c| tool_entry(c, "вызвала команда /apply")));
                        if let Some(dir) = run_dir.filter(|_| !applied.written.is_empty()) {
                            let _ = runs::mark_applied(&dir, &applied);
                        }
                        let mut text = if applied.written.is_empty() {
                            "Ничего не записано.".to_string()
                        } else {
                            format!("Записано на диск: {}", applied.written.join(", "))
                        };
                        for c in &applied.conflicts {
                            text.push_str(&format!("\nКонфликт: {} — {} (изменение осталось, /discard {} — отказаться)", c.path, c.reason, c.path));
                        }
                        let kind = if applied.conflicts.is_empty() { Kind::Info } else { Kind::Error };
                        entries.push(Entry::new(kind, text));
                    }
                    Err(err) => entries.push(Entry::new(Kind::Error, err)),
                }
                let _ = tx.send(AppEvent::Done(entries));
            });
        }
        "discard" => {
            app.busy = Some("отменяю изменения…".into());
            tokio::spawn(async move {
                let entry = match s.discard(&paths, &mut Vec::new()).await {
                    Ok(list) if list.is_empty() => Entry::new(Kind::Info, "Отменять нечего."),
                    Ok(list) => Entry::new(Kind::Info, format!("Отменены изменения: {}", list.join(", "))),
                    Err(err) => Entry::new(Kind::Error, err),
                };
                let _ = tx.send(AppEvent::Done(vec![entry]));
            });
        }
        "files" => {
            app.busy = Some("MCP fs · list_files…".into());
            let args = if rest.is_empty() { json!({ "depth": 2 }) } else { json!({ "path": rest, "depth": 2 }) };
            tokio::spawn(async move {
                let mut log = Vec::new();
                let result = s.fs("list_files", args, &mut log).await;
                let mut entries: Vec<Entry> = log.iter().map(|c| tool_entry(c, "вызвала команда /files")).collect();
                entries.push(match result {
                    Ok(text) => Entry::new(Kind::Output, format_tree(&text)),
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

fn start_task(app: &mut App, goal: String, scenario: Option<String>, tx: mpsc::UnboundedSender<AppEvent>) {
    app.busy = Some("берусь за задачу…".into());
    let s = app.session.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let progress = move |step: Step| {
            let _ = progress_tx.send(match step {
                Step::Status(msg) => AppEvent::Progress(msg),
                Step::Tool(call) => AppEvent::Entry(tool_entry(&call, "вызвала модель")),
            });
        };
        let result = match task::run(&s, &goal, scenario.as_deref(), &progress).await {
            Ok(outcome) => runs::save(&runs::runs_dir(), &outcome)
                .map(|dir| (outcome, dir))
                .map_err(|e| format!("задача выполнена, но запуск не сохранён: {e:#}")),
            Err(err) => Err(format!("{err:#}")),
        };
        let _ = tx.send(AppEvent::Finished(result));
    });
}

/// Обновить шапку: git и ожидающие изменения.
fn refresh(s: &Arc<Session>, tx: &mpsc::UnboundedSender<AppEvent>) {
    let (s, tx) = (s.clone(), tx.clone());
    tokio::spawn(async move {
        if let Ok(p) = s.pending(&mut Vec::new()).await {
            let _ = tx.send(AppEvent::Pending(p));
        }
        let line = s.git_line().await.unwrap_or_else(|e| format!("git недоступен: {e}"));
        let _ = tx.send(AppEvent::Git(line));
    });
}

/// Результат `list_files` с `depth` — для чтения человеком.
fn format_tree(text: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return text.to_string() };
    let mut out = String::new();
    let tree = v["tree"].as_array().cloned().unwrap_or_default();
    for d in &tree {
        let files = d["files"].as_u64().unwrap_or(1);
        let count = if files > 1 { format!("  ({files} файлов)") } else { String::new() };
        out.push_str(&format!("  {}{count}\n", d["path"].as_str().unwrap_or_default()));
    }
    for f in v["files"].as_array().into_iter().flatten() {
        out.push_str(&format!("  {}\n", f.as_str().unwrap_or_default()));
    }
    out.push_str(&format!("всего файлов: {}", v["total"].as_u64().unwrap_or(0)));
    if v["truncated"].as_bool() == Some(true) {
        out.push_str(" (список обрезан)");
    }
    out
}

// ---------------------------------------------------------------------------
// Отрисовка

fn server_span(s: &Session, name: &str) -> Span<'static> {
    let (text, color) = match s.mcp.server(name).map(|i| i.status) {
        Some(McpStatus::Connected) => (format!("● {name}"), Color::Green),
        Some(McpStatus::Connecting) => (format!("◌ {name}"), Color::Yellow),
        Some(McpStatus::Failed { .. }) => (format!("✗ {name}"), Color::Red),
        Some(McpStatus::Disabled) => (format!("○ {name} выкл."), Color::DarkGray),
        Some(McpStatus::Idle) => (format!("○ {name}"), Color::DarkGray),
        None => (format!("✗ {name}: нет в конфигурации"), Color::Red),
    };
    Span::styled(text, Style::default().fg(color))
}

fn draw(frame: &mut Frame, app: &App) {
    let [header, body, input, status] =
        Layout::vertical([Constraint::Length(3), Constraint::Min(3), Constraint::Length(3), Constraint::Length(1)])
            .areas(frame.area());

    let s = &app.session;
    let (pending_text, pending_color) = if app.pending.is_empty() {
        ("изменений нет".to_string(), Color::DarkGray)
    } else {
        (format!("ожидают записи: {} — /diff /apply", app.pending.summary()), Color::Yellow)
    };
    let head = Paragraph::new(Line::from(vec![
        Span::styled("✦ file-assistant", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  MCP "),
        server_span(s, &s.fs_server),
        Span::raw(" "),
        server_span(s, &s.git_server),
        Span::raw("  ·  "),
        Span::styled(pending_text, Style::default().fg(pending_color).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  "),
        Span::styled(app.git.clone(), Style::default().fg(Color::Cyan)),
        Span::raw("  ·  модель "),
        Span::styled(s.client.model().to_string(), Style::default().fg(Color::Cyan)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
    frame.render_widget(head, header);

    app.feed.draw(frame, body, COMMAND_HINTS);
    app.prompt.draw(frame, input, "цель словами или /команда (/? — список)");
    let line = app.prompt.status_line(
        app.busy.as_deref(),
        app.spinner,
        "Enter — поставить задачу · /scenario — готовые цели · /diff /apply — изменения · Esc — выход",
    );
    frame.render_widget(Paragraph::new(line), status);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::PendingFile;

    #[test]
    fn tree_is_readable() {
        let text = r#"{"total":5,"truncated":false,"tree":[{"path":"docs/","files":3},{"path":"README.md","files":1}]}"#;
        assert_eq!(format_tree(text), "  docs/  (3 файлов)\n  README.md\nвсего файлов: 5");
    }

    #[test]
    fn outcome_shows_report_and_diff() {
        let mut o = Outcome {
            goal: "g".into(),
            scenario: None,
            report: "## Итог\nок".into(),
            pending: Pending::default(),
            tool_calls: vec![],
            rounds: 2,
            hit_limit: false,
            model: "m".into(),
            temperature: 0.0,
            usage: None,
            notes: vec![],
        };
        let dir = std::path::Path::new("files/runs/x");
        let entries = outcome_entries(&o, dir);
        assert_eq!(entries[0].kind, Kind::Answer);
        assert!(entries[1].text.contains("изменений в файлах нет"));
        o.pending = Pending {
            files: vec![PendingFile { path: "a.md".into(), status: "modified".into(), added: 1, deleted: 1 }],
            added: 1,
            deleted: 1,
            patch: "diff --git a/a.md b/a.md\n".into(),
        };
        let entries = outcome_entries(&o, dir);
        assert_eq!(entries[2].kind, Kind::Output);
        assert!(entries[3].text.contains("git apply files/runs/x/changes.patch"));
    }

    /// Кадр целиком: шапка с серверами и изменениями, лента с диффом, ввод.
    #[test]
    fn draws_header_feed_and_pending() {
        std::env::set_var("LLM_API_URL", "http://127.0.0.1:9");
        std::env::set_var("LLM_API_KEY", "test");
        let session = Session {
            client: llm_core::LlmClient::from_env().unwrap(),
            mcp: llm_core::McpManager::empty(),
            fs_server: "fs".into(),
            git_server: "git".into(),
        };
        let mut app = App {
            session: Arc::new(session),
            feed: Feed::default(),
            prompt: Prompt::new(COMMAND_HINTS),
            busy: None,
            spinner: 0,
            git: "ветка day-34".into(),
            pending: Pending {
                files: vec![PendingFile { path: "CHANGELOG.md".into(), status: "created".into(), added: 3, deleted: 0 }],
                added: 3,
                deleted: 0,
                patch: String::new(),
            },
            last_run: None,
        };
        app.feed.push(Kind::Output, "+++ b/CHANGELOG.md\n+# Changelog");
        app.prompt.input.set("/apply");
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(200, 20)).unwrap();
        term.draw(|f| draw(f, &app)).unwrap();
        let buf = term.backend().buffer();
        let text: String = (0..buf.area.height)
            .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>() + "\n")
            .collect();
        assert!(text.contains("file-assistant"), "{text}");
        assert!(text.contains("ожидают записи: 1 файл (+3 −0)"), "{text}");
        assert!(text.contains("fs: нет в конфигурации"), "{text}");
        assert!(text.contains("+# Changelog"), "{text}");
        assert!(text.contains("/apply [путь…] — записать"), "{text}");
    }
}
