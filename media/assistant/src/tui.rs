//! TUI ассистента записей: шапка (сервер MCP, каталог записи, следующий шаг),
//! лента, поле ввода. Реплика выполняется в фоне; вызовы инструментов
//! появляются в ленте по мере работы, а ход идущего задания (`12:01 из
//! 58:10`, `чанки: 7 из 14`) — в строке состояния. Лента, ввод и Markdown —
//! общие с другими ассистентами (`assistant-tui`).

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
use tokio::sync::{mpsc, Mutex};

use crate::dialog::{Dialog, Step, Turn};
use crate::runs;
use crate::session::{short, Session};

const COMMANDS: &str = "Команды:
  <текст>               — реплика ассистенту: «обработай запись», «да», «возьми meeting-2.webm»
  /dir [каталог]        — каталог записи (от корня MEDIA_ROOT) и новый диалог; без аргумента — где есть записи
  /status               — состояние каталога: чекпоинты и следующий шаг (без модели)
  /jobs                 — задания сервера
  /new                  — новый диалог о той же записи
  /calls [off|brief|debug] — вызовы MCP в ленте: скрыть, кратко или для отладки (доводы и результат целиком)
  /clear — очистить ленту · /? — эта справка · /quit или Esc — выход
Долгие шаги (конвертация, транскрипция, саммари) — задания сервера: ассистент ждёт их сам, ход виден внизу.
Выход во время задания останавливает его (ffmpeg тоже) — Esc нужно нажать дважды.
Каждый диалог сохраняется в media/runs/<id>/ — dialog.md, tools.json, run.json.
Ввод: ↑/↓ — история · Alt+←/→ — по словам · Ctrl+U/K — удалить до начала/конца
Лента: колесо мыши или PgUp/PgDn — прокрутка · Ctrl+Home/Ctrl+End — в начало/конец · Ctrl+L — очистить
      Ctrl+O — вызовы: кратко → отладка → скрыты";

const COMMAND_HINTS: Hints = &[
    ("dir", "/dir [каталог] — выбрать запись (новый диалог); без аргумента — список каталогов с записями"),
    ("status", "/status — чекпоинты каталога и следующий шаг, без модели"),
    ("jobs", "/jobs — задания сервера: что идёт, что закончилось"),
    ("new", "/new — новый диалог о той же записи"),
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
    Entry(Entry),
    Done(Vec<Entry>),
    Replied(Result<(Turn, Option<PathBuf>), String>),
    Stage(String),
    /// Новый каталог записи принят.
    Dir(String, Vec<Entry>),
}

struct App {
    session: Arc<Session>,
    dialog: Arc<Mutex<Dialog>>,
    dir: String,
    feed: Feed,
    prompt: Prompt,
    busy: Option<String>,
    spinner: usize,
    stage: String,
    /// Каталог сохранения текущего диалога.
    run_dir: Option<PathBuf>,
    /// Esc во время работы нажат один раз — второй выходит.
    quit_armed: bool,
}

pub async fn run(session: Session, dir: String) -> Result<()> {
    let mut terminal = terminal::enter()?;
    let result = event_loop(&mut terminal, Arc::new(session), dir).await;
    terminal::leave(&mut terminal)?;
    result
}

async fn event_loop(terminal: &mut Term, session: Arc<Session>, dir: String) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut app = App {
        session: session.clone(),
        dialog: Arc::new(Mutex::new(Dialog::new(&dir))),
        dir: dir.clone(),
        feed: Feed { view: llm_core::CallsView::from_env(), ..Feed::default() },
        prompt: Prompt::new(COMMAND_HINTS),
        busy: None,
        spinner: 0,
        stage: "подключаю MCP media…".into(),
        run_dir: None,
        quit_armed: false,
    };
    app.feed.push(
        Kind::Info,
        format!("Запись: `{dir}`. Напишите «обработай» — ассистент сам найдёт, с чего продолжить. /dir — другая запись, /? — команды."),
    );
    refresh(&session, &app.dir, &tx);

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
                AppEvent::Entry(entry) => app.feed.entries.push(entry),
                AppEvent::Done(entries) => {
                    app.feed.entries.extend(entries);
                    finish(&mut app, &tx);
                }
                AppEvent::Replied(result) => {
                    match result {
                        Ok((turn, dir)) => {
                            app.feed.entries.extend(turn_entries(&turn));
                            if dir.is_some() {
                                app.run_dir = dir;
                            }
                        }
                        Err(err) => app.feed.push(Kind::Error, err),
                    }
                    finish(&mut app, &tx);
                }
                AppEvent::Stage(line) => app.stage = line,
                AppEvent::Dir(dir, entries) => {
                    app.dialog = Arc::new(Mutex::new(Dialog::new(&dir)));
                    app.dir = dir;
                    app.run_dir = None;
                    app.feed.entries.extend(entries);
                    finish(&mut app, &tx);
                }
            },
            Some(Ok(event)) = events.next() => {
                match app.prompt.handle(event, &mut app.feed) {
                    Action::Quit if app.busy.is_some() && !app.quit_armed => {
                        app.quit_armed = true;
                        app.feed.push(Kind::Error, "Идёт работа: выход остановит задание (и ffmpeg). Esc ещё раз — выйти.");
                    }
                    Action::Quit => return Ok(()),
                    Action::Submit(input) => {
                        app.quit_armed = false;
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

fn finish(app: &mut App, tx: &mpsc::UnboundedSender<AppEvent>) {
    app.busy = None;
    app.quit_armed = false;
    app.feed.scroll_back = 0;
    refresh(&app.session, &app.dir, tx);
}

/// Строки ленты после реплики: замечания и ответ.
fn turn_entries(t: &Turn) -> Vec<Entry> {
    let mut out: Vec<Entry> = t.notes.iter().map(|n| Entry::new(Kind::Context, format!("⚠ {n}"))).collect();
    out.push(Entry::new(Kind::Answer, with_usage(t.reply.clone(), t.usage)));
    out
}

/// `true` — выйти.
fn submit(app: &mut App, input: String, tx: &mpsc::UnboundedSender<AppEvent>) -> bool {
    let (cmd, rest) = match input.strip_prefix('/') {
        Some(c) => {
            let (cmd, rest) = c.split_once(char::is_whitespace).unwrap_or((c, ""));
            (cmd.to_string(), rest.trim().to_string())
        }
        None => (String::new(), input.clone()),
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
        _ => {}
    }
    if app.busy.is_some() {
        app.feed.push(Kind::Error, "Ассистент ещё работает — дождитесь ответа.");
        return false;
    }
    app.feed.push(Kind::User, input.clone());
    let s = app.session.clone();
    let tx = tx.clone();
    match cmd.as_str() {
        "" => {
            app.busy = Some("берусь…".into());
            let dialog = app.dialog.clone();
            let run_dir = app.run_dir.clone();
            tokio::spawn(async move {
                let progress_tx = tx.clone();
                let progress = move |step: Step| {
                    let _ = progress_tx.send(match step {
                        Step::Status(msg) | Step::Job(msg) => AppEvent::Progress(msg),
                        Step::Tool(call) => AppEvent::Entry(tool_entry(&call, "")),
                    });
                };
                let mut d = dialog.lock().await;
                let result = match d.send(&s, &rest, &progress).await {
                    Ok(turn) => {
                        // Не сохранился диалог — ответ всё равно показать.
                        let saved = runs::save(&runs::runs_dir(), run_dir.as_deref(), &d, s.client.model()).ok();
                        Ok((turn, saved))
                    }
                    Err(err) => Err(format!("{err:#}")),
                };
                let _ = tx.send(AppEvent::Replied(result));
            });
        }
        "dir" if rest.is_empty() => {
            app.busy = Some("MCP media · list_recordings…".into());
            tokio::spawn(async move {
                let entry = match s.call("list_recordings", json!({})).await {
                    Ok(text) => Entry::new(Kind::Output, format_recordings(&text)),
                    Err(err) => Entry::new(Kind::Error, err),
                };
                let _ = tx.send(AppEvent::Done(vec![entry]));
            });
        }
        "dir" => {
            app.busy = Some("MCP media · inspect…".into());
            tokio::spawn(async move {
                match s.call("inspect", json!({ "dir": rest })).await {
                    Ok(text) => {
                        let dir = serde_json::from_str::<Value>(&text).ok().and_then(|v| v["dir"].as_str().map(str::to_string));
                        let entries = vec![
                            Entry::new(Kind::Output, format_inspection(&text)),
                            Entry::new(Kind::Info, "Новый диалог об этой записи. Напишите «обработай»."),
                        ];
                        let _ = tx.send(AppEvent::Dir(dir.unwrap_or(rest), entries));
                    }
                    Err(err) => {
                        let _ = tx.send(AppEvent::Done(vec![Entry::new(Kind::Error, err)]));
                    }
                }
            });
        }
        "status" => {
            app.busy = Some("MCP media · inspect…".into());
            let dir = app.dir.clone();
            tokio::spawn(async move {
                let entry = match s.call("inspect", json!({ "dir": dir })).await {
                    Ok(text) => Entry::new(Kind::Output, format_inspection(&text)),
                    Err(err) => Entry::new(Kind::Error, err),
                };
                let _ = tx.send(AppEvent::Done(vec![entry]));
            });
        }
        "jobs" => {
            app.busy = Some("MCP media · job_status…".into());
            tokio::spawn(async move {
                let entry = match s.call("job_status", json!({})).await {
                    Ok(text) => Entry::new(Kind::Output, format_jobs(&text)),
                    Err(err) => Entry::new(Kind::Error, err),
                };
                let _ = tx.send(AppEvent::Done(vec![entry]));
            });
        }
        "new" => {
            app.dialog = Arc::new(Mutex::new(Dialog::new(&app.dir)));
            app.run_dir = None;
            app.feed.push(Kind::Info, format!("Новый диалог о записи `{}`.", app.dir));
        }
        other => {
            app.feed.entries.pop();
            app.feed.push(Kind::Error, format!("Неизвестная команда /{other}. /? — список команд."));
        }
    }
    false
}

/// Обновить шапку: следующий шаг каталога.
fn refresh(s: &Arc<Session>, dir: &str, tx: &mpsc::UnboundedSender<AppEvent>) {
    let (s, tx, dir) = (s.clone(), tx.clone(), dir.to_string());
    tokio::spawn(async move {
        let line = match s.call("inspect", json!({ "dir": dir })).await {
            Ok(text) => stage_line(&text),
            Err(err) => format!("недоступно: {}", short(&err, 80)),
        };
        let _ = tx.send(AppEvent::Stage(line));
    });
}

/// `дальше: transcribe — есть 14 MP3-чанков…` для шапки.
fn stage_line(text: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return short(text, 80) };
    let step = v["next"]["step"].as_str().unwrap_or("?");
    match step {
        "done" => "готово: транскрипт и саммари есть".to_string(),
        "nothing" => "нечего обрабатывать".to_string(),
        "ask_summary" => "транскрипт готов · саммари?".to_string(),
        "telegram_summary" => "дальше: версия для Telegram".to_string(),
        "choose_source" => "нужно выбрать исходник".to_string(),
        "wait_job" => format!("идёт {}", v["job"]["kind"].as_str().unwrap_or("задание")),
        other => format!("дальше: {other}"),
    }
}

fn state_word(v: &Value) -> &'static str {
    match v.as_str() {
        Some("valid") => "есть",
        Some("empty") => "пустой — не годится",
        Some("not_a_file") => "не файл",
        Some("stale") => "устарел — саммари новее",
        _ => "нет",
    }
}

/// Результат `inspect` — для чтения человеком.
fn format_inspection(text: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return text.to_string() };
    let chunks = &v["chunks"];
    let n = chunks["files"].as_array().map(Vec::len).unwrap_or(0);
    let chunks_line = match chunks["state"].as_str() {
        Some("valid") => format!("{n} MP3 — годны"),
        Some("invalid") => format!(
            "{n} MP3 — негодны: {}",
            chunks["problems"].as_array().into_iter().flatten().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")
        ),
        _ => "нет".into(),
    };
    let mp4 = &v["mp4"];
    let secs = |k: &str| mp4[k].as_i64().map(|s| s.to_string()).unwrap_or_else(|| "?".into());
    let mp4_line = match mp4["state"].as_str() {
        Some("valid") => format!("совпал с {} ({} с)", mp4["webm"].as_str().unwrap_or("?"), secs("mp4_seconds")),
        Some("mismatch") => format!("НЕ совпал с {}: WebM {} с, MP4 {} с", mp4["webm"].as_str().unwrap_or("?"), secs("webm_seconds"), secs("mp4_seconds")),
        Some("unverified") => "есть (не сверялся)".into(),
        Some("unreadable") => format!("не читается: {}", short(mp4["error"].as_str().unwrap_or(""), 80)),
        Some("ambiguous") => "есть, но WebM несколько — не ясно, с каким сверять".into(),
        Some("busy") => "идёт задание — не проверялся".into(),
        Some("empty") => "пустой — не годится".into(),
        _ => "нет".into(),
    };
    let list = |k: &str| {
        let items: Vec<&str> = v["sources"][k].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if items.is_empty() {
            "—".to_string()
        } else {
            items.join(", ")
        }
    };
    let mut out = format!("Каталог: {} (корень {})\n", v["dir"].as_str().unwrap_or("?"), v["root"].as_str().unwrap_or("?"));
    out.push_str(&format!("  транскрипт   {}\n", state_word(&v["transcript"])));
    out.push_str(&format!("  чанки        {chunks_line}\n"));
    out.push_str(&format!("  output.mp3   {}\n", state_word(&v["mp3"])));
    out.push_str(&format!("  output.mp4   {mp4_line}\n"));
    out.push_str(&format!("  исходники    WebM: {} · MP4: {}\n", list("webm"), list("mp4")));
    out.push_str(&format!("  саммари      {} · промпт саммари: {}\n", state_word(&v["summary"]), state_word(&v["summary_prompt"])));
    out.push_str(&format!("  для Telegram {}\n", state_word(&v["telegram_summary"])));
    let next = &v["next"];
    out.push_str(&format!("Дальше: {} — {}", next["step"].as_str().unwrap_or("?"), next["why"].as_str().unwrap_or("")));
    if let Some(choices) = next["choices"].as_array().filter(|c| !c.is_empty()) {
        out.push_str(&format!("\n  варианты: {}", choices.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")));
    }
    out
}

fn format_recordings(text: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return text.to_string() };
    let items = v["recordings"].as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        return format!("Под корнем {} записей не найдено.", v["root"].as_str().unwrap_or("?"));
    }
    let mut out = format!("Записи под {} — /dir <каталог>:\n", v["root"].as_str().unwrap_or("?"));
    for r in items {
        let stage = match r["stage"].as_str() {
            Some("summary") => "саммари готово",
            Some("transcript") => "транскрипт готов",
            Some("chunks") => "нарезано на чанки",
            Some("mp3") => "есть MP3",
            Some("mp4") => "есть output.mp4",
            _ => "только видео",
        };
        out.push_str(&format!("  {}  — {stage}\n", r["dir"].as_str().unwrap_or("?")));
    }
    out.trim_end().to_string()
}

fn format_jobs(text: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return text.to_string() };
    let jobs = v["jobs"].as_array().cloned().unwrap_or_default();
    if jobs.is_empty() {
        return "Заданий не было.".into();
    }
    jobs.iter()
        .map(|j| {
            let secs = j["elapsed_sec"].as_u64().unwrap_or(0);
            let tail = match (j["state"].as_str(), j["error"].as_str()) {
                (Some("failed"), Some(e)) => format!("ошибка: {}", short(e, 120)),
                (Some(s), _) => format!("{s} · {}", j["progress"].as_str().unwrap_or("")),
                _ => String::new(),
            };
            format!(
                "  {} {} в {} · {}:{:02} · {tail}",
                j["id"].as_str().unwrap_or("?"),
                j["kind"].as_str().unwrap_or("?"),
                j["dir"].as_str().unwrap_or("?"),
                secs / 60,
                secs % 60
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Отрисовка

fn server_span(s: &Session) -> Span<'static> {
    let name = &s.server;
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
    let head = Paragraph::new(Line::from(vec![
        Span::styled("✦ media-assistant", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  MCP "),
        server_span(s),
        Span::raw("  ·  запись "),
        Span::styled(app.dir.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  "),
        Span::styled(app.stage.clone(), Style::default().fg(Color::Yellow)),
        Span::raw("  ·  модель "),
        Span::styled(s.client.model().to_string(), Style::default().fg(Color::Cyan)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
    frame.render_widget(head, header);

    app.feed.draw(frame, body, COMMAND_HINTS);
    app.prompt.draw(frame, input, "реплика ассистенту или /команда (/? — список)");
    let line = app.prompt.status_line(
        app.busy.as_deref(),
        app.spinner,
        "Enter — отправить · /dir — запись · /status — чекпоинты · /jobs — задания · Esc — выход",
    );
    frame.render_widget(Paragraph::new(line), status);
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSPECTION: &str = r#"{"root":"/rec","dir":"2026-10-15","transcript":"missing",
        "chunks":{"state":"missing","files":[]},"mp3":"missing",
        "mp4":{"state":"mismatch","webm":"meeting.webm","webm_seconds":3605,"mp4_seconds":3401},
        "sources":{"webm":["meeting.webm"],"mp4":[]},"summary":"valid","summary_prompt":"missing","telegram_summary":"stale",
        "next":{"step":"convert_to_mp4","tool":"convert_to_mp4","args":{"source":"meeting.webm"},"why":"output.mp4 негоден"}}"#;

    #[test]
    fn inspection_is_readable() {
        let text = format_inspection(INSPECTION);
        assert!(text.starts_with("Каталог: 2026-10-15 (корень /rec)\n"), "{text}");
        assert!(text.contains("output.mp4   НЕ совпал с meeting.webm: WebM 3605 с, MP4 3401 с"), "{text}");
        assert!(text.contains("исходники    WebM: meeting.webm · MP4: —"), "{text}");
        assert!(text.contains("для Telegram устарел — саммари новее"), "{text}");
        assert!(text.ends_with("Дальше: convert_to_mp4 — output.mp4 негоден"), "{text}");
        assert_eq!(stage_line(INSPECTION), "дальше: convert_to_mp4");
        assert_eq!(stage_line(r#"{"next":{"step":"ask_summary"}}"#), "транскрипт готов · саммари?");
    }

    #[test]
    fn recordings_and_jobs_are_readable() {
        let rec = r#"{"root":"/rec","recordings":[{"dir":"2026-10-01","stage":"summary"},{"dir":"2026-10-15","stage":"source"}]}"#;
        assert_eq!(format_recordings(rec), "Записи под /rec — /dir <каталог>:\n  2026-10-01  — саммари готово\n  2026-10-15  — только видео");
        let jobs = r#"{"jobs":[{"id":"j1","kind":"transcribe","dir":"2026-10-15","state":"failed","elapsed_sec":75,"progress":"ошибка","error":"чанк 3: HTTP 500"}]}"#;
        assert_eq!(format_jobs(jobs), "  j1 transcribe в 2026-10-15 · 1:15 · ошибка: чанк 3: HTTP 500");
        assert_eq!(format_jobs(r#"{"jobs":[]}"#), "Заданий не было.");
    }

    /// Кадр целиком: шапка с записью и шагом, лента, ход задания внизу.
    #[test]
    fn draws_header_and_job_progress() {
        std::env::set_var("LLM_API_URL", "http://127.0.0.1:9");
        std::env::set_var("LLM_API_KEY", "test");
        let session = Session { client: llm_core::LlmClient::from_env().unwrap(), mcp: llm_core::McpManager::empty(), server: "media".into() };
        let mut app = App {
            session: Arc::new(session),
            dialog: Arc::new(Mutex::new(Dialog::new("2026-10-15"))),
            dir: "2026-10-15".into(),
            feed: Feed::default(),
            prompt: Prompt::new(COMMAND_HINTS),
            busy: Some("convert_to_mp4 j1 · 12:01 из 58:10 (20%) · идёт 12:30".into()),
            spinner: 0,
            stage: "дальше: convert_to_mp4".into(),
            run_dir: None,
            quit_armed: false,
        };
        app.feed.push(Kind::Answer, "Сделать саммари? (да/нет)");
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(200, 16)).unwrap();
        term.draw(|f| draw(f, &app)).unwrap();
        let buf = term.backend().buffer();
        let text: String = (0..buf.area.height)
            .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>() + "\n")
            .collect();
        assert!(text.contains("media-assistant"), "{text}");
        assert!(text.contains("запись 2026-10-15"), "{text}");
        assert!(text.contains("media: нет в конфигурации"), "{text}");
        assert!(text.contains("12:01 из 58:10 (20%)"), "{text}");
        assert!(text.contains("Сделать саммари? (да/нет)"), "{text}");
    }
}
