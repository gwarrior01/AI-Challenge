//! Терминальный интерфейс (TUI) для LLM-агента: ratatui + crossterm.
//! Enter — отправить запрос, Esc — выход. Ответ приходит асинхронно и не блокирует интерфейс,
//! пока ждём — крутится спиннер. Внизу отображается расход токенов на последний запрос и за сессию.
//! Tab — показать/скрыть сырой JSON запроса и ответа модели под каждым сообщением ассистента.

use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use llm_core::{ChatCompletion, LlmClient, Usage};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::time::Duration;
use tokio::sync::mpsc;

const SPINNER_FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

enum Role {
    User,
    Assistant,
    System,
    Error,
}

struct HistoryItem {
    role: Role,
    text: String,
    /// (JSON запроса, JSON ответа) — заполняется только для ответов ассистента.
    debug: Option<(String, String)>,
}

#[derive(Default)]
struct SessionStats {
    requests: u32,
    tokens: u64,
}

enum AppEvent {
    Response(Result<ChatCompletion>),
}

#[derive(Clone, Copy)]
struct DrawState<'a> {
    client: &'a LlmClient,
    chat_lines: &'a [Line<'static>],
    scroll: u16,
    input: &'a str,
    waiting: bool,
    spinner_frame: usize,
    stats: &'a SessionStats,
    last_usage: Option<Usage>,
    show_debug: bool,
}

/// Вертикальная раскладка экрана: шапка, чат, строка статистики, поле ввода.
/// Вынесена отдельно, чтобы run() мог посчитать высоту области чата для скролла
/// точно так же, как это делает draw() при рендере.
fn layout_chunks(area: Rect) -> [Rect; 4] {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = LlmClient::from_env()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal, client).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, client: LlmClient) -> Result<()> {
    let mut input = String::new();
    let mut history: Vec<HistoryItem> = vec![HistoryItem {
        role: Role::System,
        text: "Challenger (TUI). Enter — отправить, Esc — выход, Tab — JSON запроса/ответа."
            .to_string(),
        debug: None,
    }];
    let mut waiting = false;
    let mut spinner_frame = 0usize;
    let mut stats = SessionStats::default();
    let mut last_usage: Option<Usage> = None;
    let mut show_debug = false;
    // Текущая позиция скролла чата и флаг "прижато к низу" (авто-прокрутка к новым сообщениям).
    let mut scroll: u16 = 0;
    let mut follow_bottom = true;
    const PAGE_STEP: u16 = 8;

    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut events = EventStream::new();
    let mut spinner_tick = tokio::time::interval(Duration::from_millis(120));

    loop {
        let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
        let chunks = layout_chunks(Rect::new(0, 0, cols, rows));
        let inner_width = chunks[1].width.saturating_sub(4).max(10) as usize;
        let visible_height = chunks[1].height.saturating_sub(2);

        let mut chat_lines: Vec<Line<'static>> = Vec::new();
        for item in &history {
            chat_lines.extend(history_item_to_lines(item, inner_width, show_debug));
        }
        let max_scroll = (chat_lines.len() as u16).saturating_sub(visible_height);
        if follow_bottom || scroll >= max_scroll {
            scroll = max_scroll;
            follow_bottom = true;
        } else {
            scroll = scroll.min(max_scroll);
        }

        let draw_state = DrawState {
            client: &client,
            chat_lines: &chat_lines,
            scroll,
            input: &input,
            waiting,
            spinner_frame,
            stats: &stats,
            last_usage,
            show_debug,
        };
        terminal.draw(|frame| draw(frame, &draw_state))?;

        tokio::select! {
            maybe_event = events.next() => {
                match maybe_event {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        match key.code {
                            KeyCode::Esc => break,
                            KeyCode::Tab => show_debug = !show_debug,
                            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                show_debug = !show_debug;
                            }
                            KeyCode::PageUp => {
                                follow_bottom = false;
                                scroll = scroll.saturating_sub(PAGE_STEP);
                            }
                            KeyCode::PageDown => {
                                scroll = scroll.saturating_add(PAGE_STEP);
                            }
                            KeyCode::End => follow_bottom = true,
                            KeyCode::Enter if !waiting => {
                                let prompt = input.trim().to_string();
                                if prompt.is_empty() {
                                    continue;
                                }
                                input.clear();
                                history.push(HistoryItem {
                                    role: Role::User,
                                    text: prompt.clone(),
                                    debug: None,
                                });
                                waiting = true;

                                let client = client.clone();
                                let tx = tx.clone();
                                tokio::spawn(async move {
                                    let response = client.ask(&prompt).await;
                                    let _ = tx.send(AppEvent::Response(response));
                                });
                            }
                            KeyCode::Char(c) if !waiting => input.push(c),
                            KeyCode::Backspace if !waiting => {
                                input.pop();
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) | None => break,
                }
            }
            Some(app_event) = rx.recv() => {
                match app_event {
                    AppEvent::Response(Ok(completion)) => {
                        if let Some(usage) = completion.usage {
                            stats.requests += 1;
                            stats.tokens += usage.total_tokens as u64;
                            last_usage = Some(usage);
                        }
                        history.push(HistoryItem {
                            role: Role::Assistant,
                            text: completion.content,
                            debug: Some((completion.request_json, completion.response_json)),
                        });
                    }
                    AppEvent::Response(Err(err)) => {
                        history.push(HistoryItem {
                            role: Role::Error,
                            text: format!("{err:#}"),
                            debug: None,
                        });
                    }
                }
                waiting = false;
            }
            _ = spinner_tick.tick(), if waiting => {
                spinner_frame = (spinner_frame + 1) % SPINNER_FRAMES.len();
            }
        }
    }

    Ok(())
}

fn draw(frame: &mut Frame, state: &DrawState) {
    let DrawState {
        client,
        chat_lines,
        scroll,
        input,
        waiting,
        spinner_frame,
        stats,
        last_usage,
        show_debug,
    } = *state;

    let area = frame.area();
    let chunks = layout_chunks(area);

    let debug_status = if show_debug {
        Span::styled("JSON: вкл (Tab)", Style::default().fg(Color::Yellow))
    } else {
        Span::styled("JSON: выкл (Tab)", Style::default().fg(Color::DarkGray))
    };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  модель: "),
        Span::styled(client.model(), Style::default().fg(Color::Cyan)),
        Span::raw("  ·  "),
        debug_status,
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let chat_title = if waiting {
        format!(" Диалог {} ", SPINNER_FRAMES[spinner_frame])
    } else {
        " Диалог — PageUp/PageDown скролл, End — в конец ".to_string()
    };

    // scroll и chat_lines уже посчитаны в run() (там же, где решается, прижат ли вид к низу) —
    // здесь просто рендерим готовый текст. List не умеет скроллить внутри одного слишком
    // высокого элемента, из-за чего длинный JSON запроса/ответа мог обрезаться за пределами
    // экрана и становиться невидимым — Paragraph со скроллом лишён этой проблемы.
    let chat = Paragraph::new(chat_lines.to_vec())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(chat_title),
        )
        .scroll((scroll, 0));
    frame.render_widget(chat, chunks[1]);

    let stats_line = match last_usage {
        Some(usage) => format!(
            " Последний запрос: {} + {} = {} ток.  ·  За сессию: {} запрос(ов), {} ток.",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, stats.requests, stats.tokens
        ),
        None => " Расход токенов появится после первого ответа".to_string(),
    };
    let stats_para = Paragraph::new(stats_line).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(stats_para, chunks[2]);

    let (input_title, border_color) = if waiting {
        (
            format!(" Ожидание ответа {} ", SPINNER_FRAMES[spinner_frame]),
            Color::Yellow,
        )
    } else {
        (
            " Запрос — Enter отправить, Esc выход, Tab JSON запроса/ответа ".to_string(),
            Color::Reset,
        )
    };
    let input_para = Paragraph::new(input).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .title(input_title),
    );
    frame.render_widget(input_para, chunks[3]);
}

fn history_item_to_lines(item: &HistoryItem, width: usize, show_debug: bool) -> Vec<Line<'static>> {
    let (label, color) = match item.role {
        Role::User => ("Вы", Color::Cyan),
        Role::Assistant => ("LLM", Color::Green),
        Role::System => ("Инфо", Color::DarkGray),
        Role::Error => ("Ошибка", Color::Red),
    };

    let prefix_width = label.chars().count() + 2;
    let wrap_width = width.saturating_sub(prefix_width).max(10);
    let wrapped = textwrap::wrap(&item.text, wrap_width);

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(wrapped.len() + 1);
    if wrapped.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            format!("{label}: "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )]));
    }
    for (i, part) in wrapped.iter().enumerate() {
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{label}: "),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(part.to_string()),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(" ".repeat(prefix_width)),
                Span::raw(part.to_string()),
            ]));
        }
    }
    if show_debug {
        if let Some((request_json, response_json)) = &item.debug {
            lines.push(debug_heading_line("→ Запрос модели (JSON):"));
            lines.extend(debug_body_lines(request_json, width));
            lines.push(debug_heading_line("← Ответ модели (JSON):"));
            lines.extend(debug_body_lines(response_json, width));
        }
    }

    lines.push(Line::from(""));

    lines
}

fn debug_heading_line(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
    ))
}

fn debug_body_lines(json: &str, width: usize) -> Vec<Line<'static>> {
    // Обычный (не тусклый) цвет — DarkGray почти не виден на многих тёмных темах терминала.
    textwrap::wrap(json, width.max(10))
        .into_iter()
        .map(|part| Line::from(Span::raw(part.into_owned())))
        .collect()
}
