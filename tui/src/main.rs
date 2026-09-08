//! Терминальный интерфейс (TUI) для LLM-агента: ratatui + crossterm.
//!
//! Два экрана:
//!   • Чат — прямой диалог с LLM (как раньше). Enter — отправить, Esc — выход,
//!     Tab — показать/скрыть сырой JSON запроса/ответа.
//!   • Агенты (F2) — управление именованными агентами из общего реестра
//!     (тот же AGENTS_STORE_PATH, что у CLI и веб-интерфейса): список, создание
//!     (n), запуск/остановка (s), удаление (d), чат с запущенным (Enter).
//!
//! Создание агента (n) начинается с выбора режима: быстрый — только имя и
//! системный промпт, остальные параметры берутся по умолчанию; расширенный —
//! полный набор (модель, лимиты, temperature, top_p, reasoning, показ токенов).
//!
//! F2 переключает между экранами Чат ⇄ Агенты (из экрана создания/чата с агентом
//! тоже возвращает в список агентов/чат соответственно) — можно сделать это в
//! любой момент, даже пока агент ещё генерирует ответ: запрос продолжает
//! выполняться в фоне, а ответ появится в истории этого агента независимо от
//! того, какой экран открыт. История диалога каждого агента хранится в SQLite
//! (AGENTS_STORE_PATH) и переживает и остановку/запуск, и перезапуск всего
//! приложения — при повторном открытии чата агент помнит прошлые сообщения.

use anyhow::Result;
use crossterm::{
    event::{DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use llm_core::{AgentConfig, AgentInfo, AgentManager, ChatCompletion, ChatMessage, LlmClient, Usage};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame, Terminal,
};
use std::collections::HashMap;
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
    /// (JSON запроса, JSON ответа) — заполняется только для ответов ассистента в прямом чате.
    debug: Option<(String, String)>,
}

/// Преобразует сообщение из истории агента (роль "user"/"assistant", хранимой
/// в SQLite) в элемент для отображения в чате — используется при открытии
/// чата с агентом, чтобы показать восстановленный после перезапуска диалог.
fn history_item_from_message(message: &ChatMessage) -> HistoryItem {
    let role = match message.role.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => Role::System,
    };
    HistoryItem { role, text: message.content.clone(), debug: None }
}

#[derive(Default)]
struct SessionStats {
    requests: u32,
    tokens: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Chat,
    AgentsList,
    AgentCreate,
    AgentChat,
}

/// Шаги мастера создания агента — по одному вопросу за раз в нижнем поле ввода.
enum CreateStep {
    Name,
    System,
    Model,
    MaxTokens,
    Temperature,
    TopP,
    Reasoning,
    ShowTokens,
}

const CREATE_STEPS_TOTAL_QUICK: usize = 2;
const CREATE_STEPS_TOTAL_ADVANCED: usize = 8;

impl CreateStep {
    fn label(&self) -> &'static str {
        match self {
            CreateStep::Name => " Имя агента ",
            CreateStep::System => " Системный промпт (Enter — пропустить) ",
            CreateStep::Model => " Модель (Enter — по умолчанию) ",
            CreateStep::MaxTokens => " Макс. токенов ответа, число (Enter — без ограничения) ",
            CreateStep::Temperature => " Temperature, число (Enter — по умолчанию) ",
            CreateStep::TopP => " Top P, число (Enter — по умолчанию) ",
            CreateStep::Reasoning => " Reasoning: on / off (Enter — по умолчанию) ",
            CreateStep::ShowTokens => " Показывать токены в ответах? y/n ",
        }
    }

    fn index(&self) -> usize {
        match self {
            CreateStep::Name => 1,
            CreateStep::System => 2,
            CreateStep::Model => 3,
            CreateStep::MaxTokens => 4,
            CreateStep::Temperature => 5,
            CreateStep::TopP => 6,
            CreateStep::Reasoning => 7,
            CreateStep::ShowTokens => 8,
        }
    }

    /// Следующий шаг мастера. В быстром режиме (`quick`) после системного
    /// промпта мастер сразу завершается — остальные параметры остаются
    /// значениями по умолчанию.
    fn next(&self, quick: bool) -> Option<CreateStep> {
        use CreateStep::*;
        match self {
            Name => Some(System),
            System if quick => None,
            System => Some(Model),
            Model => Some(MaxTokens),
            MaxTokens => Some(Temperature),
            Temperature => Some(TopP),
            TopP => Some(Reasoning),
            Reasoning => Some(ShowTokens),
            ShowTokens => None,
        }
    }
}

struct CreateWizard {
    /// `None`, пока пользователь не выбрал режим создания (быстрый/расширенный).
    quick: Option<bool>,
    step: CreateStep,
    config: AgentConfig,
    error: Option<String>,
}

impl CreateWizard {
    fn new() -> Self {
        Self { quick: None, step: CreateStep::Name, config: AgentConfig::new(String::new()), error: None }
    }

    /// Обрабатывает ввод текущего шага. Возвращает true, когда мастер завершён
    /// (после последнего шага) — тогда `config` готова для создания агента.
    fn submit(&mut self, raw: &str) -> bool {
        let quick = self.quick.unwrap_or(false);
        let value = raw.trim();
        self.error = None;
        match self.step {
            CreateStep::Name => {
                if value.is_empty() {
                    self.error = Some("Имя не может быть пустым".to_string());
                    return false;
                }
                self.config.name = value.to_string();
            }
            CreateStep::System => {
                self.config.system_prompt = if value.is_empty() { None } else { Some(value.to_string()) };
            }
            CreateStep::Model => {
                self.config.model = if value.is_empty() { None } else { Some(value.to_string()) };
            }
            CreateStep::MaxTokens => {
                if value.is_empty() {
                    self.config.max_tokens = None;
                } else {
                    match value.parse::<u32>() {
                        Ok(n) if n > 0 => self.config.max_tokens = Some(n),
                        _ => {
                            self.error = Some("Введите целое положительное число или оставьте пустым".to_string());
                            return false;
                        }
                    }
                }
            }
            CreateStep::Temperature => {
                if value.is_empty() {
                    self.config.temperature = None;
                } else {
                    match value.parse::<f32>() {
                        Ok(n) => self.config.temperature = Some(n),
                        Err(_) => {
                            self.error = Some("Введите число или оставьте пустым".to_string());
                            return false;
                        }
                    }
                }
            }
            CreateStep::TopP => {
                if value.is_empty() {
                    self.config.top_p = None;
                } else {
                    match value.parse::<f32>() {
                        Ok(n) => self.config.top_p = Some(n),
                        Err(_) => {
                            self.error = Some("Введите число или оставьте пустым".to_string());
                            return false;
                        }
                    }
                }
            }
            CreateStep::Reasoning => {
                self.config.reasoning = match value.to_lowercase().as_str() {
                    "" => None,
                    "on" => Some(true),
                    "off" => Some(false),
                    _ => {
                        self.error = Some("Введите on, off или оставьте пустым".to_string());
                        return false;
                    }
                };
            }
            CreateStep::ShowTokens => {
                self.config.show_tokens = matches!(value.to_lowercase().as_str(), "y" | "yes" | "д" | "да");
            }
        }

        match self.step.next(quick) {
            Some(next) => {
                self.step = next;
                false
            }
            None => true,
        }
    }
}

enum AppEvent {
    DirectResponse(Result<ChatCompletion>),
    AgentResponse { name: String, result: Result<llm_core::AgentReply> },
}

struct DrawState<'a> {
    screen: Screen,
    client: &'a LlmClient,
    chat_lines: &'a [Line<'static>],
    scroll: u16,
    input: &'a str,
    waiting: bool,
    waiting_agent: Option<&'a str>,
    spinner_frame: usize,
    stats: &'a SessionStats,
    last_usage: Option<Usage>,
    show_debug: bool,
    agents: &'a [AgentInfo],
    agents_selected: usize,
    confirm_delete: Option<&'a str>,
    wizard: Option<&'a CreateWizard>,
    agent_chat_name: Option<&'a str>,
    agent_chat_running: bool,
}

/// Высота поля ввода по умолчанию (1 строка текста + рамка сверху/снизу).
const INPUT_HEIGHT_DEFAULT: u16 = 3;
/// Высота поля ввода в чате с агентом — вдвое больше, чтобы было удобнее
/// работать с более длинными или вставленными многострочными запросами.
const INPUT_HEIGHT_AGENT_CHAT: u16 = 6;

/// Вертикальная раскладка экрана: шапка, основная область, строка статистики, поле ввода.
/// Одна и та же раскладка используется всеми экранами (Чат/Агенты/создание/чат с агентом),
/// но высота поля ввода настраивается — в чате с агентом оно крупнее.
fn layout_chunks(area: Rect) -> [Rect; 4] {
    layout_chunks_with_input_height(area, INPUT_HEIGHT_DEFAULT)
}

fn layout_chunks_with_input_height(area: Rect, input_height: u16) -> [Rect; 4] {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(input_height),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

/// Высота поля ввода для текущего экрана — используется и при расчёте
/// раскладки для скролла в run(), и при отрисовке, чтобы оба места
/// согласованно резервировали одинаковое место под ввод.
fn input_height_for(screen: Screen) -> u16 {
    match screen {
        Screen::AgentChat => INPUT_HEIGHT_AGENT_CHAT,
        _ => INPUT_HEIGHT_DEFAULT,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = LlmClient::from_env()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Bracketed paste: терминал присылает вставленный текст одним событием
    // Event::Paste целиком, а не потоком отдельных нажатий клавиш — без этого
    // встроенные в буфер обмена переводы строк читались бы как нажатия Enter
    // и обрывали сообщение раньше времени, посреди вставки.
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal, client).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableBracketedPaste, LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, client: LlmClient) -> Result<()> {
    let mut input = String::new();
    let mut history: Vec<HistoryItem> = vec![HistoryItem {
        role: Role::System,
        text: "Challenger (TUI). Enter — отправить, Esc — выход, Tab — JSON запроса/ответа, \
               F2 — управление агентами."
            .to_string(),
        debug: None,
    }];
    let mut waiting = false;
    // Имя агента, чей ответ сейчас ожидается (None — если ждём прямой чат).
    // Позволяет показать индикатор на экране списка агентов даже когда его
    // диалог сейчас не открыт (пользователь вышел из него клавишей Esc/F2).
    let mut waiting_agent: Option<String> = None;
    let mut spinner_frame = 0usize;
    let mut stats = SessionStats::default();
    let mut last_usage: Option<Usage> = None;
    let mut show_debug = false;
    // Текущая позиция скролла чата и флаг "прижато к низу" (авто-прокрутка к новым сообщениям).
    let mut scroll: u16 = 0;
    let mut follow_bottom = true;
    const PAGE_STEP: u16 = 8;

    let mut screen = Screen::Chat;
    // Экран чата с агентом использует более высокое поле ввода, чем остальные
    // экраны, — при переключении между ними геометрия блоков не совпадает, и
    // ratatui, перерисовывающий только изменившиеся ячейки, может оставить на
    // экране обрывок текста из области, которая только что принадлежала другому
    // виджету. Отслеживаем предыдущий экран и при каждой смене форсируем полную
    // перерисовку терминала, чтобы такой "хвост" не оставался виден.
    let mut previous_screen = screen;
    let agent_manager = AgentManager::from_env(client.clone())?;
    let mut agents_selected: usize = 0;
    let mut confirm_delete: Option<String> = None;
    let mut wizard: Option<CreateWizard> = None;
    let mut agent_chat_name: Option<String> = None;
    let mut agent_histories: HashMap<String, Vec<HistoryItem>> = HashMap::new();
    let mut agent_last_usage: HashMap<String, Usage> = HashMap::new();

    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut events = EventStream::new();
    let mut spinner_tick = tokio::time::interval(Duration::from_millis(120));

    loop {
        if screen != previous_screen {
            terminal.clear()?;
            previous_screen = screen;
        }

        let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
        let chunks = layout_chunks_with_input_height(Rect::new(0, 0, cols, rows), input_height_for(screen));
        let inner_width = chunks[1].width.saturating_sub(4).max(10) as usize;
        let visible_height = chunks[1].height.saturating_sub(2);

        let active_history: &[HistoryItem] = match &screen {
            Screen::AgentChat => agent_chat_name
                .as_ref()
                .and_then(|n| agent_histories.get(n))
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            _ => &history,
        };
        let mut chat_lines: Vec<Line<'static>> = Vec::new();
        for item in active_history {
            chat_lines.extend(history_item_to_lines(item, inner_width, show_debug));
        }
        let max_scroll = (chat_lines.len() as u16).saturating_sub(visible_height);
        if follow_bottom || scroll >= max_scroll {
            scroll = max_scroll;
            follow_bottom = true;
        } else {
            scroll = scroll.min(max_scroll);
        }

        let agents_snapshot = agent_manager.list();
        if agents_selected >= agents_snapshot.len() {
            agents_selected = agents_snapshot.len().saturating_sub(1);
        }
        let agent_chat_running = agent_chat_name
            .as_ref()
            .and_then(|n| agent_manager.get(n))
            .map(|a| a.is_running())
            .unwrap_or(false);
        let screen_last_usage = match &screen {
            Screen::AgentChat => agent_chat_name.as_ref().and_then(|n| agent_last_usage.get(n)).copied(),
            _ => last_usage,
        };

        let draw_state = DrawState {
            screen,
            client: &client,
            chat_lines: &chat_lines,
            scroll,
            input: &input,
            waiting,
            waiting_agent: waiting_agent.as_deref(),
            spinner_frame,
            stats: &stats,
            last_usage: screen_last_usage,
            show_debug,
            agents: &agents_snapshot,
            agents_selected,
            confirm_delete: confirm_delete.as_deref(),
            wizard: wizard.as_ref(),
            agent_chat_name: agent_chat_name.as_deref(),
            agent_chat_running,
        };
        terminal.draw(|frame| draw(frame, &draw_state))?;

        tokio::select! {
            maybe_event = events.next() => {
                match maybe_event {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        // В экране "Чат" Esc всегда завершает программу — как и раньше.
                        if key.code == KeyCode::Esc && matches!(screen, Screen::Chat) {
                            break;
                        }

                        // F2 переключает экраны даже пока ждём ответа — запрос агенту/модели
                        // продолжает выполняться в фоне независимо от того, какой экран открыт.
                        if key.code == KeyCode::F(2) {
                            screen = match screen {
                                Screen::Chat => Screen::AgentsList,
                                Screen::AgentsList => Screen::Chat,
                                Screen::AgentCreate => { wizard = None; Screen::AgentsList }
                                Screen::AgentChat => { agent_chat_name = None; Screen::AgentsList }
                            };
                            input.clear();
                            confirm_delete = None;
                            continue;
                        }

                        match screen {
                            Screen::Chat => match key.code {
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
                                        let _ = tx.send(AppEvent::DirectResponse(response));
                                    });
                                }
                                KeyCode::Char(c) if !waiting => input.push(c),
                                KeyCode::Backspace if !waiting => {
                                    input.pop();
                                }
                                _ => {}
                            },
                            Screen::AgentsList => {
                                if let Some(pending) = confirm_delete.clone() {
                                    if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
                                        let _ = agent_manager.remove(&pending);
                                        if agent_chat_name.as_deref() == Some(pending.as_str()) {
                                            agent_chat_name = None;
                                        }
                                        agent_histories.remove(&pending);
                                        agent_last_usage.remove(&pending);
                                    }
                                    confirm_delete = None;
                                    continue;
                                }
                                match key.code {
                                    KeyCode::Esc => screen = Screen::Chat,
                                    KeyCode::Up => {
                                        agents_selected = agents_selected.saturating_sub(1);
                                    }
                                    KeyCode::Down => {
                                        if agents_selected + 1 < agents_snapshot.len() {
                                            agents_selected += 1;
                                        }
                                    }
                                    KeyCode::Char('n') => {
                                        wizard = Some(CreateWizard::new());
                                        screen = Screen::AgentCreate;
                                        input.clear();
                                    }
                                    KeyCode::Char('s') => {
                                        if let Some(info) = agents_snapshot.get(agents_selected) {
                                            let name = info.config.name.clone();
                                            let _ = if info.running {
                                                agent_manager.stop(&name)
                                            } else {
                                                agent_manager.start(&name)
                                            };
                                        }
                                    }
                                    KeyCode::Char('d') => {
                                        if let Some(info) = agents_snapshot.get(agents_selected) {
                                            confirm_delete = Some(info.config.name.clone());
                                        }
                                    }
                                    KeyCode::Enter => {
                                        if let Some(info) = agents_snapshot.get(agents_selected) {
                                            if info.running {
                                                let name = info.config.name.clone();
                                                agent_histories.entry(name.clone()).or_insert_with(|| {
                                                    agent_manager
                                                        .get(&name)
                                                        .map(|agent| {
                                                            agent
                                                                .history()
                                                                .iter()
                                                                .map(history_item_from_message)
                                                                .collect()
                                                        })
                                                        .unwrap_or_default()
                                                });
                                                agent_chat_name = Some(name);
                                                screen = Screen::AgentChat;
                                                follow_bottom = true;
                                                input.clear();
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Screen::AgentCreate => {
                                if let Some(w) = wizard.as_mut() {
                                    if w.quick.is_none() {
                                        // Первый шаг — выбор режима: быстрый (только имя и
                                        // системный промпт) или расширенный (все параметры).
                                        match key.code {
                                            KeyCode::Esc => {
                                                wizard = None;
                                                screen = Screen::AgentsList;
                                            }
                                            KeyCode::Char('1') | KeyCode::Char('q') | KeyCode::Char('Q') => {
                                                w.quick = Some(true);
                                            }
                                            KeyCode::Char('2') | KeyCode::Char('a') | KeyCode::Char('A') => {
                                                w.quick = Some(false);
                                            }
                                            _ => {}
                                        }
                                        continue;
                                    }
                                    match key.code {
                                        KeyCode::Esc => {
                                            wizard = None;
                                            screen = Screen::AgentsList;
                                            input.clear();
                                        }
                                        KeyCode::Enter => {
                                            let raw = input.clone();
                                            let done = w.submit(&raw);
                                            input.clear();
                                            if done {
                                                match agent_manager.create(w.config.clone()) {
                                                    Ok(info) => {
                                                        let created_name = info.config.name.clone();
                                                        wizard = None;
                                                        screen = Screen::AgentsList;
                                                        let refreshed = agent_manager.list();
                                                        agents_selected = refreshed
                                                            .iter()
                                                            .position(|a| a.config.name == created_name)
                                                            .unwrap_or(0);
                                                    }
                                                    Err(err) => {
                                                        w.error = Some(err.to_string());
                                                        w.step = CreateStep::Name;
                                                    }
                                                }
                                            }
                                        }
                                        KeyCode::Char(c) => input.push(c),
                                        KeyCode::Backspace => {
                                            input.pop();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Screen::AgentChat => match key.code {
                                // Esc возвращает к списку агентов даже если этот агент ещё
                                // не ответил — запрос продолжает выполняться в фоне, а ответ
                                // (или ошибка) появится в его истории независимо от того,
                                // какой экран открыт в момент получения.
                                KeyCode::Esc => {
                                    screen = Screen::AgentsList;
                                    agent_chat_name = None;
                                    input.clear();
                                }
                                KeyCode::Tab => show_debug = !show_debug,
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
                                    if let Some(name) = agent_chat_name.clone() {
                                        if let Some(agent) = agent_manager.get(&name) {
                                            input.clear();
                                            agent_histories.entry(name.clone()).or_default().push(HistoryItem {
                                                role: Role::User,
                                                text: prompt.clone(),
                                                debug: None,
                                            });
                                            waiting = true;
                                            waiting_agent = Some(name.clone());
                                            let tx = tx.clone();
                                            tokio::spawn(async move {
                                                let response = agent.handle_request(&prompt).await;
                                                let _ = tx.send(AppEvent::AgentResponse { name, result: response });
                                            });
                                        }
                                    }
                                }
                                KeyCode::Char(c) if !waiting => input.push(c),
                                KeyCode::Backspace if !waiting => {
                                    input.pop();
                                }
                                _ => {}
                            },
                        }
                    }
                    // Вставленный текст приходит одним куском — просто дописываем его в
                    // поле ввода, не трактуя содержащиеся в нём переводы строк как Enter.
                    // Отправка по-прежнему происходит только по явному нажатию Enter.
                    Some(Ok(Event::Paste(text))) => match screen {
                        Screen::Chat | Screen::AgentChat if !waiting => input.push_str(&text),
                        Screen::AgentCreate if wizard.as_ref().is_some_and(|w| w.quick.is_some()) => {
                            input.push_str(&text);
                        }
                        _ => {}
                    },
                    Some(Ok(_)) => {}
                    Some(Err(_)) | None => break,
                }
            }
            Some(app_event) = rx.recv() => {
                match app_event {
                    AppEvent::DirectResponse(Ok(completion)) => {
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
                    AppEvent::DirectResponse(Err(err)) => {
                        history.push(HistoryItem {
                            role: Role::Error,
                            text: format!("{err:#}"),
                            debug: None,
                        });
                    }
                    AppEvent::AgentResponse { name, result } => {
                        let entry = agent_histories.entry(name.clone()).or_default();
                        match result {
                            Ok(reply) => {
                                if let Some(usage) = reply.usage {
                                    agent_last_usage.insert(name, usage);
                                }
                                entry.push(HistoryItem {
                                    role: Role::Assistant,
                                    text: reply.text,
                                    debug: None,
                                });
                            }
                            Err(err) => {
                                entry.push(HistoryItem {
                                    role: Role::Error,
                                    text: format!("{err:#}"),
                                    debug: None,
                                });
                            }
                        }
                    }
                }
                waiting = false;
                waiting_agent = None;
            }
            _ = spinner_tick.tick(), if waiting => {
                spinner_frame = (spinner_frame + 1) % SPINNER_FRAMES.len();
            }
        }
    }

    Ok(())
}

fn draw(frame: &mut Frame, state: &DrawState) {
    match state.screen {
        Screen::Chat => draw_chat(frame, state),
        Screen::AgentsList => draw_agents_list(frame, state),
        Screen::AgentCreate => draw_agent_create(frame, state),
        Screen::AgentChat => draw_agent_chat(frame, state),
    }
}

fn draw_chat(frame: &mut Frame, state: &DrawState) {
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
        ..
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
        Span::raw("  ·  F2 агенты"),
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
    render_input_box(frame, chunks[3], input, input_title, border_color);
}

/// Короткая строка с ключевыми параметрами конфигурации агента — используется
/// и в списке агентов, и в детальной панели под ним.
fn agent_meta_line(config: &AgentConfig) -> String {
    let mut parts = vec![format!("модель: {}", config.model.as_deref().unwrap_or("по умолчанию"))];
    parts.push(format!(
        "токены: {}",
        if config.show_tokens { "показывать" } else { "скрывать" }
    ));
    if let Some(t) = config.temperature {
        parts.push(format!("temperature: {t}"));
    }
    if let Some(t) = config.top_p {
        parts.push(format!("top_p: {t}"));
    }
    if let Some(m) = config.max_tokens {
        parts.push(format!("макс. токенов: {m}"));
    }
    if let Some(r) = config.reasoning {
        parts.push(format!("reasoning: {}", if r { "on" } else { "off" }));
    }
    parts.join(" · ")
}

fn agents_list_lines(
    agents: &[AgentInfo],
    selected: usize,
    waiting_agent: Option<&str>,
    spinner_frame: usize,
) -> Vec<Line<'static>> {
    if agents.is_empty() {
        return vec![Line::from(Span::styled(
            "Агентов пока нет — нажмите 'n', чтобы создать первого.",
            Style::default().fg(Color::DarkGray),
        ))];
    }
    agents
        .iter()
        .enumerate()
        .map(|(i, info)| {
            let marker = if i == selected { "» " } else { "  " };
            let is_waiting = waiting_agent == Some(info.config.name.as_str());
            let (dot, dot_color) = if is_waiting {
                (SPINNER_FRAMES[spinner_frame], Color::Yellow)
            } else if info.running {
                ("●", Color::Green)
            } else {
                ("○", Color::DarkGray)
            };
            let name_style = if i == selected {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let mut spans = vec![
                Span::raw(marker),
                Span::styled(dot, Style::default().fg(dot_color)),
                Span::raw(" "),
                Span::styled(info.config.name.clone(), name_style),
                Span::styled(
                    format!("  ·  {}", agent_meta_line(&info.config)),
                    Style::default().fg(Color::DarkGray),
                ),
            ];
            if is_waiting {
                spans.push(Span::styled("  ·  отвечает…", Style::default().fg(Color::Yellow)));
            }
            Line::from(spans)
        })
        .collect()
}

fn draw_agents_list(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks(area);

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  Агенты  ·  F2/Esc — назад к чату"),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let lines = agents_list_lines(state.agents, state.agents_selected, state.waiting_agent, state.spinner_frame);
    let list = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Агенты "),
    );
    frame.render_widget(list, chunks[1]);

    let (stats_line, stats_color) = if let Some(name) = state.confirm_delete {
        (
            format!(" Удалить агента «{name}»? y — да, любая другая клавиша — отмена"),
            Color::Yellow,
        )
    } else {
        (
            " ↑/↓ выбор · n новый · s старт/стоп · Enter чат с запущенным · d удалить · Esc/F2 назад "
                .to_string(),
            Color::DarkGray,
        )
    };
    frame.render_widget(Paragraph::new(stats_line).style(Style::default().fg(stats_color)), chunks[2]);

    // chunks[3] с рамкой даёт всего одну строку содержимого — показываем только
    // сводку параметров, системный промпт целиком виден в форме создания.
    let detail_text = match state.agents.get(state.agents_selected) {
        Some(info) => match &info.config.system_prompt {
            Some(sp) => format!("{}  ·  промпт: {sp}", agent_meta_line(&info.config)),
            None => agent_meta_line(&info.config),
        },
        None => "Агент не выбран".to_string(),
    };
    let detail = Paragraph::new(detail_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Выбранный агент "),
    );
    frame.render_widget(detail, chunks[3]);
}

/// Рендерит поле ввода с переносом длинных строк по ширине и автопрокруткой
/// к последней введённой строке. Без переноса `Paragraph` обрезает (а не
/// переносит) строки шире области — это делает невидимым «хвост» длинного
/// или вставленного многострочного сообщения, из-за чего казалось, что текст
/// вылезает за рамки поля и не даёт увидеть, что реально введено.
fn render_input_box(frame: &mut Frame, area: Rect, input: &str, title: String, border_color: Color) {
    let inner_width = area.width.saturating_sub(2).max(1) as usize;
    let visible_height = area.height.saturating_sub(2);

    let mut lines: Vec<Line<'static>> = textwrap::wrap(input, inner_width)
        .into_iter()
        .map(|s| Line::from(s.into_owned()))
        .collect();
    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    let scroll_y = (lines.len() as u16).saturating_sub(visible_height);

    let input_para = Paragraph::new(lines).scroll((scroll_y, 0)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .title(title),
    );
    frame.render_widget(input_para, area);
}

fn summary_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label}: "), Style::default().fg(Color::DarkGray)),
        Span::raw(value.to_string()),
    ])
}

fn draw_agent_create(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks(area);
    let Some(wizard) = state.wizard else { return };

    let Some(quick) = wizard.quick else {
        draw_agent_create_mode_choice(frame, &chunks);
        return;
    };

    let total = if quick { CREATE_STEPS_TOTAL_QUICK } else { CREATE_STEPS_TOTAL_ADVANCED };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  ·  Новый агент ({}) · шаг {} из {total}",
            if quick { "быстро" } else { "расширенно" },
            wizard.step.index()
        )),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let mut lines = vec![
        summary_line("Имя", &wizard.config.name),
        summary_line("Системный промпт", wizard.config.system_prompt.as_deref().unwrap_or("—")),
    ];
    if quick {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Остальные параметры — по умолчанию (модель, лимиты, температура и т.д.). \
             Чтобы задать их явно, отмените (Esc) и выберите расширенный режим.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.push(summary_line("Модель", wizard.config.model.as_deref().unwrap_or("по умолчанию")));
        lines.push(summary_line(
            "Макс. токенов",
            &wizard.config.max_tokens.map(|n| n.to_string()).unwrap_or_else(|| "без ограничения".into()),
        ));
        lines.push(summary_line(
            "Temperature",
            &wizard.config.temperature.map(|n| n.to_string()).unwrap_or_else(|| "по умолчанию".into()),
        ));
        lines.push(summary_line(
            "Top P",
            &wizard.config.top_p.map(|n| n.to_string()).unwrap_or_else(|| "по умолчанию".into()),
        ));
        lines.push(summary_line(
            "Reasoning",
            match wizard.config.reasoning {
                Some(true) => "on",
                Some(false) => "off",
                None => "по умолчанию",
            },
        ));
        lines.push(summary_line("Показывать токены", if wizard.config.show_tokens { "да" } else { "нет" }));
    }
    if let Some(err) = &wizard.error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(err.clone(), Style::default().fg(Color::Red))));
    }

    let body = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Параметры агента "),
    );
    frame.render_widget(body, chunks[1]);

    frame.render_widget(
        Paragraph::new(" Enter — далее · Esc — отмена ").style(Style::default().fg(Color::DarkGray)),
        chunks[2],
    );

    render_input_box(frame, chunks[3], state.input, wizard.step.label().to_string(), Color::Cyan);
}

/// Первый шаг создания агента: выбор между быстрым режимом (только имя и
/// системный промпт, остальное — по умолчанию) и расширенным (все параметры).
fn draw_agent_create_mode_choice(frame: &mut Frame, chunks: &[Rect; 4]) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  Новый агент — выбор режима"),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let body = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("1 / q", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("  —  Быстро: только имя и системный промпт, всё остальное — по умолчанию"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("2 / a", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("  —  Расширенно: модель, лимиты токенов, temperature, top_p, reasoning, показ токенов"),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Режим создания агента "),
    );
    frame.render_widget(body, chunks[1]);

    frame.render_widget(
        Paragraph::new(" 1/q — быстро · 2/a — расширенно · Esc — отмена ")
            .style(Style::default().fg(Color::DarkGray)),
        chunks[2],
    );
    frame.render_widget(
        Paragraph::new("").block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded)),
        chunks[3],
    );
}

fn draw_agent_chat(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks_with_input_height(area, INPUT_HEIGHT_AGENT_CHAT);
    let name = state.agent_chat_name.unwrap_or("?");

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  агент: "),
        Span::styled(name, Style::default().fg(Color::Cyan)),
        Span::raw(if state.agent_chat_running { "  ·  запущен" } else { "  ·  остановлен" }),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let this_agent_waiting = state.waiting && state.waiting_agent == Some(name);
    let chat_title = if this_agent_waiting {
        format!(" Диалог {} ", SPINNER_FRAMES[state.spinner_frame])
    } else {
        " Диалог — PageUp/PageDown скролл, End в конец, Esc назад к списку агентов ".to_string()
    };
    let chat = Paragraph::new(state.chat_lines.to_vec())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(chat_title),
        )
        .scroll((state.scroll, 0));
    frame.render_widget(chat, chunks[1]);

    let stats_line = match state.last_usage {
        Some(usage) => format!(
            " Последний запрос: {} + {} = {} ток.",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        ),
        None => " Расход токенов появится после первого ответа агента".to_string(),
    };
    frame.render_widget(Paragraph::new(stats_line).style(Style::default().fg(Color::DarkGray)), chunks[2]);

    let (input_title, border_color) = if this_agent_waiting {
        (format!(" Ожидание ответа {} ", SPINNER_FRAMES[state.spinner_frame]), Color::Yellow)
    } else if state.waiting {
        (
            " Ожидаем ответ другого агента — можно выйти (Esc) и посмотреть остальных ".to_string(),
            Color::DarkGray,
        )
    } else if !state.agent_chat_running {
        (" Агент остановлен — вернитесь (Esc) и запустите его (s) ".to_string(), Color::DarkGray)
    } else {
        (" Запрос — Enter отправить, Esc назад к списку агентов ".to_string(), Color::Reset)
    };
    render_input_box(frame, chunks[3], state.input, input_title, border_color);
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
