//! Терминальный интерфейс (TUI) для LLM-агента: ratatui + crossterm.
//!
//! Два экрана:
//!   • Чат — прямой диалог с LLM (как раньше). Enter — отправить, Esc — выход,
//!     Tab — показать/скрыть сырой JSON запроса/ответа.
//!   • Агенты (F2) — управление именованными агентами из общего реестра
//!     (тот же AGENTS_STORE_PATH, что у CLI и веб-интерфейса): список, создание
//!     (n), запуск/остановка (s), удаление (d), чат с запущенным (Enter). В чате
//!     с агентом Tab так же показывает/скрывает JSON запроса/ответа — но только
//!     для сообщений этой сессии, история из SQLite его не несёт.
//!
//! Создание агента (n) начинается с выбора режима: быстрый — только имя и
//! системный промпт, остальные параметры берутся по умолчанию; расширенный —
//! полный набор (модель, лимиты, temperature, top_p, reasoning, показ токенов,
//! сжатие контекста).
//!
//! F2 переключает между экранами Чат ⇄ Агенты (из экрана создания/чата с агентом
//! тоже возвращает в список агентов/чат соответственно) — можно сделать это в
//! любой момент, даже пока агент ещё генерирует ответ: запрос продолжает
//! выполняться в фоне, а ответ появится в истории этого агента независимо от
//! того, какой экран открыт. История диалога каждого агента хранится в SQLite
//! (AGENTS_STORE_PATH) и переживает и остановку/запуск, и перезапуск всего
//! приложения — при повторном открытии чата агент помнит прошлые сообщения.
//!
//! ## Модель памяти агента (F3 из чата с агентом)
//!
//! Из чата с конкретным агентом F3 открывает экран его памяти (Esc/F3 —
//! обратно к диалогу), показывающий три уровня раздельно (см. llm_core::memory):
//! краткосрочную (текущий диалог — обзорной строкой, сам диалог виден в чате),
//! рабочую (данные ОБЩЕЙ ЗАДАЧИ, видны только присоединившимся к ней агентам)
//! и долговременную (решения/знания — ОБЩАЯ для ВСЕХ агентов сразу,
//! правка через одного агента видна через любого другого). Рабочая и
//! долговременная память заполняются только явно, командами в поле ввода
//! этого экрана — никакого автоматического попадания данных не по адресу:
//! `remember <ключ> <значение>
//! [--category CAT]`, `forget <ключ>`, `task start <название> [--goal ТЕКСТ]`
//! (создаёт общую задачу и сразу присоединяет к ней агента), `task join
//! <название>` (присоединяет к УЖЕ существующей задаче — созданной этим же
//! или другим агентом; экран списком показывает задачи, доступные для join),
//! `task set <ключ> <значение>` (видно сразу всем присоединённым агентам),
//! `task advance <planning|execution|validation|done> [--step T] [--expect T]`
//! (переход конечного автомата задачи — см. llm_core::memory::Stage — только
//! на легальный следующий этап), `task step <текст>`/`task expect <текст>`
//! (правка текущего шага/ожидаемого действия без смены этапа), `task pause`/
//! `task resume` (пауза независима от этапа — ставится на любом; резюме не
//! требует переобъяснять контекст агенту, т.к. этап/шаг/ожидание читаются из
//! БД при каждом запросе), `task finish` (завершает задачу и удаляет её данные
//! ДЛЯ ВСЕХ участников разом — синтаксис совпадает с одноимёнными
//! подкомандами `llm-cli agent`, см. cli/src/main.rs, — поведение не
//! расходится между интерфейсами).
//!
//! Этот же экран показывает **персонализацию** (см. llm_core::profile) —
//! отдельную от трёх уровней памяти ось: markdown-профиль, описывающий манеру
//! общения/язык ответа/ограничения, подключаемый к КАЖДОМУ запросу агента.
//! Команда `profile <профиль|none>` меняет его на лету, `profile new <имя>`
//! заводит новый профиль (пустой шаблон на диске, см. llm_core::profile::create)
//! прямо из интерфейса — без правки файлов руками (создание агента (n) в
//! расширенном режиме тоже спрашивает профиль отдельным шагом мастера).
//!
//! И **инварианты** (см. llm_core::invariants) — самая приоритетная ось,
//! показанная первой в этом же экране: жёсткие правила (архитектура, стек,
//! бизнес-правила), общие для ВСЕХ агентов (не выбираются на агента, как
//! профиль), подключаются первым системным сообщением к каждому запросу.
//! Команды `invariants new <id>`/`invariants remove <id>` заводят/снимают
//! глобальный (файловый) инвариант прямо из интерфейса; список действующих
//! виден в самом экране. Инварианты ТОЛЬКО этой задачи (текстовые — `task
//! invariant set <id> <текст>`/`invariant remove <id>`, и структурные —
//! `task forbid/allow/require-approval/unrequire-approval <из> <в>`, реально
//! проверяемые кодом на переходах автомата, а не только текстом) показаны в
//! секции "Рабочая — данные ОБЩЕЙ задачи" того же экрана.
//!
//! ## MCP (F4 с любого экрана)
//!
//! Экран MCP-серверов (см. llm_core::mcp) из файла `mcp.json` (или
//! `LLM_MCP_CONFIG`): статус подключения каждого сервера и список его
//! инструментов с описаниями и параметрами. Пробел/`e` включает или выключает
//! выбранный сервер (флаг сохраняется в файле — выключенный сервер остаётся в
//! конфигурации, но не подключается), `r` переподключает его, `l` перечитывает
//! файл целиком. Tab переводит фокус на инструменты выбранного сервера:
//! ↑/↓ — выбор инструмента, Enter — ввести доводы (JSON-объект, заготовка с
//! обязательными параметрами подставляется сама) и ещё раз Enter — вызвать
//! инструмент напрямую, без модели; результат виден под инструментом.
//! К включённым серверам TUI подключается в фоне при запуске;
//! инструменты подключённых серверов агенты предлагают модели в каждом
//! запросе, а каждый вызов инструмента виден в диалоге строкой «Тул» перед
//! ответом агента.

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
    /// Момент пересчёта сводки сжатого контекста или обновления фактов (см.
    /// AgentConfig::context_strategy) — отдельная от System роль, чтобы этот
    /// момент визуально выделялся среди обычных информационных сообщений.
    Compression,
    /// Вызов инструмента моделью (MCP-сервера или автомата задачи) — см.
    /// llm_core::ToolCallRecord; `ToolError` — вызов завершился ошибкой.
    Tool,
    ToolError,
}

struct HistoryItem {
    role: Role,
    text: String,
    /// (JSON запроса, JSON ответа) — заполняется только для ответов ассистента в прямом чате.
    debug: Option<(String, String)>,
    /// Токены этого конкретного сообщения: prompt_tokens для запроса пользователя,
    /// completion_tokens для ответа ассистента — API отдаёт их одной суммой на весь
    /// обмен (см. агент.rs), поэтому раскладываются по паре сообщений при построении
    /// истории (см. [`history_items_from_messages`]). `None`, если неизвестны.
    tokens: Option<u32>,
    /// Денежная стоимость `tokens` этого сообщения (см. llm_core::pricing) — та же
    /// раскладка запрос/ответ, что и у `tokens`. `None` для сообщений прямого чата
    /// (там стоимость не считается — только у именованных агентов) и когда её
    /// неоткуда взять (провайдер её не прислал и ставки цены не заданы).
    cost: Option<f64>,
    /// true, если `cost` — оценка по ставкам (см. llm_core::pricing::CostSource),
    /// а не реальная стоимость, которую вернул провайдер. Не имеет значения,
    /// если `cost` — `None`. Показывается как "≈" перед суммой.
    cost_approx: bool,
}

/// Преобразует историю агента (роль + текст + метрики токенов, хранимые в SQLite)
/// в элементы для отображения в чате — используется при открытии чата с агентом,
/// чтобы показать восстановленный после перезапуска диалог с токенами под каждым
/// сообщением, как в веб-интерфейсе. Токены приходят от API одной суммой на весь
/// обмен и хранятся на сообщении ассистента — раскладываем prompt_tokens на
/// предыдущее сообщение пользователя, completion_tokens оставляем на ответе.
/// Стоимость (см. llm_core::pricing) — реальная от провайдера, если она была
/// сохранена вместе с сообщением, иначе оценка по текущим ставкам; `None`, если
/// ни того ни другого нет.
fn history_items_from_messages(messages: &[(ChatMessage, Option<Usage>)]) -> Vec<HistoryItem> {
    let is_approx = |usage: &Usage| llm_core::pricing::source(usage) == Some(llm_core::pricing::CostSource::Estimated);
    let mut items: Vec<HistoryItem> = messages
        .iter()
        .map(|(message, usage)| {
            let role = match message.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::System,
            };
            let tokens = if matches!(role, Role::Assistant) { usage.map(|u| u.completion_tokens) } else { None };
            let cost = if matches!(role, Role::Assistant) {
                usage.and_then(|u| llm_core::pricing::resolve_output(&u))
            } else {
                None
            };
            let cost_approx = usage.map(|u| is_approx(&u)).unwrap_or(false);
            HistoryItem { role, text: message.content.clone(), debug: None, tokens, cost, cost_approx }
        })
        .collect();
    for (i, (_, usage)) in messages.iter().enumerate() {
        let Some(usage) = usage else { continue };
        if i == 0 {
            continue;
        }
        if let Role::User = items[i - 1].role {
            if items[i - 1].tokens.is_none() {
                items[i - 1].tokens = Some(usage.prompt_tokens);
            }
            if items[i - 1].cost.is_none() {
                items[i - 1].cost = llm_core::pricing::resolve_input(usage);
                items[i - 1].cost_approx = is_approx(usage);
            }
        }
    }
    items
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
    /// Модель памяти агента (см. llm_core::memory) — рабочая память текущей
    /// задачи и долговременная память, показанные и редактируемые отдельно от
    /// диалога (F3 из [`Screen::AgentChat`], Esc/F3 обратно). Команды в поле
    /// ввода этого экрана — `remember`/`forget`/`task ...` — синтаксически
    /// совпадают с подкомандами `llm-cli agent remember/forget/task`, чтобы
    /// поведение не расходилось между интерфейсами (см. [`run_memory_command`]).
    AgentMemory,
    /// MCP-серверы и их инструменты (F4 с любого экрана, F4/Esc — обратно).
    Mcp,
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
    ContextStrategy,
    Profile,
}

const CREATE_STEPS_TOTAL_QUICK: usize = 2;
const CREATE_STEPS_TOTAL_ADVANCED: usize = 10;

impl CreateStep {
    fn label(&self) -> String {
        match self {
            CreateStep::Name => " Имя агента ".to_string(),
            CreateStep::System => " Системный промпт (Enter — пропустить) ".to_string(),
            CreateStep::Model => " Модель (Enter — по умолчанию) ".to_string(),
            CreateStep::MaxTokens => " Макс. токенов ответа, число (Enter — без ограничения) ".to_string(),
            CreateStep::Temperature => " Temperature, число (Enter — по умолчанию) ".to_string(),
            CreateStep::TopP => " Top P, число (Enter — по умолчанию) ".to_string(),
            CreateStep::Reasoning => " Reasoning: on / off (Enter — по умолчанию) ".to_string(),
            CreateStep::ShowTokens => " Показывать токены в ответах? y/n ".to_string(),
            // Размер окна берётся из llm_core::context::sliding_window_size()
            // (LLM_SLIDING_WINDOW_SIZE), поэтому подпись собирается динамически.
            CreateStep::ContextStrategy => format!(
                " Стратегия контекста: full / summary / sliding-window / facts / branching \
                 (сводка — каждые {} сообщ.; окно/facts — последние {} сообщ.; Enter — full) ",
                llm_core::context::context_summary_chunk(),
                llm_core::context::sliding_window_size(),
            ),
            // Список доступных профилей собирается динамически из каталога
            // профилей (см. llm_core::profile), чтобы подсказка не расходилась
            // с тем, что реально можно выбрать.
            CreateStep::Profile => {
                let available = llm_core::list_profiles();
                let hint = if available.is_empty() {
                    "профилей в каталоге пока нет".to_string()
                } else {
                    available.join(", ")
                };
                format!(" Профиль персонализации: {hint} / none (Enter — default) ")
            }
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
            CreateStep::ContextStrategy => 9,
            CreateStep::Profile => 10,
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
            ShowTokens => Some(ContextStrategy),
            ContextStrategy => Some(Profile),
            Profile => None,
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
            CreateStep::ContextStrategy => {
                if value.is_empty() {
                    self.config.context_strategy = llm_core::ContextStrategy::default();
                } else {
                    match value.to_lowercase().parse() {
                        Ok(strategy) => self.config.context_strategy = strategy,
                        Err(_) => {
                            self.error = Some(
                                "Введите full, summary, sliding-window, facts, branching или оставьте пустым"
                                    .to_string(),
                            );
                            return false;
                        }
                    }
                }
            }
            CreateStep::Profile => {
                self.config.profile = if value.is_empty() { None } else { Some(value.to_string()) };
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
    /// `prompt` — то, что было отправлено (нужно, чтобы дописать его в chat_history
    /// вместе с ответом — см. обработчик события).
    DirectResponse { prompt: String, result: Result<ChatCompletion> },
    AgentResponse { name: String, result: Result<llm_core::AgentReply> },
    /// Фоновое действие с MCP-серверами (подключение, включение/выключение)
    /// завершилось — перерисовать экран; `message` — итог для строки статуса
    /// экрана MCP (текст, признак ошибки).
    McpUpdated { message: Option<(String, bool)> },
    /// Ручной вызов инструмента с экрана MCP завершился.
    McpCalled(McpCallView),
}

/// Ручной вызов инструмента с экрана MCP — показывается под этим инструментом.
#[derive(Clone)]
struct McpCallView {
    server: String,
    tool: String,
    arguments: String,
    /// `None`, пока вызов идёт.
    result: Option<String>,
    is_error: bool,
    elapsed_ms: u128,
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
    /// total_tokens самого последнего ответа открытого агента — как его вернула
    /// модель, без нашего суммирования (None — вне экрана чата с агентом, или
    /// агент ещё не отвечал).
    context_tokens: Option<u64>,
    /// Размер контекстного окна модели открытого агента (None — вне экрана чата с агентом).
    context_window: Option<u32>,
    /// Готовые фрагменты строки статуса активной стратегии управления
    /// контекстом открытого агента (см. [`agent_strategy_spans`]) — `None` вне
    /// экрана чата с агентом, при стратегии `full`, или если статус ещё не
    /// удалось определить.
    agent_strategy_badge: Option<Vec<Span<'static>>>,
    /// Стоимость всего диалога с открытым агентом (сумма, признак "это оценка,
    /// не реальная стоимость от провайдера" — см. llm_core::pricing) — `None`
    /// вне экрана чата с агентом или если стоимость взять неоткуда.
    agent_dialogue_cost: Option<(f64, bool)>,
    show_debug: bool,
    agents: &'a [AgentInfo],
    agents_selected: usize,
    confirm_delete: Option<&'a str>,
    wizard: Option<&'a CreateWizard>,
    agent_chat_name: Option<&'a str>,
    agent_chat_running: bool,
    /// Задача агента, чей диалог открыт, — читается заново на каждой
    /// перерисовке, поэтому смена этапа посреди ответа (move_stage) видна
    /// в шапке сразу, а не после ответа.
    agent_task: Option<llm_core::TaskState>,
    /// Число записей в истории диалога открытого агента (см. [`Agent::history`])
    /// — показывается в кратком виде на экране памяти ([`Screen::AgentMemory`])
    /// как обзор краткосрочной памяти; `None` вне экранов чата/памяти агента.
    agent_history_len: Option<usize>,
    /// Результат последней выполненной команды на экране памяти (текст, признак
    /// ошибки) — `None`, пока ни одна команда ещё не выполнялась в этой сессии
    /// экрана (тогда вместо него показывается подсказка по синтаксису команд).
    memory_status: Option<(&'a str, bool)>,
    /// Все существующие общие задачи (см. llm_core::memory) с их участниками —
    /// только на экране памяти агента, чтобы показать, к каким задачам можно
    /// присоединиться командой `task join`; `&[]` на остальных экранах.
    shared_tasks: &'a [llm_core::SharedTaskSummary],
    /// Снимок MCP-серверов (см. llm_core::mcp) — берётся на каждой перерисовке,
    /// поэтому смена статуса подключения видна сразу.
    mcp_servers: &'a [llm_core::McpServerInfo],
    mcp_selected: usize,
    /// Прокрутка списка инструментов выбранного сервера на экране MCP.
    mcp_scroll: u16,
    mcp_config_path: String,
    mcp_config_error: Option<String>,
    /// Итог последнего действия на экране MCP (текст, признак ошибки).
    mcp_status: Option<(&'a str, bool)>,
    /// Фокус на списке инструментов выбранного сервера (Tab), а не на серверах.
    mcp_focus_tools: bool,
    mcp_tool_selected: usize,
    /// Вводятся доводы для вызова выбранного инструмента (нижнее поле ввода).
    mcp_editing: bool,
    /// Последний ручной вызов инструмента (идущий или завершённый).
    mcp_call: Option<&'a McpCallView>,
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
        Screen::AgentChat | Screen::AgentMemory => INPUT_HEIGHT_AGENT_CHAT,
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
               F2 — управление агентами, F4 — MCP-серверы."
            .to_string(),
        debug: None,
        tokens: None,
        cost: None,
        cost_approx: false,
    }];
    let mut waiting = false;
    // Имя агента, чей ответ сейчас ожидается (None — если ждём прямой чат).
    // Позволяет показать индикатор на экране списка агентов даже когда его
    // диалог сейчас не открыт (пользователь вышел из него клавишей Esc/F2).
    let mut waiting_agent: Option<String> = None;
    let mut spinner_frame = 0usize;
    let mut stats = SessionStats::default();
    let mut last_usage: Option<Usage> = None;
    // Память прямого чата: сервер (веб) её ни в чём не хранит, история просто
    // пересылается целиком с каждым новым запросом — здесь то же самое, только
    // накопитель живёт в памяти процесса, а не в браузерной вкладке. Сбрасывается
    // явно по Ctrl+N (аналог "Новая сессия" в вебе), не переживает выход из TUI.
    let mut chat_history: Vec<ChatMessage> = Vec::new();
    // total_tokens самого последнего обмена прямого чата — как его вернула модель,
    // без нашего суммирования (см. context_bar в draw_chat).
    let mut chat_context_tokens: Option<u64> = None;
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
    // Реестр MCP-серверов общий с агентами: подключение идёт в фоне, чтобы
    // медленный сервер (npx скачивает пакет) не задерживал запуск TUI, — по
    // готовности приходит AppEvent::McpUpdated и экран перерисовывается.
    let mcp = agent_manager.mcp();
    let mut mcp_selected: usize = 0;
    let mut mcp_scroll: u16 = 0;
    let mut mcp_status: Option<(String, bool)> = None;
    // Экран, на который вернуть по F4/Esc с экрана MCP.
    let mut mcp_return = Screen::Chat;
    let mut mcp_focus_tools = false;
    let mut mcp_tool_selected: usize = 0;
    let mut mcp_editing = false;
    let mut mcp_call: Option<McpCallView> = None;
    let mut agents_selected: usize = 0;
    let mut confirm_delete: Option<String> = None;
    let mut wizard: Option<CreateWizard> = None;
    let mut agent_chat_name: Option<String> = None;
    let mut agent_histories: HashMap<String, Vec<HistoryItem>> = HashMap::new();
    // total_tokens самого последнего ответа каждого агента — как его вернула
    // модель, без нашего суммирования (см. context_bar в draw_agent_chat).
    // При первом открытии чата восстанавливается из истории в SQLite.
    let mut agent_context_tokens: HashMap<String, u64> = HashMap::new();
    // Результат последней команды на экране памяти агента (Screen::AgentMemory,
    // см. run_memory_command) — сбрасывается при входе на экран и при смене
    // открытого агента, чтобы не показывать результат команды для другого агента.
    let mut memory_status: Option<(String, bool)> = None;

    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    {
        let mcp = mcp.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            mcp.connect_all().await;
            let _ = tx.send(AppEvent::McpUpdated { message: None });
        });
    }
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
        let screen_context_tokens = match &screen {
            Screen::AgentChat => agent_chat_name.as_ref().and_then(|n| agent_context_tokens.get(n)).copied(),
            Screen::Chat => chat_context_tokens,
            _ => None,
        };
        // Размер контекстного окна не зависит от истории — у агента берём прямо из
        // его снимка (см. Agent::context_window), у прямого чата — тот же глобальный
        // LlmClient::context_window (у него нет своей модели/конфигурации, как у агента).
        let screen_context_window = match &screen {
            Screen::AgentChat => agent_chat_name
                .as_ref()
                .and_then(|n| agents_snapshot.iter().find(|a| &a.config.name == n))
                .map(|a| a.context_window),
            Screen::Chat => Some(client.context_window()),
            _ => None,
        };
        // Тот же снимок agents_snapshot, что и выше, — статус стратегии в нём уже
        // посчитан на момент вызова agent_manager.list() (см. Agent::info()), без
        // отдельного похода к агенту.
        let screen_agent_strategy_badge = match &screen {
            Screen::AgentChat => agent_chat_name
                .as_ref()
                .and_then(|n| agents_snapshot.iter().find(|a| &a.config.name == n))
                .and_then(agent_strategy_spans),
            _ => None,
        };
        // Стоимость всего диалога с открытым агентом (см. llm_core::pricing) —
        // сумма HistoryItem::cost по всей известной истории (восстановленной из
        // SQLite при открытии чата + отправленной в этой сессии TUI), а не
        // только новых сообщений: пользователь видит, сколько уже стоил диалог
        // целиком. `None`, если ни для одного сообщения стоимость не удалось
        // определить (ни от провайдера, ни оценкой) — тогда бейдж скрыт; второй
        // элемент пары — true, если хоть один вклад в сумму был оценкой, а не
        // реальной стоимостью от провайдера (тогда сумма помечается "≈").
        let screen_agent_dialogue_cost = match &screen {
            Screen::AgentChat => agent_chat_name.as_ref().and_then(|n| agent_histories.get(n)).and_then(|items| {
                let mut sum = 0.0;
                let mut any = false;
                let mut approx = false;
                for item in items {
                    if let Some(c) = item.cost {
                        sum += c;
                        any = true;
                        approx |= item.cost_approx;
                    }
                }
                any.then_some((sum, approx))
            }),
            _ => None,
        };
        let screen_agent_history_len = match &screen {
            Screen::AgentChat | Screen::AgentMemory => {
                agent_chat_name.as_ref().and_then(|n| agent_histories.get(n)).map(|v| v.len())
            }
            _ => None,
        };
        // Список общих задач (см. llm_core::memory) только на экране памяти —
        // нужен, чтобы показать, к каким задачам можно присоединиться (`task
        // join`); в остальных экранах не запрашивается, чтобы не дёргать БД
        // на каждой перерисовке зря.
        let screen_shared_tasks =
            if matches!(screen, Screen::AgentMemory) { agent_manager.list_tasks() } else { Vec::new() };
        let screen_agent_task = match &screen {
            Screen::AgentChat => {
                agent_chat_name.as_ref().and_then(|n| agent_manager.get(n)).and_then(|a| a.task_state())
            }
            _ => None,
        };

        let mcp_servers = mcp.servers();
        if mcp_selected >= mcp_servers.len() {
            mcp_selected = mcp_servers.len().saturating_sub(1);
        }
        let mcp_tools_count = mcp_servers.get(mcp_selected).map(|s| s.tools.len()).unwrap_or(0);
        if mcp_tool_selected >= mcp_tools_count {
            mcp_tool_selected = mcp_tools_count.saturating_sub(1);
        }
        if mcp_tools_count == 0 {
            mcp_focus_tools = false;
            mcp_editing = false;
        }
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
            last_usage,
            context_tokens: screen_context_tokens,
            context_window: screen_context_window,
            agent_strategy_badge: screen_agent_strategy_badge,
            agent_dialogue_cost: screen_agent_dialogue_cost,
            show_debug,
            agents: &agents_snapshot,
            agents_selected,
            confirm_delete: confirm_delete.as_deref(),
            wizard: wizard.as_ref(),
            agent_chat_name: agent_chat_name.as_deref(),
            agent_chat_running,
            agent_task: screen_agent_task,
            agent_history_len: screen_agent_history_len,
            memory_status: memory_status.as_ref().map(|(text, is_error)| (text.as_str(), *is_error)),
            shared_tasks: &screen_shared_tasks,
            mcp_servers: &mcp_servers,
            mcp_selected,
            mcp_scroll,
            mcp_config_path: mcp.config_path().display().to_string(),
            mcp_config_error: mcp.config_error(),
            mcp_status: mcp_status.as_ref().map(|(text, is_error)| (text.as_str(), *is_error)),
            mcp_focus_tools,
            mcp_tool_selected,
            mcp_editing,
            mcp_call: mcp_call.as_ref(),
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
                                Screen::AgentChat | Screen::AgentMemory => { agent_chat_name = None; Screen::AgentsList }
                                Screen::Mcp => { mcp_editing = false; Screen::AgentsList }
                            };
                            input.clear();
                            confirm_delete = None;
                            continue;
                        }

                        // F4 открывает экран MCP-серверов с любого экрана и возвращает
                        // обратно — как и F2, даже пока агент отвечает.
                        if key.code == KeyCode::F(4) {
                            if screen == Screen::Mcp {
                                screen = mcp_return;
                                if mcp_editing {
                                    mcp_editing = false;
                                    input.clear();
                                }
                            } else {
                                mcp_return = screen;
                                screen = Screen::Mcp;
                                mcp_status = None;
                                mcp_scroll = 0;
                            }
                            continue;
                        }

                        // F3 переключается между диалогом агента и его рабочей/долговременной
                        // памятью (см. Screen::AgentMemory) — доступно только пока чат с
                        // конкретным агентом открыт; в остальных экранах ничего не делает.
                        if key.code == KeyCode::F(3) {
                            screen = match screen {
                                Screen::AgentChat => Screen::AgentMemory,
                                Screen::AgentMemory => Screen::AgentChat,
                                other => other,
                            };
                            input.clear();
                            memory_status = None;
                            continue;
                        }

                        match screen {
                            Screen::Chat => match key.code {
                                KeyCode::Tab => show_debug = !show_debug,
                                KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    show_debug = !show_debug;
                                }
                                // Ctrl+N вместо простого 'n' — поле ввода тут текстовое, обычная
                                // 'n' должна просто печататься. Аналог кнопки "Новая сессия" в вебе:
                                // очищает и видимую историю, и память, которая уходит в запрос.
                                KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    history.clear();
                                    history.push(HistoryItem {
                                        role: Role::System,
                                        text: "Новая сессия — история сброшена.".to_string(),
                                        debug: None,
                                        tokens: None,
                                        cost: None,
                                        cost_approx: false,
                                    });
                                    chat_history.clear();
                                    chat_context_tokens = None;
                                    stats = SessionStats::default();
                                    last_usage = None;
                                    follow_bottom = true;
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
                                        tokens: None,
                                        cost: None,
                                        cost_approx: false,
                                    });
                                    waiting = true;

                                    let client = client.clone();
                                    let tx = tx.clone();
                                    // Пересылаем всю накопленную историю + новое сообщение — модель
                                    // должна видеть предыдущие реплики (см. chat_history выше).
                                    let mut messages = chat_history.clone();
                                    messages.push(ChatMessage::user(prompt.clone()));
                                    tokio::spawn(async move {
                                        let response = client.chat(&messages).await;
                                        let _ = tx.send(AppEvent::DirectResponse { prompt, result: response });
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
                                        agent_context_tokens.remove(&pending);
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
                                                        .map(|agent| history_items_from_messages(&agent.history_with_usage()))
                                                        .unwrap_or_default()
                                                });
                                                // total_tokens последнего обмена из истории — то же число,
                                                // что вернула модель, без нашего суммирования.
                                                agent_context_tokens.entry(name.clone()).or_insert_with(|| {
                                                    agent_manager
                                                        .get(&name)
                                                        .and_then(|agent| {
                                                            agent.history_with_usage().into_iter().rev().find_map(
                                                                |(_, usage)| usage.map(|u| u.total_tokens as u64),
                                                            )
                                                        })
                                                        .unwrap_or(0)
                                                });
                                                agent_chat_name = Some(name);
                                                screen = Screen::AgentChat;
                                                follow_bottom = true;
                                                input.clear();
                                                memory_status = None;
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
                                // Ctrl+S вместо простого 's' — здесь поле ввода текстовое (не
                                // список команд, как в Screen::AgentsList), поэтому обычная 's'
                                // должна просто печататься в сообщение.
                                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    if let Some(name) = agent_chat_name.clone() {
                                        if let Some(agent) = agent_manager.get(&name) {
                                            let _ = if agent.is_running() {
                                                agent_manager.stop(&name)
                                            } else {
                                                agent_manager.start(&name)
                                            };
                                        }
                                    }
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
                                    if let Some(name) = agent_chat_name.clone() {
                                        if let Some(agent) = agent_manager.get(&name) {
                                            input.clear();
                                            waiting = true;
                                            waiting_agent = Some(name.clone());
                                            send_to_agent(agent, name, prompt, true, &mut agent_histories, &tx);
                                        }
                                    }
                                }
                                KeyCode::Char(c) if !waiting => input.push(c),
                                KeyCode::Backspace if !waiting => {
                                    input.pop();
                                }
                                _ => {}
                            },
                            // Экран памяти агента (см. Screen::AgentMemory) — команды
                            // remember/forget/task выполняются сразу, синхронно (никакого
                            // обращения к LLM здесь нет), поэтому `waiting` не проверяется:
                            // это не мешает и не мешается запросу к самой LLM, если он
                            // сейчас выполняется в фоне для этого же или другого агента.
                            Screen::AgentMemory => match key.code {
                                KeyCode::Esc => {
                                    screen = Screen::AgentChat;
                                    input.clear();
                                }
                                KeyCode::Enter => {
                                    let raw = input.trim().to_string();
                                    input.clear();
                                    if raw.is_empty() {
                                        continue;
                                    }
                                    if let Some(name) = agent_chat_name.clone() {
                                        if let Some(agent) = agent_manager.get(&name) {
                                            // task reject/resume/approve — не просто команды памяти: после
                                            // них агент сразу продолжает работу (перерабатывает отклонённый
                                            // этап, продолжает после паузы, начинает утверждённый этап),
                                            // поэтому переключаемся в диалог, где виден его ответ. Причина
                                            // отклонения видна в чате как реплика человека; продолжения после
                                            // resume/approve служебные — в чате их нет.
                                            let tokens: Vec<&str> = raw.split_whitespace().collect();
                                            let continuation = match tokens.as_slice() {
                                                ["task", "reject", note @ ..] => Some(
                                                    agent.task_reject(&note.join(" ")).map(|f| Some((f, true))),
                                                ),
                                                ["task", "resume"] => {
                                                    Some(agent.task_resume().map(|f| f.map(|f| (f, false))))
                                                }
                                                ["task", "approve"] => {
                                                    Some(agent.task_approve().map(|f| Some((f, false))))
                                                }
                                                _ => None,
                                            };
                                            if let Some(result) = continuation {
                                                match result {
                                                    Ok(Some((followup, from_human))) if !waiting => {
                                                        memory_status = None;
                                                        screen = Screen::AgentChat;
                                                        follow_bottom = true;
                                                        waiting = true;
                                                        waiting_agent = Some(name.clone());
                                                        send_to_agent(
                                                            agent,
                                                            name,
                                                            followup,
                                                            from_human,
                                                            &mut agent_histories,
                                                            &tx,
                                                        );
                                                    }
                                                    Ok(Some(_)) => {
                                                        memory_status = Some((
                                                            "Готово. Агент сейчас отвечает на другой запрос — всё нужное \
                                                             уже в контексте задачи, он учтёт это в следующем ответе."
                                                                .to_string(),
                                                            false,
                                                        ));
                                                    }
                                                    Ok(None) => {
                                                        memory_status = Some((
                                                            "Готово. Агенту сейчас нечего продолжать самому: задача не была \
                                                             на паузе, завершена или ждёт вашего решения по переходу."
                                                                .to_string(),
                                                            false,
                                                        ));
                                                    }
                                                    Err(err) => memory_status = Some((err.to_string(), true)),
                                                }
                                                continue;
                                            }
                                            memory_status = Some(match run_memory_command(&agent, &raw) {
                                                Ok(text) => (text, false),
                                                Err(text) => (text, true),
                                            });
                                        }
                                    }
                                }
                                KeyCode::Char(c) => input.push(c),
                                KeyCode::Backspace => {
                                    input.pop();
                                }
                                _ => {}
                            },
                            // Ввод доводов для вызова инструмента: все клавиши идут в поле
                            // ввода (иначе Пробел/e/r/l переключали бы серверы).
                            Screen::Mcp if mcp_editing => match key.code {
                                KeyCode::Esc => {
                                    mcp_editing = false;
                                    input.clear();
                                }
                                KeyCode::Enter => {
                                    let tool = mcp_servers
                                        .get(mcp_selected)
                                        .and_then(|s| s.tools.get(mcp_tool_selected).map(|t| (s.name.clone(), t.name.clone())));
                                    let running = mcp_call.as_ref().is_some_and(|c| c.result.is_none());
                                    if let (Some((server, tool)), false) = (tool, running) {
                                        let arguments = input.trim().to_string();
                                        mcp_editing = false;
                                        input.clear();
                                        mcp_call = Some(McpCallView {
                                            server: server.clone(),
                                            tool: tool.clone(),
                                            arguments: arguments.clone(),
                                            result: None,
                                            is_error: false,
                                            elapsed_ms: 0,
                                        });
                                        let (mcp, tx) = (mcp.clone(), tx.clone());
                                        tokio::spawn(async move {
                                            let started = std::time::Instant::now();
                                            let (result, is_error) = match mcp.call_tool(&server, &tool, &arguments).await {
                                                Some(r) => (r.text, r.is_error),
                                                None => (format!("у сервера «{server}» нет инструмента «{tool}» (или сервер не подключён)"), true),
                                            };
                                            let _ = tx.send(AppEvent::McpCalled(McpCallView {
                                                server,
                                                tool,
                                                arguments,
                                                result: Some(result),
                                                is_error,
                                                elapsed_ms: started.elapsed().as_millis(),
                                            }));
                                        });
                                    }
                                }
                                KeyCode::Char(c) => input.push(c),
                                KeyCode::Backspace => {
                                    input.pop();
                                }
                                _ => {}
                            },
                            Screen::Mcp if mcp_focus_tools => match key.code {
                                KeyCode::Esc | KeyCode::Tab | KeyCode::BackTab => mcp_focus_tools = false,
                                KeyCode::Up => mcp_tool_selected = mcp_tool_selected.saturating_sub(1),
                                KeyCode::Down => {
                                    if mcp_tool_selected + 1 < mcp_tools_count {
                                        mcp_tool_selected += 1;
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some(tool) = mcp_servers.get(mcp_selected).and_then(|s| s.tools.get(mcp_tool_selected)) {
                                        // Повторный вызов того же инструмента — с прежними доводами.
                                        input = match &mcp_call {
                                            Some(c) if c.tool == tool.name && mcp_servers[mcp_selected].name == c.server => c.arguments.clone(),
                                            _ => tool.arguments_template().to_string(),
                                        };
                                        mcp_editing = true;
                                    }
                                }
                                _ => {}
                            },
                            Screen::Mcp => match key.code {
                                KeyCode::Esc => screen = mcp_return,
                                KeyCode::Tab => {
                                    if mcp_tools_count > 0 {
                                        mcp_focus_tools = true;
                                    }
                                }
                                KeyCode::Up => {
                                    mcp_selected = mcp_selected.saturating_sub(1);
                                    mcp_scroll = 0;
                                    mcp_tool_selected = 0;
                                }
                                KeyCode::Down => {
                                    if mcp_selected + 1 < mcp_servers.len() {
                                        mcp_selected += 1;
                                        mcp_scroll = 0;
                                        mcp_tool_selected = 0;
                                    }
                                }
                                KeyCode::PageUp => mcp_scroll = mcp_scroll.saturating_sub(PAGE_STEP),
                                KeyCode::PageDown => mcp_scroll = mcp_scroll.saturating_add(PAGE_STEP),
                                KeyCode::Char(' ') | KeyCode::Char('e') => {
                                    if let Some(server) = mcp_servers.get(mcp_selected) {
                                        let (name, enable) = (server.name.clone(), !server.enabled);
                                        mcp_status = Some((
                                            if enable { format!("Включаю «{name}»…") } else { format!("Выключаю «{name}»…") },
                                            false,
                                        ));
                                        let (mcp, tx) = (mcp.clone(), tx.clone());
                                        tokio::spawn(async move {
                                            let message = match mcp.set_enabled(&name, enable).await {
                                                Ok(()) if enable => (format!("«{name}» включён."), false),
                                                Ok(()) => (format!("«{name}» выключен — остаётся в конфигурации."), false),
                                                Err(err) => (format!("{err:#}"), true),
                                            };
                                            let _ = tx.send(AppEvent::McpUpdated { message: Some(message) });
                                        });
                                    }
                                }
                                KeyCode::Char('r') => {
                                    if let Some(server) = mcp_servers.get(mcp_selected).filter(|s| s.enabled) {
                                        let name = server.name.clone();
                                        mcp_status = Some((format!("Переподключаю «{name}»…"), false));
                                        let (mcp, tx) = (mcp.clone(), tx.clone());
                                        tokio::spawn(async move {
                                            mcp.connect(&name).await;
                                            let _ = tx.send(AppEvent::McpUpdated { message: None });
                                        });
                                    }
                                }
                                KeyCode::Char('l') => {
                                    mcp_status = Some(("Перечитываю конфигурацию…".to_string(), false));
                                    let (mcp, tx) = (mcp.clone(), tx.clone());
                                    tokio::spawn(async move {
                                        mcp.reload().await;
                                        let message = ("Конфигурация перечитана.".to_string(), false);
                                        let _ = tx.send(AppEvent::McpUpdated { message: Some(message) });
                                    });
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
                        Screen::AgentMemory => input.push_str(&text),
                        Screen::Mcp if mcp_editing => input.push_str(&text),
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
                    // Не ответ на запрос — флаги ожидания ниже не трогаем.
                    AppEvent::McpUpdated { message } => {
                        if message.is_some() {
                            mcp_status = message;
                        }
                        continue;
                    }
                    AppEvent::McpCalled(call) => {
                        mcp_call = Some(call);
                        continue;
                    }
                    AppEvent::DirectResponse { prompt, result: Ok(completion) } => {
                        let response_tokens = completion.usage.map(|u| u.completion_tokens);
                        if let Some(usage) = completion.usage {
                            stats.requests += 1;
                            stats.tokens += usage.total_tokens as u64;
                            last_usage = Some(usage);
                            chat_context_tokens = Some(usage.total_tokens as u64);
                            if let Some(last) = history.last_mut() {
                                if matches!(last.role, Role::User) && last.tokens.is_none() {
                                    last.tokens = Some(usage.prompt_tokens);
                                }
                            }
                        }
                        // В память кладём независимо от того, пришёл ли usage, — модель должна
                        // помнить этот обмен в любом случае.
                        chat_history.push(ChatMessage::user(prompt));
                        chat_history.push(ChatMessage::assistant(completion.content.clone()));
                        history.push(HistoryItem {
                            role: Role::Assistant,
                            text: completion.content,
                            debug: Some((completion.request_json, completion.response_json)),
                            tokens: response_tokens,
                            cost: None,
                            cost_approx: false,
                        });
                    }
                    AppEvent::DirectResponse { result: Err(err), .. } => {
                        history.push(HistoryItem {
                            role: Role::Error,
                            text: format!("{err:#}"),
                            debug: None,
                            tokens: None,
                            cost: None,
                            cost_approx: false,
                        });
                    }
                    AppEvent::AgentResponse { name, result } => {
                        let entry = agent_histories.entry(name.clone()).or_default();
                        match result {
                            // Пауза, поставленная во время ответа, оборвала его — вместо
                            // ответа пометка; запрос уйдёт заново после task resume.
                            Ok(reply) if reply.interrupted => {
                                entry.push(HistoryItem {
                                    role: Role::System,
                                    text: reply.text,
                                    debug: None,
                                    tokens: None,
                                    cost: None,
                                    cost_approx: false,
                                });
                            }
                            Ok(reply) => {
                                let response_tokens = reply.usage.map(|u| u.completion_tokens);
                                let response_cost = reply.cost.map(|c| c.output);
                                let cost_approx =
                                    reply.cost.map(|c| c.source == llm_core::pricing::CostSource::Estimated);
                                if let Some(usage) = reply.usage {
                                    if let Some(last) = entry.last_mut() {
                                        if matches!(last.role, Role::User) && last.tokens.is_none() {
                                            last.tokens = Some(usage.prompt_tokens);
                                        }
                                    }
                                }
                                if let Some(cost) = reply.cost {
                                    if let Some(last) = entry.last_mut() {
                                        if matches!(last.role, Role::User) && last.cost.is_none() {
                                            last.cost = Some(cost.input);
                                            last.cost_approx = cost.source == llm_core::pricing::CostSource::Estimated;
                                        }
                                    }
                                }
                                if let Some(usage) = reply.usage {
                                    agent_context_tokens.insert(name, usage.total_tokens as u64);
                                }
                                let summarized = reply.summarized;
                                let summary_covers = reply.summary_covers;
                                let facts_updated = reply.facts_updated;
                                // Вызовы инструментов — отдельными строками перед ответом,
                                // в том порядке, в каком их делала модель.
                                for call in &reply.tool_calls {
                                    entry.push(HistoryItem {
                                        role: if call.is_error { Role::ToolError } else { Role::Tool },
                                        text: format!("{}\n→ {}", call.headline(160), call.result_preview(300)),
                                        debug: None,
                                        tokens: None,
                                        cost: None,
                                        cost_approx: false,
                                    });
                                }
                                entry.push(HistoryItem {
                                    role: Role::Assistant,
                                    text: reply.text,
                                    // Tab (см. show_debug) показывает/скрывает это так же,
                                    // как и в прямом чате — раньше у ответов агента debug
                                    // всегда был пуст, потому что AgentReply не нёс JSON.
                                    debug: Some((reply.request_json, reply.response_json)),
                                    tokens: response_tokens,
                                    cost: response_cost,
                                    cost_approx: cost_approx.unwrap_or(false),
                                });
                                // Момент пересчёта сводки сжатого контекста (стратегия
                                // summary) — показываем его отдельной строкой сразу после
                                // ответа, в рамках которого он произошёл.
                                if summarized {
                                    entry.push(HistoryItem {
                                        role: Role::Compression,
                                        text: format!(
                                            "Контекст сжат в сводку — она теперь охватывает {} более ранних сообщений.",
                                            summary_covers.unwrap_or(0)
                                        ),
                                        debug: None,
                                        tokens: None,
                                        cost: None,
                                        cost_approx: false,
                                    });
                                }
                                // Момент обновления фактов (стратегия facts) — аналогичная метка.
                                if facts_updated {
                                    entry.push(HistoryItem {
                                        role: Role::Compression,
                                        text: "📌 Факты обновлены.".to_string(),
                                        debug: None,
                                        tokens: None,
                                        cost: None,
                                        cost_approx: false,
                                    });
                                }
                            }
                            Err(err) => {
                                entry.push(HistoryItem {
                                    role: Role::Error,
                                    text: format!("{err:#}"),
                                    debug: None,
                                    tokens: None,
                                    cost: None,
                                    cost_approx: false,
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
        Screen::AgentMemory => draw_agent_memory(frame, state),
        Screen::Mcp => draw_mcp(frame, state),
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
        context_tokens,
        context_window,
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
        Span::raw("  ·  F2 агенты  ·  Ctrl+N новая сессия"),
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

    let mut stats_spans = vec![Span::raw(match last_usage {
        Some(usage) => format!(
            " Последний запрос: {} + {} = {} ток.  ·  За сессию: {} запрос(ов), {} ток.",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, stats.requests, stats.tokens
        ),
        None => " Расход токенов появится после первого ответа".to_string(),
    })];
    // Заполнение контекста — total_tokens САМОГО ПОСЛЕДНЕГО обмена (как его вернула
    // модель, без нашего суммирования), а не сумма по сессии — та же логика, что и
    // в чате агента (см. context_bar) и в вебе.
    if let Some(window) = context_window {
        let used = context_tokens.unwrap_or(0);
        let percent = if window == 0 { 0 } else { ((used as f64 / window as f64) * 100.0).min(100.0).round() as u32 };
        stats_spans.push(Span::raw("  ·  Контекст: "));
        stats_spans.push(Span::styled(context_bar(used, window, 20), Style::default().fg(context_bar_color(percent))));
        stats_spans.push(Span::raw(format!(" {percent}% ({}/{})", format_tokens(used), format_tokens(window as u64))));
    }
    let stats_para = Paragraph::new(Line::from(stats_spans)).style(Style::default().fg(Color::DarkGray));
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
    parts.push(format!("стратегия контекста: {}", config.context_strategy));
    parts.push(format!("профиль: {}", llm_core::profile::resolve_name(config.profile.as_deref())));
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
        Span::raw("  ·  Агенты  ·  F2/Esc — назад к чату  ·  F4 — MCP"),
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
/// Бейдж MCP для строки статуса чата с агентом: сколько инструментов
/// подключённых серверов агент сейчас предлагает модели. Пусто, если в
/// конфигурации нет включённых серверов.
fn mcp_badge_spans(servers: &[llm_core::McpServerInfo]) -> Vec<Span<'static>> {
    let enabled: Vec<_> = servers.iter().filter(|s| s.enabled).collect();
    if enabled.is_empty() {
        return Vec::new();
    }
    let connected = enabled.iter().filter(|s| s.status == llm_core::McpStatus::Connected).count();
    let tools: usize = enabled.iter().map(|s| s.tools.len()).sum();
    let color = if connected == enabled.len() { Color::Magenta } else { Color::Yellow };
    vec![
        Span::raw("  ·  🔌 "),
        Span::styled(format!("MCP {connected}/{} · {tools} инстр.", enabled.len()), Style::default().fg(color)),
    ]
}

fn mcp_status_color(status: &llm_core::McpStatus) -> Color {
    match status {
        llm_core::McpStatus::Connected => Color::Green,
        llm_core::McpStatus::Connecting | llm_core::McpStatus::Idle => Color::Yellow,
        llm_core::McpStatus::Failed { .. } => Color::Red,
        llm_core::McpStatus::Disabled => Color::DarkGray,
    }
}

/// Экран MCP-серверов (F4): слева список серверов со статусами, справа —
/// выбранный сервер и его инструменты (описание + параметры).
fn draw_mcp(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks(area);

    let header = Paragraph::new(Line::from(vec![
        Span::styled("✦ Challenger", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  MCP-серверы  ·  F4/Esc — назад"),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
    frame.render_widget(header, chunks[0]);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(chunks[1]);

    let mut list_lines: Vec<Line<'static>> = Vec::new();
    if state.mcp_servers.is_empty() {
        list_lines.push(Line::from(Span::styled(
            "Серверов нет — добавьте их в файл конфигурации (см. mcp.example.json).",
            Style::default().fg(Color::DarkGray),
        )));
    }
    for (i, server) in state.mcp_servers.iter().enumerate() {
        let selected = i == state.mcp_selected;
        let marker = if selected { "▸ " } else { "  " };
        let name_style = if selected {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let mut spans = vec![
            Span::raw(marker),
            Span::styled(server.name.clone(), name_style),
            Span::raw("  "),
            Span::styled(server.status.label(), Style::default().fg(mcp_status_color(&server.status))),
        ];
        if server.status == llm_core::McpStatus::Connected {
            spans.push(Span::styled(
                format!(" · {} инстр.", server.tools.len()),
                Style::default().fg(Color::DarkGray),
            ));
        }
        list_lines.push(Line::from(spans));
    }
    let list = Paragraph::new(list_lines)
        .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Серверы "));
    frame.render_widget(list, columns[0]);

    let detail_width = columns[1].width.saturating_sub(4).max(10) as usize;
    let mut detail: Vec<Line<'static>> = Vec::new();
    // Строка выбранного инструмента — к ней прокручивается список, пока фокус на инструментах.
    let mut selected_line: Option<usize> = None;
    let dim = Style::default().fg(Color::DarkGray);
    let wrapped = |text: &str, indent: usize, style: Style| -> Vec<Line<'static>> {
        textwrap::wrap(text, detail_width.saturating_sub(indent).max(10))
            .into_iter()
            .map(|part| Line::from(vec![Span::raw(" ".repeat(indent)), Span::styled(part.to_string(), style)]))
            .collect()
    };
    if let Some(server) = state.mcp_servers.get(state.mcp_selected) {
        detail.push(Line::from(vec![
            Span::styled(server.name.clone(), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(server.status.label(), Style::default().fg(mcp_status_color(&server.status))),
            Span::styled(server.server_version.as_ref().map(|v| format!("  ·  {v}")).unwrap_or_default(), dim),
        ]));
        detail.extend(wrapped(&server.transport, 0, dim));
        if let Some(description) = &server.description {
            detail.extend(wrapped(description, 0, Style::default()));
        }
        if let llm_core::McpStatus::Failed { error } = &server.status {
            for line in error.lines() {
                detail.extend(wrapped(line, 0, Style::default().fg(Color::Red)));
            }
        }
        if !server.tools.is_empty() {
            detail.push(Line::from(""));
            detail.push(Line::from(Span::styled(
                format!("Инструменты ({}):", server.tools.len()),
                Style::default().add_modifier(Modifier::BOLD),
            )));
        }
        for (i, tool) in server.tools.iter().enumerate() {
            let selected = state.mcp_focus_tools && i == state.mcp_tool_selected;
            detail.push(Line::from(""));
            if selected {
                selected_line = Some(detail.len());
            }
            let name_style = if selected {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Magenta)
            };
            detail.push(Line::from(vec![
                Span::styled(format!("{} {}", if selected { "▸" } else { "•" }, tool.name), name_style),
                Span::styled(tool.title.as_ref().map(|t| format!("  ({t})")).unwrap_or_default(), dim),
            ]));
            if let Some(description) = &tool.description {
                for line in description.lines().filter(|l| !l.trim().is_empty()) {
                    detail.extend(wrapped(line.trim(), 2, Style::default()));
                }
            }
            detail.extend(wrapped(&format!("параметры: {}", tool.params_summary()), 2, dim));
            // Результат ручного вызова — прямо под тем инструментом, который вызывали.
            if let Some(call) = state.mcp_call.filter(|c| c.server == server.name && c.tool == tool.name) {
                detail.extend(wrapped(&format!("вызов: {}", call.arguments), 2, Style::default().fg(Color::Cyan)));
                match &call.result {
                    None => detail.extend(wrapped("⏳ выполняется…", 2, Style::default().fg(Color::Yellow))),
                    Some(result) => {
                        let (label, color) = if call.is_error { ("⚠ ошибка", Color::Red) } else { ("→ результат", Color::Green) };
                        detail.extend(wrapped(&format!("{label} · {} мс:", call.elapsed_ms), 2, Style::default().fg(color)));
                        let lines: Vec<&str> = result.lines().collect();
                        for line in lines.iter().take(MCP_RESULT_MAX_LINES) {
                            detail.extend(wrapped(line, 4, Style::default()));
                        }
                        if lines.len() > MCP_RESULT_MAX_LINES {
                            detail.extend(wrapped(
                                &format!("… ещё {} строк", lines.len() - MCP_RESULT_MAX_LINES),
                                4,
                                dim,
                            ));
                        }
                    }
                }
            }
        }
    }
    let max_scroll = (detail.len() as u16).saturating_sub(columns[1].height.saturating_sub(2));
    let scroll = match selected_line {
        Some(line) => (line as u16).saturating_sub(1),
        None => state.mcp_scroll,
    };
    let (detail_title, detail_border) = if state.mcp_focus_tools {
        (" Инструменты — ↑/↓ выбор, Enter вызвать, Tab/Esc к серверам ", Color::Cyan)
    } else {
        (" Сервер и инструменты — Tab к инструментам, PageUp/PageDown прокрутка ", Color::Reset)
    };
    let details = Paragraph::new(detail)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(detail_border))
                .title(detail_title),
        )
        .scroll((scroll.min(max_scroll), 0));
    frame.render_widget(details, columns[1]);

    let (status_line, status_color) = match state.mcp_status {
        Some((text, true)) => (format!(" {text}"), Color::Red),
        Some((text, false)) => (format!(" {text}"), Color::Yellow),
        None if state.mcp_focus_tools => (
            " ↑/↓ инструмент · Enter ввести доводы и вызвать · Tab/Esc к серверам · F4 назад ".to_string(),
            Color::DarkGray,
        ),
        None => (
            " ↑/↓ выбор · Tab инструменты · Пробел/e вкл/выкл · r переподключить · l перечитать файл · F4/Esc назад "
                .to_string(),
            Color::DarkGray,
        ),
    };
    frame.render_widget(Paragraph::new(status_line).style(Style::default().fg(status_color)), chunks[2]);

    let footer = match &state.mcp_config_error {
        Some(err) => Paragraph::new(err.clone()).style(Style::default().fg(Color::Red)),
        None => Paragraph::new(format!(
            "{} — выключенный сервер (\"enabled\": false) остаётся в файле, но не подключается",
            state.mcp_config_path
        ))
        .style(Style::default().fg(Color::DarkGray)),
    };
    if state.mcp_editing {
        let tool = state
            .mcp_servers
            .get(state.mcp_selected)
            .and_then(|s| s.tools.get(state.mcp_tool_selected))
            .map(|t| t.name.as_str())
            .unwrap_or("?");
        let title = format!(" Доводы {tool} (JSON-объект) — Enter вызвать, Esc отмена ");
        render_input_box(frame, chunks[3], state.input, title, Color::Cyan);
    } else {
        frame.render_widget(
            footer.block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Конфигурация ")),
            chunks[3],
        );
    }
}

/// Сколько строк результата ручного вызова показывать под инструментом.
const MCP_RESULT_MAX_LINES: usize = 30;

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
        lines.push(summary_line("Стратегия контекста", &wizard.config.context_strategy.to_string()));
        lines.push(summary_line(
            "Профиль",
            &llm_core::profile::resolve_name(wizard.config.profile.as_deref()),
        ));
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

    render_input_box(frame, chunks[3], state.input, wizard.step.label(), Color::Cyan);
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
            Span::raw(
                "  —  Расширенно: модель, лимиты токенов, temperature, top_p, reasoning, \
                 показ токенов, сжатие контекста",
            ),
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

/// Сокращённая запись количества токенов (как formatTokenCount в веб-интерфейсе):
/// 96000 → "96K", 1234567 → "1.2M" — иначе большие числа неудобно читать в узкой
/// строке статуса терминала.
fn format_tokens(n: u64) -> String {
    fn trimmed(v: f64) -> String {
        let s = format!("{v:.1}");
        s.trim_end_matches(".0").to_string()
    }
    if n >= 1_000_000 {
        format!("{}M", trimmed(n as f64 / 1_000_000.0))
    } else if n >= 1000 {
        format!("{}K", trimmed(n as f64 / 1000.0))
    } else {
        n.to_string()
    }
}

/// Текстовая полоса заполнения контекстного окна — терминал не рисует круги, как
/// веб-интерфейс, поэтому здесь тот же смысл передаёт горизонтальная полоса из
/// символов-блоков.
fn context_bar(used: u64, window: u32, width: usize) -> String {
    let ratio = if window == 0 { 0.0 } else { (used as f64 / window as f64).min(1.0) };
    let filled = ((ratio * width as f64).round() as usize).min(width);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(width - filled))
}

/// Зелёный → жёлтый → красный по мере приближения к лимиту контекста — те же
/// пороги, что и у кольца в веб-интерфейсе (см. contextRingColor в index.html).
fn context_bar_color(percent: u32) -> Color {
    if percent >= 90 {
        Color::Red
    } else if percent >= 70 {
        Color::Yellow
    } else {
        Color::Green
    }
}

/// Строит фрагменты строки статуса активной стратегии управления контекстом
/// агента — рядом с индикатором заполнения контекста в шапке чата (см.
/// `DrawState::agent_strategy_badge`). `None`, если активна стратегия `full`
/// (управлять нечем).
fn agent_strategy_spans(info: &AgentInfo) -> Option<Vec<Span<'static>>> {
    if let Some(c) = info.compression {
        return Some(vec![
            Span::raw("  ·  🗜 ещё "),
            Span::styled(c.messages_until_summary.to_string(), Style::default().fg(Color::Yellow)),
            Span::raw(" сообщ. до сжатия"),
        ]);
    }
    if let Some(f) = &info.facts {
        return Some(vec![
            Span::raw("  ·  📌 facts: "),
            Span::styled(f.facts.len().to_string(), Style::default().fg(Color::Yellow)),
            Span::raw(format!(" (окно {})", f.window_size)),
        ]);
    }
    if let Some(w) = info.sliding_window {
        return Some(vec![
            Span::raw("  ·  🪟 окно: "),
            Span::styled(format!("{}/{}", w.kept_messages, w.total_messages), Style::default().fg(Color::Yellow)),
            Span::raw(" сообщ."),
        ]);
    }
    if let Some(b) = &info.branching {
        return Some(vec![
            Span::raw("  ·  🌿 ветка: "),
            Span::styled(b.current_branch.clone(), Style::default().fg(Color::Yellow)),
        ]);
    }
    None
}

/// Этап задачи для шапки диалога агента: бейдж этапа, пометки паузы и
/// ожидания утверждения, «работает…» пока идёт ответ; без задачи — подсказка,
/// как её завести.
fn task_stage_spans(task: Option<&llm_core::TaskState>, working: bool) -> Vec<Span<'static>> {
    let Some(task) = task else {
        return vec![Span::styled("  ·  задачи нет (F3: task start)", Style::default().fg(Color::DarkGray))];
    };
    let stage_style = if task.paused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
    };
    let mut spans =
        vec![Span::raw("  ·  этап: "), Span::styled(task.stage.as_str().to_uppercase(), stage_style)];
    // Пока агент отвечает, «ждёт утверждения» не показываем: модель
    // предлагает переход раньше, чем дописывает сам ответ (план, итог), —
    // просьба утвердить появлялась бы раньше того, что утверждать.
    if task.paused {
        spans.push(Span::styled(" ⏸ пауза", Style::default().fg(Color::Yellow)));
    } else if working {
        spans.push(Span::styled(" — агент работает…", Style::default().fg(Color::DarkGray)));
    } else if let Some(pending) = task.pending_stage {
        spans.push(Span::styled(
            format!(" ⏳ ждёт утверждения → {pending}"),
            Style::default().fg(Color::Yellow),
        ));
    }
    spans
}

fn draw_agent_chat(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks_with_input_height(area, INPUT_HEIGHT_AGENT_CHAT);
    let name = state.agent_chat_name.unwrap_or("?");

    let this_agent_waiting = state.waiting && state.waiting_agent == Some(name);
    let mut header_spans = vec![
        Span::styled(
            "✦ Challenger",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  агент: "),
        Span::styled(name, Style::default().fg(Color::Cyan)),
        Span::raw(if state.agent_chat_running { "  ·  запущен" } else { "  ·  остановлен" }),
    ];
    header_spans.extend(task_stage_spans(state.agent_task.as_ref(), this_agent_waiting));
    header_spans.push(Span::raw("  ·  Ctrl+S старт/стоп  ·  F3 память  ·  F4 MCP"));
    let header = Paragraph::new(Line::from(header_spans))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    frame.render_widget(header, chunks[0]);

    let chat_title = if this_agent_waiting {
        format!(" Диалог {} ", SPINNER_FRAMES[state.spinner_frame])
    } else {
        " Диалог — PageUp/PageDown скролл, End в конец, Tab JSON запроса/ответа, Esc назад к списку агентов "
            .to_string()
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

    // Заполнение контекстного окна — числитель это total_tokens САМОГО ПОСЛЕДНЕГО
    // ответа (как его вернула модель, без нашего суммирования — так же теперь
    // делает и веб-интерфейс, см. agentChatStats.usedTokens в index.html),
    // знаменатель — размер окна модели агента. Токены отдельного запроса/ответа
    // показываются под каждым сообщением в самом диалоге (см. history_item_to_lines).
    let context_line = match state.context_window {
        Some(window) => {
            let used = state.context_tokens.unwrap_or(0);
            let percent = if window == 0 {
                0
            } else {
                ((used as f64 / window as f64) * 100.0).min(100.0).round() as u32
            };
            let mut spans = vec![
                Span::raw(" Контекст: "),
                Span::styled(context_bar(used, window, 24), Style::default().fg(context_bar_color(percent))),
                Span::raw(format!(" {percent}% ({}/{})", format_tokens(used), format_tokens(window as u64))),
            ];
            // Наглядно показывает статус активной стратегии управления контекстом
            // (сколько сообщений осталось до сжатия, окно, факты или ветка) — прямо
            // рядом с заполнением контекста (см. agent_strategy_spans).
            if let Some(badge) = &state.agent_strategy_badge {
                spans.extend(badge.iter().cloned());
            }
            // Стоимость всего диалога (см. llm_core::pricing) — скрыта, если стоимость
            // взять неоткуда (ни от провайдера, ни оценкой по LLM_PRICE_*_PER_1M).
            if let Some((cost, approx)) = state.agent_dialogue_cost {
                let mark = if approx { "≈" } else { "" };
                spans.push(Span::raw("  ·  💰 "));
                spans.push(Span::styled(
                    format!("{mark}{}{cost:.8}", llm_core::pricing::currency()),
                    Style::default().fg(Color::Yellow),
                ));
            }
            spans.extend(mcp_badge_spans(state.mcp_servers));
            Line::from(spans)
        }
        None => Line::from(" Расход токенов появится после первого ответа агента"),
    };
    frame.render_widget(Paragraph::new(context_line).style(Style::default().fg(Color::DarkGray)), chunks[2]);

    let (input_title, border_color) = if this_agent_waiting {
        (format!(" Ожидание ответа {} ", SPINNER_FRAMES[state.spinner_frame]), Color::Yellow)
    } else if state.waiting {
        (
            " Ожидаем ответ другого агента — можно выйти (Esc) и посмотреть остальных ".to_string(),
            Color::DarkGray,
        )
    } else if !state.agent_chat_running {
        (" Агент остановлен — Ctrl+S запустить (или Esc и 's' в списке) ".to_string(), Color::DarkGray)
    } else {
        (" Запрос — Enter отправить, Esc назад к списку агентов ".to_string(), Color::Reset)
    };
    render_input_box(frame, chunks[3], state.input, input_title, border_color);
}

/// Экран модели памяти агента (см. llm_core::memory) — F3 из [`Screen::AgentChat`],
/// Esc/F3 обратно. Показывает все три уровня разом (краткосрочная — только
/// обзорная строка, т.к. сам диалог виден в чате; рабочая и долговременная —
/// целиком, поскольку это единственное место, где их вообще видно) и исполняет
/// команды `remember`/`forget`/`task ...` из поля ввода (см. [`run_memory_command`]) —
/// синтаксис намеренно совпадает с одноимёнными подкомандами `llm-cli agent`.
fn draw_agent_memory(frame: &mut Frame, state: &DrawState) {
    let area = frame.area();
    let chunks = layout_chunks_with_input_height(area, INPUT_HEIGHT_AGENT_CHAT);
    let name = state.agent_chat_name.unwrap_or("?");
    let info = state.agents.iter().find(|a| a.config.name == name);

    let header = Paragraph::new(Line::from(vec![
        Span::styled("✦ Challenger", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::raw("  ·  память агента: "),
        Span::styled(name, Style::default().fg(Color::Cyan)),
        Span::raw("  ·  F3/Esc — назад к диалогу"),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
    frame.render_widget(header, chunks[0]);

    let section_style = Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD);
    let hint_style = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Инварианты — жёсткие правила (общие для ВСЕХ агентов, приоритетнее всего остального)",
        section_style,
    )));
    let invariant_ids = llm_core::invariants::list_ids();
    if invariant_ids.is_empty() {
        lines.push(Line::from(Span::styled(
            "  пусто — команда: invariants new <id>",
            hint_style,
        )));
    } else {
        for id in &invariant_ids {
            lines.push(Line::from(format!("  - {id}")));
        }
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "Персонализация — профиль (отдельная ось: КАК отвечать, не память)",
        section_style,
    )));
    match info.map(|i| &i.profile) {
        Some(profile) if !profile.enabled => {
            lines.push(Line::from("  отключена явно — команда: profile <профиль>"));
        }
        Some(profile) if profile.found => {
            lines.push(Line::from(format!("  «{}» — подключён к каждому запросу", profile.name)));
        }
        Some(profile) => {
            lines.push(Line::from(Span::styled(
                format!("  «{}» — файл не найден, персонализация сейчас не применяется", profile.name),
                hint_style,
            )));
        }
        None => lines.push(Line::from(Span::styled("  ?", hint_style))),
    }
    if let Some(profile) = info.map(|i| &i.profile) {
        if !profile.available.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  доступные профили: {}", profile.available.join(", ")),
                hint_style,
            )));
        }
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled("Краткосрочная — текущий диалог", section_style)));
    lines.push(Line::from(format!(
        "  записей в истории: {} · стратегия контекста: {}",
        state.agent_history_len.unwrap_or(0),
        info.map(|i| i.config.context_strategy.to_string()).unwrap_or_else(|| "?".to_string()),
    )));
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled("Рабочая — данные ОБЩЕЙ задачи", section_style)));
    match info.and_then(|i| i.task.as_ref()) {
        Some(task) => {
            let goal_suffix = task.goal.as_deref().map(|g| format!(" (цель: {g})")).unwrap_or_default();
            lines.push(Line::from(format!("  задача «{}»{goal_suffix}", task.name)));
            let members = state
                .shared_tasks
                .iter()
                .find(|t| t.name == task.name)
                .map(|t| t.members.join(", "))
                .unwrap_or_default();
            if !members.is_empty() {
                lines.push(Line::from(Span::styled(format!("  участники: {members}"), hint_style)));
            }
            let stage_style = if task.paused {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green)
            };
            let pause_suffix = if task.paused { "  ⏸ НА ПАУЗЕ" } else { "" };
            lines.push(Line::from(vec![
                Span::raw("  этап: "),
                Span::styled(format!("{}{pause_suffix}", task.stage), stage_style),
            ]));
            if let Some(step) = task.current_step.as_deref().filter(|s| !s.is_empty()) {
                lines.push(Line::from(format!("  текущий шаг: {step}")));
            }
            if let Some(expect) = task.expected_action.as_deref().filter(|s| !s.is_empty()) {
                lines.push(Line::from(format!("  ожидаемое действие: {expect}")));
            }
            if let Some(pending) = task.pending_stage {
                let outcome = task.pending_outcome.as_deref().unwrap_or("(не указан)");
                lines.push(Line::from(Span::styled(
                    format!("  ⏳ предложен переход в «{pending}» (итог: {outcome}) — approve/reject"),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                )));
            }
            if !task.blocked_transitions.is_empty() {
                let items: Vec<String> =
                    task.blocked_transitions.iter().map(|(f, t)| format!("{f}->{t}")).collect();
                lines.push(Line::from(Span::styled(
                    format!("  🚫 запрещено инвариантом задачи: {}", items.join(", ")),
                    Style::default().fg(Color::Red),
                )));
            }
            if !task.extra_approval_transitions.is_empty() {
                let items: Vec<String> =
                    task.extra_approval_transitions.iter().map(|(f, t)| format!("{f}->{t}")).collect();
                lines.push(Line::from(Span::styled(
                    format!("  🔒 доп. согласие для модели: {}", items.join(", ")),
                    hint_style,
                )));
            }
            if !task.transitions.is_empty() {
                // Панель тесная — только последние записи журнала.
                const SHOWN: usize = 5;
                let skipped = task.transitions.len().saturating_sub(SHOWN);
                lines.push(Line::from(if skipped > 0 {
                    format!("  журнал переходов (последние {SHOWN} из {}):", task.transitions.len())
                } else {
                    "  журнал переходов:".to_string()
                }));
                for record in &task.transitions[skipped..] {
                    let color = match record.outcome {
                        llm_core::TransitionOutcome::Applied | llm_core::TransitionOutcome::Approved => Color::Green,
                        llm_core::TransitionOutcome::Proposed => Color::Yellow,
                        llm_core::TransitionOutcome::Rejected | llm_core::TransitionOutcome::Refused => Color::Red,
                    };
                    lines.push(Line::from(Span::styled(format!("    {record}"), Style::default().fg(color))));
                }
            }
            if !task.invariants.is_empty() {
                lines.push(Line::from("  инварианты этой задачи:"));
                for (id, text) in &task.invariants {
                    lines.push(Line::from(format!("    - [{id}] {text}")));
                }
            }
            if task.data.is_empty() {
                lines.push(Line::from(Span::styled("  (данных пока нет)", hint_style)));
            } else {
                for (key, value) in &task.data {
                    lines.push(Line::from(format!("  - {key} = {value}")));
                }
            }
        }
        None => {
            lines.push(Line::from(Span::styled(
                "  активной задачи нет — команда: task start <название> [--goal ТЕКСТ] или task join <название>",
                hint_style,
            )));
            if !state.shared_tasks.is_empty() {
                lines.push(Line::from(Span::styled("  доступные задачи для join:", hint_style)));
                for task in state.shared_tasks {
                    let goal = task.goal.as_deref().map(|g| format!(" (цель: {g})")).unwrap_or_default();
                    let members =
                        if task.members.is_empty() { "без участников".to_string() } else { task.members.join(", ") };
                    lines.push(Line::from(Span::styled(
                        format!("    - {}{goal} · участники: {members}", task.name),
                        hint_style,
                    )));
                }
            }
        }
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "Долговременная — решения, знания (общая для всех агентов)",
        section_style,
    )));
    match info.map(|i| &i.long_term) {
        Some(map) if !map.is_empty() => {
            for (key, item) in map {
                lines.push(Line::from(format!("  - [{}] {key} = {}", item.category, item.value)));
            }
        }
        _ => lines.push(Line::from(Span::styled(
            "  пусто — команда: remember <ключ> <значение> [--category CAT]",
            hint_style,
        ))),
    }

    let body = Paragraph::new(lines).block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Память агента "),
    );
    frame.render_widget(body, chunks[1]);

    let (status_text, status_color) = match state.memory_status {
        Some((text, is_error)) => (format!(" {text}"), if is_error { Color::Red } else { Color::Green }),
        None => (
            " Команды: remember <ключ> <значение> [--category CAT] · forget <ключ> · \
             task start <имя> [--goal ТЕКСТ] · task join <имя> · task set <ключ> <значение> · \
             task advance <этап> [--step T] [--expect T] · task pause · task resume · \
             task approve · task reject <причина> · task finish · task invariant set/remove <id> [текст] · \
             task forbid/allow/require-approval/unrequire-approval <из> <в> · profile <профиль|none> · \
             profile new <имя> · invariants new <id> · invariants remove <id>"
                .to_string(),
            Color::DarkGray,
        ),
    };
    frame.render_widget(Paragraph::new(status_text).style(Style::default().fg(status_color)), chunks[2]);

    render_input_box(
        frame,
        chunks[3],
        state.input,
        " Команда памяти — Enter выполнить, Esc/F3 назад к диалогу ".to_string(),
        Color::Reset,
    );
}

/// Разбирает и выполняет одну команду экрана памяти агента ([`Screen::AgentMemory`]) —
/// `remember`/`forget`/`task start|join|set|show|advance|step|expect|pause|resume|finish`,
/// синтаксически совпадающие с одноимёнными подкомандами `llm-cli agent` (см.
/// cli/src/main.rs), чтобы поведение этой модели памяти не расходилось между
/// интерфейсами. `task` работает с ОБЩЕЙ задачей (см. `crate::memory` в
/// llm-core) — `start` создаёт её и сразу присоединяет агента, `join`
/// присоединяет к уже существующей (созданной другим агентом), `finish`
/// завершает её для ВСЕХ присоединённых агентов разом. Все операции — быстрые
/// синхронные вызовы над [`llm_core::Agent`] (никакого обращения к LLM),
/// поэтому выполняются прямо в цикле событий. Возвращает текст сообщения для
/// строки статуса — `Ok` при успехе, `Err` при ошибке (некорректная команда
/// или отказ метода `Agent`, например повторный `task start`/`join` без
/// предварительного `finish`).
/// Разбирает пару этапов `<из> <в>` из хвоста команд forbid/allow/
/// require-approval/unrequire-approval (см. [`run_memory_command`]) — общая
/// для всех четырёх: `tokens[2]` — исходный этап, `tokens[3]` — целевой.
fn parse_tui_transition_pair(tokens: &[&str], action: &str) -> Result<(llm_core::Stage, llm_core::Stage), String> {
    let from_raw = tokens.get(2).copied().ok_or_else(|| format!("укажите этапы: task {action} <из> <в>"))?;
    let to_raw = tokens.get(3).copied().ok_or_else(|| format!("укажите этапы: task {action} <из> <в>"))?;
    let from: llm_core::Stage = from_raw.parse().map_err(|err: anyhow::Error| err.to_string())?;
    let to: llm_core::Stage = to_raw.parse().map_err(|err: anyhow::Error| err.to_string())?;
    Ok((from, to))
}

/// Отправляет сообщение агенту так же, как Enter в его диалоге: реплика
/// человека сразу появляется в истории, ответ приходит событием
/// [`AppEvent::AgentResponse`]. `from_human = false` — служебное продолжение
/// (после resume/approve, см. `Agent::continue_task`): в диалоге его нет.
/// Флаги ожидания выставляет вызывающая сторона.
fn send_to_agent(
    agent: std::sync::Arc<llm_core::Agent>,
    name: String,
    prompt: String,
    from_human: bool,
    histories: &mut HashMap<String, Vec<HistoryItem>>,
    tx: &mpsc::UnboundedSender<AppEvent>,
) {
    if from_human {
        histories.entry(name.clone()).or_default().push(HistoryItem {
            role: Role::User,
            text: prompt.clone(),
            debug: None,
            tokens: None,
            cost: None,
            cost_approx: false,
        });
    }
    let tx = tx.clone();
    tokio::spawn(async move {
        let response =
            if from_human { agent.handle_request(&prompt).await } else { agent.continue_task(&prompt).await };
        let _ = tx.send(AppEvent::AgentResponse { name, result: response });
    });
}

fn run_memory_command(agent: &llm_core::Agent, raw: &str) -> Result<String, String> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    match tokens.first().copied() {
        Some("remember") => {
            let key = tokens.get(1).copied().ok_or("укажите ключ: remember <ключ> <значение> [--category CAT]")?;
            let mut category = "knowledge".to_string();
            let mut value_tokens: Vec<&str> = Vec::new();
            let mut i = 2;
            while i < tokens.len() {
                if tokens[i] == "--category" {
                    i += 1;
                    category = tokens.get(i).copied().ok_or("--category требует значение")?.to_string();
                } else {
                    value_tokens.push(tokens[i]);
                }
                i += 1;
            }
            if value_tokens.is_empty() {
                return Err("укажите значение: remember <ключ> <значение> [--category CAT]".to_string());
            }
            let value = value_tokens.join(" ");
            agent.remember(key, &value, &category).map_err(|err| err.to_string())?;
            Ok(format!("Сохранено в долговременную память: [{category}] {key} = {value}"))
        }
        Some("forget") => {
            let key = tokens.get(1).copied().ok_or("укажите ключ: forget <ключ>")?;
            match agent.forget(key) {
                Ok(true) => Ok(format!("Запись «{key}» удалена из долговременной памяти.")),
                Ok(false) => Ok(format!("В долговременной памяти не было записи «{key}».")),
                Err(err) => Err(err.to_string()),
            }
        }
        Some("task") => match tokens.get(1).copied() {
            Some("start") => {
                let task_name =
                    tokens.get(2).copied().ok_or("укажите название: task start <название> [--goal ТЕКСТ]")?;
                let rest = &tokens[3.min(tokens.len())..];
                let goal = if rest.first().copied() == Some("--goal") {
                    Some(rest[1..].join(" "))
                } else {
                    None
                };
                agent.task_start(task_name, goal.as_deref()).map_err(|err| err.to_string())?;
                Ok(format!("Задача «{task_name}» создана, агент присоединён к ней."))
            }
            Some("join") => {
                let task_name = tokens.get(2).copied().ok_or("укажите название: task join <название>")?;
                agent.task_join(task_name).map_err(|err| err.to_string())?;
                Ok(format!(
                    "Агент присоединён к задаче «{task_name}» — видит и меняет её рабочую память \
                     наравне с остальными участниками."
                ))
            }
            Some("set") => {
                let key = tokens.get(2).copied().ok_or("укажите ключ: task set <ключ> <значение>")?;
                let value_tokens = &tokens[3.min(tokens.len())..];
                if value_tokens.is_empty() {
                    return Err("укажите значение: task set <ключ> <значение>".to_string());
                }
                let value = value_tokens.join(" ");
                agent.task_set(key, &value).map_err(|err| err.to_string())?;
                Ok(format!("Рабочая память обновлена (видно всем участникам): {key} = {value}"))
            }
            Some("advance") => {
                let stage_raw =
                    tokens.get(2).copied().ok_or("укажите этап: task advance <planning|execution|validation|done> [--step T] [--expect T]")?;
                let stage: llm_core::Stage = stage_raw.parse().map_err(|err: anyhow::Error| err.to_string())?;
                let mut step: Option<String> = None;
                let mut expect: Option<String> = None;
                let mut i = 3;
                while i < tokens.len() {
                    match tokens[i] {
                        "--step" => {
                            i += 1;
                            step = Some(tokens.get(i).copied().ok_or("--step требует значение")?.to_string());
                        }
                        "--expect" => {
                            i += 1;
                            expect = Some(tokens.get(i).copied().ok_or("--expect требует значение")?.to_string());
                        }
                        other => return Err(format!("неизвестный флаг «{other}»")),
                    }
                    i += 1;
                }
                agent.task_advance(stage, step.as_deref(), expect.as_deref()).map_err(|err| err.to_string())?;
                Ok(format!("Задача переведена на этап «{stage_raw}»."))
            }
            Some("step") => {
                let text = tokens.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty())
                    .ok_or("укажите текущий шаг: task step <текст>")?;
                agent.task_step(&text).map_err(|err| err.to_string())?;
                Ok(format!("Текущий шаг обновлён: {text}"))
            }
            Some("expect") => {
                let text = tokens.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty())
                    .ok_or("укажите ожидаемое действие: task expect <текст>")?;
                agent.task_expect(&text).map_err(|err| err.to_string())?;
                Ok(format!("Ожидаемое действие обновлено: {text}"))
            }
            Some("pause") => {
                agent.task_pause().map_err(|err| err.to_string())?;
                Ok("Задача поставлена на паузу.".to_string())
            }
            Some("finish") => match agent.task_finish() {
                Ok(Some(task)) => {
                    Ok(format!("Задача «{}» завершена и удалена ДЛЯ ВСЕХ участников.", task.name))
                }
                Ok(None) => Ok("Активной задачи не было.".to_string()),
                Err(err) => Err(err.to_string()),
            },
            Some("invariant") => match tokens.get(1).copied() {
                Some("set") => {
                    let id = tokens.get(2).copied().ok_or("укажите id: task invariant set <id> <текст>")?;
                    let text = tokens.get(3..).map(|s| s.join(" ")).filter(|s| !s.is_empty())
                        .ok_or("укажите текст: task invariant set <id> <текст>")?;
                    agent.task_invariant_set(id, &text).map_err(|err| err.to_string())?;
                    Ok(format!("Инвариант задачи «{id}» сохранён."))
                }
                Some("remove") => {
                    let id = tokens.get(2).copied().ok_or("укажите id: task invariant remove <id>")?;
                    match agent.task_invariant_remove(id).map_err(|err| err.to_string())? {
                        true => Ok(format!("Инвариант задачи «{id}» удалён.")),
                        false => Err(format!("инварианта задачи «{id}» нет")),
                    }
                }
                _ => Err("укажите действие: task invariant set <id> <текст> | task invariant remove <id>".to_string()),
            },
            Some("forbid") => {
                let (from, to) = parse_tui_transition_pair(&tokens, "forbid")?;
                agent.task_forbid_transition(from, to).map_err(|err| err.to_string())?;
                Ok(format!("Переход «{from}» -> «{to}» запрещён для этой задачи (абсолютно)."))
            }
            Some("allow") => {
                let (from, to) = parse_tui_transition_pair(&tokens, "allow")?;
                match agent.task_allow_transition(from, to).map_err(|err| err.to_string())? {
                    true => Ok(format!("Запрет на «{from}» -> «{to}» снят.")),
                    false => Err(format!("переход «{from}» -> «{to}» и не был запрещён")),
                }
            }
            Some("require-approval") => {
                let (from, to) = parse_tui_transition_pair(&tokens, "require-approval")?;
                agent.task_require_approval(from, to).map_err(|err| err.to_string())?;
                Ok(format!("Переход «{from}» -> «{to}» (модели) теперь требует подтверждения человеком."))
            }
            Some("unrequire-approval") => {
                let (from, to) = parse_tui_transition_pair(&tokens, "unrequire-approval")?;
                match agent.task_unrequire_approval(from, to).map_err(|err| err.to_string())? {
                    true => Ok(format!("Доп. требование подтверждения на «{from}» -> «{to}» снято.")),
                    false => Err(format!("для «{from}» -> «{to}» доп. требования и не было")),
                }
            }
            Some("show") => Ok("Текущее состояние показано выше.".to_string()),
            Some(other) => Err(format!(
                "неизвестное действие «{other}» — task start/join/set/show/advance/step/expect/pause/resume/\
                 approve/reject/finish/invariant/forbid/allow/require-approval/unrequire-approval"
            )),
            None => Err(
                "укажите действие: task start/join/set/show/advance/step/expect/pause/resume/approve/reject/\
                 finish/invariant/forbid/allow/require-approval/unrequire-approval"
                    .to_string(),
            ),
        },
        Some("profile") => match tokens.get(1).copied() {
            Some("new") => {
                let profile_name = tokens.get(2).copied().ok_or("укажите имя: profile new <имя>")?;
                llm_core::profile::create(profile_name, &llm_core::profile::template(profile_name))
                    .map_err(|err| err.to_string())?;
                Ok(format!("Профиль «{profile_name}» создан — подключить: profile {profile_name}"))
            }
            Some(new_profile) => {
                let mut config = agent.config();
                config.profile = Some(new_profile.to_string());
                agent.set_config(config).map_err(|err| err.to_string())?;
                if llm_core::profile::is_disabled(new_profile) {
                    Ok("Персонализация отключена.".to_string())
                } else {
                    Ok(format!("Профиль переключён на «{new_profile}»."))
                }
            }
            None => Err("укажите профиль: profile <профиль|none> или profile new <имя>".to_string()),
        },
        Some("invariants") => match tokens.get(1).copied() {
            Some("new") => {
                let id = tokens.get(2).copied().ok_or("укажите id: invariants new <id>")?;
                llm_core::invariants::create(id, &llm_core::invariants::template(id))
                    .map_err(|err| err.to_string())?;
                Ok(format!("Инвариант «{id}» создан — отредактируйте файл, чтобы описать правило."))
            }
            Some("remove") => {
                let id = tokens.get(2).copied().ok_or("укажите id: invariants remove <id>")?;
                match llm_core::invariants::remove(id).map_err(|err| err.to_string())? {
                    true => Ok(format!("Инвариант «{id}» удалён — его действие снято для всех агентов.")),
                    false => Err(format!("инварианта «{id}» нет в каталоге")),
                }
            }
            _ => Err("укажите действие: invariants new <id> или invariants remove <id>".to_string()),
        },
        Some(other) => Err(format!("неизвестная команда «{other}» — remember/forget/task/profile/invariants")),
        None => Ok(String::new()),
    }
}

fn history_item_to_lines(item: &HistoryItem, width: usize, show_debug: bool) -> Vec<Line<'static>> {
    let (label, color) = match item.role {
        Role::User => ("Вы", Color::Cyan),
        Role::Assistant => ("LLM", Color::Green),
        Role::System => ("Инфо", Color::DarkGray),
        Role::Error => ("Ошибка", Color::Red),
        Role::Compression => ("Сжатие", Color::Yellow),
        Role::Tool => ("Тул", Color::Magenta),
        Role::ToolError => ("Тул ⚠", Color::Red),
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
    // Токены этого конкретного сообщения — как в веб-интерфейсе, прямо под
    // текстом: "токены запроса" у сообщения пользователя, "токены ответа" у
    // ответа ассистента (см. HistoryItem::tokens); стоимость (см.
    // HistoryItem::cost) дописывается на ту же строку, если заданы ставки цены.
    if let Some(tokens) = item.tokens {
        let token_label = match item.role {
            Role::User => "токены запроса",
            Role::Assistant => "токены ответа",
            _ => "токены",
        };
        let mut text = format!("{token_label}: {tokens}");
        if let Some(cost) = item.cost {
            let mark = if item.cost_approx { "≈" } else { "" };
            text.push_str(&format!(" · {mark}{}{:.8}", llm_core::pricing::currency(), cost));
        }
        lines.push(Line::from(vec![
            Span::raw(" ".repeat(prefix_width)),
            Span::styled(text, Style::default().fg(Color::DarkGray)),
        ]));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// usage приходит от API одной суммой на весь обмен и хранится в БД только на
    /// сообщении ассистента (см. Db::append_message в agent.rs) — эта проверка
    /// фиксирует, что history_items_from_messages правильно раскладывает её:
    /// prompt_tokens уходит на предшествующее сообщение пользователя, а
    /// completion_tokens остаётся на самом ответе.
    #[test]
    fn pairs_usage_across_user_and_assistant_messages() {
        let usage1 = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cost: None,
            cost_input: None,
            cost_output: None,
        };
        let usage2 = Usage {
            prompt_tokens: 20,
            completion_tokens: 8,
            total_tokens: 28,
            cost: None,
            cost_input: None,
            cost_output: None,
        };
        let messages = vec![
            (ChatMessage::user("привет"), None),
            (ChatMessage::assistant("здравствуйте"), Some(usage1)),
            (ChatMessage::user("как дела"), None),
            (ChatMessage::assistant("хорошо"), Some(usage2)),
        ];

        let items = history_items_from_messages(&messages);

        assert_eq!(items.len(), 4);
        assert_eq!(items[0].tokens, Some(10)); // запрос пользователя #1
        assert_eq!(items[1].tokens, Some(5)); // ответ ассистента #1
        assert_eq!(items[2].tokens, Some(20)); // запрос пользователя #2
        assert_eq!(items[3].tokens, Some(8)); // ответ ассистента #2
    }

    /// Сообщение без usage (например, ассистент так и не ответил) не должно
    /// падать и должно оставлять токены неизвестными (None), а не паниковать
    /// на индексации предыдущего элемента.
    #[test]
    fn leaves_tokens_none_without_usage() {
        let messages = vec![(ChatMessage::user("привет"), None)];
        let items = history_items_from_messages(&messages);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tokens, None);
    }
}
