//! Агент — самостоятельная сущность поверх [`LlmClient`]: у неё есть своя
//! конфигурация (системный промпт, модель, параметры генерации, показывать ли
//! токены, стратегия управления контекстом) и жизненный цикл (запущен/остановлен).
//! Запрос пользователя обрабатывается агентом, а не единичным вызовом клиента
//! напрямую — пока агент остановлен, он отказывается отвечать.
//!
//! ## Управление контекстом
//!
//! Реализовано 4 стратегии (см. [`context::ContextStrategy`] и документацию
//! модуля [`crate::context`]), переключаемые через `AgentConfig::context_strategy`:
//! `full` (без управления), `summary` (сжатие в сводку), `sliding-window`
//! (последние N сообщений), `facts` (sticky facts + последние N сообщений) и
//! `branching` (ветки диалога с checkpoint'ами). История диалога агента всегда
//! хранится как набор именованных веток (см. [`Branches`]) — даже когда
//! активна не-ветвящаяся стратегия, просто вся работа идёт с единственной
//! веткой `main`, поэтому переключение стратегии не требует миграции данных.
//!
//! Конфигурация агентов, история их диалогов, сводки, факты и ветки хранятся в
//! SQLite (см. [`Db`]), поэтому они переживают перезапуск приложения: агент,
//! запущенный до перезапуска, при следующем старте снова видит все прежние
//! сообщения и продолжает диалог, как будто его не выключали. Удаление агента
//! удаляет и всё это (каскадно, через `ON DELETE CASCADE`).
//!
//! ## Модель памяти
//!
//! Помимо истории диалога, у агента есть ещё два независимых, явно
//! заполняемых уровня памяти — см. модуль [`crate::memory`] за полным
//! описанием модели из трёх уровней (краткосрочная/рабочая/долговременная).
//! Оба этих уровня — ОБЩИЕ, не приватные для отдельного агента:
//! [`Agent::remember`]/[`Agent::forget`] (долговременная) читают и пишут
//! ЕДИНЫЙ набор данных на всё приложение — правка через одного агента сразу
//! видна через любого другого, независимо от того, какую задачу он выполняет;
//! [`Agent::task_start`]/[`Agent::task_join`]/[`Agent::task_set`]/
//! [`Agent::task_finish`] (рабочая) устроены детальнее — она принадлежит не
//! агенту, а ОБЩЕЙ ЗАДАЧЕ (см. [`crate::memory::TaskState`]), и видна только
//! агентам, явно присоединившимся к этой конкретной задаче по имени — см.
//! [`AgentManager::list_tasks`] для списка всех существующих задач и их
//! участников.
//!
//! ## Персонализация
//!
//! Независимая от памяти ось — см. модуль [`crate::profile`]: markdown-файл в
//! каталоге профилей описывает манеру общения, язык ответа и ограничения и
//! подключается к КАЖДОМУ запросу ([`Agent::handle_request`]), независимо от
//! `context_strategy` и от памяти выше. `AgentConfig::profile` хранит только
//! ИМЯ файла — [`Agent::profile_status`] читает его содержимое с диска заново
//! при каждом обращении, как и долговременную/рабочую память.
//!
//! ## Инварианты
//!
//! Ещё одна независимая, самая приоритетная ось — см. модуль
//! [`crate::invariants`]: жёсткие правила (архитектура, технические решения,
//! ограничения по стеку, бизнес-правила), которые ассистент не имеет права
//! нарушать ни на одном этапе работы. Три источника, каждый со своим
//! масштабом действия:
//!
//! - **Глобальные — файлы** каталога инвариантов (markdown, отдельно от
//!   диалога и от SQLite) — один общий набор для ВСЕХ агентов и задач, как
//!   долговременная память.
//! - **Глобальные — долговременная память**: записи с категорией
//!   `"invariant"` ([`crate::invariants::MEMORY_CATEGORY`]) — та же сила
//!   действия, что и у файловых, но заводятся обычной командой `agent
//!   remember`, без отдельного файла (например, короткое правило "не
//!   транслитерировать английские термины").
//! - **Задачи** — [`crate::memory::TaskState::invariants`] (текстовые) и
//!   `blocked_transitions`/`extra_approval_transitions` (структурные —
//!   дополнительные запреты/гейты согласия на конкретные переходы автомата,
//!   см. [`Agent::task_forbid_transition`]/[`Agent::task_require_approval`]) —
//!   действуют ТОЛЬКО пока агент работает над этой конкретной задачей, и
//!   структурные из них не просто описаны в промпте, а реально проверяются
//!   кодом в [`Agent::task_advance`]/`tool_move_stage`, а не только текстом.
//!
//! Оба глобальных источника объединяются в один блок и подключаются ПЕРВЫМ
//! системным сообщением у каждого запроса ([`Agent::handle_request`]), раньше
//! системного промпта и персонализации, потому что они могут ему
//! противоречить, но не отменяют его. Задачные — часть блока задачи
//! ([`crate::memory::format_task_block`]), видны только пока задача активна.

use crate::memory::{
    LongTermItem, LongTermMemory, SharedTaskSummary, Stage, TaskState, TransitionActor, TransitionOutcome,
    TransitionRecord,
};
use crate::{context, context::ContextStrategy, ChatMessage, ChatOptions, LlmClient, Usage};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Имя ветки, с которой начинается диалог любого агента и которая остаётся
/// единственной, пока активна не-ветвящаяся стратегия контекста.
pub const MAIN_BRANCH: &str = "main";

/// Предел кругов "вызов инструмента -> результат -> продолжение" внутри
/// одного обмена (см. [`Agent::run_tool_loop`]) — защита от зацикливания
/// модели на вызовах инструментов автомата задачи, а не архитектурное
/// ограничение самого автомата.
const MAX_TOOL_ROUNDS: usize = 6;

/// Как часто [`Agent::handle_request`] проверяет, не поставили ли задачу на
/// паузу, пока ждёт ответа LLM.
const PAUSE_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Ответ человеку вместо ответа модели, прерванного паузой (см. [`AgentReply::interrupted`]).
const INTERRUPTED_BY_PAUSE: &str = "⏸ Задача поставлена на паузу.";

/// Предлагать ли модели инструменты автомата (`move_stage`/`update_step`):
/// только пока есть активная задача, она не на паузе, не конечная и не ждёт
/// утверждения человеком уже предложенного перехода — во всех этих случаях
/// модель и так не должна двигать автомат (см. [`Agent::handle_request`]).
fn task_tools_offered(task: Option<&TaskState>) -> bool {
    task.map(|t| !t.paused && t.stage != Stage::Done && t.pending_stage.is_none()).unwrap_or(false)
}

/// Складывает метрики токенов/стоимости двух обменов — используется, чтобы
/// показать пользователю честную сумму по всем кругам вызова инструментов
/// внутри одного обмена ([`Agent::run_tool_loop`]), а не только последний.
fn sum_usage(a: Option<Usage>, b: Option<Usage>) -> Option<Usage> {
    fn add_opt(a: Option<f64>, b: Option<f64>) -> Option<f64> {
        match (a, b) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => Some(Usage {
            prompt_tokens: a.prompt_tokens + b.prompt_tokens,
            completion_tokens: a.completion_tokens + b.completion_tokens,
            total_tokens: a.total_tokens + b.total_tokens,
            cost: add_opt(a.cost, b.cost),
            cost_input: add_opt(a.cost_input, b.cost_input),
            cost_output: add_opt(a.cost_output, b.cost_output),
        }),
    }
}

/// Настраиваемое поведение агента: системный промпт, модель и параметры
/// генерации, стратегия управления контекстом, а также флаг вывода количества
/// использованных токенов.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Модель для запросов этого агента; если не задана — берётся модель клиента (LLM_MODEL).
    #[serde(default)]
    pub model: Option<String>,
    /// Если true — в текст ответа агента добавляется строка с расходом токенов.
    #[serde(default)]
    pub show_tokens: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub reasoning: Option<bool>,
    /// Активная стратегия управления контекстом — см. документацию модуля и
    /// [`context::ContextStrategy`]. По умолчанию [`ContextStrategy::Full`]
    /// (поведение как раньше: вся история целиком в каждом запросе).
    #[serde(default)]
    pub context_strategy: ContextStrategy,
    /// Переопределение размера окна (в сообщениях пользователя) для стратегий
    /// `SlidingWindow` и `Facts` — см. [`context::sliding_window_size`].
    /// `None` — используется общее значение из `LLM_SLIDING_WINDOW_SIZE`.
    #[serde(default)]
    pub window_size: Option<usize>,
    /// Имя профиля персонализации (см. [`crate::profile`]) — markdown-файл в
    /// каталоге профилей, описывающий манеру общения, язык ответа и
    /// ограничения для конкретного человека. `None` разрешается в
    /// [`crate::profile::DEFAULT_PROFILE`] (см. [`crate::profile::resolve_name`]);
    /// значение [`crate::profile::NONE_PROFILE`] явно выключает персонализацию
    /// для этого агента. Это отдельная ось от долговременной/рабочей памяти —
    /// профиль описывает КАК отвечать, память — ЧТО агент знает.
    #[serde(default)]
    pub profile: Option<String>,
}

impl AgentConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: None,
            model: None,
            show_tokens: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            reasoning: None,
            context_strategy: ContextStrategy::default(),
            window_size: None,
            profile: None,
        }
    }
}

/// Живой статус персонализации агента (см. [`crate::profile`]): какое имя
/// профиля реально используется, включена ли персонализация вообще,
/// существует ли файл этого профиля на диске, его содержимое (если найден) и
/// список всех профилей, доступных в каталоге — для интерфейсов, предлагающих
/// выбор.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStatus {
    /// Имя профиля, разрешённое из `AgentConfig::profile` (см.
    /// [`crate::profile::resolve_name`]) — либо явно заданное, либо
    /// [`crate::profile::DEFAULT_PROFILE`].
    pub name: String,
    /// `false`, только если имя профиля — [`crate::profile::NONE_PROFILE`]
    /// (персонализация явно выключена для этого агента).
    pub enabled: bool,
    /// `true`, если файл `<name>.md` найден в каталоге профилей — именно этот
    /// признак определяет, попадёт ли `content` в запрос к LLM.
    pub found: bool,
    /// Содержимое файла профиля, если он найден и персонализация включена —
    /// то же самое, что подмешивается в запрос (см. [`crate::profile::format_profile_block`]).
    pub content: Option<String>,
    /// Все профили, найденные сейчас в каталоге профилей (см. [`crate::profile::list_profiles`]).
    pub available: Vec<String>,
}

/// Снимок состояния агента для отображения в интерфейсах и для сохранения на диск.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub config: AgentConfig,
    pub running: bool,
    /// Размер контекстного окна (см. [`LlmClient::context_window`]) — используется
    /// интерфейсом для индикатора заполнения контекста. Числитель этого индикатора —
    /// total_tokens самого последнего ответа (см. [`AgentReply::usage`]), а не сумма
    /// по всему диалогу: так это число всегда в точности то, что вернула модель,
    /// без вычислений с нашей стороны.
    pub context_window: u32,
    /// Живой статус стратегии [`ContextStrategy::Summary`] — `Some` только пока
    /// она активна.
    pub compression: Option<CompressionInfo>,
    /// Живой статус стратегий [`ContextStrategy::SlidingWindow`] и
    /// [`ContextStrategy::Facts`] (обе используют одно и то же скользящее окно) —
    /// `Some` только пока одна из них активна.
    pub sliding_window: Option<SlidingWindowInfo>,
    /// Текущий набор фактов стратегии [`ContextStrategy::Facts`] — `Some` только
    /// пока она активна.
    pub facts: Option<FactsInfo>,
    /// Живой статус стратегии [`ContextStrategy::Branching`] — `Some` только
    /// пока она активна.
    pub branching: Option<BranchingInfo>,
    /// Долговременная память — ОБЩАЯ для ВСЕХ агентов (решения/знания,
    /// см. [`crate::memory`]), одинакова независимо от того, через какого
    /// агента её читают или какую задачу он выполняет; всегда присутствует
    /// (может быть пустой), не зависит от `context_strategy`: заполняется
    /// только явно, через [`Agent::remember`]. Прочитано заново из БД на
    /// момент вызова, а не из кеша — эти данные могли только что измениться
    /// через любого другого агента (см. [`Agent::long_term_memory`]).
    pub long_term: LongTermMemory,
    /// Рабочая память ОБЩЕЙ задачи, к которой сейчас присоединён этот агент
    /// (см. [`crate::memory::TaskState`]) — `None`, если агент ни к какой
    /// задаче не присоединён. Прочитано заново из БД на момент вызова, а не
    /// из кеша — эти данные могли только что измениться другим агентом,
    /// участвующим в той же задаче (см. [`Agent::task_state`]).
    pub task: Option<TaskState>,
    /// Персонализация (см. [`crate::profile`]) — независимая от памяти ось:
    /// описывает КАК агент должен отвечать (манера общения, язык, ограничения),
    /// а не что он знает. Прочитано заново с диска на момент вызова, а не из
    /// кеша — файл профиля мог только что измениться (правка вручную или через
    /// другого агента, если несколько агентов используют один и тот же профиль).
    pub profile: ProfileStatus,
}

/// Единица счёта здесь — сообщение ПОЛЬЗОВАТЕЛЯ (один его запрос + один ответ
/// ассистента = один "обмен"), а не отдельная запись в истории: интерфейсы
/// уменьшают счётчик на 1 за каждое отправленное пользователем сообщение, а не
/// на 2 (запрос + ответ) — см. [`context::RAW_MESSAGES_PER_EXCHANGE`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Сколько сообщений пользователя от начала диалога уже покрывает текущая сводка.
    pub summarized_count: usize,
    /// Сколько сообщений пользователя накопилось с прошлого пересчёта сводки и
    /// ещё не попало в неё. Растёт на 1 с каждым отправленным сообщением и
    /// сбрасывается в 0 сразу после очередного пересчёта.
    pub pending_messages: usize,
    /// Сколько ещё сообщений должно добавиться, прежде чем сводка будет
    /// пересчитана снова — `context_summary_chunk() - pending_messages` (см.
    /// [`context::context_summary_chunk`]).
    pub messages_until_summary: usize,
}

/// Живой статус стратегий [`ContextStrategy::SlidingWindow`] / [`ContextStrategy::Facts`]:
/// сколько сообщений пользователя всего в текущей ветке и сколько из них
/// реально попадёт в следующий запрос (последние `window_size`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SlidingWindowInfo {
    pub window_size: usize,
    pub total_messages: usize,
    pub kept_messages: usize,
    pub dropped_messages: usize,
}

/// Живой статус стратегии [`ContextStrategy::Facts`] — сам набор фактов плюс
/// размер окна сообщений, с которым он комбинируется в запросе.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactsInfo {
    pub facts: BTreeMap<String, String>,
    pub window_size: usize,
}

/// Живой статус стратегии [`ContextStrategy::Branching`] — текущая ветка,
/// список всех существующих веток и список сохранённых checkpoint'ов.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingInfo {
    pub current_branch: String,
    pub branches: Vec<String>,
    pub checkpoints: Vec<String>,
}

/// Результат обработки запроса агентом: итоговый текст (уже с учётом
/// `show_tokens`) и сырые метрики расхода токенов за этот запрос — как их
/// вернула модель, без пересчёта.
#[derive(Debug, Clone)]
pub struct AgentReply {
    pub text: String,
    pub usage: Option<Usage>,
    /// true, если в рамках обработки именно этого запроса была пересчитана
    /// сводка сжатого контекста (стратегия [`ContextStrategy::Summary`]) —
    /// сигнал для интерфейсов показать пользователю момент сжатия.
    pub summarized: bool,
    /// Сколько сообщений от начала диалога сейчас покрывает сводка. Задано,
    /// только если `summarized == true`.
    pub summary_covers: Option<usize>,
    /// true, если в рамках обработки этого запроса был обновлён набор фактов
    /// (стратегия [`ContextStrategy::Facts`]).
    pub facts_updated: bool,
    /// Сырые JSON запроса к LLM и его ответа за этот обмен (как в
    /// [`crate::ChatCompletion`]) — для отладочного просмотра в интерфейсах,
    /// по галочке "показывать JSON запроса/ответа".
    pub request_json: String,
    pub response_json: String,
    /// Денежная стоимость этого обмена по ставкам [`crate::pricing::from_env`] —
    /// `None`, если ставки не заданы (фича выключена) или модель не вернула
    /// `usage`. Не персистится: интерфейсы, которым нужна накопительная
    /// стоимость всего диалога (а не только этого обмена), пересчитывают её
    /// сами по сохранённым в истории метрикам токенов.
    pub cost: Option<AgentCost>,
    /// `true`, если задачу поставили на паузу, пока шёл запрос к LLM: запрос
    /// оборван, ответ модели отброшен и не сохранён в историю, а `text` —
    /// пояснение для человека. Сам запрос запомнен в задаче и уйдёт заново
    /// при возобновлении (см. [`Agent::task_resume`]).
    pub interrupted: bool,
    /// Инструменты, которые модель вызвала в рамках этого обмена, по порядку —
    /// и MCP-серверов (см. [`crate::mcp`]), и встроенные инструменты автомата
    /// задачи. Как и `cost`, не персистится: интерфейсы показывают эти вызовы
    /// рядом с ответом, пока открыт диалог.
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Один вызов инструмента моделью за обмен — для показа в интерфейсах.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// MCP-сервер, которому ушёл вызов; `None` — встроенный инструмент
    /// автомата задачи (`move_stage`/`update_step`).
    pub server: Option<String>,
    /// Имя инструмента — у MCP-инструментов исходное, как его назвал сервер
    /// (без префикса `mcp__<сервер>__`, см. [`crate::mcp::qualified_tool_name`]).
    pub tool: String,
    /// Доводы вызова — сырой JSON-текст, как его прислала модель.
    pub arguments: String,
    /// Результат, который ушёл модели.
    pub result: String,
    /// Вызов завершился ошибкой (MCP-сервер вернул `isError`, вызов не удался,
    /// или инструмент неизвестен).
    pub is_error: bool,
}

impl ToolCallRecord {
    /// Однострочное описание вызова для чата: `сервер · инструмент(доводы)`,
    /// доводы укорочены до `max_args` символов.
    pub fn headline(&self, max_args: usize) -> String {
        let args = self.arguments.trim();
        let args = if args.is_empty() || args == "{}" {
            String::new()
        } else if args.chars().count() > max_args {
            args.chars().take(max_args).collect::<String>() + "…"
        } else {
            args.to_string()
        };
        match &self.server {
            Some(server) => format!("{server} · {}({args})", self.tool),
            None => format!("автомат задачи · {}({args})", self.tool),
        }
    }

    /// Результат одной строкой (переводы строк и повторные пробелы схлопнуты),
    /// укороченный до `max` символов — полный результат ушёл модели.
    pub fn result_preview(&self, max: usize) -> String {
        let flat = self.result.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.chars().count() > max {
            flat.chars().take(max).collect::<String>() + "…"
        } else {
            flat
        }
    }
}

/// Стоимость одного обмена, разложенная на входную и выходную часть — так
/// интерфейсы, показывающие токены запроса и ответа раздельно (под разными
/// сообщениями, как в TUI и веб-интерфейсе), могут показать и стоимость
/// раздельно, тем же способом. См. [`crate::pricing`] для того, откуда берутся
/// `input`/`output`/`total` — реальная сумма от провайдера (`source == Api`)
/// или оценка по ставкам (`source == Estimated`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentCost {
    pub input: f64,
    pub output: f64,
    pub total: f64,
    pub source: crate::pricing::CostSource,
}

/// Состояние сжатия контекста конкретного агента (стратегия
/// [`ContextStrategy::Summary`]): текст текущей сводки и сколько сообщений
/// истории текущей ветки (считая от начала) она уже покрывает — эти сообщения
/// больше не отправляются в LLM по отдельности, вместо них в запрос
/// подставляется сводка (см. [`Agent::handle_request`]). Саму сводку строит
/// [`context::summarize_chunk`] — общая логика с обычным чатом веб-интерфейса.
#[derive(Debug, Clone, Default)]
struct CompressionState {
    summary: String,
    summarized_count: usize,
}

/// Состояние фактов конкретного агента (стратегия [`ContextStrategy::Facts`]).
#[derive(Debug, Clone, Default)]
struct FactsState {
    facts: BTreeMap<String, String>,
}

/// Checkpoint — именованная точка в истории одной из веток, зафиксированная
/// как (имя ветки, сколько сырых сообщений в ней было на тот момент). От
/// checkpoint'а можно позже ответвить новую ветку через [`Agent::branch_from`].
#[derive(Debug, Clone)]
struct Checkpoint {
    branch: String,
    len: usize,
}

/// История диалога агента как набор именованных веток (стратегия
/// [`ContextStrategy::Branching`]) — плюс то, какая из них сейчас активна, и
/// сохранённые checkpoint'ы. Для не-ветвящихся стратегий используется только
/// ветка [`MAIN_BRANCH`], поэтому это же хранилище — обычная линейная история.
#[derive(Debug, Clone)]
struct Branches {
    current: String,
    data: HashMap<String, Vec<(ChatMessage, Option<Usage>)>>,
    checkpoints: HashMap<String, Checkpoint>,
}

impl Default for Branches {
    fn default() -> Self {
        let mut data = HashMap::new();
        data.insert(MAIN_BRANCH.to_string(), Vec::new());
        Self { current: MAIN_BRANCH.to_string(), data, checkpoints: HashMap::new() }
    }
}

impl Branches {
    fn current_messages(&self) -> &Vec<(ChatMessage, Option<Usage>)> {
        self.data.get(&self.current).expect("текущая ветка всегда существует в data")
    }

    fn current_messages_mut(&mut self) -> &mut Vec<(ChatMessage, Option<Usage>)> {
        self.data.get_mut(&self.current).expect("текущая ветка всегда существует в data")
    }
}

/// Агент — отдельная сущность, инкапсулирующая обращение к LLM через API.
/// Хранит свою конфигурацию и состояние запуска; пока агент не запущен,
/// обработка запросов отклоняется без обращения к API.
pub struct Agent {
    name: String,
    client: LlmClient,
    config: RwLock<AgentConfig>,
    running: AtomicBool,
    /// Накопленная история диалога, организованная по веткам (см. [`Branches`]),
    /// восстановленная из SQLite при создании агента. Каждое новое сообщение
    /// дописывается и в эту память, и в БД — без этого при перезапуске
    /// приложения агент забывал бы прошлые реплики и начинал диалог с чистого
    /// листа.
    branches: Mutex<Branches>,
    /// Сводка устаревшей части истории текущей ветки (стратегия `summary`),
    /// восстановленная из SQLite при создании агента.
    summary: Mutex<CompressionState>,
    /// Набор фактов (стратегия `facts`), восстановленный из SQLite при создании агента.
    facts: Mutex<FactsState>,
    // Ни долговременная, ни рабочая память НЕ кешируются здесь — обе общие
    // (долговременная — сразу для всех агентов, рабочая — для всех участников
    // задачи), поэтому любой другой агент мог изменить их только что; см.
    // crate::memory (module doc), Agent::long_term_memory и Agent::task_state,
    // которые читают их из БД заново при каждом обращении.
    db: Arc<Db>,
    /// MCP-серверы (см. [`crate::mcp`]) — общие для всех агентов: инструменты
    /// всех подключённых сейчас серверов предлагаются модели в каждом запросе.
    mcp: Arc<crate::mcp::McpManager>,
}

impl Agent {
    fn new(
        config: AgentConfig,
        client: LlmClient,
        db: Arc<Db>,
        mcp: Arc<crate::mcp::McpManager>,
        branches: Branches,
        summary: CompressionState,
        facts: FactsState,
    ) -> Self {
        let name = config.name.clone();
        Self {
            name,
            client,
            config: RwLock::new(config),
            running: AtomicBool::new(false),
            branches: Mutex::new(branches),
            summary: Mutex::new(summary),
            facts: Mutex::new(facts),
            db,
            mcp,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> AgentConfig {
        self.config.read().expect("конфигурация агента отравлена паникой").clone()
    }

    /// Заменяет конфигурацию агента (например, чтобы переключить стратегию
    /// управления контекстом на лету) и сохраняет её в БД.
    pub fn set_config(&self, config: AgentConfig) -> Result<()> {
        self.db.update_config(&self.name, &config)?;
        *self.config.write().expect("конфигурация агента отравлена паникой") = config;
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Текущая история диалога **активной ветки** (для отображения в
    /// интерфейсах, например при открытии чата с агентом после перезапуска
    /// приложения).
    pub fn history(&self) -> Vec<ChatMessage> {
        self.branches
            .lock()
            .expect("ветки агента отравлены паникой")
            .current_messages()
            .iter()
            .map(|(m, _)| m.clone())
            .collect()
    }

    /// История диалога активной ветки вместе с метриками токенов каждого
    /// сообщения — для веб-интерфейса, которому нужно показать расход токенов
    /// рядом с каждым запросом и ответом, а не только суммарно по диалогу.
    pub fn history_with_usage(&self) -> Vec<(ChatMessage, Option<Usage>)> {
        self.branches.lock().expect("ветки агента отравлены паникой").current_messages().clone()
    }

    /// Имя текущей активной ветки (всегда [`MAIN_BRANCH`], если стратегия
    /// [`ContextStrategy::Branching`] не активна).
    pub fn current_branch(&self) -> String {
        self.branches.lock().expect("ветки агента отравлены паникой").current.clone()
    }

    /// Список имён всех существующих веток, отсортированный.
    pub fn list_branches(&self) -> Vec<String> {
        let branches = self.branches.lock().expect("ветки агента отравлены паникой");
        let mut names: Vec<String> = branches.data.keys().cloned().collect();
        names.sort();
        names
    }

    /// Список имён всех сохранённых checkpoint'ов, отсортированный.
    pub fn list_checkpoints(&self) -> Vec<String> {
        let branches = self.branches.lock().expect("ветки агента отравлены паникой");
        let mut names: Vec<String> = branches.checkpoints.keys().cloned().collect();
        names.sort();
        names
    }

    /// Отмечает текущую позицию (текущую ветку + сколько сообщений в ней сейчас
    /// накоплено) как checkpoint с именем `label` — точку, от которой позже
    /// можно ответвиться через [`Agent::branch_from`]. Требует активную
    /// стратегию [`ContextStrategy::Branching`].
    pub fn checkpoint(&self, label: &str) -> Result<()> {
        self.require_branching_strategy()?;
        let label = label.trim().to_string();
        if label.is_empty() {
            bail!("имя checkpoint не может быть пустым");
        }
        let (branch, len) = {
            let branches = self.branches.lock().expect("ветки агента отравлены паникой");
            (branches.current.clone(), branches.current_messages().len())
        };
        self.branches
            .lock()
            .expect("ветки агента отравлены паникой")
            .checkpoints
            .insert(label.clone(), Checkpoint { branch: branch.clone(), len });
        if let Err(err) = self.db.save_checkpoint(&self.name, &label, &branch, len) {
            eprintln!("не удалось сохранить checkpoint «{label}» агента «{}» в БД: {err:#}", self.name);
        }
        Ok(())
    }

    /// Создаёт новую ветку `new_branch`, ответвляя её от `checkpoint` (история
    /// исходной ветки на момент checkpoint'а) или, если `checkpoint` не задан,
    /// от текущего конца активной ветки ("ответвиться прямо сейчас"). Не
    /// переключает на новую ветку автоматически — см. [`Agent::switch_branch`].
    /// Требует активную стратегию [`ContextStrategy::Branching`].
    pub fn branch_from(&self, checkpoint: Option<&str>, new_branch: &str) -> Result<()> {
        self.require_branching_strategy()?;
        let new_branch = new_branch.trim().to_string();
        if new_branch.is_empty() {
            bail!("имя ветки не может быть пустым");
        }

        let cloned = {
            let mut branches = self.branches.lock().expect("ветки агента отравлены паникой");
            if branches.data.contains_key(&new_branch) {
                bail!("ветка «{new_branch}» уже существует");
            }
            let (source_branch, len) = match checkpoint {
                Some(name) => {
                    let cp = branches
                        .checkpoints
                        .get(name)
                        .ok_or_else(|| anyhow!("checkpoint «{name}» не найден"))?;
                    (cp.branch.clone(), cp.len)
                }
                None => (branches.current.clone(), branches.current_messages().len()),
            };
            let source_messages = branches
                .data
                .get(&source_branch)
                .ok_or_else(|| anyhow!("ветка «{source_branch}» не найдена"))?;
            let cloned: Vec<(ChatMessage, Option<Usage>)> =
                source_messages.iter().take(len).cloned().collect();
            branches.data.insert(new_branch.clone(), cloned.clone());
            cloned
        };

        if let Err(err) = self.db.create_branch(&self.name, &new_branch, &cloned) {
            eprintln!("не удалось сохранить ветку «{new_branch}» агента «{}» в БД: {err:#}", self.name);
        }
        Ok(())
    }

    /// Переключает активную ветку на `name`. Требует активную стратегию
    /// [`ContextStrategy::Branching`].
    pub fn switch_branch(&self, name: &str) -> Result<()> {
        self.require_branching_strategy()?;
        {
            let mut branches = self.branches.lock().expect("ветки агента отравлены паникой");
            if !branches.data.contains_key(name) {
                bail!("ветка «{name}» не найдена");
            }
            branches.current = name.to_string();
        }
        if let Err(err) = self.db.save_current_branch(&self.name, name) {
            eprintln!("не удалось сохранить текущую ветку агента «{}» в БД: {err:#}", self.name);
        }
        Ok(())
    }

    // --- Персонализация (см. модуль crate::profile) --------------------------
    //
    // Отдельная от памяти ось: профиль описывает КАК агент должен отвечать
    // (манера общения, язык ответа, ограничения), а не что он знает. Хранится
    // не в SQLite, а markdown-файлом в каталоге профилей — конфигурация
    // агента содержит только ИМЯ профиля (`AgentConfig::profile`), поэтому
    // несколько агентов могут ссылаться на один и тот же файл без дублирования
    // текста. Ничего не кешируется в памяти процесса — читается с диска заново
    // при каждом обращении, как и долговременная/рабочая память, потому что
    // файл мог только что измениться вручную или через другого агента.

    /// Снимок статуса персонализации — имя используемого профиля, найден ли он
    /// на диске, его содержимое (если найден) и список всех доступных профилей.
    pub fn profile_status(&self) -> ProfileStatus {
        let name = crate::profile::resolve_name(self.config().profile.as_deref());
        let enabled = !crate::profile::is_disabled(&name);
        let content = if enabled { crate::profile::load(&name) } else { None };
        ProfileStatus { found: content.is_some(), name, enabled, content, available: crate::profile::list_profiles() }
    }

    // --- Долговременная память — ОБЩАЯ для всех агентов (см. модуль crate::memory) ---
    //
    // В отличие от рабочей памяти задачи (которую видят только присоединённые
    // к ней агенты), долговременная память не имеет вообще никакой изоляции —
    // это один-единственный набор ключ/значение на всё приложение. Метод всё
    // ещё вызывается через конкретного агента (`agent.remember(...)`), но имя
    // этого агента не участвует в хранении: это просто точка входа в общий
    // API, а не владелец данных. Ничего не кешируется в памяти процесса — как
    // и рабочая память задачи, читается из БД заново при каждом обращении,
    // потому что её мог только что изменить любой другой агент.

    /// Явно сохраняет (или обновляет) одну запись долговременной памяти —
    /// единственный способ туда что-то положить: ничего не пишется сюда
    /// автоматически по ответу модели, только по прямому вызову этого метода
    /// (за ним стоит команда `agent remember` в CLI и аналоги в других
    /// интерфейсах). Изменение сразу видно всем агентам, не только этому.
    pub fn remember(&self, key: &str, value: &str, category: &str) -> Result<()> {
        let key = key.trim();
        if key.is_empty() {
            bail!("ключ долговременной памяти не может быть пустым");
        }
        let category = if category.trim().is_empty() { "knowledge" } else { category.trim() };
        let item = LongTermItem { category: category.to_string(), value: value.to_string() };
        self.db.save_long_term(key, &item)?;
        Ok(())
    }

    /// Явно удаляет запись долговременной памяти (для всех агентов разом).
    /// Возвращает `true`, если она существовала.
    pub fn forget(&self, key: &str) -> Result<bool> {
        self.db.delete_long_term(key)
    }

    /// Снимок всей долговременной памяти — общей для всех агентов приложения.
    pub fn long_term_memory(&self) -> LongTermMemory {
        self.db.load_long_term().unwrap_or_default()
    }

    // --- Рабочая память ОБЩЕЙ задачи (см. модуль crate::memory) ----------------
    //
    // Задача — общая сущность, адресуемая по имени, а не приватная для этого
    // агента: task_start создаёт её и сразу присоединяет к ней вызывающего
    // агента, task_join присоединяет к уже существующей (созданной другим
    // агентом) задаче. Ничего не кешируется в памяти процесса — каждое чтение
    // идёт в БД заново (см. Db::agent_task_name/load_shared_task), потому что
    // данные могут в любой момент измениться другим участником той же задачи.

    /// Создаёт НОВУЮ общую задачу с этим именем и сразу присоединяет к ней
    /// вызывающего агента. Ошибка, если у агента уже есть активная задача
    /// (сначала завершите её — [`Agent::task_finish`]) или если задача с таким
    /// именем уже существует (тогда нужно [`Agent::task_join`], а не `task_start`).
    pub fn task_start(&self, name: &str, goal: Option<&str>) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            bail!("имя задачи не может быть пустым");
        }
        if let Some(current) = self.db.agent_task_name(&self.name)? {
            bail!(
                "у агента «{}» уже есть активная задача «{current}» — сначала завершите её, прежде чем \
                 начинать новую",
                self.name
            );
        }
        if self.db.shared_task_exists(name)? {
            bail!(
                "задача «{name}» уже существует — присоединитесь к ней вместо создания новой"
            );
        }
        self.db.create_shared_task(name, goal)?;
        self.db.set_agent_task(&self.name, name)?;
        Ok(())
    }

    /// Присоединяет вызывающего агента к УЖЕ СУЩЕСТВУЮЩЕЙ общей задаче (созданной
    /// им самим или другим агентом через [`Agent::task_start`]) — с этого момента
    /// он видит и меняет её рабочую память наравне со всеми остальными
    /// участниками. Ошибка, если у агента уже есть активная задача, или если
    /// задачи с таким именем не существует (тогда нужен [`Agent::task_start`]).
    pub fn task_join(&self, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            bail!("имя задачи не может быть пустым");
        }
        if let Some(current) = self.db.agent_task_name(&self.name)? {
            bail!(
                "у агента «{}» уже есть активная задача «{current}» — сначала завершите её, прежде чем \
                 присоединяться к другой",
                self.name
            );
        }
        if !self.db.shared_task_exists(name)? {
            bail!(
                "задача «{name}» не найдена — сначала создайте её"
            );
        }
        self.db.set_agent_task(&self.name, name)?;
        Ok(())
    }

    /// Явно сохраняет пару ключ/значение в рабочую память задачи, к которой
    /// сейчас присоединён агент — видна сразу всем остальным её участникам.
    /// Ошибка, если агент ни к какой задаче не присоединён.
    pub fn task_set(&self, key: &str, value: &str) -> Result<()> {
        let key = key.trim();
        if key.is_empty() {
            bail!("ключ рабочей памяти не может быть пустым");
        }
        let task_name = self.db.agent_task_name(&self.name)?.ok_or_else(|| {
            anyhow!(
                "у агента «{}» нет активной задачи — сначала создайте задачу или присоединитесь \
                 к существующей",
                self.name
            )
        })?;
        self.db.save_shared_task_data(&task_name, key, value)?;
        Ok(())
    }

    /// Явно сохраняет (или обновляет) текстовый инвариант, привязанный
    /// ТОЛЬКО к задаче, к которой сейчас присоединён агент (см.
    /// [`crate::memory::TaskState::invariants`] и
    /// [`crate::memory::format_task_invariants_block`]) — обязателен наравне
    /// с глобальными инвариантами ([`crate::invariants`]), но действует и
    /// виден только пока идёт работа над этой конкретной задачей: снимается
    /// сам собой при `task finish`, без отдельной очистки. Ошибка, если
    /// агент ни к какой задаче не присоединён.
    pub fn task_invariant_set(&self, id: &str, text: &str) -> Result<()> {
        let id = id.trim();
        if id.is_empty() {
            bail!("идентификатор инварианта задачи не может быть пустым");
        }
        let text = text.trim();
        if text.is_empty() {
            bail!("текст инварианта задачи не может быть пустым");
        }
        let task_name = self.current_task_name()?;
        self.db.save_task_invariant(&task_name, id, text)?;
        Ok(())
    }

    /// Удаляет инвариант задачи, к которой сейчас присоединён агент.
    /// Возвращает `true`, если он существовал.
    pub fn task_invariant_remove(&self, id: &str) -> Result<bool> {
        let task_name = self.current_task_name()?;
        self.db.delete_task_invariant(&task_name, id)
    }

    /// Снимок задачи, к которой сейчас присоединён агент, — читается из БД
    /// заново при каждом вызове (см. документацию поля [`Agent`] выше), `None`,
    /// если агент ни к какой задаче не присоединён.
    pub fn task_state(&self) -> Option<TaskState> {
        let task_name = self.db.agent_task_name(&self.name).ok().flatten()?;
        self.db.load_shared_task(&task_name).ok().flatten()
    }

    /// Завершает задачу, к которой присоединён агент, — целиком, **для всех**
    /// её участников разом (не только для вызывающего агента): её рабочая
    /// память безвозвратно удаляется, а membership всех агентов, что были к
    /// ней присоединены, снимается автоматически (`ON DELETE CASCADE`, см.
    /// схему `Db`). Возвращает снимок задачи, каким он был перед завершением —
    /// если что-то из него должно пережить задачу, это нужно явно перенести
    /// через [`Agent::remember`] *до* вызова этого метода. `Ok(None)`, если
    /// агент ни к какой задаче не был присоединён.
    pub fn task_finish(&self) -> Result<Option<TaskState>> {
        let Some(task_name) = self.db.agent_task_name(&self.name)? else {
            return Ok(None);
        };
        let snapshot = self.db.load_shared_task(&task_name)?;
        self.db.delete_shared_task(&task_name)?;
        Ok(snapshot)
    }

    // --- Конечный автомат задачи (Task State Machine, см. crate::memory::Stage) ---
    //
    // Слой поверх рабочей памяти задачи выше: та же общая задача, к которой
    // присоединён агент, получает формализованное состояние — этап, текущий
    // шаг и ожидаемое действие. Переходы между этапами не произвольны (см.
    // Stage::allowed_next), пауза же независима от этапа и ставится на любом
    // из них. Всё читается из БД заново при каждом обращении, как и остальная
    // рабочая память — см. Agent::task_state.

    fn current_task_name(&self) -> Result<String> {
        self.db.agent_task_name(&self.name)?.ok_or_else(|| {
            anyhow!(
                "у агента «{}» нет активной задачи — сначала создайте задачу или присоединитесь \
                 к существующей",
                self.name
            )
        })
    }

    /// Переводит задачу, к которой присоединён агент, на следующий этап
    /// конечного автомата — только если переход легален (см.
    /// [`Stage::allowed_next`]); иначе — ошибка с перечислением того, куда
    /// можно перейти прямо сейчас. Задачу нельзя двигать по этапам, пока она
    /// на паузе — сначала [`Agent::task_resume`]. `step`/`expected_action`,
    /// если заданы, обновляются вместе с переходом (см. [`Agent::task_step`]/
    /// [`Agent::task_expect`] для правки без смены этапа).
    pub fn task_advance(&self, stage: Stage, step: Option<&str>, expected_action: Option<&str>) -> Result<()> {
        let task_name = self.current_task_name()?;
        let current = self.db.load_shared_task(&task_name)?.ok_or_else(|| {
            anyhow!("задача «{task_name}» не найдена (была удалена параллельно?)")
        })?;
        let changes_stage = current.stage != stage;
        let refusal = if current.paused {
            Some(format!(
                "задача «{task_name}» на паузе — сначала возобновите её, прежде чем переходить на другой этап"
            ))
        } else if changes_stage && !current.stage.allowed_next().contains(&stage) {
            let allowed = current.stage.allowed_next();
            let options = if allowed.is_empty() {
                "нет — это конечный этап".to_string()
            } else {
                allowed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            };
            Some(format!(
                "нельзя перейти из этапа «{}» сразу в «{stage}» — допустимые следующие этапы: {options}",
                current.stage
            ))
        // Абсолютный запрет инвариантом ЭТОЙ задачи (см.
        // TaskState::blocked_transitions) — в отличие от гейтов согласия,
        // действует одинаково и на ручной переход человеком, и на переход,
        // предложенный моделью (см. tool_move_stage): снять его может только
        // человек, а не сам этот вызов.
        } else if changes_stage && current.transition_blocked(current.stage, stage) {
            Some(format!(
                "переход «{}» -> «{stage}» запрещён инвариантом задачи «{task_name}» (не временное \
                 ограничение, а прямой запрет) — сначала снимите запрет на этот переход",
                current.stage
            ))
        } else {
            None
        };
        if let Some(reason) = refusal {
            if changes_stage {
                self.db.log_transition(
                    &task_name,
                    current.stage,
                    stage,
                    TransitionActor::Human,
                    TransitionOutcome::Refused,
                    &reason,
                )?;
            }
            bail!(reason);
        }
        self.db.save_task_stage(&task_name, stage, step, expected_action)?;
        if !changes_stage {
            return Ok(());
        }
        self.db.log_transition(
            &task_name,
            current.stage,
            stage,
            TransitionActor::Human,
            TransitionOutcome::Applied,
            "ручной переход",
        )?;
        // Ручной переход — решение человека, и оно заменяет ждущее утверждения
        // предложение модели: сделанное с прежнего этапа, оно устарело, а его
        // поздний approve увёл бы задачу туда, откуда её уже провели дальше.
        if let Some(pending) = current.pending_stage {
            self.db.clear_task_pending(&task_name)?;
            self.db.log_transition(
                &task_name,
                current.stage,
                pending,
                TransitionActor::Human,
                TransitionOutcome::Rejected,
                &format!("предложение снято: человек сам перевёл задачу в «{stage}»"),
            )?;
        }
        Ok(())
    }

    /// Дополнительно ЗАПРЕЩАЕТ переход `from -> to` для задачи, к которой
    /// присоединён агент, сверх общей карты [`Stage::allowed_next`] (см.
    /// [`crate::memory::TaskState::blocked_transitions`]) — абсолютный
    /// запрет, действует и на [`Agent::task_advance`], и на переход,
    /// предложенный моделью.
    pub fn task_forbid_transition(&self, from: Stage, to: Stage) -> Result<()> {
        let task_name = self.current_task_name()?;
        self.db.save_task_transition_rule(&task_name, from, to, "block")
    }

    /// Снимает запрет, поставленный [`Agent::task_forbid_transition`].
    /// Возвращает `true`, если он существовал.
    pub fn task_allow_transition(&self, from: Stage, to: Stage) -> Result<bool> {
        let task_name = self.current_task_name()?;
        self.db.delete_task_transition_rule(&task_name, from, to, "block")
    }

    /// Требует для этой задачи подтверждения человеком на переход `from ->
    /// to` ДОПОЛНИТЕЛЬНО к двум переходам, гейтящимся всегда (см.
    /// [`crate::memory::TaskState::extra_approval_transitions`]) — действует
    /// только на переход, который предлагает МОДЕЛЬ (`move_stage`); ручной
    /// [`Agent::task_advance`] — это и так решение человека.
    pub fn task_require_approval(&self, from: Stage, to: Stage) -> Result<()> {
        let task_name = self.current_task_name()?;
        self.db.save_task_transition_rule(&task_name, from, to, "approve")
    }

    /// Снимает дополнительный гейт, поставленный [`Agent::task_require_approval`].
    /// Возвращает `true`, если он существовал.
    pub fn task_unrequire_approval(&self, from: Stage, to: Stage) -> Result<bool> {
        let task_name = self.current_task_name()?;
        self.db.delete_task_transition_rule(&task_name, from, to, "approve")
    }

    /// Обновляет текущий шаг задачи, не меняя этап.
    pub fn task_step(&self, step: &str) -> Result<()> {
        let task_name = self.current_task_name()?;
        let stage = self.db.load_shared_task(&task_name)?.map(|t| t.stage).unwrap_or_default();
        self.db.save_task_stage(&task_name, stage, Some(step), None)?;
        Ok(())
    }

    /// Обновляет ожидаемое действие задачи, не меняя этап.
    pub fn task_expect(&self, expected_action: &str) -> Result<()> {
        let task_name = self.current_task_name()?;
        let stage = self.db.load_shared_task(&task_name)?.map(|t| t.stage).unwrap_or_default();
        self.db.save_task_stage(&task_name, stage, None, Some(expected_action))?;
        Ok(())
    }

    /// Ставит задачу на паузу — на ЛЮБОМ этапе, не меняя его. С этого момента
    /// [`crate::memory::format_stage_block`] добавляет в каждый запрос к LLM
    /// явную пометку "на паузе", чтобы модель не предпринимала новых действий
    /// по задаче до [`Agent::task_resume`].
    pub fn task_pause(&self) -> Result<()> {
        let task_name = self.current_task_name()?;
        self.db.save_task_paused(&task_name, true)?;
        Ok(())
    }

    /// Снимает паузу — задача продолжается с того же этапа/шага/ожидаемого
    /// действия, на которых была приостановлена: ничего из этого не терялось
    /// (хранилось в БД, не в памяти процесса), поэтому агенту не нужно заново
    /// объяснять контекст — он снова придёт в каждом запросе автоматически,
    /// вместе с итогами пройденных этапов (`TaskState::stage_path`).
    ///
    /// Возвращает служебное продолжение, которое интерфейс сразу отправляет
    /// агенту через [`Agent::continue_task`] (в чате его не видно): ответить на
    /// запрос, ответ на который оборвала пауза ([`TaskState::interrupted_prompt`],
    /// сам запрос уже в истории), иначе — продолжить незавершённый этап
    /// ([`crate::memory::resume_followup`]);
    /// `None` — если задача не была на паузе или агенту сейчас нечего делать
    /// самому (задача завершена или ждёт решения человека по переходу).
    pub fn task_resume(&self) -> Result<Option<String>> {
        let task_name = self.current_task_name()?;
        let current = self
            .db
            .load_shared_task(&task_name)?
            .ok_or_else(|| anyhow!("задача «{task_name}» не найдена (была удалена параллельно?)"))?;
        if !current.paused {
            return Ok(None);
        }
        self.db.save_task_paused(&task_name, false)?;
        // Ответ на запрос человека оборвала пауза — агент отвечает на него.
        if let Some(prompt) = current.interrupted_prompt {
            self.db.save_task_interrupted_prompt(&task_name, None)?;
            return Ok(Some(crate::memory::interrupted_request_followup(&prompt)));
        }
        Ok(crate::memory::resume_followup(&current))
    }

    /// Применяет переход, который модель предложила вызовом `move_stage`, но
    /// который требовал утверждения человеком (см. [`Stage::requires_approval_to`],
    /// `TaskState::pending_stage`) — единственный способ провести такой переход.
    /// Ошибка, если утверждать сейчас нечего или задача на паузе — как и
    /// [`Agent::task_advance`], утверждение сдвигает этап, а на паузе автомат
    /// заморожен: сначала [`Agent::task_resume`] (предложение при этом не теряется).
    /// Ошибка и тогда, когда после предложения переход запретили инвариантом
    /// задачи ([`Agent::task_forbid_transition`]).
    ///
    /// Возвращает служебное продолжение ([`crate::memory::approval_followup`]),
    /// которое интерфейс сразу отправляет агенту через [`Agent::continue_task`],
    /// чтобы работа на новом этапе началась без ещё одного ручного сообщения.
    pub fn task_approve(&self) -> Result<String> {
        let task_name = self.current_task_name()?;
        let current = self
            .db
            .load_shared_task(&task_name)?
            .ok_or_else(|| anyhow!("задача «{task_name}» не найдена (была удалена параллельно?)"))?;
        let Some(pending) = current.pending_stage else {
            bail!("у задачи «{task_name}» нет перехода, ожидающего утверждения");
        };
        if current.paused {
            let reason = format!(
                "задача «{task_name}» на паузе — сначала возобновите её, прежде чем утверждать переход \
                 (предложение сохранено и никуда не денется)"
            );
            self.db.log_transition(
                &task_name,
                current.stage,
                pending,
                TransitionActor::Human,
                TransitionOutcome::Refused,
                &reason,
            )?;
            bail!(reason);
        }
        // Предложение проверялось в момент вызова move_stage, но с тех пор человек
        // мог запретить этот переход инвариантом задачи — утверждение не должно
        // обходить такой запрет. Предложение при отказе не снимается: его можно
        // отклонить (reject) или, сняв запрет, утвердить. (Этап же под
        // предложением сменить нельзя — task_advance снимает предложение сам.)
        if current.transition_blocked(current.stage, pending) {
            let reason = format!(
                "переход «{}» -> «{pending}» запрещён инвариантом задачи «{task_name}» — утвердить его нельзя; \
                 отклоните предложение или сначала снимите запрет на этот переход",
                current.stage
            );
            self.db.log_transition(
                &task_name,
                current.stage,
                pending,
                TransitionActor::Human,
                TransitionOutcome::Refused,
                &reason,
            )?;
            bail!(reason);
        }
        self.db.save_task_stage(&task_name, pending, None, None)?;
        self.db.clear_task_pending(&task_name)?;
        self.db.log_transition(
            &task_name,
            current.stage,
            pending,
            TransitionActor::Human,
            TransitionOutcome::Approved,
            current.pending_outcome.as_deref().unwrap_or(""),
        )?;
        Ok(crate::memory::approval_followup(current.stage, pending))
    }

    /// Отклоняет предложенный моделью переход — этап остаётся прежним,
    /// `pending_stage`/`pending_outcome` снимаются. Причина обязательна: это
    /// единственное, из чего модель узнаёт, что исправлять, — она ложится в
    /// журнал переходов, а блок статуса задачи (см.
    /// [`crate::memory::format_task_block`]) показывает модели последнее
    /// отклонение, пока она не предложит переход заново. В «Ожидаемое
    /// действие» причина не пишется: после пересмотра она осталась бы там
    /// устаревшей, а журнал хранит её и так.
    ///
    /// Возвращает сообщение агенту (см. [`crate::memory::rejection_followup`]),
    /// которое интерфейс отправляет от имени человека сразу после отклонения,
    /// чтобы этап был переработан без ещё одного ручного сообщения. Ошибка, если
    /// отклонять сейчас нечего или причина пустая.
    pub fn task_reject(&self, note: &str) -> Result<String> {
        let task_name = self.current_task_name()?;
        let current = self
            .db
            .load_shared_task(&task_name)?
            .ok_or_else(|| anyhow!("задача «{task_name}» не найдена (была удалена параллельно?)"))?;
        let Some(pending) = current.pending_stage else {
            bail!("у задачи «{task_name}» нет перехода, ожидающего утверждения");
        };
        let note = note.trim();
        if note.is_empty() {
            bail!("укажите причину отклонения — без неё агенту нечего исправлять");
        }
        self.db.clear_task_pending(&task_name)?;
        self.db.log_transition(
            &task_name,
            current.stage,
            pending,
            TransitionActor::Human,
            TransitionOutcome::Rejected,
            note,
        )?;
        Ok(crate::memory::rejection_followup(current.stage, pending, note))
    }

    /// Описания инструментов автомата задачи (function calling) — единственная
    /// инструкция модели о том, как ими пользоваться (см. документацию
    /// [`crate::memory::Stage::directive`]): текст описания стабилен между
    /// запросами (в отличие от блока статуса, см. [`crate::memory::format_stage_block`]),
    /// поэтому не мешает кэшированию промпта на стороне провайдера.
    fn task_tool_definitions() -> Vec<crate::ToolDefinition> {
        vec![
            crate::ToolDefinition {
                name: "move_stage".to_string(),
                description:
                    "Предложить переход конечного автомата задачи на другой этап (planning/execution/\
                     validation/done). Программа сверяет переход с картой — легально только: planning->execution, \
                     execution->validation, execution->planning (пересмотр плана), validation->execution \
                     (доработка), validation->done. НЕЛЬЗЯ прыгать \
                     через этап (например, из planning сразу в done, минуя execution/validation) — такой вызов \
                     будет отклонён с объяснением, куда можно перейти прямо сейчас; если это случилось, вызови \
                     move_stage ЕЩЁ РАЗ с тем этапом, что назван в объяснении как легальный, а не бросай работу \
                     над задачей на середине. Переходы planning->execution и validation->done требуют \
                     подтверждения человека: вызов НЕ применит переход сразу, а поставит его в ожидание — после \
                     такого вызова остановись, сообщи человеку, что план/итог готов и ждёт его решения, и НЕ \
                     продолжай работу дальше как будто переход уже произошёл. Переходы execution->validation, \
                     execution->planning и validation->execution применяются сразу. КОНКРЕТНАЯ ЗАДАЧА может дополнительно (см. блок \
                     «Состояние задачи» в этом же запросе) запрещать отдельные переходы совсем (тогда вызов \
                     отклоняется — это инвариант, не обойти) или требовать подтверждения человека там, где по \
                     умолчанию оно не нужно (тогда вызов ставится в ожидание, как и planning->execution/validation->done)."
                        .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "stage": {
                            "type": "string",
                            "enum": ["planning", "execution", "validation", "done"],
                            "description": "Целевой этап перехода."
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Итог текущего этапа — что сделано или к чему пришли, коротко."
                        }
                    },
                    "required": ["stage", "outcome"]
                }),
            },
            crate::ToolDefinition {
                name: "update_step".to_string(),
                description: "Обновить текущий шаг и/или ожидаемое действие задачи, не меняя этап — используй, \
                     чтобы отметить прогресс внутри этапа."
                    .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "description": "Текущий шаг работы."},
                        "expected_action": {
                            "type": "string",
                            "description": "Чего ожидается дальше (необязательно)."
                        }
                    },
                    "required": ["step"]
                }),
            },
        ]
    }

    /// Крутит обмен через круги "запрос с инструментами -> (вызовы инструментов
    /// -> результаты) -> запрос снова", пока модель не ответит текстом без
    /// вызовов — см. документацию [`crate::RequestMessage`] за тем, почему
    /// промежуточные сообщения этого круга никогда не персистятся в историю
    /// диалога: они существуют только в границах этого одного обмена. Число
    /// кругов ограничено [`MAX_TOOL_ROUNDS`] — если модель зациклилась на
    /// вызовах, последний запрос идёт уже без инструментов, чтобы вынудить
    /// текстовый ответ, а не оставить пользователя без ответа вовсе.
    ///
    /// Первый круг идёт с `require_tool_call = true` (`tool_choice: "required"`) —
    /// живой прогон против реальной модели (локальный Qwen3 через LM Studio)
    /// показал, что с `"auto"` модель нередко просто решает задачу текстом в
    /// первом же ответе, не пытаясь вызвать ни одного инструмента, несмотря на
    /// прямую инструкцию в [`crate::memory::Stage::directive`] — то есть именно
    /// то поведение, из-за которого автомат вообще делался. Принуждение к вызову
    /// действует только на первом круге: получив результат вызова, модель на
    /// следующем круге (`"auto"`) уже свободно отвечает текстом — иначе она не
    /// смогла бы задать пользователю уточняющий вопрос или сообщить, что ждёт
    /// утверждения перехода.
    async fn run_tool_loop(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        task_tools: bool,
        mcp_tools: Vec<crate::ToolDefinition>,
    ) -> Result<(crate::ChatCompletion, Vec<ToolCallRecord>)> {
        let mut tools = if task_tools { Self::task_tool_definitions() } else { Vec::new() };
        tools.extend(mcp_tools);
        let mut wire: Vec<crate::RequestMessage> = messages.iter().map(crate::RequestMessage::from).collect();
        let mut usage_total: Option<Usage> = None;
        let mut records: Vec<ToolCallRecord> = Vec::new();
        // Модель нередко пишет содержательный текст (например, сам черновик)
        // В ТОМ ЖЕ круге, где вызывает инструмент — например, если предложенный
        // переход отклонён программой, следующий круг ссылается на этот текст
        // ("письмо выше"), не повторяя его. Если сохранять только контент
        // последнего круга, такой текст молча теряется и пользователь видит
        // ссылку на текст, которого не видел. Поэтому копится контент КАЖДОГО
        // круга с непустым текстом, а не только финального.
        let mut visible_parts: Vec<String> = Vec::new();

        for round in 0..MAX_TOOL_ROUNDS {
            // Принуждение к вызову нужно только автомату задачи (см. документацию
            // выше): без задачи, с одними MCP-инструментами, модель сама решает,
            // нужен ли ей инструмент для ответа.
            let require_tool_call = round == 0 && task_tools;
            let completion = self.client.chat_with_tools(model, &wire, &tools, require_tool_call, options).await?;
            usage_total = sum_usage(usage_total, completion.usage);
            if !completion.content.trim().is_empty() {
                visible_parts.push(completion.content.clone());
            }

            if completion.tool_calls.is_empty() {
                // Живой прогон показал: после отклонения модель переписывает
                // план по замечанию, но может так и не вызвать move_stage —
                // тогда человеку нечего утверждать, а модель просит
                // «подтвердить». Один принудительный круг, где доступен
                // только move_stage, доводит переработку до нового предложения.
                if task_tools && self.awaiting_reproposal() {
                    wire.push(crate::RequestMessage::from(&ChatMessage::assistant(completion.content.clone())));
                    wire.push(crate::RequestMessage::from(&ChatMessage::user(
                        "[Автомат задачи] Ты переработал результат этапа по замечанию, но не предложил переход \
                         заново — человеку нечего утверждать. Вызови move_stage: в outcome — кратко переработанный \
                         результат и что изменено по замечанию.",
                    )));
                    let move_stage_only: Vec<_> =
                        tools.iter().filter(|t| t.name == "move_stage").cloned().collect();
                    let forced =
                        self.client.chat_with_tools(model, &wire, &move_stage_only, true, options).await?;
                    usage_total = sum_usage(usage_total, forced.usage);
                    for call in &forced.tool_calls {
                        records.push(self.execute_tool(call).await);
                    }
                }
                let completion = crate::ChatCompletion {
                    content: visible_parts.join("\n\n"),
                    usage: usage_total,
                    tool_calls: Vec::new(),
                    ..completion
                };
                return Ok((completion, records));
            }

            wire.push(crate::RequestMessage::assistant_tool_calls(
                completion.content.clone(),
                completion.tool_calls.clone(),
            ));
            for call in &completion.tool_calls {
                let record = self.execute_tool(call).await;
                wire.push(crate::RequestMessage::tool_result(call.id.clone(), record.result.clone()));
                records.push(record);
            }
        }

        // Предел кругов достигнут (см. документацию выше) — принудительно просим
        // текстовый ответ без инструментов, чтобы обмен не завершился без ответа.
        let completion = self.client.chat_with_tools(model, &wire, &[], false, options).await?;
        usage_total = sum_usage(usage_total, completion.usage);
        if !completion.content.trim().is_empty() {
            visible_parts.push(completion.content.clone());
        }
        let completion = crate::ChatCompletion {
            content: visible_parts.join("\n\n"),
            usage: usage_total,
            tool_calls: Vec::new(),
            ..completion
        };
        Ok((completion, records))
    }

    /// Выполняет один вызов инструмента моделью: инструмент MCP-сервера уходит
    /// на свой сервер (см. [`crate::mcp::McpManager::call`]), остальное —
    /// встроенные инструменты автомата задачи.
    async fn execute_tool(&self, call: &crate::ToolCallWire) -> ToolCallRecord {
        let arguments = call.function.arguments.clone();
        if let Some(result) = self.mcp.call(&call.function.name, &arguments).await {
            return ToolCallRecord {
                server: Some(result.server),
                tool: result.tool,
                arguments,
                result: result.text,
                is_error: result.is_error,
            };
        }
        let is_known = matches!(call.function.name.as_str(), "move_stage" | "update_step");
        ToolCallRecord {
            server: None,
            tool: call.function.name.clone(),
            result: self.execute_task_tool(call),
            arguments,
            is_error: !is_known,
        }
    }

    /// `true`, если человек отклонил переход с текущего этапа, а модель ещё не
    /// предложила его заново (см. [`crate::memory::active_rejection`]) — и
    /// автомат сейчас позволяет ей это сделать.
    fn awaiting_reproposal(&self) -> bool {
        self.task_state().is_some_and(|task| {
            task_tools_offered(Some(&task)) && crate::memory::active_rejection(&task).is_some()
        })
    }

    /// Выполняет один вызов инструмента модели, разбирая доводы как JSON —
    /// никогда не паникует на данных модели: неразбираемые или некорректные
    /// доводы дают текстовый результат с объяснением, а не ошибку выполнения.
    fn execute_task_tool(&self, call: &crate::ToolCallWire) -> String {
        match call.function.name.as_str() {
            "move_stage" => self.tool_move_stage(&call.function.arguments),
            "update_step" => self.tool_update_step(&call.function.arguments),
            other => format!("нет такого инструмента: {other}"),
        }
    }

    fn tool_move_stage(&self, arguments: &str) -> String {
        let value: serde_json::Value = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(v) if v.is_object() => v,
            _ => return "доводы не разобраны: ожидался JSON-объект с полями stage и outcome".to_string(),
        };
        let Some(stage_raw) = value.get("stage").and_then(|v| v.as_str()) else {
            return "доводы не разобраны: отсутствует поле stage".to_string();
        };
        let outcome = value.get("outcome").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if outcome.is_empty() {
            return "доводы не разобраны: поле outcome не может быть пустым".to_string();
        }
        let stage: Stage = match stage_raw.parse() {
            Ok(s) => s,
            Err(err) => return format!("{err:#}"),
        };

        let task_name = match self.current_task_name() {
            Ok(name) => name,
            Err(err) => return format!("{err:#}"),
        };
        let current = match self.db.load_shared_task(&task_name) {
            Ok(Some(t)) => t,
            Ok(None) => return format!("задачи «{task_name}» больше нет — вызов не применён"),
            Err(err) => return format!("не удалось прочитать задачу: {err:#}"),
        };
        // Журнал — лучшая попытка: результат вызова модели не должен теряться
        // из-за сбоя записи, как и с историей диалога в handle_request.
        let log = |outcome_kind: TransitionOutcome, reason: &str| {
            if let Err(err) =
                self.db.log_transition(&task_name, current.stage, stage, TransitionActor::Model, outcome_kind, reason)
            {
                eprintln!("не удалось записать переход в журнал задачи: {err:#}");
            }
        };
        let allowed = current.stage.allowed_next();
        let refusal = if current.paused {
            Some("задача на паузе — вызовы инструментов сейчас не обрабатываются".to_string())
        } else if let Some(pending) = current.pending_stage {
            Some(format!("уже ждёт утверждения перехода в «{pending}» — дождись решения человека"))
        } else if current.stage == stage || !allowed.contains(&stage) {
            Some(if allowed.is_empty() {
                format!("нет такого перехода: этап «{}» конечный", current.stage)
            } else {
                format!(
                    "нет такого перехода: из «{}» можно в {}",
                    current.stage,
                    allowed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                )
            })
        // Абсолютный запрет инвариантом ЭТОЙ задачи (см.
        // TaskState::blocked_transitions) — в отличие от гейта согласия ниже,
        // модель не может провести этот переход НИКАК, даже поставив его в
        // ожидание: снять запрет может только человек (agent task ... allow).
        } else if current.transition_blocked(current.stage, stage) {
            Some(format!(
                "нет такого перехода: «{}» -> «{stage}» запрещён инвариантом ЭТОЙ задачи — это прямой \
                 запрет, а не гейт согласия, снять его может только человек",
                current.stage
            ))
        } else {
            None
        };
        if let Some(reason) = refusal {
            log(TransitionOutcome::Refused, &reason);
            return reason;
        }

        if current.transition_requires_approval(current.stage, stage) {
            if let Err(err) = self.db.save_task_pending(&task_name, stage, &outcome) {
                return format!("не удалось сохранить предложение перехода: {err:#}");
            }
            log(TransitionOutcome::Proposed, &outcome);
            format!(
                "Переход в «{stage}» предложен и ждёт подтверждения человека — он может утвердить переход или \
                 отклонить его с причиной. Остановись, сообщи об этом человеку и жди — не веди себя так, будто \
                 переход уже произошёл. {}",
                crate::memory::APPROVAL_IS_NOT_A_CHAT_MESSAGE
            )
        } else {
            let previous = current.stage;
            if let Err(err) = self.db.save_task_stage(&task_name, stage, None, None) {
                return format!("не удалось применить переход: {err:#}");
            }
            log(TransitionOutcome::Applied, &outcome);
            format!("Переход применён: этап теперь «{stage}» (было «{previous}»). Итог: {outcome}")
        }
    }

    /// Если `prompt` просит перескочить обязательный этап (см.
    /// [`crate::memory::skip_request_target`]), записывает отказ в журнал
    /// переходов и возвращает текст ответа, который уйдёт пользователю вместо
    /// ответа модели. На паузе не срабатывает — там модель и так ничего не
    /// делает по задаче (см. [`crate::memory::format_task_block`]).
    fn refuse_stage_skip(&self, task: &TaskState, prompt: &str) -> Option<String> {
        if task.paused {
            return None;
        }
        let target = crate::memory::skip_request_target(task.stage, prompt)?;
        let reply = match task.stage {
            Stage::Planning => "⛔ Этап планирования пропустить нельзя: реализация начинается только после \
                 утверждённого плана (сейчас этап «planning»). Опишите требования или попросите предложить план — \
                 когда я предложу переход в execution, утвердите его. Если план действительно не нужен, переведите \
                 задачу в execution вручную."
                .to_string(),
            _ => format!(
                "⛔ Завершить задачу без проверки нельзя: из «{}» путь только в validation, а в done — только \
                 после проверки и подтверждения человеком. Когда реализация будет готова, я переведу задачу на \
                 проверку.",
                task.stage
            ),
        };
        if let Err(err) = self.db.log_transition(
            &task.name,
            task.stage,
            target,
            TransitionActor::Human,
            TransitionOutcome::Refused,
            "просьба в сообщении перескочить этап — перехвачена до обращения к модели",
        ) {
            eprintln!("не удалось записать переход в журнал задачи: {err:#}");
        }
        Some(reply)
    }

    fn tool_update_step(&self, arguments: &str) -> String {
        let value: serde_json::Value = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(v) if v.is_object() => v,
            _ => return "доводы не разобраны: ожидался JSON-объект с полем step".to_string(),
        };
        let Some(step) = value.get("step").and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty())
        else {
            return "доводы не разобраны: поле step не может быть пустым".to_string();
        };
        let expected_action =
            value.get("expected_action").and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty());

        let task_name = match self.current_task_name() {
            Ok(name) => name,
            Err(err) => return format!("{err:#}"),
        };
        let current = match self.db.load_shared_task(&task_name) {
            Ok(Some(t)) => t,
            Ok(None) => return format!("задачи «{task_name}» больше нет — вызов не применён"),
            Err(err) => return format!("не удалось прочитать задачу: {err:#}"),
        };
        if current.paused {
            return "задача на паузе — вызовы инструментов сейчас не обрабатываются".to_string();
        }
        if let Err(err) = self.db.save_task_stage(&task_name, current.stage, Some(step), expected_action) {
            return format!("не удалось обновить шаг: {err:#}");
        }
        "Шаг обновлён.".to_string()
    }

    fn require_branching_strategy(&self) -> Result<()> {
        if self.config().context_strategy != ContextStrategy::Branching {
            bail!(
                "ветки доступны только при стратегии контекста «branching» \
                 (сейчас у агента «{}» другая стратегия)",
                self.name
            );
        }
        Ok(())
    }

    /// Запускает агента. История диалога не сбрасывается — она хранится в
    /// SQLite и переживает и остановку/запуск, и перезапуск всего приложения,
    /// так что диалог продолжается так, будто агент не выключался.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Размер контекстного окна (см. [`LlmClient::context_window`]).
    pub fn context_window(&self) -> u32 {
        self.client.context_window()
    }

    pub fn info(&self) -> AgentInfo {
        let config = self.config();
        let compression = self.compression_status(&config);
        let sliding_window = self.sliding_window_status(&config);
        let facts = self.facts_status(&config);
        let branching = self.branching_status(&config);
        let long_term = self.long_term_memory();
        let task = self.task_state();
        let profile = self.profile_status();
        AgentInfo {
            running: self.is_running(),
            context_window: self.context_window(),
            compression,
            sliding_window,
            facts,
            branching,
            long_term,
            task,
            profile,
            config,
        }
    }

    /// Считает [`CompressionInfo`] по текущей истории активной ветки и сводке —
    /// `None`, если активна не стратегия [`ContextStrategy::Summary`]. Счёт
    /// ведётся в сообщениях пользователя (обменах), а не в сырых записях
    /// истории — см. [`context::RAW_MESSAGES_PER_EXCHANGE`].
    fn compression_status(&self, config: &AgentConfig) -> Option<CompressionInfo> {
        if config.context_strategy != ContextStrategy::Summary {
            return None;
        }
        let history_len =
            self.branches.lock().expect("ветки агента отравлены паникой").current_messages().len();
        let summarized_count_raw =
            self.summary.lock().expect("сводка агента отравлена паникой").summarized_count;
        let pending = history_len.saturating_sub(summarized_count_raw) / context::RAW_MESSAGES_PER_EXCHANGE;
        Some(CompressionInfo {
            summarized_count: summarized_count_raw / context::RAW_MESSAGES_PER_EXCHANGE,
            pending_messages: pending,
            messages_until_summary: context::context_summary_chunk().saturating_sub(pending),
        })
    }

    /// Считает [`SlidingWindowInfo`] — `None`, если активна не `SlidingWindow`/`Facts`.
    fn sliding_window_status(&self, config: &AgentConfig) -> Option<SlidingWindowInfo> {
        if !matches!(config.context_strategy, ContextStrategy::SlidingWindow | ContextStrategy::Facts) {
            return None;
        }
        let window = config.window_size.unwrap_or_else(context::sliding_window_size);
        let total_raw =
            self.branches.lock().expect("ветки агента отравлены паникой").current_messages().len();
        let total = total_raw / context::RAW_MESSAGES_PER_EXCHANGE;
        let kept = total.min(window);
        Some(SlidingWindowInfo {
            window_size: window,
            total_messages: total,
            kept_messages: kept,
            dropped_messages: total.saturating_sub(kept),
        })
    }

    /// Возвращает текущий набор фактов — `None`, если активна не стратегия `Facts`.
    fn facts_status(&self, config: &AgentConfig) -> Option<FactsInfo> {
        if config.context_strategy != ContextStrategy::Facts {
            return None;
        }
        let window = config.window_size.unwrap_or_else(context::sliding_window_size);
        let facts = self.facts.lock().expect("факты агента отравлены паникой").facts.clone();
        Some(FactsInfo { facts, window_size: window })
    }

    /// Возвращает статус веток — `None`, если активна не стратегия `Branching`.
    fn branching_status(&self, config: &AgentConfig) -> Option<BranchingInfo> {
        if config.context_strategy != ContextStrategy::Branching {
            return None;
        }
        let branches = self.branches.lock().expect("ветки агента отравлены паникой");
        let mut names: Vec<String> = branches.data.keys().cloned().collect();
        names.sort();
        let mut checkpoints: Vec<String> = branches.checkpoints.keys().cloned().collect();
        checkpoints.sort();
        Some(BranchingInfo { current_branch: branches.current.clone(), branches: names, checkpoints })
    }

    /// Принимает запрос пользователя и обращается к LLM через API, добавляя его
    /// в контекст текущего диалога (активной ветки) с учётом активной стратегии
    /// управления контекстом. Если агент остановлен, запрос отклоняется без
    /// обращения к сети.
    pub async fn handle_request(&self, prompt: &str) -> Result<AgentReply> {
        self.exchange(prompt, true).await
    }

    /// Служебное продолжение работы по задаче — после утверждения перехода,
    /// снятия паузы и т.п. (см. [`Agent::task_approve`], [`Agent::task_resume`]):
    /// `instruction` уходит модели только в этом запросе и НЕ сохраняется в
    /// историю диалога — человек его не писал, поэтому ни в чате, ни при
    /// загрузке истории его не видно; сохраняется лишь ответ агента.
    pub async fn continue_task(&self, instruction: &str) -> Result<AgentReply> {
        self.exchange(instruction, false).await
    }

    /// Общий путь [`Agent::handle_request`] и [`Agent::continue_task`]:
    /// `from_human` — реплику написал человек (сохраняется в историю и
    /// проверяется на просьбу перескочить этап) или это служебное продолжение.
    async fn exchange(&self, prompt: &str, from_human: bool) -> Result<AgentReply> {
        if !self.is_running() {
            bail!("агент «{}» остановлен — сначала запустите его", self.name);
        }

        let config = self.config();
        let mut messages = Vec::new();

        // Долговременная память читается здесь, раньше остальных блоков ниже,
        // потому что она — один из ТРЁХ источников глобальных инвариантов
        // (файлы каталога инвариантов, записи долговременной памяти с
        // категорией "invariant" — см. crate::invariants::from_long_term — и,
        // отдельно, текстовые/структурные инварианты активной задачи ниже) —
        // без повторного обращения к БД для одного и того же снимка.
        let long_term = self.long_term_memory();

        // Инварианты (см. crate::invariants) подмешиваются ПЕРВЫМ системным
        // сообщением — раньше системного промпта, персонализации и памяти —
        // потому что это самый приоритетный слой: все они могут ему
        // противоречить, но не могут его отменить. Читаются заново на каждый
        // запрос, как и профиль ниже, а не кешируются — файл каталога или
        // запись долговременной памяти могли только что измениться. Пусто ->
        // блок не добавляется, как и с пустой долговременной памятью/профилем.
        let mut invariants = crate::invariants::load_all();
        invariants.extend(crate::invariants::from_long_term(&long_term));
        let invariants_block = crate::invariants::format_invariants_block(&invariants);
        if !invariants_block.is_empty() {
            messages.push(ChatMessage::system(invariants_block));
        }

        if let Some(system_prompt) = &config.system_prompt {
            if !system_prompt.trim().is_empty() {
                messages.push(ChatMessage::system(system_prompt.clone()));
            }
        }

        // Персонализация (см. crate::profile) подмешивается ПЕРЕД памятью и
        // независимо от неё — это отдельная ось (КАК отвечать, а не что агент
        // знает), поэтому подключена к каждому запросу так же безусловно, как
        // системный промпт, а не завязана на context_strategy. Пусто, только
        // если профиль явно отключён (`NONE_PROFILE`) или файл не найден на
        // диске — тогда сообщение просто не добавляется, как и с пустой
        // долговременной/рабочей памятью ниже.
        let profile_name = crate::profile::resolve_name(config.profile.as_deref());
        if let Some(profile_content) = crate::profile::load(&profile_name) {
            messages.push(ChatMessage::system(crate::profile::format_profile_block(
                &profile_name,
                &profile_content,
            )));
        }

        // Долговременная и рабочая память (см. crate::memory) подмешиваются в
        // запрос независимо от `context_strategy` — та управляет только тем,
        // как в запрос попадает КРАТКОСРОЧНАЯ память (история диалога ниже);
        // это ортогональная ось и её выключить нельзя, зато обе заполняются
        // только явно (Agent::remember/task_set), поэтому попадание сюда
        // лишнего исключено самой моделью данных, а не проверкой здесь.
        if !long_term.is_empty() {
            messages.push(ChatMessage::system(crate::memory::format_long_term_block(&long_term)));
        }
        let active_task = self.task_state();
        if let Some(task) = &active_task {
            messages.push(ChatMessage::system(crate::memory::format_task_block(task)));
        }
        // Инструменты конечного автомата (move_stage/update_step, см. task_tool_definitions
        // ниже) предлагаются модели, только пока есть активная задача, она не на паузе, не
        // конечная и не ждёт утверждения человеком уже предложенного перехода — во всех этих
        // случаях модель и так не должна двигать автомат, поэтому инструменты просто не даются
        // (см. документацию Stage::directive/format_stage_block, откуда модель узнаёт, почему).
        let task_tools = task_tools_offered(active_task.as_ref());
        // Инструменты подключённых MCP-серверов (см. crate::mcp) предлагаются в
        // каждом запросе, независимо от задачи — список читается заново, т.к.
        // серверы могли только что подключиться/отключиться.
        let mcp_tools = self.mcp.tool_definitions();
        // Просьба перескочить обязательный этап перехватывается ДО модели (см.
        // crate::memory::skip_request_target): переход автомат не пропустит и
        // так, но без этого модель могла бы послушаться и написать реализацию
        // или финал прямо текстом ответа, оставив этап прежним.
        let stage_skip_refusal = active_task
            .as_ref()
            .filter(|_| from_human)
            .and_then(|task| self.refuse_stage_skip(task, prompt));

        let window = config.window_size.unwrap_or_else(context::sliding_window_size);
        let window_raw = window * context::RAW_MESSAGES_PER_EXCHANGE;

        {
            let branches = self.branches.lock().expect("ветки агента отравлены паникой");
            let history = branches.current_messages();
            match config.context_strategy {
                ContextStrategy::Full | ContextStrategy::Branching => {
                    // Branching управляет контекстом изоляцией веток, а не
                    // обрезанием истории — внутри одной ветки уходит вся её история.
                    messages.extend(history.iter().map(|(m, _)| m.clone()));
                }
                ContextStrategy::Summary => {
                    let summary = self.summary.lock().expect("сводка агента отравлена паникой");
                    if !summary.summary.is_empty() {
                        messages.push(ChatMessage::system(format!(
                            "Сводка более ранней части этого диалога (сообщения до неё не включены в \
                             запрос дословно, чтобы не раздувать контекст):\n\n{}",
                            summary.summary
                        )));
                    }
                    messages.extend(history.iter().skip(summary.summarized_count).map(|(m, _)| m.clone()));
                }
                ContextStrategy::SlidingWindow => {
                    let start = history.len().saturating_sub(window_raw);
                    messages.extend(history[start..].iter().map(|(m, _)| m.clone()));
                }
                ContextStrategy::Facts => {
                    let facts = self.facts.lock().expect("факты агента отравлены паникой");
                    if !facts.facts.is_empty() {
                        messages.push(ChatMessage::system(context::format_facts_block(&facts.facts)));
                    }
                    drop(facts);
                    let start = history.len().saturating_sub(window_raw);
                    messages.extend(history[start..].iter().map(|(m, _)| m.clone()));
                }
            }
        }
        messages.push(ChatMessage::user(prompt));

        let options = ChatOptions {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            reasoning: config.reasoning,
            ..ChatOptions::default()
        };

        let model = config.model.clone().unwrap_or_else(|| self.client.model().to_string());
        // Пауза, поставленная ПОКА идёт этот запрос, обрывает его (см.
        // unless_paused): ответ, полученный уже после паузы, не проверяется и
        // не сохраняется — запрос уйдёт заново при возобновлении. Задача, уже
        // стоявшая на паузе к началу запроса, не отслеживается: на паузе с
        // агентом можно разговаривать, он лишь не двигает задачу.
        let pause_watch = active_task.as_ref().filter(|t| !t.paused).map(|t| t.name.clone());
        let (completion, tool_calls) = if let Some(refusal) = &stage_skip_refusal {
            let completion = crate::ChatCompletion {
                content: refusal.clone(),
                usage: None,
                request_json: String::new(),
                response_json: String::new(),
                tool_calls: Vec::new(),
            };
            (completion, Vec::new())
        } else {
            let exchange = async {
                if task_tools || !mcp_tools.is_empty() {
                    self.run_tool_loop(&model, &messages, &options, task_tools, mcp_tools).await
                } else {
                    self.client.chat_with_model(&model, &messages, &options).await.map(|c| (c, Vec::new()))
                }
            };
            let completion = self.unless_paused(pause_watch.as_deref(), exchange).await?;
            // Ответ мог прийти в те доли секунды между постановкой паузы и
            // очередной её проверкой — тогда он тоже отбрасывается.
            let paused_meanwhile = match &pause_watch {
                Some(task_name) => self.db.task_paused(task_name).unwrap_or(false),
                None => false,
            };
            match completion {
                Some(completion) if !paused_meanwhile => completion,
                _ => {
                    // Реплика человека остаётся в истории (в чате она уже есть) и
                    // запоминается в задаче: при возобновлении агент ответит на
                    // неё (см. task_resume). Прерванное служебное продолжение не
                    // запоминается — возобновление и так продолжит работу.
                    if from_human {
                        let task_name =
                            pause_watch.as_deref().expect("прерывание возможно только при наблюдении");
                        self.save_to_history(Some(&ChatMessage::user(prompt.to_string())), None);
                        self.db.save_task_interrupted_prompt(task_name, Some(prompt))?;
                    }
                    return Ok(AgentReply {
                        text: INTERRUPTED_BY_PAUSE.to_string(),
                        usage: None,
                        summarized: false,
                        summary_covers: None,
                        facts_updated: false,
                        request_json: String::new(),
                        response_json: String::new(),
                        cost: None,
                        interrupted: true,
                        tool_calls: Vec::new(),
                    });
                }
            }
        };

        let user_message = ChatMessage::user(prompt.to_string());
        let assistant_message = ChatMessage::assistant(completion.content.clone());
        self.save_to_history(from_human.then_some(&user_message), Some((&assistant_message, completion.usage)));

        let summary_covers = self.update_summary_if_needed(&config).await;
        let facts_updated = if config.context_strategy == ContextStrategy::Facts
            && stage_skip_refusal.is_none()
            && from_human
        {
            self.update_facts_after_message(&config, &user_message, &assistant_message).await
        } else {
            false
        };

        let mut text = completion.content;
        if config.show_tokens {
            if let Some(usage) = completion.usage {
                text.push_str(&format!(
                    "\n\n[токены: запрос {} + ответ {} = {}]",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                ));
            }
        }

        // Не подмешивается в `text` (в отличие от токенов выше) — остаётся структурным
        // полем, как `usage`/`summarized`/`facts_updated`, чтобы каждый интерфейс сам решал,
        // как и когда её показывать. Предпочитает реальную стоимость от провайдера
        // (usage.cost), если он её прислал, оценке по ставкам (см. crate::pricing).
        let cost = completion.usage.and_then(|usage| {
            crate::pricing::source(&usage).map(|source| AgentCost {
                input: crate::pricing::resolve_input(&usage).unwrap_or(0.0),
                output: crate::pricing::resolve_output(&usage).unwrap_or(0.0),
                total: crate::pricing::resolve_total(&usage).unwrap_or(0.0),
                source,
            })
        });

        Ok(AgentReply {
            text,
            usage: completion.usage,
            summarized: summary_covers.is_some(),
            summary_covers,
            facts_updated,
            request_json: completion.request_json,
            response_json: completion.response_json,
            cost,
            interrupted: false,
            tool_calls,
        })
    }

    /// Дописывает реплику человека и/или ответ агента в активную ветку — и в
    /// память, и в БД. Лучшая попытка: ответ уже получен и не должен потеряться
    /// для пользователя из-за сбоя записи в БД — при ошибке лишь предупреждаем.
    /// Метрики токенов приходят от API одной суммой на весь обмен (запрос +
    /// ответ), поэтому хранятся при ответе агента — иначе они задвоились бы
    /// при подсчёте суммы по диалогу.
    fn save_to_history(&self, user: Option<&ChatMessage>, assistant: Option<(&ChatMessage, Option<Usage>)>) {
        let branch_name = {
            let mut branches = self.branches.lock().expect("ветки агента отравлены паникой");
            let current = branches.current.clone();
            let history = branches.current_messages_mut();
            if let Some(user) = user {
                history.push((user.clone(), None));
            }
            if let Some((assistant, usage)) = assistant {
                history.push((assistant.clone(), usage));
            }
            current
        };
        if let Some(user) = user {
            if let Err(err) = self.db.append_message(&self.name, &branch_name, user, None) {
                eprintln!("не удалось сохранить сообщение пользователя в БД: {err:#}");
            }
        }
        if let Some((assistant, usage)) = assistant {
            if let Err(err) = self.db.append_message(&self.name, &branch_name, assistant, usage) {
                eprintln!("не удалось сохранить ответ агента в БД: {err:#}");
            }
        }
    }

    /// Выполняет `exchange` (обмен с LLM), пока задача `task_name` не на
    /// паузе: как только её ставят на паузу, future обмена отбрасывается —
    /// HTTP-запрос к LLM обрывается, а инструменты, которые модель могла бы
    /// вызвать дальше, не выполняются. `Ok(None)` — обмен прерван паузой.
    /// Вызовы инструментов, успевшие выполниться ДО паузы, остаются в силе:
    /// между ожиданиями ответа модели код синхронный, прервать его посередине
    /// нельзя. Без `task_name` просто ждёт обмен.
    async fn unless_paused<T>(
        &self,
        task_name: Option<&str>,
        exchange: impl std::future::Future<Output = Result<T>>,
    ) -> Result<Option<T>> {
        let Some(task_name) = task_name else {
            return exchange.await.map(Some);
        };
        let paused = async {
            loop {
                tokio::time::sleep(PAUSE_POLL_INTERVAL).await;
                if self.db.task_paused(task_name).unwrap_or(false) {
                    return;
                }
            }
        };
        tokio::select! {
            result = exchange => result.map(Some),
            () = paused => Ok(None),
        }
    }

    /// Пересчитывает сводку истории активной ветки, если активна стратегия
    /// [`ContextStrategy::Summary`] и с прошлого пересчёта пользователь отправил
    /// не менее [`context::context_summary_chunk`] новых сообщений — тогда
    /// сводка перестраивается заново, целиком охватывая весь накопленный с
    /// прошлого раза "хвост" диалога. Сводку строит [`context::summarize_chunk`]
    /// (общая логика с обычным чатом веб-интерфейса) — ошибка здесь не должна
    /// портить уже полученный пользователем ответ, поэтому лишь логируется, как
    /// и ошибки записи в БД выше.
    ///
    /// Возвращает `Some(N)`, если сводка была пересчитана и теперь покрывает
    /// `N` сообщений пользователя от начала диалога — этот момент интерфейсы
    /// показывают пользователю (см. [`AgentReply::summarized`]); `None`, если
    /// пересчёт в этот раз не потребовался или завершился ошибкой.
    async fn update_summary_if_needed(&self, config: &AgentConfig) -> Option<usize> {
        if config.context_strategy != ContextStrategy::Summary {
            return None;
        }

        let (chunk, prev_summary, new_summarized_count) = {
            let branches = self.branches.lock().expect("ветки агента отравлены паникой");
            let history = branches.current_messages();
            let summary = self.summary.lock().expect("сводка агента отравлена паникой");
            let total = history.len();
            let pending_raw = total.saturating_sub(summary.summarized_count);
            let chunk_threshold_raw = context::context_summary_chunk() * context::RAW_MESSAGES_PER_EXCHANGE;
            if pending_raw < chunk_threshold_raw {
                return None;
            }
            let chunk: Vec<ChatMessage> =
                history[summary.summarized_count..total].iter().map(|(m, _)| m.clone()).collect();
            (chunk, summary.summary.clone(), total)
        };

        let model = config.model.clone().unwrap_or_else(|| self.client.model().to_string());
        match context::summarize_chunk(&self.client, &model, &prev_summary, &chunk).await {
            Ok(new_summary) => {
                let mut summary = self.summary.lock().expect("сводка агента отравлена паникой");
                summary.summary = new_summary;
                summary.summarized_count = new_summarized_count;
                if let Err(err) = self.db.save_summary(&self.name, &summary.summary, summary.summarized_count) {
                    eprintln!("не удалось сохранить сводку контекста агента «{}» в БД: {err:#}", self.name);
                }
                Some(new_summarized_count / context::RAW_MESSAGES_PER_EXCHANGE)
            }
            Err(err) => {
                eprintln!("не удалось построить сводку контекста для агента «{}»: {err:#}", self.name);
                None
            }
        }
    }

    /// Обновляет набор фактов (стратегия [`ContextStrategy::Facts`]) отдельным
    /// запросом к LLM после каждого сообщения пользователя. Ошибка не портит
    /// уже полученный пользователем ответ — только логируется, как и в
    /// [`Self::update_summary_if_needed`]. Возвращает `true`, только если набор
    /// фактов **реально изменился** по содержанию — сама LLM вызывается после
    /// каждого сообщения (так задумано, чтобы не пропустить новый факт), но
    /// когда в сообщении не было ничего нового, модель возвращает тот же набор
    /// без изменений, и это не должно выглядеть в интерфейсах как «факты
    /// обновлены» после каждой реплики.
    async fn update_facts_after_message(
        &self,
        config: &AgentConfig,
        user_message: &ChatMessage,
        assistant_message: &ChatMessage,
    ) -> bool {
        let previous = self.facts.lock().expect("факты агента отравлены паникой").facts.clone();
        let model = config.model.clone().unwrap_or_else(|| self.client.model().to_string());
        let exchange = [user_message.clone(), assistant_message.clone()];
        match context::update_facts(&self.client, &model, &previous, &exchange).await {
            Ok(new_facts) => {
                if new_facts == previous {
                    return false;
                }
                self.facts.lock().expect("факты агента отравлены паникой").facts = new_facts.clone();
                if let Err(err) = self.db.save_facts(&self.name, &new_facts) {
                    eprintln!("не удалось сохранить facts агента «{}» в БД: {err:#}", self.name);
                }
                true
            }
            Err(err) => {
                eprintln!("не удалось обновить facts агента «{}»: {err:#}", self.name);
                false
            }
        }
    }
}

/// Тонкая обёртка над SQLite-соединением: хранит реестр агентов (таблица
/// `agents`), их сообщения по веткам (`messages`), сводки (`context_summaries`),
/// факты (`agent_facts`), текущую активную ветку (`agent_branch_state`) и
/// checkpoint'ы (`agent_checkpoints`) — все с `ON DELETE CASCADE` при удалении
/// агента. Соединение защищено мьютексом — операции короткие и нечастые,
/// отдельный пул не нужен.
struct Db(Mutex<Connection>);

impl Db {
    fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("не удалось открыть БД агентов: {}", path.display()))?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS agents (
                 name    TEXT PRIMARY KEY,
                 config  TEXT NOT NULL,
                 running INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS messages (
                 id               INTEGER PRIMARY KEY AUTOINCREMENT,
                 agent_name       TEXT NOT NULL REFERENCES agents(name) ON DELETE CASCADE,
                 role             TEXT NOT NULL,
                 content          TEXT NOT NULL,
                 created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                 prompt_tokens     INTEGER,
                 completion_tokens INTEGER,
                 total_tokens      INTEGER,
                 cost              REAL,
                 cost_input        REAL,
                 cost_output       REAL
             );
             CREATE INDEX IF NOT EXISTS idx_messages_agent_name ON messages(agent_name);
             CREATE TABLE IF NOT EXISTS context_summaries (
                 agent_name       TEXT PRIMARY KEY REFERENCES agents(name) ON DELETE CASCADE,
                 summary          TEXT NOT NULL,
                 summarized_count INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS agent_facts (
                 agent_name TEXT PRIMARY KEY REFERENCES agents(name) ON DELETE CASCADE,
                 facts_json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS agent_branch_state (
                 agent_name     TEXT PRIMARY KEY REFERENCES agents(name) ON DELETE CASCADE,
                 current_branch TEXT NOT NULL DEFAULT 'main'
             );
             CREATE TABLE IF NOT EXISTS agent_checkpoints (
                 agent_name TEXT NOT NULL REFERENCES agents(name) ON DELETE CASCADE,
                 name       TEXT NOT NULL,
                 branch     TEXT NOT NULL,
                 msg_count  INTEGER NOT NULL,
                 PRIMARY KEY (agent_name, name)
             );
             CREATE TABLE IF NOT EXISTS long_term_memory (
                 key        TEXT PRIMARY KEY,
                 category   TEXT NOT NULL,
                 value      TEXT NOT NULL,
                 updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
             );
             CREATE TABLE IF NOT EXISTS shared_tasks (
                 name            TEXT PRIMARY KEY,
                 goal            TEXT,
                 stage           TEXT NOT NULL DEFAULT 'planning',
                 current_step    TEXT,
                 expected_action TEXT,
                 paused          INTEGER NOT NULL DEFAULT 0,
                 pending_stage   TEXT,
                 pending_outcome TEXT,
                 interrupted_prompt TEXT
             );
             CREATE TABLE IF NOT EXISTS shared_task_data (
                 task_name TEXT NOT NULL REFERENCES shared_tasks(name) ON DELETE CASCADE,
                 key       TEXT NOT NULL,
                 value     TEXT NOT NULL,
                 PRIMARY KEY (task_name, key)
             );
             CREATE TABLE IF NOT EXISTS shared_task_invariants (
                 task_name TEXT NOT NULL REFERENCES shared_tasks(name) ON DELETE CASCADE,
                 id        TEXT NOT NULL,
                 text      TEXT NOT NULL,
                 PRIMARY KEY (task_name, id)
             );
             CREATE TABLE IF NOT EXISTS shared_task_transition_rules (
                 task_name  TEXT NOT NULL REFERENCES shared_tasks(name) ON DELETE CASCADE,
                 from_stage TEXT NOT NULL,
                 to_stage   TEXT NOT NULL,
                 kind       TEXT NOT NULL,
                 PRIMARY KEY (task_name, from_stage, to_stage, kind)
             );
             CREATE TABLE IF NOT EXISTS shared_task_transitions (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 task_name  TEXT NOT NULL REFERENCES shared_tasks(name) ON DELETE CASCADE,
                 from_stage TEXT NOT NULL,
                 to_stage   TEXT NOT NULL,
                 actor      TEXT NOT NULL,
                 outcome    TEXT NOT NULL,
                 reason     TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
             );
             CREATE TABLE IF NOT EXISTS agent_current_task (
                 agent_name TEXT PRIMARY KEY REFERENCES agents(name) ON DELETE CASCADE,
                 task_name  TEXT NOT NULL REFERENCES shared_tasks(name) ON DELETE CASCADE
             );",
        )
        .context("не удалось создать таблицы БД агентов")?;
        Self::migrate_token_columns(&conn).context("не удалось обновить схему БД агентов")?;
        Self::migrate_branch_column(&conn).context("не удалось обновить схему БД агентов (ветки)")?;
        Self::migrate_cost_columns(&conn).context("не удалось обновить схему БД агентов (стоимость)")?;
        Self::migrate_task_stage_columns(&conn)
            .context("не удалось обновить схему БД агентов (состояние задачи)")?;
        Ok(Self(Mutex::new(conn)))
    }

    /// Добавляет столбцы токенов в таблицу `messages`, созданную более ранней
    /// версией приложения (до появления подсчёта токенов за сообщение) — без этой
    /// миграции у уже существующих файлов БД агентов (`agents.db`) не было бы
    /// этих столбцов и `append_message`/`load_messages` завершались бы ошибкой.
    fn migrate_token_columns(conn: &Connection) -> Result<()> {
        let existing = Self::table_columns(conn, "messages")?;
        for column in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            if !existing.iter().any(|c| c == column) {
                conn.execute(&format!("ALTER TABLE messages ADD COLUMN {column} INTEGER"), [])?;
            }
        }
        Ok(())
    }

    /// Добавляет столбец `branch` в таблицу `messages`, созданную до появления
    /// стратегии ветвления — существующие сообщения при этом относятся к
    /// [`MAIN_BRANCH`] (значение по умолчанию у столбца).
    fn migrate_branch_column(conn: &Connection) -> Result<()> {
        let existing = Self::table_columns(conn, "messages")?;
        if !existing.iter().any(|c| c == "branch") {
            conn.execute(
                &format!("ALTER TABLE messages ADD COLUMN branch TEXT NOT NULL DEFAULT '{MAIN_BRANCH}'"),
                [],
            )?;
        }
        Ok(())
    }

    /// Добавляет столбцы реальной стоимости запроса в таблицу `messages`,
    /// созданную до появления [`crate::pricing`] (`usage.cost` от провайдера —
    /// см. документацию модуля `pricing`) — у уже существующих сообщений эти
    /// столбцы останутся NULL, что означает "провайдер стоимость не прислал",
    /// то же самое, что и у только что созданной БД.
    fn migrate_cost_columns(conn: &Connection) -> Result<()> {
        let existing = Self::table_columns(conn, "messages")?;
        for column in ["cost", "cost_input", "cost_output"] {
            if !existing.iter().any(|c| c == column) {
                conn.execute(&format!("ALTER TABLE messages ADD COLUMN {column} REAL"), [])?;
            }
        }
        Ok(())
    }

    /// Добавляет столбцы конечного автомата (`stage`/`current_step`/
    /// `expected_action`/`paused`) в таблицу `shared_tasks`, созданную до
    /// появления Task State Machine — существующие задачи при этом получают
    /// `stage = 'planning'`, `paused = 0` (значения по умолчанию у столбцов).
    fn migrate_task_stage_columns(conn: &Connection) -> Result<()> {
        let existing = Self::table_columns(conn, "shared_tasks")?;
        if !existing.iter().any(|c| c == "stage") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN stage TEXT NOT NULL DEFAULT 'planning'", [])?;
        }
        if !existing.iter().any(|c| c == "current_step") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN current_step TEXT", [])?;
        }
        if !existing.iter().any(|c| c == "expected_action") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN expected_action TEXT", [])?;
        }
        if !existing.iter().any(|c| c == "paused") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN paused INTEGER NOT NULL DEFAULT 0", [])?;
        }
        if !existing.iter().any(|c| c == "pending_stage") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN pending_stage TEXT", [])?;
        }
        if !existing.iter().any(|c| c == "pending_outcome") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN pending_outcome TEXT", [])?;
        }
        if !existing.iter().any(|c| c == "interrupted_prompt") {
            conn.execute("ALTER TABLE shared_tasks ADD COLUMN interrupted_prompt TEXT", [])?;
        }
        Ok(())
    }

    fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let columns: Vec<String> =
            stmt.query_map([], |row| row.get::<_, String>(1))?.collect::<std::result::Result<_, _>>()?;
        Ok(columns)
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.0.lock().expect("соединение с БД агентов отравлено паникой")
    }

    /// Загружает все агенты (конфигурация + флаг запуска) для восстановления реестра при старте.
    fn load_agents(&self) -> Result<Vec<(AgentConfig, bool)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT config, running FROM agents")?;
        let rows = stmt
            .query_map([], |row| {
                let config_json: String = row.get(0)?;
                let running: i64 = row.get(1)?;
                Ok((config_json, running != 0))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        rows.into_iter()
            .map(|(config_json, running)| {
                let config: AgentConfig = serde_json::from_str(&config_json)
                    .context("не удалось разобрать конфигурацию агента из БД")?;
                Ok((config, running))
            })
            .collect()
    }

    /// Загружает историю сообщений одного агента по всем веткам, в порядке их
    /// появления, вместе с метриками токенов, и собирает их в [`Branches`]
    /// (текущая активная ветка и checkpoint'ы подгружаются отдельно). Ветка
    /// [`MAIN_BRANCH`] гарантированно присутствует в результате, даже если у
    /// агента пока нет сообщений.
    fn load_branches(&self, name: &str) -> Result<Branches> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT branch, role, content, prompt_tokens, completion_tokens, total_tokens, \
                    cost, cost_input, cost_output \
             FROM messages WHERE agent_name = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map([name], |row| {
                let branch: String = row.get(0)?;
                let prompt_tokens: Option<u32> = row.get(3)?;
                let completion_tokens: Option<u32> = row.get(4)?;
                let total_tokens: Option<u32> = row.get(5)?;
                // Реальная стоимость от провайдера (см. crate::pricing) хранится
                // независимо от токенов — может отсутствовать, даже когда токены есть
                // (провайдер их не прислал), поэтому читается отдельно, без такой же
                // связки "всё или ничего", как у prompt/completion/total_tokens выше.
                let cost: Option<f64> = row.get(6)?;
                let cost_input: Option<f64> = row.get(7)?;
                let cost_output: Option<f64> = row.get(8)?;
                let usage = match (prompt_tokens, completion_tokens, total_tokens) {
                    (Some(prompt_tokens), Some(completion_tokens), Some(total_tokens)) => {
                        Some(Usage { prompt_tokens, completion_tokens, total_tokens, cost, cost_input, cost_output })
                    }
                    _ => None,
                };
                Ok((branch, ChatMessage { role: row.get(1)?, content: row.get(2)? }, usage))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);

        let mut data: HashMap<String, Vec<(ChatMessage, Option<Usage>)>> = HashMap::new();
        data.entry(MAIN_BRANCH.to_string()).or_default();
        for (branch, message, usage) in rows {
            data.entry(branch).or_default().push((message, usage));
        }

        let current = conn
            .query_row(
                "SELECT current_branch FROM agent_branch_state WHERE agent_name = ?1",
                [name],
                |row| row.get::<_, String>(0),
            )
            .unwrap_or_else(|_| MAIN_BRANCH.to_string());
        data.entry(current.clone()).or_default();

        let mut checkpoints = HashMap::new();
        let mut cp_stmt = conn.prepare("SELECT name, branch, msg_count FROM agent_checkpoints WHERE agent_name = ?1")?;
        let cp_rows = cp_stmt
            .query_map([name], |row| {
                let cp_name: String = row.get(0)?;
                let branch: String = row.get(1)?;
                let msg_count: i64 = row.get(2)?;
                Ok((cp_name, Checkpoint { branch, len: msg_count as usize }))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (cp_name, cp) in cp_rows {
            checkpoints.insert(cp_name, cp);
        }

        Ok(Branches { current, data, checkpoints })
    }

    /// Загружает сводку сжатого контекста агента (стратегия `summary`) —
    /// пустая сводка с нулевым счётчиком, если её ещё нет.
    fn load_summary(&self, name: &str) -> Result<CompressionState> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT summary, summarized_count FROM context_summaries WHERE agent_name = ?1")?;
        let mut rows = stmt.query_map([name], |row| {
            let summarized_count: i64 = row.get(1)?;
            Ok(CompressionState { summary: row.get(0)?, summarized_count: summarized_count as usize })
        })?;
        rows.next().transpose().map(|opt| opt.unwrap_or_default()).context("не удалось прочитать сводку контекста")
    }

    fn save_summary(&self, agent_name: &str, summary: &str, summarized_count: usize) -> Result<()> {
        self.conn().execute(
            "INSERT INTO context_summaries (agent_name, summary, summarized_count) VALUES (?1, ?2, ?3)
             ON CONFLICT(agent_name) DO UPDATE SET summary = excluded.summary, summarized_count = excluded.summarized_count",
            rusqlite::params![agent_name, summary, summarized_count as i64],
        )?;
        Ok(())
    }

    /// Загружает набор фактов агента (стратегия `facts`) — пустой набор, если его ещё нет.
    fn load_facts(&self, name: &str) -> Result<FactsState> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT facts_json FROM agent_facts WHERE agent_name = ?1")?;
        let mut rows = stmt.query_map([name], |row| row.get::<_, String>(0))?;
        match rows.next().transpose()? {
            Some(json) => {
                let facts: BTreeMap<String, String> =
                    serde_json::from_str(&json).context("не удалось разобрать facts_json из БД")?;
                Ok(FactsState { facts })
            }
            None => Ok(FactsState::default()),
        }
    }

    fn save_facts(&self, agent_name: &str, facts: &BTreeMap<String, String>) -> Result<()> {
        let json = serde_json::to_string(facts)?;
        self.conn().execute(
            "INSERT INTO agent_facts (agent_name, facts_json) VALUES (?1, ?2)
             ON CONFLICT(agent_name) DO UPDATE SET facts_json = excluded.facts_json",
            rusqlite::params![agent_name, json],
        )?;
        Ok(())
    }

    /// Загружает всю долговременную память — общую для всех агентов приложения
    /// (см. [`crate::memory`]) — пустая, если в ней ещё ничего нет.
    fn load_long_term(&self) -> Result<LongTermMemory> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT key, category, value FROM long_term_memory")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, LongTermItem { category: row.get(1)?, value: row.get(2)? }))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows.into_iter().collect())
    }

    fn save_long_term(&self, key: &str, item: &LongTermItem) -> Result<()> {
        self.conn().execute(
            "INSERT INTO long_term_memory (key, category, value) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET category = excluded.category, value = excluded.value, \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            rusqlite::params![key, item.category, item.value],
        )?;
        Ok(())
    }

    /// Удаляет запись долговременной памяти. Возвращает `true`, если она существовала.
    fn delete_long_term(&self, key: &str) -> Result<bool> {
        let affected = self.conn().execute("DELETE FROM long_term_memory WHERE key = ?1", [key])?;
        Ok(affected > 0)
    }

    /// Имя общей задачи, к которой сейчас присоединён агент (см.
    /// [`crate::memory`]) — `None`, если он ни к какой задаче не присоединён.
    fn agent_task_name(&self, agent_name: &str) -> Result<Option<String>> {
        let conn = self.conn();
        match conn.query_row(
            "SELECT task_name FROM agent_current_task WHERE agent_name = ?1",
            [agent_name],
            |row| row.get::<_, String>(0),
        ) {
            Ok(name) => Ok(Some(name)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// `true`, если общая задача с этим именем уже существует (независимо от
    /// того, есть ли у неё сейчас хоть один участник).
    fn shared_task_exists(&self, task_name: &str) -> Result<bool> {
        self.conn()
            .query_row("SELECT EXISTS(SELECT 1 FROM shared_tasks WHERE name = ?1)", [task_name], |row| row.get(0))
            .map_err(Into::into)
    }

    /// Загружает общую задачу по имени (метаданные + все её данные) — видна
    /// одинаково всем агентам, которые к ней присоединены. `None`, если задачи
    /// с таким именем не существует.
    fn load_shared_task(&self, task_name: &str) -> Result<Option<TaskState>> {
        type SharedTaskRow = (
            Option<String>,
            String,
            Option<String>,
            Option<String>,
            bool,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let conn = self.conn();
        let row: Option<SharedTaskRow> = match conn.query_row(
            "SELECT goal, stage, current_step, expected_action, paused, pending_stage, pending_outcome, \
             interrupted_prompt FROM shared_tasks WHERE name = ?1",
            [task_name],
            |row| {
                let paused: i64 = row.get(4)?;
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    paused != 0,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        ) {
            Ok(row) => Some(row),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(err) => return Err(err.into()),
        };
        let Some((
            goal,
            stage_raw,
            current_step,
            expected_action,
            paused,
            pending_stage_raw,
            pending_outcome,
            interrupted_prompt,
        )) = row
        else {
            return Ok(None);
        };
        let stage: Stage = stage_raw
            .parse()
            .with_context(|| format!("некорректное значение этапа «{stage_raw}» в БД для задачи «{task_name}»"))?;
        let pending_stage = pending_stage_raw
            .map(|s| {
                s.parse::<Stage>().with_context(|| {
                    format!("некорректное значение предложенного этапа «{s}» в БД для задачи «{task_name}»")
                })
            })
            .transpose()?;
        let mut stmt = conn.prepare("SELECT key, value FROM shared_task_data WHERE task_name = ?1")?;
        let data: BTreeMap<String, String> = stmt
            .query_map([task_name], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        let mut inv_stmt = conn.prepare("SELECT id, text FROM shared_task_invariants WHERE task_name = ?1")?;
        let invariants: BTreeMap<String, String> = inv_stmt
            .query_map([task_name], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        let mut rules_stmt =
            conn.prepare("SELECT from_stage, to_stage, kind FROM shared_task_transition_rules WHERE task_name = ?1")?;
        let rule_rows: Vec<(String, String, String)> = rules_stmt
            .query_map([task_name], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<std::result::Result<_, _>>()?;
        let mut blocked_transitions = BTreeSet::new();
        let mut extra_approval_transitions = BTreeSet::new();
        for (from_raw, to_raw, kind) in rule_rows {
            let from: Stage = from_raw.parse().with_context(|| {
                format!("некорректный этап «{from_raw}» в правиле перехода задачи «{task_name}»")
            })?;
            let to: Stage = to_raw.parse().with_context(|| {
                format!("некорректный этап «{to_raw}» в правиле перехода задачи «{task_name}»")
            })?;
            match kind.as_str() {
                "block" => blocked_transitions.insert((from, to)),
                "approve" => extra_approval_transitions.insert((from, to)),
                other => bail!("неизвестный тип правила перехода «{other}» в БД для задачи «{task_name}»"),
            };
        }
        type TransitionRow = (String, String, String, String, String, String);
        let parse_record = |(from, to, actor, outcome, reason, at): TransitionRow| -> Result<TransitionRecord> {
            Ok(TransitionRecord {
                from: from.parse()?,
                to: to.parse()?,
                actor: actor.parse()?,
                outcome: outcome.parse()?,
                reason,
                at,
            })
        };
        let mut log_stmt = conn.prepare(
            "SELECT from_stage, to_stage, actor, outcome, reason, created_at FROM shared_task_transitions \
             WHERE task_name = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let log_rows: Vec<TransitionRow> = log_stmt
            .query_map(rusqlite::params![task_name, crate::memory::TRANSITION_LOG_LIMIT as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        let transitions = log_rows
            .into_iter()
            .rev()
            .map(parse_record)
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("некорректная запись журнала переходов задачи «{task_name}»"))?;
        let mut accepted_stmt = conn.prepare(
            "SELECT from_stage, to_stage, actor, outcome, reason, created_at FROM shared_task_transitions \
             WHERE task_name = ?1 AND outcome IN ('applied', 'approved') ORDER BY id",
        )?;
        let accepted_rows: Vec<TransitionRow> = accepted_stmt
            .query_map([task_name], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        let accepted = accepted_rows
            .into_iter()
            .map(parse_record)
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("некорректная запись журнала переходов задачи «{task_name}»"))?;
        let stage_path = crate::memory::stage_path(&accepted, stage);
        Ok(Some(TaskState {
            name: task_name.to_string(),
            goal,
            data,
            stage,
            current_step,
            expected_action,
            paused,
            pending_stage,
            pending_outcome,
            invariants,
            blocked_transitions,
            extra_approval_transitions,
            transitions,
            stage_path,
            interrupted_prompt,
        }))
    }

    /// Дописывает запись в журнал переходов задачи (см. [`TaskState::transitions`]).
    fn log_transition(
        &self,
        task_name: &str,
        from: Stage,
        to: Stage,
        actor: TransitionActor,
        outcome: TransitionOutcome,
        reason: &str,
    ) -> Result<()> {
        self.conn().execute(
            "INSERT INTO shared_task_transitions (task_name, from_stage, to_stage, actor, outcome, reason) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![task_name, from.as_str(), to.as_str(), actor.as_str(), outcome.as_str(), reason.trim()],
        )?;
        Ok(())
    }

    /// Обновляет состояние конечного автомата задачи (этап/шаг/ожидаемое
    /// действие) — `None` в `step`/`expected_action` оставляет соответствующее
    /// поле как есть (не затирает его пустым значением), в отличие от `stage`,
    /// который всегда перезаписывается явно.
    fn save_task_stage(
        &self,
        task_name: &str,
        stage: Stage,
        step: Option<&str>,
        expected_action: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn();
        conn.execute(
            "UPDATE shared_tasks SET stage = ?2 WHERE name = ?1",
            rusqlite::params![task_name, stage.as_str()],
        )?;
        if let Some(step) = step {
            conn.execute(
                "UPDATE shared_tasks SET current_step = ?2 WHERE name = ?1",
                rusqlite::params![task_name, step],
            )?;
        }
        if let Some(expected_action) = expected_action {
            conn.execute(
                "UPDATE shared_tasks SET expected_action = ?2 WHERE name = ?1",
                rusqlite::params![task_name, expected_action],
            )?;
        }
        Ok(())
    }

    /// Ставит/снимает паузу задачи — независимо от этапа (см. документацию [`TaskState`]).
    /// Стоит ли задача на паузе — лёгкий запрос для частой проверки во время
    /// ожидания ответа LLM (см. `Agent::unless_paused`). `false`, если задачи нет.
    fn task_paused(&self, task_name: &str) -> Result<bool> {
        match self.conn().query_row("SELECT paused FROM shared_tasks WHERE name = ?1", [task_name], |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(paused) => Ok(paused != 0),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
            Err(err) => Err(err.into()),
        }
    }

    fn save_task_paused(&self, task_name: &str, paused: bool) -> Result<()> {
        self.conn().execute(
            "UPDATE shared_tasks SET paused = ?2 WHERE name = ?1",
            rusqlite::params![task_name, paused as i64],
        )?;
        Ok(())
    }

    /// Кладёт предложенный моделью переход (`move_stage` на этап, требующий
    /// утверждения) на ожидание человека — `stage` задачи не меняется, см.
    /// [`Stage::requires_approval_to`] и `TaskState::pending_stage`.
    fn save_task_pending(&self, task_name: &str, pending_stage: Stage, pending_outcome: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE shared_tasks SET pending_stage = ?2, pending_outcome = ?3 WHERE name = ?1",
            rusqlite::params![task_name, pending_stage.as_str(), pending_outcome],
        )?;
        Ok(())
    }

    /// Снимает ожидающее утверждения предложение (без применения) — используется
    /// и при отклонении ([`crate::agent::Agent::task_reject`]), и после
    /// применения ([`crate::agent::Agent::task_approve`]), и когда человек сам
    /// ставит другой этап (см. `Agent::task_advance`).
    /// Запоминает (`Some`) или снимает (`None`) запрос, ответ на который
    /// прервала пауза (см. [`TaskState::interrupted_prompt`]).
    fn save_task_interrupted_prompt(&self, task_name: &str, prompt: Option<&str>) -> Result<()> {
        self.conn().execute(
            "UPDATE shared_tasks SET interrupted_prompt = ?2 WHERE name = ?1",
            rusqlite::params![task_name, prompt],
        )?;
        Ok(())
    }

    fn clear_task_pending(&self, task_name: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE shared_tasks SET pending_stage = NULL, pending_outcome = NULL WHERE name = ?1",
            [task_name],
        )?;
        Ok(())
    }

    /// Создаёт новую общую задачу — вызывающая сторона уже проверила, что
    /// задачи с таким именем ещё нет ([`Db::shared_task_exists`]).
    fn create_shared_task(&self, task_name: &str, goal: Option<&str>) -> Result<()> {
        self.conn().execute(
            "INSERT INTO shared_tasks (name, goal) VALUES (?1, ?2)",
            rusqlite::params![task_name, goal],
        )?;
        Ok(())
    }

    /// Присоединяет агента `agent_name` к задаче `task_name` (или переносит его
    /// членство, если оно уже было где-то ещё — вызывающая сторона в
    /// [`Agent::task_start`]/[`Agent::task_join`] такое не допускает, но сам
    /// метод остаётся простым upsert'ом).
    fn set_agent_task(&self, agent_name: &str, task_name: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO agent_current_task (agent_name, task_name) VALUES (?1, ?2)
             ON CONFLICT(agent_name) DO UPDATE SET task_name = excluded.task_name",
            rusqlite::params![agent_name, task_name],
        )?;
        Ok(())
    }

    fn save_shared_task_data(&self, task_name: &str, key: &str, value: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO shared_task_data (task_name, key, value) VALUES (?1, ?2, ?3)
             ON CONFLICT(task_name, key) DO UPDATE SET value = excluded.value",
            rusqlite::params![task_name, key, value],
        )?;
        Ok(())
    }

    /// Сохраняет (или обновляет) текстовый инвариант, привязанный ТОЛЬКО к
    /// этой задаче (см. [`crate::agent::Agent::task_invariant_set`]).
    fn save_task_invariant(&self, task_name: &str, id: &str, text: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO shared_task_invariants (task_name, id, text) VALUES (?1, ?2, ?3)
             ON CONFLICT(task_name, id) DO UPDATE SET text = excluded.text",
            rusqlite::params![task_name, id, text],
        )?;
        Ok(())
    }

    /// Удаляет инвариант задачи. Возвращает `true`, если он существовал.
    fn delete_task_invariant(&self, task_name: &str, id: &str) -> Result<bool> {
        let affected = self.conn().execute(
            "DELETE FROM shared_task_invariants WHERE task_name = ?1 AND id = ?2",
            rusqlite::params![task_name, id],
        )?;
        Ok(affected > 0)
    }

    /// Добавляет структурное правило автомата задачи — `kind` — `"block"`
    /// (см. [`crate::memory::TaskState::blocked_transitions`]) или `"approve"`
    /// (см. [`crate::memory::TaskState::extra_approval_transitions`]).
    fn save_task_transition_rule(&self, task_name: &str, from: Stage, to: Stage, kind: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO shared_task_transition_rules (task_name, from_stage, to_stage, kind) \
             VALUES (?1, ?2, ?3, ?4) ON CONFLICT(task_name, from_stage, to_stage, kind) DO NOTHING",
            rusqlite::params![task_name, from.as_str(), to.as_str(), kind],
        )?;
        Ok(())
    }

    /// Снимает структурное правило автомата задачи. Возвращает `true`, если оно существовало.
    fn delete_task_transition_rule(&self, task_name: &str, from: Stage, to: Stage, kind: &str) -> Result<bool> {
        let affected = self.conn().execute(
            "DELETE FROM shared_task_transition_rules \
             WHERE task_name = ?1 AND from_stage = ?2 AND to_stage = ?3 AND kind = ?4",
            rusqlite::params![task_name, from.as_str(), to.as_str(), kind],
        )?;
        Ok(affected > 0)
    }

    /// Удаляет общую задачу целиком — каскадно удаляет и её данные
    /// (`shared_task_data`), и членство ВСЕХ агентов, которые были к ней
    /// присоединены (`agent_current_task`, `ON DELETE CASCADE`), поэтому
    /// завершение задачи одним участником завершает её для всех разом (см.
    /// [`Agent::task_finish`]).
    fn delete_shared_task(&self, task_name: &str) -> Result<()> {
        self.conn().execute("DELETE FROM shared_tasks WHERE name = ?1", [task_name])?;
        Ok(())
    }

    /// Список всех существующих общих задач с их участниками — используется
    /// интерфейсами, чтобы показать, к каким задачам вообще можно
    /// присоединиться (см. [`AgentManager::list_tasks`]).
    fn list_shared_tasks(&self) -> Result<Vec<SharedTaskSummary>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT name, goal FROM shared_tasks ORDER BY name")?;
        let tasks: Vec<(String, Option<String>)> = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        drop(stmt);

        let mut members_stmt = conn.prepare("SELECT agent_name FROM agent_current_task WHERE task_name = ?1")?;
        let mut summaries = Vec::with_capacity(tasks.len());
        for (name, goal) in tasks {
            let members: Vec<String> = members_stmt
                .query_map([&name], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<_, _>>()?;
            summaries.push(SharedTaskSummary { name, goal, members });
        }
        Ok(summaries)
    }

    fn save_current_branch(&self, agent_name: &str, branch: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO agent_branch_state (agent_name, current_branch) VALUES (?1, ?2)
             ON CONFLICT(agent_name) DO UPDATE SET current_branch = excluded.current_branch",
            rusqlite::params![agent_name, branch],
        )?;
        Ok(())
    }

    fn save_checkpoint(&self, agent_name: &str, name: &str, branch: &str, msg_count: usize) -> Result<()> {
        self.conn().execute(
            "INSERT INTO agent_checkpoints (agent_name, name, branch, msg_count) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(agent_name, name) DO UPDATE SET branch = excluded.branch, msg_count = excluded.msg_count",
            rusqlite::params![agent_name, name, branch, msg_count as i64],
        )?;
        Ok(())
    }

    /// Вставляет ветку `branch` целиком (список сообщений, склонированный от
    /// checkpoint'а или от текущего конца исходной ветки) одной транзакцией.
    fn create_branch(&self, agent_name: &str, branch: &str, messages: &[(ChatMessage, Option<Usage>)]) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        for (message, usage) in messages {
            tx.execute(
                "INSERT INTO messages (agent_name, branch, role, content, prompt_tokens, completion_tokens, total_tokens, cost, cost_input, cost_output)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    agent_name,
                    branch,
                    message.role,
                    message.content,
                    usage.map(|u| u.prompt_tokens),
                    usage.map(|u| u.completion_tokens),
                    usage.map(|u| u.total_tokens),
                    usage.and_then(|u| u.cost),
                    usage.and_then(|u| u.cost_input),
                    usage.and_then(|u| u.cost_output),
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn insert_agent(&self, config: &AgentConfig, running: bool) -> Result<()> {
        let config_json = serde_json::to_string(config)?;
        self.conn().execute(
            "INSERT INTO agents (name, config, running) VALUES (?1, ?2, ?3)",
            rusqlite::params![config.name, config_json, running as i64],
        )?;
        Ok(())
    }

    fn update_config(&self, name: &str, config: &AgentConfig) -> Result<()> {
        let config_json = serde_json::to_string(config)?;
        self.conn().execute("UPDATE agents SET config = ?1 WHERE name = ?2", rusqlite::params![config_json, name])?;
        Ok(())
    }

    fn set_running(&self, name: &str, running: bool) -> Result<()> {
        self.conn().execute(
            "UPDATE agents SET running = ?1 WHERE name = ?2",
            rusqlite::params![running as i64, name],
        )?;
        Ok(())
    }

    /// Удаляет агента; связанные сообщения, сводка, факты, ветки и checkpoint'ы
    /// удаляются каскадно (`ON DELETE CASCADE`).
    fn delete_agent(&self, name: &str) -> Result<()> {
        self.conn().execute("DELETE FROM agents WHERE name = ?1", [name])?;
        Ok(())
    }

    fn append_message(&self, agent_name: &str, branch: &str, message: &ChatMessage, usage: Option<Usage>) -> Result<()> {
        self.conn().execute(
            "INSERT INTO messages (agent_name, branch, role, content, prompt_tokens, completion_tokens, total_tokens, cost, cost_input, cost_output)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                agent_name,
                branch,
                message.role,
                message.content,
                usage.map(|u| u.prompt_tokens),
                usage.map(|u| u.completion_tokens),
                usage.map(|u| u.total_tokens),
                usage.and_then(|u| u.cost),
                usage.and_then(|u| u.cost_input),
                usage.and_then(|u| u.cost_output),
            ],
        )?;
        Ok(())
    }
}

/// Реестр именованных агентов с сохранением в SQLite, общий для CLI, TUI и
/// веб-интерфейса — агент, добавленный в одном из них, виден в других, а его
/// диалог переживает перезапуск процесса.
pub struct AgentManager {
    client: LlmClient,
    db: Arc<Db>,
    mcp: Arc<crate::mcp::McpManager>,
    agents: RwLock<HashMap<String, Arc<Agent>>>,
}

impl AgentManager {
    /// Реестр без MCP-серверов — см. [`AgentManager::with_mcp`].
    pub fn new(client: LlmClient, store_path: impl Into<PathBuf>) -> Result<Self> {
        Self::with_mcp(client, store_path, Arc::new(crate::mcp::McpManager::empty()))
    }

    /// Реестр, агенты которого предлагают модели инструменты MCP-серверов
    /// `mcp` (см. [`crate::mcp`]). Подключаться к серверам — забота
    /// вызывающего ([`crate::mcp::McpManager::connect_all`]): интерфейс сам
    /// решает, ждать подключения или вести его в фоне.
    pub fn with_mcp(
        client: LlmClient,
        store_path: impl Into<PathBuf>,
        mcp: Arc<crate::mcp::McpManager>,
    ) -> Result<Self> {
        let db = Arc::new(Db::open(&store_path.into())?);
        let manager = Self { client, db, mcp, agents: RwLock::new(HashMap::new()) };
        manager.load()?;
        Ok(manager)
    }

    /// Путь к файлу БД берётся из AGENTS_STORE_PATH, по умолчанию — `agents.db`
    /// в текущей рабочей директории; MCP-серверы — из файла
    /// [`crate::mcp::config_path`] (ещё не подключены).
    pub fn from_env(client: LlmClient) -> Result<Self> {
        let path = std::env::var("AGENTS_STORE_PATH").unwrap_or_else(|_| "agents.db".to_string());
        Self::with_mcp(client, path, Arc::new(crate::mcp::McpManager::from_env()))
    }

    /// Реестр MCP-серверов, общий для всех агентов.
    pub fn mcp(&self) -> Arc<crate::mcp::McpManager> {
        self.mcp.clone()
    }

    fn load(&self) -> Result<()> {
        let records = self.db.load_agents()?;
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        for (config, running) in records {
            let branches = self.db.load_branches(&config.name)?;
            let summary = self.db.load_summary(&config.name)?;
            let facts = self.db.load_facts(&config.name)?;
            let agent = Agent::new(
                config,
                self.client.clone(),
                self.db.clone(),
                self.mcp.clone(),
                branches,
                summary,
                facts,
            );
            if running {
                agent.start();
            }
            agents.insert(agent.name().to_string(), Arc::new(agent));
        }
        Ok(())
    }

    pub fn list(&self) -> Vec<AgentInfo> {
        let agents = self.agents.read().expect("реестр агентов отравлен паникой");
        let mut list: Vec<AgentInfo> = agents.values().map(|a| a.info()).collect();
        list.sort_by(|a, b| a.config.name.cmp(&b.config.name));
        list
    }

    pub fn get(&self, name: &str) -> Option<Arc<Agent>> {
        self.agents.read().expect("реестр агентов отравлен паникой").get(name).cloned()
    }

    pub fn create(&self, mut config: AgentConfig) -> Result<AgentInfo> {
        let name = config.name.trim().to_string();
        if name.is_empty() {
            bail!("имя агента не может быть пустым");
        }
        config.name = name.clone();

        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        if agents.contains_key(&name) {
            bail!("агент с именем «{name}» уже существует");
        }
        self.db.insert_agent(&config, false)?;
        let agent = Agent::new(
            config,
            self.client.clone(),
            self.db.clone(),
            self.mcp.clone(),
            Branches::default(),
            CompressionState::default(),
            FactsState::default(),
        );
        let info = agent.info();
        agents.insert(name, Arc::new(agent));

        Ok(info)
    }

    /// Список всех существующих общих задач (рабочая память, см.
    /// [`crate::memory`]) с их участниками — не привязан к конкретному агенту,
    /// т.к. задача — общая сущность; используется интерфейсами, чтобы показать,
    /// к каким задачам можно присоединиться командой `agent task <имя> join`.
    pub fn list_tasks(&self) -> Vec<SharedTaskSummary> {
        self.db.list_shared_tasks().unwrap_or_default()
    }

    /// Снимок всей долговременной памяти — как и [`Agent::long_term_memory`],
    /// но без привязки к конкретному агенту (данные ОДНИ на всё приложение,
    /// см. [`crate::memory`]): удобно там, где нужно показать (например,
    /// глобальные инварианты категории `"invariant"`, см.
    /// [`crate::invariants::from_long_term`]) их до выбора конкретного агента.
    pub fn long_term_memory(&self) -> LongTermMemory {
        self.db.load_long_term().unwrap_or_default()
    }

    /// Удаляет агента вместе со всей его историей диалога в БД.
    pub fn remove(&self, name: &str) -> Result<()> {
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        if agents.remove(name).is_none() {
            bail!("агент «{name}» не найден");
        }
        drop(agents);
        self.db.delete_agent(name)
    }

    pub fn start(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.start();
        self.db.set_running(name, true)?;
        Ok(agent.info())
    }

    pub fn stop(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.stop();
        self.db.set_running(name, false)?;
        Ok(agent.info())
    }
}

#[cfg(test)]
mod tests {
    //! Проверки конечного автомата задачи: недопустимые переходы (ручные и
    //! предложенные моделью через `move_stage`), реакция на них, гейты
    //! подтверждения человеком и продолжение после паузы — в том числе после
    //! перезапуска приложения (новый `AgentManager` над тем же файлом БД).

    use super::*;
    use crate::{ToolCallFunction, ToolCallWire};
    use std::sync::atomic::AtomicUsize;

    /// Временный файл БД, удаляемый по завершении теста.
    struct TempStore(PathBuf);

    impl TempStore {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            Self(std::env::temp_dir().join(format!("llm-core-test-{}-{n}.db", std::process::id())))
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn open_agent(store: &TempStore) -> (AgentManager, Arc<Agent>) {
        let manager = AgentManager::new(LlmClient::for_tests(), store.0.clone()).unwrap();
        if manager.get("tester").is_none() {
            manager.create(AgentConfig::new("tester")).unwrap();
        }
        let agent = manager.get("tester").unwrap();
        (manager, agent)
    }

    /// Агент с только что начатой задачей (этап `planning`).
    fn agent_with_task(store: &TempStore) -> (AgentManager, Arc<Agent>) {
        let (manager, agent) = open_agent(store);
        agent.task_start("demo", Some("написать отчёт")).unwrap();
        (manager, agent)
    }

    /// Вызов инструмента так, как его сделала бы модель.
    fn call_tool(agent: &Agent, name: &str, arguments: serde_json::Value) -> String {
        agent.execute_task_tool(&ToolCallWire {
            id: "call-1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction { name: name.to_string(), arguments: arguments.to_string() },
        })
    }

    fn move_stage(agent: &Agent, stage: &str) -> String {
        call_tool(agent, "move_stage", serde_json::json!({ "stage": stage, "outcome": "итог этапа" }))
    }

    fn task(agent: &Agent) -> TaskState {
        agent.task_state().expect("агент должен быть присоединён к задаче")
    }

    // --- Недопустимые переходы ---

    #[test]
    fn manual_advance_cannot_skip_stages() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        for target in [Stage::Validation, Stage::Done] {
            let err = agent.task_advance(target, None, None).unwrap_err().to_string();
            assert!(err.contains("допустимые следующие этапы: execution"), "{err}");
        }
        assert_eq!(task(&agent).stage, Stage::Planning);

        agent.task_advance(Stage::Execution, None, None).unwrap();
        let err = agent.task_advance(Stage::Done, None, None).unwrap_err().to_string();
        assert!(err.contains("допустимые следующие этапы: validation"), "{err}");
        assert_eq!(task(&agent).stage, Stage::Execution);
    }

    #[test]
    fn done_is_terminal() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        for stage in [Stage::Execution, Stage::Validation, Stage::Done] {
            agent.task_advance(stage, None, None).unwrap();
        }

        let err = agent.task_advance(Stage::Execution, None, None).unwrap_err().to_string();
        assert!(err.contains("конечный этап"), "{err}");
        let reply = move_stage(&agent, "execution");
        assert!(reply.contains("этап «done» конечный"), "{reply}");
        assert_eq!(task(&agent).stage, Stage::Done);
        assert!(!task_tools_offered(Some(&task(&agent))));
    }

    #[test]
    fn model_cannot_start_implementation_before_plan() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        for target in ["validation", "done"] {
            let reply = move_stage(&agent, target);
            assert!(reply.contains("нет такого перехода: из «planning» можно в execution"), "{reply}");
        }
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, None);
    }

    #[test]
    fn model_cannot_finish_without_validation() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        agent.task_advance(Stage::Execution, None, None).unwrap();

        let reply = move_stage(&agent, "done");
        assert!(reply.contains("нет такого перехода: из «execution» можно в validation"), "{reply}");
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Execution);
        assert_eq!(state.pending_stage, None);
    }

    #[test]
    fn model_gets_explanation_for_malformed_calls() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        let reply = move_stage(&agent, "deploy");
        assert!(reply.contains("неизвестный этап «deploy»"), "{reply}");
        let reply = call_tool(&agent, "move_stage", serde_json::json!({ "stage": "execution", "outcome": "  " }));
        assert!(reply.contains("outcome не может быть пустым"), "{reply}");
        let reply = call_tool(&agent, "move_stage", serde_json::json!("execution"));
        assert!(reply.contains("ожидался JSON-объект"), "{reply}");
        let reply = call_tool(&agent, "jump_to_done", serde_json::json!({}));
        assert!(reply.contains("нет такого инструмента"), "{reply}");

        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, None);
    }

    #[test]
    fn task_invariant_blocks_otherwise_legal_transition() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        agent.task_advance(Stage::Execution, None, None).unwrap();
        agent.task_advance(Stage::Validation, None, None).unwrap();
        agent.task_forbid_transition(Stage::Validation, Stage::Execution).unwrap();

        let reply = move_stage(&agent, "execution");
        assert!(reply.contains("запрещён инвариантом"), "{reply}");
        assert!(agent.task_advance(Stage::Execution, None, None).is_err());
        assert_eq!(task(&agent).stage, Stage::Validation);

        assert!(agent.task_allow_transition(Stage::Validation, Stage::Execution).unwrap());
        assert!(move_stage(&agent, "execution").contains("Переход применён"));
        assert_eq!(task(&agent).stage, Stage::Execution);
    }

    #[test]
    fn replanning_from_execution_needs_new_approval() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");
        agent.task_approve().unwrap();

        // План оказался негодным — откат в planning применяется сразу…
        assert!(move_stage(&agent, "planning").contains("Переход применён"));
        assert_eq!(task(&agent).stage, Stage::Planning);
        // …а обратно в execution — снова только через утверждение.
        assert!(move_stage(&agent, "execution").contains("ждёт подтверждения"));
        assert_eq!(task(&agent).stage, Stage::Planning);
    }

    // --- Журнал переходов ---

    #[test]
    fn journal_records_every_attempt() {
        use crate::memory::{TransitionActor as A, TransitionOutcome as O};
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        move_stage(&agent, "done"); // запрещено картой
        move_stage(&agent, "execution"); // ждёт утверждения
        agent.task_reject("добавь сроки").unwrap();
        move_stage(&agent, "execution");
        agent.task_approve().unwrap();
        let _ = agent.task_advance(Stage::Done, None, None); // ручной прыжок — тоже отказ
        move_stage(&agent, "validation"); // применяется сразу

        let log: Vec<_> = task(&agent).transitions.iter().map(|r| (r.from, r.to, r.actor, r.outcome)).collect();
        assert_eq!(
            log,
            vec![
                (Stage::Planning, Stage::Done, A::Model, O::Refused),
                (Stage::Planning, Stage::Execution, A::Model, O::Proposed),
                (Stage::Planning, Stage::Execution, A::Human, O::Rejected),
                (Stage::Planning, Stage::Execution, A::Model, O::Proposed),
                (Stage::Planning, Stage::Execution, A::Human, O::Approved),
                (Stage::Execution, Stage::Done, A::Human, O::Refused),
                (Stage::Execution, Stage::Validation, A::Model, O::Applied),
            ]
        );
        let records = task(&agent).transitions;
        assert!(records[0].reason.contains("нет такого перехода"), "{}", records[0].reason);
        assert_eq!(records[2].reason, "добавь сроки");
        assert!(records[0].to_string().starts_with("✖ planning -> done [model, refused]"), "{}", records[0]);
    }

    #[test]
    fn journal_survives_restart_and_is_capped() {
        let store = TempStore::new();
        {
            let (_m, agent) = agent_with_task(&store);
            for _ in 0..crate::memory::TRANSITION_LOG_LIMIT + 5 {
                move_stage(&agent, "done");
            }
            move_stage(&agent, "execution");
        }
        let (_m, agent) = open_agent(&store);
        let log = task(&agent).transitions;
        assert_eq!(log.len(), crate::memory::TRANSITION_LOG_LIMIT);
        assert_eq!(log.last().unwrap().to, Stage::Execution, "новейшая запись — последней");
    }

    /// Тексты ядра читают в TUI, вебе и CLI, а часть из них — модель, которая
    /// пересказывает их человеку, поэтому в них не должно быть команд
    /// конкретного интерфейса.
    #[test]
    fn messages_do_not_mention_interface_commands() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        let mut texts = vec![
            agent.task_start("другая", None).unwrap_err().to_string(),
            agent.task_join("нет-такой").unwrap_err().to_string(),
            move_stage(&agent, "execution"),
            crate::memory::format_task_block(&task(&agent)),
        ];
        agent.task_pause().unwrap();
        texts.push(agent.task_approve().unwrap_err().to_string());
        texts.push(agent.task_advance(Stage::Execution, None, None).unwrap_err().to_string());
        texts.push(crate::memory::format_task_block(&task(&agent)));
        agent.task_resume().unwrap();
        agent.task_forbid_transition(Stage::Planning, Stage::Execution).unwrap();
        texts.push(agent.task_approve().unwrap_err().to_string());
        texts.push(agent.task_reject("").unwrap_err().to_string());
        texts.push(agent.task_reject("нужно разбить на шаги").unwrap());
        texts.push(crate::memory::format_task_block(&task(&agent)));
        texts.push(move_stage(&agent, "execution"));
        texts.push(agent.refuse_stage_skip(&task(&agent), "пропусти планирование").unwrap());
        agent.task_finish().unwrap();
        texts.push(agent.task_advance(Stage::Execution, None, None).unwrap_err().to_string());

        for text in texts {
            assert!(!text.contains("agent task") && !text.contains("agent remember"), "{text}");
        }
    }

    // --- Просьба пропустить этап в сообщении ---

    #[tokio::test]
    async fn request_to_skip_planning_is_answered_without_model() {
        let store = TempStore::new();
        let (manager, agent) = agent_with_task(&store);
        manager.start("tester").unwrap();

        // LlmClient::for_tests смотрит на недоступный адрес — дойди запрос до
        // модели, handle_request вернул бы ошибку сети.
        let reply = agent.handle_request("Пропусти планирование, сразу пиши код").await.unwrap();
        assert!(reply.text.contains("Этап планирования пропустить нельзя"), "{}", reply.text);
        assert!(reply.usage.is_none());

        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, None);
        let last = state.transitions.last().unwrap();
        assert_eq!((last.from, last.to), (Stage::Planning, Stage::Execution));
        assert_eq!(last.outcome, crate::memory::TransitionOutcome::Refused);

        // Обмен остаётся в истории — модель увидит его в следующем запросе.
        let history = agent.history();
        assert_eq!(history.len(), 2);
        assert!(history[1].content.contains("⛔"));
    }

    #[tokio::test]
    async fn request_to_skip_validation_is_answered_without_model() {
        let store = TempStore::new();
        let (manager, agent) = agent_with_task(&store);
        manager.start("tester").unwrap();
        agent.task_advance(Stage::Execution, None, None).unwrap();

        let reply = agent.handle_request("пропусти проверку и сразу завершай").await.unwrap();
        assert!(reply.text.contains("без проверки нельзя"), "{}", reply.text);
        assert_eq!(task(&agent).stage, Stage::Execution);
        assert_eq!(task(&agent).transitions.last().unwrap().to, Stage::Done);
    }

    /// Фиктивный OpenAI-совместимый сервер: на каждый запрос к
    /// `/chat/completions` отдаёт следующее из `replies` (тело `message`,
    /// см. также [`delayed`]) и запоминает тела запросов. Возвращает base_url
    /// и журнал запросов.
    async fn mock_llm(
        replies: Vec<serde_json::Value>,
    ) -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        tokio::spawn(async move {
            for mut message in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                // Читаем заголовки, затем тело ровно по Content-Length.
                let body_start = loop {
                    let n = socket.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                    if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&raw[..body_start]).to_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                while raw.len() < body_start + length {
                    let n = socket.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                }
                seen.lock().unwrap().push(serde_json::from_slice(&raw[body_start..body_start + length]).unwrap());
                if let Some(ms) = message.get("__delay_ms").and_then(|v| v.as_u64()) {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    message = message["__message"].clone();
                }
                let body = serde_json::json!({ "choices": [{ "message": message }] }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                // Клиент мог уже оборвать запрос (пауза) — это не ошибка сервера.
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (base_url, requests)
    }

    /// Ответ фиктивного сервера, отданный через `ms` миллисекунд.
    fn delayed(ms: u64, message: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "__delay_ms": ms, "__message": message })
    }

    /// Агент с задачей на этапе execution (план утверждён) поверх фиктивного LLM.
    fn executing_agent(store: &TempStore, base_url: &str) -> (AgentManager, Arc<Agent>) {
        let manager = AgentManager::new(LlmClient::for_tests_at(base_url), store.0.clone()).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("demo", None).unwrap();
        move_stage(&agent, "execution");
        agent.task_approve().unwrap();
        (manager, agent)
    }

    #[tokio::test]
    async fn pause_during_request_discards_reply_and_resends_prompt_on_resume() {
        let (base_url, requests) = mock_llm(vec![
            delayed(2000, serde_json::json!({ "content": "ОТВЕТ, ПОЛУЧЕННЫЙ ПОСЛЕ ПАУЗЫ" })),
            tool_call("update_step", serde_json::json!({ "step": "раздел 2" })),
            serde_json::json!({ "content": "Раздел 2 готов." }),
        ])
        .await;
        let store = TempStore::new();
        let (_m, agent) = executing_agent(&store, &base_url);

        let started = std::time::Instant::now();
        let request = tokio::spawn({
            let agent = agent.clone();
            async move { agent.handle_request("напиши раздел 2").await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        agent.task_pause().unwrap();
        let reply = request.await.unwrap().unwrap();

        assert!(reply.interrupted);
        assert_eq!(reply.text, "⏸ Задача поставлена на паузу.");
        assert!(started.elapsed() < std::time::Duration::from_millis(1500), "запрос оборван, а не дождались ответа");
        // Реплика человека осталась (в чате она уже есть), отброшенный ответ — нет.
        let history = agent.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "напиши раздел 2");
        assert_eq!(task(&agent).interrupted_prompt.as_deref(), Some("напиши раздел 2"));

        // «Возобновить» — служебное продолжение: ответить на прерванный запрос.
        let followup = agent.task_resume().unwrap().unwrap();
        assert!(followup.contains("«напиши раздел 2»"), "{followup}");
        assert_eq!(task(&agent).interrupted_prompt, None);
        let reply = agent.continue_task(&followup).await.unwrap();
        assert!(!reply.interrupted);
        assert_eq!(reply.text, "Раздел 2 готов.");

        // В истории — запрос человека и ответ; ни служебного продолжения, ни
        // отброшенного ответа.
        let history = agent.history();
        let contents: Vec<_> = history.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
        assert_eq!(contents, [("user", "напиши раздел 2"), ("assistant", "Раздел 2 готов.")]);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        // Модель видит и сам запрос в истории, и указание ответить на него.
        let resumed = requests[1].to_string();
        assert!(resumed.contains("напиши раздел 2") && resumed.contains("Задача возобновлена"), "{resumed}");
    }

    #[tokio::test]
    async fn approval_continues_without_showing_a_message() {
        let (base_url, requests) = mock_llm(vec![
            tool_call("update_step", serde_json::json!({ "step": "пишу черновик" })),
            serde_json::json!({ "content": "Черновик письма: …" }),
        ])
        .await;
        let store = TempStore::new();
        let manager = AgentManager::new(LlmClient::for_tests_at(&base_url), store.0.clone()).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("demo", None).unwrap();
        move_stage(&agent, "execution");

        let followup = agent.task_approve().unwrap();
        assert!(followup.contains("утвердил переход «planning -> execution»"), "{followup}");
        let reply = agent.continue_task(&followup).await.unwrap();
        assert_eq!(reply.text, "Черновик письма: …");

        // Служебное продолжение ушло модели, но в историю попал только ответ.
        assert!(requests.lock().unwrap()[0].to_string().contains("утвердил переход"));
        let history = agent.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "assistant");
    }

    #[tokio::test]
    async fn tool_calls_after_pause_are_not_applied() {
        let (base_url, _requests) = mock_llm(vec![
            // До паузы: шаг обновлён — это остаётся в силе.
            tool_call("update_step", serde_json::json!({ "step": "черновик готов" })),
            // После паузы: переход на проверку не должен примениться.
            delayed(
                2000,
                tool_call("move_stage", serde_json::json!({ "stage": "validation", "outcome": "готово" })),
            ),
        ])
        .await;
        let store = TempStore::new();
        let (_m, agent) = executing_agent(&store, &base_url);

        let request = tokio::spawn({
            let agent = agent.clone();
            async move { agent.handle_request("доделай черновик").await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        agent.task_pause().unwrap();
        assert!(request.await.unwrap().unwrap().interrupted);

        let state = task(&agent);
        assert_eq!(state.stage, Stage::Execution, "ответ после паузы не применён");
        assert_eq!(state.current_step.as_deref(), Some("черновик готов"), "сделанное до паузы осталось");
    }

    #[tokio::test]
    async fn request_made_while_already_paused_is_answered_normally() {
        let (base_url, requests) = mock_llm(vec![serde_json::json!({ "content": "Жду возобновления." })]).await;
        let store = TempStore::new();
        let (_m, agent) = executing_agent(&store, &base_url);
        agent.task_pause().unwrap();

        let reply = agent.handle_request("на чём остановились?").await.unwrap();
        assert!(!reply.interrupted);
        assert_eq!(reply.text, "Жду возобновления.");
        assert_eq!(task(&agent).interrupted_prompt, None);
        assert!(requests.lock().unwrap()[0].get("tools").is_none(), "на паузе инструменты не предлагаются");
    }

    fn tool_call(name: &str, arguments: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "content": null,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() }
            }]
        })
    }

    #[tokio::test]
    async fn reworked_plan_is_always_resubmitted_for_approval() {
        let (base_url, requests) = mock_llm(vec![
            // Модель по замечанию обновляет шаг…
            tool_call("update_step", serde_json::json!({ "step": "план переработан" })),
            // …показывает новый план текстом, но move_stage не вызывает.
            serde_json::json!({ "content": "Новый план: 1) сбор 2) сравнение QoQ 3) прогноз" }),
            // Принудительный круг: предложение переработанного плана.
            tool_call(
                "move_stage",
                serde_json::json!({ "stage": "execution", "outcome": "добавлены QoQ и прогноз" }),
            ),
        ])
        .await;
        let store = TempStore::new();
        let manager = AgentManager::new(LlmClient::for_tests_at(&base_url), store.0.clone()).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("demo", None).unwrap();
        move_stage(&agent, "execution");
        let followup = agent.task_reject("нет сравнения с прошлым кварталом и прогноза").unwrap();

        let reply = agent.handle_request(&followup).await.unwrap();
        assert!(reply.text.contains("Новый план"), "{}", reply.text);

        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, Some(Stage::Execution), "переработанный план ждёт утверждения");
        assert_eq!(state.pending_outcome.as_deref(), Some("добавлены QoQ и прогноз"));
        assert!(crate::memory::active_rejection(&state).is_none());

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        // Причина отклонения была в контексте модели с первого же запроса.
        assert!(requests[0].to_string().contains("нет сравнения с прошлым кварталом и прогноза"));
        // В принудительном круге модели доступен только move_stage, и вызвать его обязательно.
        let forced_tools: Vec<_> =
            requests[2]["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].clone()).collect();
        assert_eq!(forced_tools, vec![serde_json::json!("move_stage")]);
        assert_eq!(requests[2]["tool_choice"], "required");
    }

    #[tokio::test]
    async fn resume_sends_agent_back_to_unfinished_stage_with_collected_results() {
        let (base_url, requests) = mock_llm(vec![
            tool_call("update_step", serde_json::json!({ "step": "раздел 3: прогноз" })),
            serde_json::json!({ "content": "Продолжаю: раздел 3, прогноз на следующий квартал." }),
        ])
        .await;
        let store = TempStore::new();
        let manager = AgentManager::new(LlmClient::for_tests_at(&base_url), store.0.clone()).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("demo", Some("квартальный отчёт")).unwrap();
        call_tool(
            &agent,
            "move_stage",
            serde_json::json!({ "stage": "execution", "outcome": "ПЛАН: сбор, сравнение QoQ, прогноз" }),
        );
        agent.task_approve().unwrap();
        call_tool(&agent, "update_step", serde_json::json!({ "step": "раздел 2 из 3 готов" }));
        agent.task_pause().unwrap();

        // Кнопка «Возобновить»: интерфейс сразу отправляет followup агенту.
        let followup = agent.task_resume().unwrap().unwrap();
        let reply = agent.handle_request(&followup).await.unwrap();
        assert!(reply.text.contains("раздел 3"), "{}", reply.text);

        let first = requests.lock().unwrap()[0].to_string();
        assert!(first.contains("Задача возобновлена после паузы"), "{first}");
        assert!(first.contains("Сейчас этап EXECUTION"), "директива незавершённого этапа: {first}");
        assert!(first.contains("ПЛАН: сбор, сравнение QoQ, прогноз"), "итог planning уходит модели: {first}");
        assert!(first.contains("раздел 2 из 3 готов"), "текущий шаг уходит модели: {first}");
        assert!(!first.contains("НА ПАУЗЕ"));
        assert!(first.contains("\"tools\""), "после паузы инструменты автомата снова доступны");
        assert_eq!(task(&agent).stage, Stage::Execution);
    }

    #[tokio::test]
    async fn no_forced_round_when_model_resubmits_itself() {
        let (base_url, requests) = mock_llm(vec![
            tool_call("move_stage", serde_json::json!({ "stage": "execution", "outcome": "план v2" })),
            serde_json::json!({ "content": "План v2 ждёт утверждения." }),
        ])
        .await;
        let store = TempStore::new();
        let manager = AgentManager::new(LlmClient::for_tests_at(&base_url), store.0.clone()).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("demo", None).unwrap();
        move_stage(&agent, "execution");
        let followup = agent.task_reject("мало деталей").unwrap();

        agent.handle_request(&followup).await.unwrap();
        assert_eq!(task(&agent).pending_outcome.as_deref(), Some("план v2"));
        assert_eq!(requests.lock().unwrap().len(), 2, "лишнего круга нет");
    }

    #[tokio::test]
    async fn ordinary_request_still_goes_to_model() {
        let store = TempStore::new();
        let (manager, agent) = agent_with_task(&store);
        manager.start("tester").unwrap();

        // Не перехвачено — значит, дошло до (недоступной) модели.
        assert!(agent.handle_request("Какие требования к отчёту?").await.is_err());
    }

    // --- Гейты подтверждения человеком ---

    #[test]
    fn plan_approval_gates_execution() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        let reply = move_stage(&agent, "execution");
        assert!(reply.contains("ждёт подтверждения человека"), "{reply}");
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning, "переход не должен примениться без утверждения");
        assert_eq!(state.pending_stage, Some(Stage::Execution));
        assert_eq!(state.pending_outcome.as_deref(), Some("итог этапа"));
        assert!(!task_tools_offered(Some(&state)), "пока ждём решения, инструменты не предлагаются");

        let reply = move_stage(&agent, "execution");
        assert!(reply.contains("уже ждёт утверждения"), "{reply}");

        agent.task_approve().unwrap();
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Execution);
        assert_eq!(state.pending_stage, None);
        assert_eq!(state.pending_outcome, None);
        assert!(task_tools_offered(Some(&state)));
        assert!(agent.task_approve().is_err(), "утверждать больше нечего");
    }

    #[test]
    fn rejected_plan_stays_in_planning_with_reason() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");

        agent.task_reject("добавь критерии готовности").unwrap();
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, None);
        assert_eq!(state.expected_action, None, "причина — в журнале и блоке отклонения, не в ожидаемом действии");
        assert_eq!(state.transitions.last().unwrap().reason, "добавь критерии готовности");
        assert!(agent.task_reject("ещё раз").is_err(), "отклонять больше нечего");

        // Модель может доработать план и предложить переход заново.
        assert!(move_stage(&agent, "execution").contains("ждёт подтверждения"));
    }

    #[test]
    fn reject_requires_reason() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");

        for note in ["", "   "] {
            let err = agent.task_reject(note).unwrap_err().to_string();
            assert!(err.contains("укажите причину отклонения"), "{err}");
        }
        let state = task(&agent);
        assert_eq!(state.pending_stage, Some(Stage::Execution), "без причины предложение остаётся");
        assert_eq!(state.transitions.last().unwrap().outcome, crate::memory::TransitionOutcome::Proposed);
    }

    #[test]
    fn rejected_plan_is_reworked_with_the_reason() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");

        // Сообщение, которое интерфейс сразу отправляет агенту от имени человека.
        let followup = agent.task_reject("нет оценки сроков по шагам").unwrap();
        assert!(followup.contains("нет оценки сроков по шагам"), "{followup}");
        assert!(followup.contains("Переработай план"), "{followup}");

        // Причина видна модели в статусе задачи — и без этого сообщения (CLI).
        let block = crate::memory::format_task_block(&task(&agent));
        assert!(block.contains("ОТКЛОНИЛ предложенный переход «planning -> execution»"), "{block}");
        assert!(block.contains("Причина: нет оценки сроков по шагам"), "{block}");
        assert!(block.contains("Сейчас этап PLANNING"), "директива этапа остаётся: {block}");

        // Отказ автомата не отменяет напоминание — модель ещё не ответила на замечание…
        move_stage(&agent, "done");
        assert!(crate::memory::format_task_block(&task(&agent)).contains("ОТКЛОНИЛ"));
        // …а новое предложение плана — отменяет.
        move_stage(&agent, "execution");
        assert!(!crate::memory::format_task_block(&task(&agent)).contains("ОТКЛОНИЛ"));
    }

    #[test]
    fn rejected_result_followup_points_back_to_rework() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        for stage in [Stage::Execution, Stage::Validation] {
            agent.task_advance(stage, None, None).unwrap();
        }
        move_stage(&agent, "done");

        let followup = agent.task_reject("не проверены граничные случаи").unwrap();
        assert!(followup.contains("Итог проверки не принят"), "{followup}");
        assert_eq!(task(&agent).stage, Stage::Validation);
    }

    #[test]
    fn full_cycle_driven_by_model_with_rework() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);

        move_stage(&agent, "execution");
        agent.task_approve().unwrap();

        // execution -> validation и откат validation -> execution — без ворот.
        assert!(move_stage(&agent, "validation").contains("Переход применён"));
        assert!(move_stage(&agent, "execution").contains("Переход применён"));
        assert!(move_stage(&agent, "validation").contains("Переход применён"));
        assert_eq!(task(&agent).stage, Stage::Validation);

        // validation -> done — только с подтверждением.
        assert!(move_stage(&agent, "done").contains("ждёт подтверждения"));
        assert_eq!(task(&agent).stage, Stage::Validation);
        agent.task_approve().unwrap();
        assert_eq!(task(&agent).stage, Stage::Done);
    }

    // --- Пауза и продолжение ---

    #[test]
    fn pause_freezes_the_state_machine() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        agent.task_advance(Stage::Execution, Some("шаг 1"), None).unwrap();
        agent.task_pause().unwrap();

        let state = task(&agent);
        assert!(!task_tools_offered(Some(&state)));
        assert!(move_stage(&agent, "validation").contains("на паузе"));
        let reply = call_tool(&agent, "update_step", serde_json::json!({ "step": "шаг 2" }));
        assert!(reply.contains("на паузе"), "{reply}");
        let err = agent.task_advance(Stage::Validation, None, None).unwrap_err().to_string();
        assert!(err.contains("на паузе"), "{err}");

        let state = task(&agent);
        assert_eq!(state.stage, Stage::Execution);
        assert_eq!(state.current_step.as_deref(), Some("шаг 1"));
    }

    #[test]
    fn approve_is_refused_while_paused() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");
        agent.task_pause().unwrap();

        let err = agent.task_approve().unwrap_err().to_string();
        assert!(err.contains("на паузе"), "{err}");
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Planning);
        assert_eq!(state.pending_stage, Some(Stage::Execution), "предложение переживает паузу");

        agent.task_resume().unwrap();
        agent.task_approve().unwrap();
        assert_eq!(task(&agent).stage, Stage::Execution);
    }

    #[test]
    fn approve_is_refused_when_transition_forbidden_after_proposal() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        for stage in [Stage::Execution, Stage::Validation] {
            agent.task_advance(stage, None, None).unwrap();
        }
        assert!(move_stage(&agent, "done").contains("ждёт подтверждения"));
        agent.task_forbid_transition(Stage::Validation, Stage::Done).unwrap();

        let err = agent.task_approve().unwrap_err().to_string();
        assert!(err.contains("запрещён инвариантом"), "{err}");
        let state = task(&agent);
        assert_eq!(state.stage, Stage::Validation);
        assert_eq!(state.pending_stage, Some(Stage::Done), "предложение остаётся до решения человека");

        // Снять запрет — и то же предложение утверждается.
        agent.task_allow_transition(Stage::Validation, Stage::Done).unwrap();
        agent.task_approve().unwrap();
        assert_eq!(task(&agent).stage, Stage::Done);
    }

    #[test]
    fn manual_advance_supersedes_pending_proposal() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        assert!(move_stage(&agent, "execution").contains("ждёт подтверждения"));

        // Человек, не дожидаясь approve, сам провёл задачу дальше —
        // предложение модели (сделанное из planning) больше не актуально.
        agent.task_advance(Stage::Execution, None, None).unwrap();
        let state = task(&agent);
        assert_eq!(state.pending_stage, None);
        assert_eq!(state.pending_outcome, None);
        assert!(task_tools_offered(Some(&state)));

        agent.task_advance(Stage::Validation, None, None).unwrap();
        let err = agent.task_approve().unwrap_err().to_string();
        assert!(err.contains("нет перехода, ожидающего утверждения"), "{err}");
        assert_eq!(task(&agent).stage, Stage::Validation, "approve не должен откатывать этап назад");
    }

    #[test]
    fn step_update_keeps_pending_proposal() {
        let store = TempStore::new();
        let (_m, agent) = agent_with_task(&store);
        move_stage(&agent, "execution");

        // Правка шага без смены этапа предложение не снимает.
        agent.task_step("уточнил критерии").unwrap();
        agent.task_advance(Stage::Planning, Some("план v2"), None).unwrap();
        assert_eq!(task(&agent).pending_stage, Some(Stage::Execution));
        agent.task_approve().unwrap();
        assert_eq!(task(&agent).stage, Stage::Execution);
    }

    #[test]
    fn resume_continues_from_same_point_after_restart() {
        let store = TempStore::new();
        {
            let (_m, agent) = agent_with_task(&store);
            move_stage(&agent, "execution");
            agent.task_approve().unwrap();
            call_tool(
                &agent,
                "update_step",
                serde_json::json!({ "step": "раздел 2 из 3", "expected_action": "дописать выводы" }),
            );
            agent.task_pause().unwrap();
        }

        // «Перезапуск приложения» — новый реестр агентов над тем же файлом БД.
        let (_m, agent) = open_agent(&store);
        let state = task(&agent);
        assert!(state.paused);
        assert_eq!(state.stage, Stage::Execution);
        assert_eq!(state.current_step.as_deref(), Some("раздел 2 из 3"));
        assert_eq!(state.expected_action.as_deref(), Some("дописать выводы"));
        assert!(crate::memory::format_task_block(&state).contains("ЗАДАЧА НА ПАУЗЕ"));

        let followup = agent.task_resume().unwrap().expect("агент должен продолжить сам");
        assert!(followup.contains("незавершённого этапа «execution»"), "{followup}");
        assert!(followup.contains("раздел 2 из 3"), "{followup}");
        assert_eq!(agent.task_resume().unwrap(), None, "повторное возобновление ничего не запускает");
        let state = task(&agent);
        assert!(!state.paused);
        assert_eq!(state.stage, Stage::Execution);
        // Утверждённый план — в статусе задачи, а не только в истории диалога.
        assert_eq!(state.stage_path.len(), 1);
        assert!(crate::memory::format_task_block(&state).contains("Итоги пройденных этапов"));
        assert_eq!(state.current_step.as_deref(), Some("раздел 2 из 3"));
        assert_eq!(state.expected_action.as_deref(), Some("дописать выводы"));
        assert!(task_tools_offered(Some(&state)));

        // Автомат продолжает работать по тем же правилам, что и до паузы.
        assert!(move_stage(&agent, "done").contains("нет такого перехода"));
        assert!(move_stage(&agent, "validation").contains("Переход применён"));
    }

    #[tokio::test]
    async fn model_calls_mcp_tool_and_sees_its_result() {
        let (base_url, requests) = mock_llm(vec![
            tool_call("mcp__calc__add", serde_json::json!({ "a": 2, "b": 40 })),
            serde_json::json!({ "content": "Получилось 42." }),
        ])
        .await;
        let store = TempStore::new();
        let mcp = Arc::new(crate::mcp::McpManager::empty());
        mcp.attach_for_tests("calc", crate::mcp::test_support::spawn_calculator()).await;
        let manager = AgentManager::with_mcp(LlmClient::for_tests_at(&base_url), store.0.clone(), mcp).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();

        let reply = agent.handle_request("сколько будет 2 + 40?").await.unwrap();
        assert_eq!(reply.text, "Получилось 42.");
        assert_eq!(reply.tool_calls.len(), 1);
        let call = &reply.tool_calls[0];
        assert_eq!((call.server.as_deref(), call.tool.as_str(), call.result.as_str()), (Some("calc"), "add", "42"));
        assert!(!call.is_error);
        assert_eq!(call.headline(80), r#"calc · add({"a":2,"b":40})"#);

        let requests = requests.lock().unwrap();
        // Без задачи — только MCP-инструменты и без принуждения к вызову.
        let offered: Vec<&str> =
            requests[0]["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        assert_eq!(offered, ["mcp__calc__add", "mcp__calc__fail"]);
        assert_eq!(requests[0]["tool_choice"], "auto");
        // Результат вызова ушёл модели во втором круге.
        let tool_message = requests[1]["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(tool_message["role"], "tool");
        assert_eq!(tool_message["content"], "42");
        // В историю попадают только реплики человека и итоговый ответ.
        assert_eq!(agent.history().len(), 2);
    }

    #[tokio::test]
    async fn mcp_tools_are_offered_alongside_task_tools() {
        let (base_url, requests) = mock_llm(vec![
            tool_call("mcp__calc__fail", serde_json::json!({ "a": 1, "b": 1 })),
            serde_json::json!({ "content": "Инструмент сломался." }),
        ])
        .await;
        let store = TempStore::new();
        let mcp = Arc::new(crate::mcp::McpManager::empty());
        mcp.attach_for_tests("calc", crate::mcp::test_support::spawn_calculator()).await;
        let manager = AgentManager::with_mcp(LlmClient::for_tests_at(&base_url), store.0.clone(), mcp).unwrap();
        manager.create(AgentConfig::new("tester")).unwrap();
        manager.start("tester").unwrap();
        let agent = manager.get("tester").unwrap();
        agent.task_start("задача", None).unwrap();

        let reply = agent.handle_request("составь план").await.unwrap();
        assert!(reply.tool_calls[0].is_error);

        let requests = requests.lock().unwrap();
        let offered: Vec<&str> =
            requests[0]["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        assert_eq!(offered, ["move_stage", "update_step", "mcp__calc__add", "mcp__calc__fail"]);
        assert_eq!(requests[0]["tool_choice"], "required");
    }
}
