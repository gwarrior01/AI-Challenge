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

use crate::{context, context::ContextStrategy, ChatMessage, ChatOptions, LlmClient, Usage};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Имя ветки, с которой начинается диалог любого агента и которая остаётся
/// единственной, пока активна не-ветвящаяся стратегия контекста.
pub const MAIN_BRANCH: &str = "main";

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
        }
    }
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
    db: Arc<Db>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: AgentConfig,
        client: LlmClient,
        db: Arc<Db>,
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
        AgentInfo {
            running: self.is_running(),
            context_window: self.context_window(),
            compression,
            sliding_window,
            facts,
            branching,
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
        if !self.is_running() {
            bail!("агент «{}» остановлен — сначала запустите его", self.name);
        }

        let config = self.config();
        let mut messages = Vec::new();
        if let Some(system_prompt) = &config.system_prompt {
            if !system_prompt.trim().is_empty() {
                messages.push(ChatMessage::system(system_prompt.clone()));
            }
        }

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
        let completion = self.client.chat_with_model(&model, &messages, &options).await?;

        let user_message = ChatMessage::user(prompt.to_string());
        let assistant_message = ChatMessage::assistant(completion.content.clone());
        let branch_name = {
            let mut branches = self.branches.lock().expect("ветки агента отравлены паникой");
            let current = branches.current.clone();
            let history = branches.current_messages_mut();
            history.push((user_message.clone(), None));
            history.push((assistant_message.clone(), completion.usage));
            current
        };
        // Лучшая попытка: ответ уже получен и не должен потеряться для
        // пользователя из-за сбоя записи в БД — при ошибке лишь предупреждаем.
        if let Err(err) = self.db.append_message(&self.name, &branch_name, &user_message, None) {
            eprintln!("не удалось сохранить сообщение пользователя в БД: {err:#}");
        }
        // Метрики токенов приходят от API одной суммой на весь обмен (запрос +
        // ответ), поэтому сохраняем их при сообщении ассистента — иначе они
        // задвоились бы при подсчёте суммы по диалогу.
        if let Err(err) = self.db.append_message(&self.name, &branch_name, &assistant_message, completion.usage)
        {
            eprintln!("не удалось сохранить ответ агента в БД: {err:#}");
        }

        let summary_covers = self.update_summary_if_needed(&config).await;
        let facts_updated = if config.context_strategy == ContextStrategy::Facts {
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
        })
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
             );",
        )
        .context("не удалось создать таблицы БД агентов")?;
        Self::migrate_token_columns(&conn).context("не удалось обновить схему БД агентов")?;
        Self::migrate_branch_column(&conn).context("не удалось обновить схему БД агентов (ветки)")?;
        Self::migrate_cost_columns(&conn).context("не удалось обновить схему БД агентов (стоимость)")?;
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
    agents: RwLock<HashMap<String, Arc<Agent>>>,
}

impl AgentManager {
    pub fn new(client: LlmClient, store_path: impl Into<PathBuf>) -> Result<Self> {
        let db = Arc::new(Db::open(&store_path.into())?);
        let manager = Self { client, db, agents: RwLock::new(HashMap::new()) };
        manager.load()?;
        Ok(manager)
    }

    /// Путь к файлу БД берётся из AGENTS_STORE_PATH, по умолчанию — `agents.db`
    /// в текущей рабочей директории.
    pub fn from_env(client: LlmClient) -> Result<Self> {
        let path = std::env::var("AGENTS_STORE_PATH").unwrap_or_else(|_| "agents.db".to_string());
        Self::new(client, path)
    }

    fn load(&self) -> Result<()> {
        let records = self.db.load_agents()?;
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        for (config, running) in records {
            let branches = self.db.load_branches(&config.name)?;
            let summary = self.db.load_summary(&config.name)?;
            let facts = self.db.load_facts(&config.name)?;
            let agent = Agent::new(config, self.client.clone(), self.db.clone(), branches, summary, facts);
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
            Branches::default(),
            CompressionState::default(),
            FactsState::default(),
        );
        let info = agent.info();
        agents.insert(name, Arc::new(agent));

        Ok(info)
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
