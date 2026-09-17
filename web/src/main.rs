//! Веб-интерфейс LLM-агента (axum): чат в браузере со счётчиком токенов на запрос и за сессию.

use anyhow::Result;
use axum::{
    extract::{Path, State},
    response::Html,
    routing::{get, post},
    Json, Router,
};
use llm_core::{AgentConfig, AgentManager, ChatMessage, ChatOptions, ContextStrategy, LlmClient};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    client: Arc<LlmClient>,
    /// Модель для запросов анализа решений (вкладка "Задача · 4 способа" -> "Проверить
    /// решения моделью"). Задаётся через LLM_ANALYSIS_MODEL; если переменная не задана,
    /// совпадает с основной моделью клиента.
    analysis_model: Arc<String>,
    index_html: Arc<String>,
    /// Реестр именованных агентов (вкладка "Агенты"): у каждого своя конфигурация и
    /// жизненный цикл (запущен/остановлен), запросы обрабатывает сам агент, а не
    /// прямой вызов клиента.
    agents: Arc<AgentManager>,
}

#[derive(Deserialize)]
struct AskRequest {
    prompt: String,
    /// Предыдущие сообщения диалога — присылает браузер (хранит их у себя в памяти вкладки,
    /// пока пользователь не нажмёт "Новая сессия"). Сервер не хранит историю обычного чата
    /// сам — в отличие от именованных агентов, у которых она в SQLite; здесь сервер просто
    /// пересылает то, что прислал клиент, добавляя новое сообщение пользователя.
    #[serde(default)]
    history: Vec<ChatMessage>,
    /// JSON Schema желаемого формата ответа — необязательная, задаётся в настройках интерфейса.
    /// Если задана, модели передаётся системная инструкция и нативный response_format.
    #[serde(default)]
    json_schema: Option<serde_json::Value>,
    /// Ограничение длины ответа в токенах (передаётся модели как max_tokens).
    #[serde(default)]
    max_tokens: Option<u32>,
    /// Стоп-последовательности — генерация обрывается, как только модель их выдаст.
    #[serde(default)]
    stop: Vec<String>,
    /// Температура сэмплирования.
    #[serde(default)]
    temperature: Option<f32>,
    /// Nucleus sampling (top_p).
    #[serde(default)]
    top_p: Option<f32>,
    /// Явно включить/выключить режим рассуждений (`enable_thinking`) у моделей, которые его
    /// поддерживают. `None` — не переопределять поведение модели по умолчанию.
    #[serde(default)]
    reasoning: Option<bool>,
    /// Если true — запрос использует модель для анализа (LLM_ANALYSIS_MODEL) вместо основной.
    #[serde(default)]
    analysis: bool,
}

/// Тело запроса на создание агента (вкладка "Агенты").
#[derive(Deserialize)]
struct CreateAgentRequest {
    name: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    show_tokens: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    reasoning: Option<bool>,
    /// Стратегия управления контекстом — "full" (по умолчанию, без управления),
    /// "summary", "sliding-window", "facts" или "branching" (см. llm_core::ContextStrategy).
    #[serde(default = "default_strategy_str")]
    context_strategy: String,
    /// Переопределение размера окна для sliding-window/facts (см. AgentConfig::window_size).
    #[serde(default)]
    window_size: Option<usize>,
    /// Имя профиля персонализации (см. llm_core::profile) — пусто/не задано
    /// означает "default", "none" явно отключает персонализацию.
    #[serde(default)]
    profile: Option<String>,
}

fn default_strategy_str() -> String {
    ContextStrategy::default().to_string()
}

#[derive(Deserialize)]
struct AgentAskRequest {
    prompt: String,
}

#[derive(Deserialize)]
struct SetStrategyRequest {
    context_strategy: String,
}

/// Тело запроса на смену профиля персонализации агента (см. llm_core::profile) —
/// `profile` может быть именем существующего файла, "default" или "none".
#[derive(Deserialize)]
struct SetProfileRequest {
    profile: String,
}

/// Тело запроса на создание нового профиля (см. llm_core::profile::create) —
/// `content` не задан → используется пустой шаблон (llm_core::profile::template).
#[derive(Deserialize)]
struct CreateProfileRequest {
    name: String,
    #[serde(default)]
    content: Option<String>,
}

/// Тело запроса на создание нового инварианта (см. llm_core::invariants::create) —
/// `content` не задан → используется пустой шаблон (llm_core::invariants::template).
/// В отличие от профиля инвариант ни к какому агенту не привязан — общий для всех.
#[derive(Deserialize)]
struct CreateInvariantRequest {
    id: String,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct CheckpointRequest {
    label: String,
}

#[derive(Deserialize)]
struct BranchRequest {
    new_branch: String,
    #[serde(default)]
    from_checkpoint: Option<String>,
}

#[derive(Deserialize)]
struct SwitchBranchRequest {
    branch: String,
}

/// Тело запроса на сохранение записи долговременной памяти (см. llm_core::memory) —
/// `category` пустая/не задана означает "knowledge" (см. `Agent::remember`).
#[derive(Deserialize)]
struct RememberRequest {
    key: String,
    value: String,
    #[serde(default)]
    category: String,
}

#[derive(Deserialize)]
struct ForgetRequest {
    key: String,
}

/// Тело запроса на создание новой ОБЩЕЙ задачи (рабочая память, см.
/// llm_core::memory) — вызывающий агент сразу к ней присоединяется.
#[derive(Deserialize)]
struct TaskStartRequest {
    name: String,
    #[serde(default)]
    goal: Option<String>,
}

/// Тело запроса на присоединение к уже существующей общей задаче.
#[derive(Deserialize)]
struct TaskJoinRequest {
    name: String,
}

#[derive(Deserialize)]
struct TaskSetRequest {
    key: String,
    value: String,
}

/// Тело запроса на переход конечного автомата задачи (см. llm_core::memory::Stage) —
/// `stage` обязателен, `step`/`expect` опциональны (не заданы = соответствующее
/// поле не меняется при этом переходе, см. `Agent::task_advance`).
#[derive(Deserialize)]
struct TaskAdvanceRequest {
    stage: String,
    #[serde(default)]
    step: Option<String>,
    #[serde(default)]
    expect: Option<String>,
}

#[derive(Deserialize)]
struct TaskStepRequest {
    step: String,
}

#[derive(Deserialize)]
struct TaskExpectRequest {
    expect: String,
}

/// Тело запроса на сжатие фрагмента истории обычного чата в сводку (см.
/// `llm_core::context`) — обычный чат не хранит состояние на сервере, поэтому
/// сама сводка и счётчик уже сжатых сообщений живут в браузере, а сервер лишь
/// выполняет отдельное обращение к LLM по готовому фрагменту.
#[derive(Deserialize)]
struct SummarizeRequest {
    #[serde(default)]
    previous_summary: String,
    messages: Vec<ChatMessage>,
}

const INDEX_TEMPLATE: &str = include_str!("index.html");

async fn index(State(state): State<AppState>) -> Html<String> {
    Html((*state.index_html).clone())
}

async fn ask(
    State(state): State<AppState>,
    Json(req): Json<AskRequest>,
) -> Json<serde_json::Value> {
    let mut messages = Vec::new();
    let mut response_format = None;
    if let Some(schema) = req.json_schema {
        let pretty_schema = serde_json::to_string_pretty(&schema).unwrap_or_default();
        messages.push(ChatMessage::system(format!(
            "Отвечай строго валидным JSON, соответствующим следующей JSON Schema. \
             Не добавляй пояснений, markdown-разметку или текст вне JSON.\n\n{pretty_schema}"
        )));
        response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema, "strict": true },
        }));
    }
    messages.extend(req.history);
    messages.push(ChatMessage::user(req.prompt));

    let options = ChatOptions {
        max_tokens: req.max_tokens.filter(|&n| n > 0),
        stop: req.stop.into_iter().filter(|s| !s.trim().is_empty()).collect(),
        temperature: req.temperature,
        top_p: req.top_p,
        response_format,
        reasoning: req.reasoning,
    };

    let model: &str = if req.analysis { state.analysis_model.as_str() } else { state.client.model() };

    match state.client.chat_with_model(model, &messages, &options).await {
        Ok(completion) => Json(serde_json::json!({
            "answer": completion.content,
            "usage": completion.usage,
            "requestJson": completion.request_json,
            "responseJson": completion.response_json,
            "context_window": state.client.context_window(),
        })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Сжимает присланный фрагмент диалога обычного чата в обновлённую сводку —
/// та же логика управления контекстом, что у именованных агентов (см.
/// `llm_core::context::summarize_chunk`), но состояние (сама сводка, счётчик
/// уже сжатых сообщений) хранится в браузере, а не на сервере, поскольку
/// обычный чат вообще не персистится на бэкенде.
async fn summarize(
    State(state): State<AppState>,
    Json(req): Json<SummarizeRequest>,
) -> Json<serde_json::Value> {
    match llm_core::context::summarize_chunk(&state.client, state.client.model(), &req.previous_summary, &req.messages)
        .await
    {
        Ok(summary) => Json(serde_json::json!({ "summary": summary })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Возвращает список всех агентов с их конфигурацией и статусом запуска.
async fn list_agents(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "agents": state.agents.list() }))
}

/// Добавляет нового агента (по умолчанию — остановленного).
async fn create_agent(
    State(state): State<AppState>,
    Json(req): Json<CreateAgentRequest>,
) -> Json<serde_json::Value> {
    let context_strategy: ContextStrategy = match req.context_strategy.parse() {
        Ok(strategy) => strategy,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    let config = AgentConfig {
        name: req.name,
        system_prompt: req.system_prompt.filter(|s| !s.trim().is_empty()),
        model: req.model.filter(|s| !s.trim().is_empty()),
        show_tokens: req.show_tokens,
        max_tokens: req.max_tokens.filter(|&n| n > 0),
        temperature: req.temperature,
        top_p: req.top_p,
        reasoning: req.reasoning,
        context_strategy,
        window_size: req.window_size.filter(|&n| n > 0),
        profile: req.profile.filter(|s| !s.trim().is_empty()),
    };
    match state.agents.create(config) {
        Ok(info) => Json(serde_json::json!({ "agent": info })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Меняет стратегию управления контекстом уже существующего агента на лету.
async fn set_agent_strategy(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SetStrategyRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let strategy: ContextStrategy = match req.context_strategy.parse() {
        Ok(strategy) => strategy,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    let mut config = agent.config();
    config.context_strategy = strategy;
    match agent.set_config(config) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Меняет профиль персонализации агента (см. llm_core::profile) — отдельная
/// от context_strategy и от памяти ось: подключается к КАЖДОМУ запросу этого
/// агента, независимо от них. `profile: "none"` явно отключает персонализацию.
async fn set_agent_profile(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SetProfileRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let mut config = agent.config();
    config.profile = Some(req.profile);
    match agent.set_config(config) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Список всех профилей, найденных в каталоге профилей (см.
/// llm_core::profile::list_profiles) — используется фронтендом, чтобы
/// предложить выбор при переключении профиля агента.
async fn list_profiles() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "profiles": llm_core::list_profiles(),
        "dir": llm_core::profile::profiles_dir().display().to_string(),
    }))
}

/// Создаёт новый файл профиля на диске (см. llm_core::profile::create) — так
/// профиль можно завести прямо из интерфейса, а не правкой файлов руками.
/// Ошибка, если имя пустое, зарезервировано ("none") или профиль с таким
/// именем уже существует.
async fn create_profile(Json(req): Json<CreateProfileRequest>) -> Json<serde_json::Value> {
    let content = req.content.unwrap_or_else(|| llm_core::profile::template(&req.name));
    match llm_core::profile::create(&req.name, &content) {
        Ok(()) => Json(serde_json::json!({ "profiles": llm_core::list_profiles() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Список ВСЕХ инвариантов (см. llm_core::invariants) вместе с их текстом —
/// в отличие от профилей это не выбор для конкретного агента, а полный набор
/// правил, действующих сразу для всех, поэтому отдаём содержимое целиком, а
/// не только идентификаторы (фронтенду нужно показать текст, не просто список).
async fn list_invariants() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "invariants": llm_core::invariants::load_all(),
        "dir": llm_core::invariants::invariants_dir().display().to_string(),
    }))
}

/// Создаёт новый инвариант на диске (см. llm_core::invariants::create) — как
/// и профиль, заводится через интерфейс, а не правкой файлов руками. Ошибка,
/// если идентификатор пустой или инвариант с таким id уже существует.
async fn create_invariant(Json(req): Json<CreateInvariantRequest>) -> Json<serde_json::Value> {
    let content = req.content.unwrap_or_else(|| llm_core::invariants::template(&req.id));
    match llm_core::invariants::create(&req.id, &content) {
        Ok(()) => Json(serde_json::json!({ "invariants": llm_core::invariants::load_all() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Удаляет инвариант по идентификатору (см. llm_core::invariants::remove) —
/// единственный способ снять его действие для всех агентов сразу.
async fn delete_invariant(Path(id): Path<String>) -> Json<serde_json::Value> {
    match llm_core::invariants::remove(&id) {
        Ok(_) => Json(serde_json::json!({ "invariants": llm_core::invariants::load_all() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Отмечает checkpoint в текущей ветке агента (стратегия branching).
async fn create_checkpoint(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<CheckpointRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.checkpoint(&req.label) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Ответвляет новую ветку от checkpoint'а (или от текущего конца текущей ветки).
async fn create_branch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BranchRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.branch_from(req.from_checkpoint.as_deref(), &req.new_branch) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Переключает активную ветку агента.
async fn switch_branch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SwitchBranchRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.switch_branch(&req.branch) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info(), "history": agent.history_with_usage()
            .into_iter()
            .map(|(m, usage)| serde_json::json!({ "role": m.role, "content": m.content, "usage": usage }))
            .collect::<Vec<_>>() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Явно сохраняет (или обновляет) одну запись долговременной памяти агента
/// (см. llm_core::memory) — единственный способ туда что-то положить, ничего
/// не пишется сюда автоматически по ответу модели.
async fn remember_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<RememberRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.remember(&req.key, &req.value, &req.category) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Удаляет запись долговременной памяти агента.
async fn forget_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ForgetRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.forget(&req.key) {
        Ok(existed) => Json(serde_json::json!({ "agent": agent.info(), "existed": existed })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Создаёт новую ОБЩУЮ задачу (рабочая память, см. llm_core::memory) и сразу
/// присоединяет к ней вызывающего агента — ошибка, если у него уже есть
/// активная задача, или если задача с таким именем уже существует (тогда
/// нужен `.../task/join`, а не `.../task/start`).
async fn task_start_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskStartRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_start(&req.name, req.goal.as_deref()) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Присоединяет агента к уже существующей общей задаче (созданной им самим
/// или другим агентом) — с этого момента он делит её рабочую память со всеми
/// остальными участниками. Ошибка, если задачи с таким именем нет, или если у
/// агента уже есть другая активная задача.
async fn task_join_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskJoinRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_join(&req.name) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Сохраняет пару ключ/значение в рабочую память задачи, к которой сейчас
/// присоединён агент — видна сразу всем остальным её участникам. Ошибка, если
/// агент ни к какой задаче не присоединён.
async fn task_set_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskSetRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_set(&req.key, &req.value) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Завершает задачу, к которой присоединён агент, — ДЛЯ ВСЕХ её участников
/// разом: рабочая память удаляется целиком, а членство остальных агентов
/// снимается автоматически (см. `Agent::task_finish`).
async fn task_finish_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_finish() {
        Ok(finished) => Json(serde_json::json!({ "agent": agent.info(), "finished": finished })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Переводит задачу, к которой присоединён агент, на следующий этап
/// конечного автомата (см. `Agent::task_advance`) — только если переход
/// легален; иначе `error` объясняет, куда можно перейти прямо сейчас.
async fn task_advance_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskAdvanceRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let stage: llm_core::Stage = match req.stage.parse() {
        Ok(stage) => stage,
        Err(err) => return Json(serde_json::json!({ "error": format!("{err:#}") })),
    };
    match agent.task_advance(stage, req.step.as_deref(), req.expect.as_deref()) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Обновляет текущий шаг задачи, не меняя этап.
async fn task_step_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskStepRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_step(&req.step) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Обновляет ожидаемое действие задачи, не меняя этап.
async fn task_expect_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskExpectRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_expect(&req.expect) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Ставит задачу на паузу — на любом этапе, независимо от него.
async fn task_pause_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_pause() {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Снимает паузу — задача продолжается с того же этапа/шага/ожидаемого
/// действия, без необходимости заново объяснять контекст агенту (см.
/// `Agent::task_resume`).
async fn task_resume_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_resume() {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Применяет переход, который модель предложила вызовом инструмента
/// `move_stage`, но который требовал утверждения человеком — см.
/// `Agent::task_approve` и `TaskState::pending_stage`.
async fn task_approve_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_approve() {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

#[derive(Deserialize)]
struct TaskRejectRequest {
    #[serde(default)]
    note: String,
}

/// Отклоняет предложенный моделью переход — см. `Agent::task_reject`.
async fn task_reject_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskRejectRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_reject(&req.note) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Тело запроса на инвариант, привязанный ТОЛЬКО к задаче (см.
/// llm_core::memory::TaskState::invariants) — в отличие от глобальных
/// (`/api/invariants`, `/api/agents/:name/remember` с category="invariant"),
/// виден и действует только пока агент работает над этой задачей.
#[derive(Deserialize)]
struct TaskInvariantRequest {
    id: String,
    text: String,
}

/// Сохраняет текстовый инвариант задачи — см. `Agent::task_invariant_set`.
async fn task_invariant_set_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskInvariantRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_invariant_set(&req.id, &req.text) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Удаляет инвариант задачи — см. `Agent::task_invariant_remove`.
async fn task_invariant_remove_agent(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    match agent.task_invariant_remove(&id) {
        Ok(_) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Тело запроса на структурное правило перехода автомата задачи (см.
/// llm_core::memory::TaskState::blocked_transitions/extra_approval_transitions) —
/// `from`/`to` — машинные имена этапов (planning/execution/validation/done).
#[derive(Deserialize)]
struct TaskTransitionRuleRequest {
    from: String,
    to: String,
}

fn parse_transition_rule_request(
    req: &TaskTransitionRuleRequest,
) -> anyhow::Result<(llm_core::Stage, llm_core::Stage)> {
    Ok((req.from.parse()?, req.to.parse()?))
}

/// Дополнительно ЗАПРЕЩАЕТ переход для этой задачи (абсолютно) — см.
/// `Agent::task_forbid_transition`.
async fn task_forbid_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskTransitionRuleRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let (from, to) = match parse_transition_rule_request(&req) {
        Ok(pair) => pair,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    match agent.task_forbid_transition(from, to) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Снимает запрет, поставленный `task_forbid_agent` — см. `Agent::task_allow_transition`.
async fn task_allow_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskTransitionRuleRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let (from, to) = match parse_transition_rule_request(&req) {
        Ok(pair) => pair,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    match agent.task_allow_transition(from, to) {
        Ok(_) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Требует для этой задачи подтверждения человеком на переход, предложенный
/// МОДЕЛЬЮ — см. `Agent::task_require_approval`.
async fn task_require_approval_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskTransitionRuleRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let (from, to) = match parse_transition_rule_request(&req) {
        Ok(pair) => pair,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    match agent.task_require_approval(from, to) {
        Ok(()) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Снимает гейт, поставленный `task_require_approval_agent` — см.
/// `Agent::task_unrequire_approval`.
async fn task_unrequire_approval_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<TaskTransitionRuleRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let (from, to) = match parse_transition_rule_request(&req) {
        Ok(pair) => pair,
        Err(err) => return Json(serde_json::json!({ "error": err.to_string() })),
    };
    match agent.task_unrequire_approval(from, to) {
        Ok(_) => Json(serde_json::json!({ "agent": agent.info() })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Список всех существующих общих задач с их участниками — не привязан к
/// конкретному агенту, используется фронтендом, чтобы предложить задачи,
/// доступные для `.../task/join`.
async fn list_tasks(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "tasks": state.agents.list_tasks() }))
}

async fn start_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    match state.agents.start(&name) {
        Ok(info) => Json(serde_json::json!({ "agent": info })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

async fn stop_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    match state.agents.stop(&name) {
        Ok(info) => Json(serde_json::json!({ "agent": info })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

async fn delete_agent(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    match state.agents.remove(&name) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Пересылает запрос конкретному агенту. Если агент остановлен или не найден,
/// возвращает ошибку — обращения к LLM не происходит.
async fn ask_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<AgentAskRequest>,
) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    ensure_task(&agent, &name);
    match agent.handle_request(&req.prompt).await {
        Ok(reply) => {
            // Свежий статус стратегии ПОСЛЕ этого обмена (и возможного пересчёта
            // сводки/фактов выше) — интерфейс показывает по нему прогресс до
            // следующего сжатия/окно/факты, рядом с индикатором заполнения контекста.
            let info = agent.info();
            Json(serde_json::json!({
                "answer": reply.text,
                "usage": reply.usage,
                "context_window": agent.context_window(),
                "summarized": reply.summarized,
                "summary_covers": reply.summary_covers,
                "facts_updated": reply.facts_updated,
                "requestJson": reply.request_json,
                "responseJson": reply.response_json,
                "compression": info.compression,
                "sliding_window": info.sliding_window,
                "facts": info.facts,
                "branching": info.branching,
                "profile": info.profile,
                // Этап конечного автомата задачи ПОСЛЕ этого обмена — задача уже
                // заведена (ensure_task выше) и могла сдвинуться вызовом move_stage
                // внутри handle_request; веб-интерфейс показывает его прямо в чате
                // (полоска вверху + панель у кнопки "Отправить"), не в панели памяти.
                "task": info.task,
            }))
        }
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

/// Веб-интерфейс ведёт задачу за пользователя, без ручного `agent task start`:
/// первый же вопрос агенту без активной задачи заводит её автоматически, сразу
/// в этапе `planning` (значение по умолчанию [`llm_core::Stage`]) — см.
/// документацию модуля [`llm_core::memory`] за тем, что это довесок только для
/// веб-интерфейса: CLI и TUI по-прежнему требуют явного `agent task start`,
/// как и весь остальной "ничего не пишется автоматически" принцип рабочей и
/// долговременной памяти.
///
/// Имя задачи — имя самого агента: у каждого именованного агента в этом
/// потоке не больше одной активной задачи одновременно, и после `task/finish`
/// то же имя свободно для следующей. Коллизия (то же имя занято ЧУЖИМ
/// агентом) — редкий случай при обычном использовании веб-интерфейса,
/// поэтому просто добавляем метку времени и пробуем ещё раз, не блокируя
/// обмен из-за этого.
fn ensure_task(agent: &llm_core::Agent, agent_name: &str) {
    if agent.task_state().is_some() {
        return;
    }
    if agent.task_start(agent_name, None).is_ok() {
        return;
    }
    let unique = format!(
        "{agent_name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
    );
    let _ = agent.task_start(&unique, None);
}

/// Возвращает историю диалога агента, восстановленную из SQLite — используется
/// интерфейсом, чтобы показать прежние сообщения при открытии чата, даже если
/// сервер был перезапущен после последнего обращения к агенту.
async fn agent_history(State(state): State<AppState>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let Some(agent) = state.agents.get(&name) else {
        return Json(serde_json::json!({ "error": format!("агент «{name}» не найден") }));
    };
    let messages: Vec<serde_json::Value> = agent
        .history_with_usage()
        .into_iter()
        .map(|(m, usage)| serde_json::json!({ "role": m.role, "content": m.content, "usage": usage }))
        .collect();
    let info = agent.info();
    Json(serde_json::json!({
        "messages": messages,
        "context_window": info.context_window,
        "compression": info.compression,
        "sliding_window": info.sliding_window,
        "facts": info.facts,
        "branching": info.branching,
        "profile": info.profile,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = LlmClient::from_env()?;
    let analysis_model =
        std::env::var("LLM_ANALYSIS_MODEL").unwrap_or_else(|_| client.model().to_string());
    // Ставки цены (см. llm_core::pricing) передаются на фронтенд одним JSON-блобом,
    // чтобы стоимость и новых, и восстановленных из SQLite сообщений в чате агента
    // считалась одним и тем же способом на клиенте, а не дублировалась ещё и в
    // Rust. Объект всегда есть (не `null`) — currency нужна независимо от того,
    // заданы ли ставки оценки: реальная стоимость от провайдера (usage.cost,
    // если он её прислал) показывается даже без LLM_PRICE_INPUT_PER_1M/OUTPUT_PER_1M.
    // inputPerMillion/outputPerMillion — `null`, если ставки не заданы (тогда
    // оценка недоступна, но реальная стоимость от провайдера всё ещё может быть).
    let pricing = llm_core::pricing::from_env();
    let pricing_json = serde_json::json!({
        "inputPerMillion": pricing.map(|p| p.input_per_million),
        "outputPerMillion": pricing.map(|p| p.output_per_million),
        "currency": llm_core::pricing::currency(),
    })
    .to_string();
    let index_html = INDEX_TEMPLATE
        .replace("__MODEL_NAME__", client.model())
        .replace("__ANALYSIS_MODEL_NAME__", &analysis_model)
        .replace("__CONTEXT_SUMMARY_CHUNK__", &llm_core::context::context_summary_chunk().to_string())
        .replace("__SLIDING_WINDOW_SIZE__", &llm_core::context::sliding_window_size().to_string())
        .replace("__PRICING_JSON__", &pricing_json);
    let agents = Arc::new(AgentManager::from_env(client.clone())?);

    let state = AppState {
        client: Arc::new(client),
        analysis_model: Arc::new(analysis_model),
        index_html: Arc::new(index_html),
        agents,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/ask", post(ask))
        .route("/api/summarize", post(summarize))
        .route("/api/agents", get(list_agents).post(create_agent))
        .route("/api/agents/:name", axum::routing::delete(delete_agent))
        .route("/api/agents/:name/start", post(start_agent))
        .route("/api/agents/:name/stop", post(stop_agent))
        .route("/api/agents/:name/ask", post(ask_agent))
        .route("/api/agents/:name/history", get(agent_history))
        .route("/api/agents/:name/strategy", post(set_agent_strategy))
        .route("/api/agents/:name/profile", post(set_agent_profile))
        .route("/api/profiles", get(list_profiles).post(create_profile))
        .route(
            "/api/invariants",
            get(list_invariants).post(create_invariant),
        )
        .route("/api/invariants/:id", axum::routing::delete(delete_invariant))
        .route("/api/agents/:name/checkpoint", post(create_checkpoint))
        .route("/api/agents/:name/branch", post(create_branch))
        .route("/api/agents/:name/switch", post(switch_branch))
        .route("/api/agents/:name/remember", post(remember_agent))
        .route("/api/agents/:name/forget", post(forget_agent))
        .route("/api/agents/:name/task/start", post(task_start_agent))
        .route("/api/agents/:name/task/join", post(task_join_agent))
        .route("/api/agents/:name/task/set", post(task_set_agent))
        .route("/api/agents/:name/task/advance", post(task_advance_agent))
        .route("/api/agents/:name/task/step", post(task_step_agent))
        .route("/api/agents/:name/task/expect", post(task_expect_agent))
        .route("/api/agents/:name/task/pause", post(task_pause_agent))
        .route("/api/agents/:name/task/resume", post(task_resume_agent))
        .route("/api/agents/:name/task/approve", post(task_approve_agent))
        .route("/api/agents/:name/task/reject", post(task_reject_agent))
        .route("/api/agents/:name/task/finish", post(task_finish_agent))
        .route("/api/agents/:name/task/invariant", post(task_invariant_set_agent))
        .route(
            "/api/agents/:name/task/invariant/:id",
            axum::routing::delete(task_invariant_remove_agent),
        )
        .route("/api/agents/:name/task/forbid", post(task_forbid_agent))
        .route("/api/agents/:name/task/allow", post(task_allow_agent))
        .route("/api/agents/:name/task/require-approval", post(task_require_approval_agent))
        .route("/api/agents/:name/task/unrequire-approval", post(task_unrequire_approval_agent))
        .route("/api/tasks", get(list_tasks))
        .with_state(state);

    let port = std::env::var("PORT").ok().and_then(|v| v.trim().parse::<u16>().ok()).unwrap_or(8080);
    let addr = format!("0.0.0.0:{port}");
    println!("Веб-интерфейс запущен: http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
