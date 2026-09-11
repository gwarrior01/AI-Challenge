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
            }))
        }
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
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
        .route("/api/agents/:name/checkpoint", post(create_checkpoint))
        .route("/api/agents/:name/branch", post(create_branch))
        .route("/api/agents/:name/switch", post(switch_branch))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    println!("Веб-интерфейс запущен: http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
