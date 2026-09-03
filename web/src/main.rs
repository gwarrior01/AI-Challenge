//! Веб-интерфейс LLM-агента (axum): чат в браузере со счётчиком токенов на запрос и за сессию.

use anyhow::Result;
use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use llm_core::{ChatMessage, ChatOptions, LlmClient};
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
}

#[derive(Deserialize)]
struct AskRequest {
    prompt: String,
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
        })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = LlmClient::from_env()?;
    let analysis_model =
        std::env::var("LLM_ANALYSIS_MODEL").unwrap_or_else(|_| client.model().to_string());
    let index_html = INDEX_TEMPLATE
        .replace("__MODEL_NAME__", client.model())
        .replace("__ANALYSIS_MODEL_NAME__", &analysis_model);

    let state = AppState {
        client: Arc::new(client),
        analysis_model: Arc::new(analysis_model),
        index_html: Arc::new(index_html),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/ask", post(ask))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    println!("Веб-интерфейс запущен: http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
