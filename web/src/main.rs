//! Веб-интерфейс LLM-агента (axum): чат в браузере со счётчиком токенов на запрос и за сессию.

use anyhow::Result;
use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use llm_core::LlmClient;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    client: Arc<LlmClient>,
    index_html: Arc<String>,
}

#[derive(Deserialize)]
struct AskRequest {
    prompt: String,
}

const INDEX_TEMPLATE: &str = include_str!("index.html");

async fn index(State(state): State<AppState>) -> Html<String> {
    Html((*state.index_html).clone())
}

async fn ask(
    State(state): State<AppState>,
    Json(req): Json<AskRequest>,
) -> Json<serde_json::Value> {
    match state.client.ask(&req.prompt).await {
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
    let index_html = INDEX_TEMPLATE.replace("__MODEL_NAME__", client.model());

    let state = AppState {
        client: Arc::new(client),
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
