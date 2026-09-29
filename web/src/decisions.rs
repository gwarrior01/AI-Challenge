//! Вкладка «Jev»: вопросы к модели решений Jev (OpenRouter Decisions API,
//! см. `llm_core::decisions`) и её ответы с вероятностями.

use axum::{
    routing::{get, post},
    Json, Router,
};
use llm_core::decisions::{self, DecisionsConfig, Question};
use serde::Deserialize;

use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/api/decisions/config", get(config)).route("/api/decisions", post(decide))
}

/// Модель, есть ли ключ и пример вопросов — для начального вида вкладки.
async fn config() -> Json<serde_json::Value> {
    let cfg = DecisionsConfig::from_env();
    Json(serde_json::json!({
        "model": cfg.model,
        "url": cfg.url,
        "hasKey": cfg.api_key.is_some(),
        "example": decisions::example_questions(),
        "exampleState": decisions::EXAMPLE_STATE,
    }))
}

#[derive(Deserialize)]
struct DecideRequest {
    questions: Vec<Question>,
    state: String,
}

async fn decide(Json(req): Json<DecideRequest>) -> Json<serde_json::Value> {
    match decisions::decide(&DecisionsConfig::from_env(), &req.questions, &req.state).await {
        Ok(decision) => Json(serde_json::json!({
            "response": decision.response,
            "requestJson": decision.request_json,
            "responseJson": decision.response_json,
        })),
        Err(err) => Json(serde_json::json!({ "error": format!("{err:#}") })),
    }
}
