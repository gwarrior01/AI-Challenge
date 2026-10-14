//! HTTP-интерфейс сервиса: страница чата поддержки и JSON API.
//!
//! ```text
//! GET  /                 страница чата: вопросы о продукте и о тикетах (номер — в тексте вопроса)
//! POST /api/ask          {question, session_id?, ticket_id?} → ответ (см. service::AskReply)
//! POST /api/reset        {session_id} — забыть разговор
//! ```

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;

use crate::service::{AskRequest, Support};

type AppState = Arc<Support>;

pub fn router(support: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/ask", post(ask))
        .route("/api/reset", post(reset))
        .with_state(support)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

fn error(code: StatusCode, message: impl Into<String>) -> Response {
    (code, Json(json!({ "error": message.into() }))).into_response()
}

async fn ask(State(s): State<AppState>, Json(req): Json<AskRequest>) -> Response {
    if req.question.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "пустой вопрос");
    }
    match s.ask(req).await {
        Ok(reply) => {
            let lines = reply.context_lines();
            let mut v = serde_json::to_value(reply).unwrap_or_default();
            v["context_lines"] = json!(lines);
            Json(v).into_response()
        }
        Err(err) => error(StatusCode::BAD_GATEWAY, format!("{err:#}")),
    }
}

#[derive(Deserialize)]
struct ResetRequest {
    session_id: String,
}

async fn reset(State(s): State<AppState>, Json(req): Json<ResetRequest>) -> StatusCode {
    s.reset(&req.session_id);
    StatusCode::NO_CONTENT
}
