//! HTTP-слой шлюза: OpenAI-совместимые эндпоинты поверх llama-server с ключами,
//! лимитами и статистикой по пользователям.

use crate::config::{Config, User};
use crate::limits::{Admission, AdmissionError, RateLimiter};
use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

const CHAT_HTML: &str = include_str!("chat.html");

#[derive(Default, Clone, Serialize)]
pub struct UserStats {
    pub requests: u64,
    pub ok: u64,
    pub rejected_rate_limit: u64,
    pub rejected_context: u64,
    pub rejected_queue: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_latency_ms: u64,
}

pub struct App {
    cfg: Config,
    http: reqwest::Client,
    rate: RateLimiter,
    admission: Admission,
    /// Ключ → пользователь.
    users: HashMap<String, User>,
    /// Имя пользователя → его собственные слоты (per_user_parallel).
    user_slots: HashMap<String, Arc<Semaphore>>,
    stats: Mutex<BTreeMap<String, UserStats>>,
}

type Shared = Arc<App>;

/// Чем закончился запрос — для статистики и лога.
enum Outcome {
    Ok { prompt: u64, completion: u64 },
    RateLimit,
    Context,
    Queue,
    Error,
}

impl App {
    pub fn new(cfg: Config) -> anyhow::Result<Shared> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.request_timeout_secs))
            .build()?;
        let users = cfg.users.iter().map(|u| (u.key.clone(), u.clone())).collect();
        let user_slots = cfg
            .users
            .iter()
            .map(|u| (u.name.clone(), Arc::new(Semaphore::new(cfg.per_user_parallel))))
            .collect();
        let stats = cfg.users.iter().map(|u| (u.name.clone(), UserStats::default())).collect();
        Ok(Arc::new(App {
            admission: Admission::new(cfg.max_concurrent, cfg.max_queue),
            rate: RateLimiter::per_minute(),
            http,
            users,
            user_slots,
            stats: Mutex::new(stats),
            cfg,
        }))
    }

    fn auth(&self, headers: &HeaderMap) -> Option<&User> {
        let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
        let key = value.strip_prefix("Bearer ")?.trim();
        self.users.get(key)
    }

    fn record(&self, user: &str, started: Instant, outcome: Outcome) {
        let ms = started.elapsed().as_millis() as u64;
        let mut stats = self.stats.lock().unwrap();
        let s = stats.entry(user.to_string()).or_default();
        s.requests += 1;
        let what = match outcome {
            Outcome::Ok { prompt, completion } => {
                s.ok += 1;
                s.prompt_tokens += prompt;
                s.completion_tokens += completion;
                s.total_latency_ms += ms;
                format!("ok prompt={prompt} completion={completion}")
            }
            Outcome::RateLimit => {
                s.rejected_rate_limit += 1;
                "429 rate limit".into()
            }
            Outcome::Context => {
                s.rejected_context += 1;
                "400 context".into()
            }
            Outcome::Queue => {
                s.rejected_queue += 1;
                "503 queue".into()
            }
            Outcome::Error => {
                s.errors += 1;
                "error".into()
            }
        };
        println!("{user}: {what} {ms}ms");
    }

    /// Сколько токенов займёт промпт в контексте модели: шаблон чата модели
    /// (/apply-template) и токенизатор (/tokenize) самого llama-server — оценка точная,
    /// а не «символы / 4». Оба эндпоинта не занимают слот генерации.
    async fn count_prompt_tokens(&self, messages: &Value) -> anyhow::Result<u64> {
        let templated: Value = self
            .http
            .post(format!("{}/apply-template", self.cfg.backend))
            .json(&json!({ "messages": messages }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let prompt = templated["prompt"].as_str().unwrap_or_default();
        let tokens: Value = self
            .http
            .post(format!("{}/tokenize", self.cfg.backend))
            .json(&json!({ "content": prompt, "add_special": true }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(tokens["tokens"].as_array().map_or(0, |t| t.len() as u64))
    }
}

/// Ошибка в формате OpenAI — клиенты SDK разбирают именно его.
fn api_error(status: StatusCode, kind: &str, code: &str, message: impl Into<String>) -> Response {
    let body = json!({ "error": { "message": message.into(), "type": kind, "code": code } });
    (status, Json(body)).into_response()
}

fn unauthorized() -> Response {
    api_error(StatusCode::UNAUTHORIZED, "invalid_request_error", "invalid_api_key", "Неверный или отсутствующий API-ключ")
}

/// Итоговый max_tokens: то, что просил клиент (`max_completion_tokens` или `max_tokens`),
/// иначе значение по умолчанию, но не больше потолка.
fn effective_max_tokens(req: &Value, cfg: &Config) -> u64 {
    req.get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(cfg.default_max_tokens)
        .min(cfg.max_output_tokens)
}

pub fn router(app: Shared) -> Router {
    Router::new()
        .route("/", get(|| async { Html(CHAT_HTML) }))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/stats", get(stats))
        .with_state(app)
}

async fn health(State(app): State<Shared>) -> Response {
    let backend = match app.http.get(format!("{}/health", app.cfg.backend)).send().await {
        Ok(r) if r.status().is_success() => "ok",
        Ok(_) => "loading",
        Err(_) => "down",
    };
    let status = if backend == "ok" { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    let body = json!({
        "gateway": "ok",
        "backend": backend,
        "model": app.cfg.model,
        "active": app.admission.active(),
        "waiting": app.admission.waiting(),
    });
    (status, Json(body)).into_response()
}

async fn models(State(app): State<Shared>, headers: HeaderMap) -> Response {
    if app.auth(&headers).is_none() {
        return unauthorized();
    }
    Json(json!({
        "object": "list",
        "data": [{ "id": app.cfg.model, "object": "model", "owned_by": "local" }],
    }))
    .into_response()
}

async fn stats(State(app): State<Shared>, headers: HeaderMap) -> Response {
    let Some(user) = app.auth(&headers) else { return unauthorized() };
    let all = app.stats.lock().unwrap().clone();
    let users: BTreeMap<_, _> = if user.admin { all } else { all.into_iter().filter(|(n, _)| n == &user.name).collect() };
    Json(json!({
        "model": app.cfg.model,
        "active": app.admission.active(),
        "waiting": app.admission.waiting(),
        "limits": {
            "max_context_tokens": app.cfg.max_context_tokens,
            "max_output_tokens": app.cfg.max_output_tokens,
            "max_concurrent": app.cfg.max_concurrent,
            "max_queue": app.cfg.max_queue,
            "per_user_parallel": app.cfg.per_user_parallel,
            "rpm": app.cfg.rpm(user),
        },
        "users": users,
    }))
    .into_response()
}

async fn chat_completions(State(app): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let Some(user) = app.auth(&headers).cloned() else { return unauthorized() };
    let started = Instant::now();

    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "invalid_json", "Тело запроса — не JSON-объект"),
    };
    if !req["messages"].as_array().is_some_and(|m| !m.is_empty()) {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "missing_messages", "Нужен непустой массив messages");
    }

    if let Err(retry) = app.rate.check(&user.name, app.cfg.rpm(&user), Instant::now()) {
        app.record(&user.name, started, Outcome::RateLimit);
        let secs = retry.as_secs().max(1);
        let mut resp = api_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "rate_limit_exceeded",
            format!("Лимит {} запросов в минуту исчерпан, повторите через {secs} с", app.cfg.rpm(&user)),
        );
        resp.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(secs));
        return resp;
    }

    let max_tokens = effective_max_tokens(&req, &app.cfg);
    let prompt_tokens = match app.count_prompt_tokens(&req["messages"]).await {
        Ok(n) => n,
        Err(e) => {
            app.record(&user.name, started, Outcome::Error);
            return api_error(StatusCode::BAD_GATEWAY, "api_error", "backend_unavailable", format!("Модель недоступна: {e}"));
        }
    };
    if prompt_tokens + max_tokens > app.cfg.max_context_tokens {
        app.record(&user.name, started, Outcome::Context);
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "context_length_exceeded",
            format!(
                "Контекст модели — {} токенов, а запрос занимает {} (промпт {prompt_tokens} + ответ до {max_tokens}). \
                 Сократите сообщения или max_tokens.",
                app.cfg.max_context_tokens,
                prompt_tokens + max_tokens
            ),
        );
    }

    // Сначала свой слот пользователя, потом общий: пока пользователь ждёт сам себя,
    // он не занимает место в общей очереди.
    let queue_timeout = Duration::from_secs(app.cfg.queue_timeout_secs);
    let user_slot = app.user_slots[&user.name].clone();
    let user_permit = match tokio::time::timeout(queue_timeout, user_slot.acquire_owned()).await {
        Ok(p) => p.expect("семафор не закрывается"),
        Err(_) => return queue_rejected(&app, &user.name, started, AdmissionError::Timeout),
    };
    let slot = match app.admission.enter(queue_timeout).await {
        Ok(p) => p,
        Err(e) => return queue_rejected(&app, &user.name, started, e),
    };

    let stream = req["stream"].as_bool().unwrap_or(false);
    let obj = req.as_object_mut().expect("проверено выше");
    obj.remove("max_completion_tokens");
    obj.insert("max_tokens".into(), json!(max_tokens));
    obj.insert("model".into(), json!(app.cfg.model));
    if stream {
        obj.insert("stream_options".into(), json!({ "include_usage": true }));
    }

    let resp = match app.http.post(format!("{}/v1/chat/completions", app.cfg.backend)).json(&req).send().await {
        Ok(r) => r,
        Err(e) => {
            app.record(&user.name, started, Outcome::Error);
            return api_error(StatusCode::BAD_GATEWAY, "api_error", "backend_error", format!("Ошибка модели: {e}"));
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if !status.is_success() {
        app.record(&user.name, started, Outcome::Error);
        let text = resp.text().await.unwrap_or_default();
        return (status, [(header::CONTENT_TYPE, "application/json")], text).into_response();
    }

    if stream {
        let mut guard = StreamGuard {
            app: app.clone(),
            user: user.name.clone(),
            started,
            buf: String::new(),
            usage: None,
            _permits: (user_permit, slot),
        };
        let body = resp.bytes_stream().map(move |chunk| {
            if let Ok(b) = &chunk {
                guard.feed(b);
            }
            chunk
        });
        return Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(body))
            .expect("корректные заголовки");
    }

    let json: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            app.record(&user.name, started, Outcome::Error);
            return api_error(StatusCode::BAD_GATEWAY, "api_error", "backend_error", format!("Ответ модели не разобран: {e}"));
        }
    };
    drop((user_permit, slot));
    let (prompt, completion) = usage_of(&json).unwrap_or((prompt_tokens, 0));
    app.record(&user.name, started, Outcome::Ok { prompt, completion });
    Json(json).into_response()
}

fn queue_rejected(app: &App, user: &str, started: Instant, e: AdmissionError) -> Response {
    app.record(user, started, Outcome::Queue);
    let msg = match e {
        AdmissionError::QueueFull => "Сервер занят: очередь к модели заполнена, повторите позже",
        AdmissionError::Timeout => "Сервер занят: слот модели не освободился вовремя, повторите позже",
    };
    api_error(StatusCode::SERVICE_UNAVAILABLE, "server_busy", "server_busy", msg)
}

fn usage_of(v: &Value) -> Option<(u64, u64)> {
    let u = v.get("usage")?;
    Some((u["prompt_tokens"].as_u64()?, u["completion_tokens"].as_u64()?))
}

/// Держит слоты, пока клиент читает поток, вылавливает usage из последнего SSE-чанка
/// и пишет статистику, когда поток закончился или клиент отключился.
struct StreamGuard {
    app: Shared,
    user: String,
    started: Instant,
    buf: String,
    usage: Option<(u64, u64)>,
    _permits: (OwnedSemaphorePermit, OwnedSemaphorePermit),
}

impl StreamGuard {
    fn feed(&mut self, chunk: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=pos).collect();
            if let Some(data) = line.trim().strip_prefix("data:") {
                if let Ok(v) = serde_json::from_str::<Value>(data.trim()) {
                    if let Some(u) = usage_of(&v) {
                        self.usage = Some(u);
                    }
                }
            }
        }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        let outcome = match self.usage {
            Some((prompt, completion)) => Outcome::Ok { prompt, completion },
            None => Outcome::Error,
        };
        self.app.record(&self.user, self.started, outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn config(backend: String) -> Config {
        toml::from_str::<Config>(&format!(
            r#"
            backend = "{backend}"
            model = "test-model"
            max_context_tokens = 100
            max_output_tokens = 40
            default_max_tokens = 10
            default_rpm = 3
            [[users]]
            name = "alice"
            key = "alice-key-0123456789"
            admin = true
            [[users]]
            name = "bob"
            key = "bob-key-0123456789ab"
            "#
        ))
        .unwrap()
    }

    /// Мок llama-server: шаблон — склейка content, токен — слово; запоминает
    /// max_tokens и model последнего запроса.
    async fn mock_backend(last_max_tokens: Arc<AtomicU64>) -> String {
        let r = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/apply-template",
                post(|Json(v): Json<Value>| async move {
                    let text: Vec<String> =
                        v["messages"].as_array().unwrap().iter().map(|m| m["content"].as_str().unwrap().to_string()).collect();
                    Json(json!({ "prompt": text.join(" ") }))
                }),
            )
            .route(
                "/tokenize",
                post(|Json(v): Json<Value>| async move {
                    let n = v["content"].as_str().unwrap().split_whitespace().count();
                    Json(json!({ "tokens": vec![1; n] }))
                }),
            )
            .route(
                "/v1/chat/completions",
                post(move |Json(v): Json<Value>| async move {
                    last_max_tokens.store(v["max_tokens"].as_u64().unwrap(), Ordering::SeqCst);
                    assert_eq!(v["model"], "test-model");
                    Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": "привет" } }],
                        "usage": { "prompt_tokens": 2, "completion_tokens": 1 }
                    }))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, r).await.unwrap() });
        format!("http://{addr}")
    }

    async fn gateway() -> (String, Arc<AtomicU64>) {
        let last = Arc::new(AtomicU64::new(0));
        let backend = mock_backend(last.clone()).await;
        let app = App::new(config(backend)).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(app)).await.unwrap() });
        (format!("http://{addr}"), last)
    }

    async fn chat(base: &str, key: &str, body: Value) -> (u16, Value) {
        let r = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .unwrap();
        (r.status().as_u16(), r.json().await.unwrap())
    }

    fn msg(text: &str) -> Value {
        json!({ "messages": [{ "role": "user", "content": text }] })
    }

    #[tokio::test]
    async fn rejects_bad_key() {
        let (base, _) = gateway().await;
        let (status, body) = chat(&base, "wrong-key-0123456789", msg("привет")).await;
        assert_eq!(status, 401);
        assert_eq!(body["error"]["code"], "invalid_api_key");
    }

    #[tokio::test]
    async fn clamps_max_tokens_and_overrides_model() {
        let (base, last) = gateway().await;
        let mut body = msg("привет");
        body["max_tokens"] = json!(1000);
        body["model"] = json!("gpt-4o");
        let (status, _) = chat(&base, "alice-key-0123456789", body).await;
        assert_eq!(status, 200);
        assert_eq!(last.load(Ordering::SeqCst), 40);

        chat(&base, "alice-key-0123456789", msg("привет")).await;
        assert_eq!(last.load(Ordering::SeqCst), 10, "без max_tokens — значение по умолчанию");
    }

    #[tokio::test]
    async fn rejects_too_long_context() {
        let (base, _) = gateway().await;
        // 95 слов промпта + 10 токенов ответа > 100.
        let (status, body) = chat(&base, "alice-key-0123456789", msg(&"слово ".repeat(95))).await;
        assert_eq!(status, 400);
        assert_eq!(body["error"]["code"], "context_length_exceeded");
    }

    #[tokio::test]
    async fn rate_limit_is_per_user() {
        let (base, _) = gateway().await;
        for _ in 0..3 {
            assert_eq!(chat(&base, "alice-key-0123456789", msg("привет")).await.0, 200);
        }
        let (status, body) = chat(&base, "alice-key-0123456789", msg("привет")).await;
        assert_eq!(status, 429);
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
        assert_eq!(chat(&base, "bob-key-0123456789ab", msg("привет")).await.0, 200);
    }

    #[tokio::test]
    async fn stats_visible_to_admin_only_in_full() {
        let (base, _) = gateway().await;
        chat(&base, "bob-key-0123456789ab", msg("привет")).await;
        let get = |key: &'static str| {
            let base = base.clone();
            async move {
                reqwest::Client::new()
                    .get(format!("{base}/stats"))
                    .bearer_auth(key)
                    .send()
                    .await
                    .unwrap()
                    .json::<Value>()
                    .await
                    .unwrap()
            }
        };
        let admin = get("alice-key-0123456789").await;
        assert_eq!(admin["users"]["bob"]["completion_tokens"], 1);
        assert!(admin["users"]["alice"].is_object());
        let bob = get("bob-key-0123456789ab").await;
        assert!(bob["users"]["alice"].is_null());
    }

    #[test]
    fn stream_guard_finds_usage_split_across_chunks() {
        let cfg = config("http://127.0.0.1:1".into());
        let app = App::new(cfg).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let a = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
            let b = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
            let mut g = StreamGuard {
                app: app.clone(),
                user: "alice".into(),
                started: Instant::now(),
                buf: String::new(),
                usage: None,
                _permits: (a, b),
            };
            g.feed(b"data: {\"choices\":[]}\n\ndata: {\"usage\":{\"prompt_");
            g.feed(b"tokens\":5,\"completion_tokens\":7}}\n\ndata: [DONE]\n\n");
            assert_eq!(g.usage, Some((5, 7)));
        });
    }
}
