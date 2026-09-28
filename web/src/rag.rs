//! Вкладка «RAG»: загрузка и векторизация документов, просмотр индекса
//! (документы, версии, чанки, карта документа), поиск, удаление.
//!
//! Индексация идёт в фоне — одна задача на сервер ([`RagJob`]), страница
//! опрашивает `GET /api/rag/job`. Просмотр читает индекс напрямую
//! (см. `llm_core::rag::browse`).

use std::sync::{Arc, Mutex};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    routing::{get, post},
    Json, Router,
};
use llm_core::rag::{self, browse, ChunkParams, Embedder, RagConfig, Strategy};
use serde::{Deserialize, Serialize};

use crate::AppState;

/// Фоновая задача индексации: загрузка файла или полная индексация.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RagJob {
    /// Номер задачи — страница отличает новую задачу от прошлой.
    pub id: u64,
    pub running: bool,
    /// Что делается: «векторизация rag/uploads/x.pdf», «индексация корпуса».
    pub title: String,
    /// Документ, если задача про один документ.
    pub source: Option<String>,
    /// Последняя строка хода («structure: эмбеддинги 32/120»).
    pub progress: String,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub error: Option<String>,
    /// Итог (IndexReport) и он же текстом.
    pub report: Option<serde_json::Value>,
    pub summary: Option<String>,
}

pub type SharedJob = Arc<Mutex<RagJob>>;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/rag/overview", get(overview))
        .route("/api/rag/docs", get(documents).delete(remove_document))
        .route("/api/rag/versions", get(versions))
        .route("/api/rag/chunks", get(chunks))
        .route("/api/rag/chunk", get(chunk))
        .route("/api/rag/map", get(map))
        .route("/api/rag/search", post(search))
        .route("/api/rag/job", get(job))
        .route("/api/rag/index", post(index_corpus))
        .route("/api/rag/rechunk", post(rechunk))
        .route(
            "/api/rag/upload",
            post(upload).layer(DefaultBodyLimit::max(rag::MAX_UPLOAD_BYTES + 1024 * 1024)),
        )
}

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn error(err: impl std::fmt::Display) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "error": err.to_string() }))
}

/// Чтение индекса (SQLite) — в пуле блокирующих задач, не в рантайме.
async fn blocking<T: Serialize + Send + 'static>(
    f: impl FnOnce(&RagConfig) -> anyhow::Result<T> + Send + 'static,
) -> Json<serde_json::Value> {
    let result = tokio::task::spawn_blocking(move || f(&RagConfig::from_env())).await;
    match result {
        Ok(Ok(value)) => Json(serde_json::to_value(value).unwrap_or_default()),
        Ok(Err(err)) => error(format!("{err:#}")),
        Err(err) => error(err),
    }
}

async fn overview() -> Json<serde_json::Value> {
    blocking(browse::overview).await
}

async fn documents() -> Json<serde_json::Value> {
    blocking(browse::documents).await
}

#[derive(Deserialize)]
struct SourceQuery {
    source: String,
}

async fn versions(Query(q): Query<SourceQuery>) -> Json<serde_json::Value> {
    blocking(move |cfg| browse::versions(cfg, &q.source)).await
}

#[derive(Deserialize)]
struct ChunksQuery {
    source: String,
    version: Option<i64>,
    strategy: String,
    #[serde(default)]
    offset: usize,
    limit: Option<usize>,
}

async fn chunks(Query(q): Query<ChunksQuery>) -> Json<serde_json::Value> {
    blocking(move |cfg| browse::chunks(cfg, &q.source, q.version, &q.strategy, q.offset, q.limit.unwrap_or(50))).await
}

#[derive(Deserialize)]
struct ChunkQuery {
    id: String,
}

async fn chunk(Query(q): Query<ChunkQuery>) -> Json<serde_json::Value> {
    blocking(move |cfg| browse::chunk(cfg, &q.id)).await
}

#[derive(Deserialize)]
struct MapQuery {
    source: String,
    version: Option<i64>,
    strategy: String,
}

async fn map(Query(q): Query<MapQuery>) -> Json<serde_json::Value> {
    blocking(move |cfg| browse::document_map(cfg, &q.source, q.version, &q.strategy)).await
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default)]
    strategies: Vec<String>,
    k: Option<usize>,
}

async fn search(Json(req): Json<SearchRequest>) -> Json<serde_json::Value> {
    if req.query.trim().is_empty() {
        return error("пустой запрос");
    }
    let cfg = RagConfig::from_env();
    // Без явного списка — только стратегии, которыми что-то нарезано.
    let strategies = if req.strategies.is_empty() {
        match rag::indexed_strategies(&cfg) {
            Ok(s) if s.is_empty() => return error("индекс пуст — сначала загрузите документ"),
            Ok(s) => s,
            Err(err) => return error(format!("{err:#}")),
        }
    } else {
        match parse_strategies(&req.strategies) {
            Ok(s) => s,
            Err(err) => return error(err),
        }
    };
    let embedder = match Embedder::from_env() {
        Ok(e) => e,
        Err(err) => return error(format!("{err:#}")),
    };
    let opts = rag::SearchOptions { strategies, k: req.k.unwrap_or(5).clamp(1, 20), ..Default::default() };
    match rag::search(&cfg, &embedder, req.query.trim(), &opts).await {
        Ok(results) => Json(serde_json::json!({
            "results": results
                .into_iter()
                .map(|(s, hits)| serde_json::json!({ "strategy": s.name(), "hits": hits }))
                .collect::<Vec<_>>(),
        })),
        Err(err) => error(format!("{err:#}")),
    }
}

fn parse_strategies(names: &[String]) -> Result<Vec<Strategy>, String> {
    if names.is_empty() {
        return Ok(Strategy::ALL.to_vec());
    }
    names
        .iter()
        .map(|n| Strategy::parse(n.trim()).ok_or_else(|| format!("неизвестная стратегия «{n}»")))
        .collect()
}

async fn job(State(state): State<AppState>) -> Json<serde_json::Value> {
    let job = state.rag_job.lock().unwrap().clone();
    Json(serde_json::to_value(job).unwrap_or_default())
}

/// Занимает слот фоновой задачи; `None`, если задача уже идёт.
fn start_job(state: &AppState, title: String, source: Option<String>) -> Option<u64> {
    let mut job = state.rag_job.lock().unwrap();
    if job.running {
        return None;
    }
    let id = job.id + 1;
    *job = RagJob { id, running: true, title, source, started_at: now(), ..Default::default() };
    Some(id)
}

/// Запускает индексацию в фоне: `run` получает функцию хода и возвращает итог.
fn spawn_job<F, Fut>(state: &AppState, run: F)
where
    F: FnOnce(Arc<dyn Fn(String) + Send + Sync>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = anyhow::Result<rag::IndexReport>> + Send + 'static,
{
    let shared = state.rag_job.clone();
    tokio::spawn(async move {
        let progress_job = shared.clone();
        let progress: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |line: String| {
            progress_job.lock().unwrap().progress = line;
        });
        let result = run(progress).await;
        let mut job = shared.lock().unwrap();
        job.running = false;
        job.finished_at = Some(now());
        match result {
            Ok(report) => {
                job.summary = Some(saved_vectors(&report));
                job.report = serde_json::to_value(&report).ok();
            }
            Err(err) => job.error = Some(format!("{err:#}")),
        }
    });
}

/// Итог одной строкой: сколько векторов сохранено (новых и из кэша) и
/// сколько чанков нарезано.
fn saved_vectors(report: &rag::IndexReport) -> String {
    let embedded: usize = report.runs.iter().map(|r| r.embedded).sum();
    let cached: usize = report.runs.iter().map(|r| r.cached).sum();
    let chunks: usize = report.runs.iter().map(|r| r.chunks).sum();
    if chunks == 0 {
        return if report.single {
            "Без изменений: документ уже нарезан с этими параметрами".to_string()
        } else {
            "Без изменений: новых версий и недостающих нарезок нет".to_string()
        };
    }
    let model = std::env::var("LLM_EMBEDDING_MODEL").unwrap_or_default();
    format!("Сохранено векторов: {} (новых {embedded}, из кэша {cached}) · чанков {chunks} · {model}", embedded + cached)
}

/// Стратегия и её параметры из строки запроса: `name`, `strategy` и
/// параметры по именам (`size=800&overlap=100`); не заданные — по умолчанию.
fn upload_settings(q: &std::collections::HashMap<String, String>) -> Result<(String, Strategy, ChunkParams), String> {
    let name = q.get("name").cloned().ok_or("не указано имя файла (name)")?;
    let (strategy, params) = strategy_settings(q)?;
    Ok((name, strategy, params))
}

/// `strategy` и её параметры по именам; не заданные — по умолчанию.
fn strategy_settings(q: &std::collections::HashMap<String, String>) -> Result<(Strategy, ChunkParams), String> {
    let strategy_name = q.get("strategy").map(String::as_str).unwrap_or("structure");
    let strategy = Strategy::parse(strategy_name).ok_or_else(|| format!("неизвестная стратегия «{strategy_name}»"))?;
    let mut params = ChunkParams::default();
    for spec in ChunkParams::specs(strategy) {
        if let Some(raw) = q.get(spec.name).filter(|v| !v.trim().is_empty()) {
            let value: usize =
                raw.trim().parse().map_err(|_| format!("{} ({}): нужно целое число, задано «{raw}»", spec.name, spec.label))?;
            params.set(spec.name, value)?;
        }
    }
    params.validate(strategy)?;
    Ok((strategy, params))
}

/// Тело запроса — сам файл; в строке запроса — имя, стратегия и её
/// параметры. Файл сохраняется в папку загрузок и векторизуется в фоне;
/// ход — `GET /api/rag/job`.
async fn upload(
    State(state): State<AppState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
    body: Bytes,
) -> Json<serde_json::Value> {
    let (name, strategy, params) = match upload_settings(&q) {
        Ok(s) => s,
        Err(err) => return error(err),
    };
    // Модель проверяется до сохранения: без неё загрузка бессмысленна.
    let embedder = match Embedder::from_env() {
        Ok(e) => e,
        Err(err) => return error(format!("{err:#}")),
    };
    let cfg = RagConfig::from_env();
    if state.rag_job.lock().unwrap().running {
        return error("идёт другая индексация — дождитесь её окончания");
    }
    let source = match rag::save_upload(&cfg, &name, &body) {
        Ok(s) => s,
        Err(err) => return error(format!("{err:#}")),
    };
    let file_name = source.rsplit('/').next().unwrap_or(&source).to_string();
    let title = format!("Векторизация {file_name}");
    let Some(id) = start_job(&state, title, Some(source.clone())) else {
        return error("идёт другая индексация — дождитесь её окончания");
    };
    let job_source = source.clone();
    spawn_job(&state, move |progress| async move {
        rag::index_file(&cfg, &embedder, &job_source, &[strategy], &params, &*progress).await
    });
    Json(serde_json::json!({ "job": id, "source": source, "strategy": strategy.name() }))
}

#[derive(Deserialize)]
struct RechunkRequest {
    source: String,
    strategy: String,
    #[serde(default)]
    params: std::collections::HashMap<String, serde_json::Value>,
}

/// Нарезать уже проиндексированный документ стратегией — из просмотра, без
/// повторной загрузки файла. Файл берётся с диска (для загруженных — из
/// папки загрузок); изменился — это станет новой версией.
async fn rechunk(State(state): State<AppState>, Json(req): Json<RechunkRequest>) -> Json<serde_json::Value> {
    let mut q: std::collections::HashMap<String, String> =
        req.params.iter().map(|(k, v)| (k.clone(), v.to_string().trim_matches('"').to_string())).collect();
    q.insert("strategy".into(), req.strategy.clone());
    let (strategy, params) = match strategy_settings(&q) {
        Ok(s) => s,
        Err(err) => return error(err),
    };
    let embedder = match Embedder::from_env() {
        Ok(e) => e,
        Err(err) => return error(format!("{err:#}")),
    };
    let cfg = RagConfig::from_env();
    if !cfg.base.join(&req.source).is_file() {
        return error(format!("файла {} нет на диске — загрузите его заново", req.source));
    }
    let file_name = req.source.rsplit('/').next().unwrap_or(&req.source).to_string();
    let Some(id) = start_job(&state, format!("Нарезка {file_name}: {}", strategy.name()), Some(req.source.clone())) else {
        return error("идёт другая индексация — дождитесь её окончания");
    };
    let source = req.source.clone();
    spawn_job(&state, move |progress| async move {
        rag::index_file(&cfg, &embedder, &source, &[strategy], &params, &*progress).await
    });
    Json(serde_json::json!({ "job": id, "source": req.source, "strategy": strategy.name() }))
}

#[derive(Deserialize)]
struct IndexRequest {
    #[serde(default)]
    strategies: Vec<String>,
}

/// Полная индексация корпуса — как `index` в TUI.
async fn index_corpus(State(state): State<AppState>, Json(req): Json<IndexRequest>) -> Json<serde_json::Value> {
    let strategies = match parse_strategies(&req.strategies) {
        Ok(s) => s,
        Err(err) => return error(err),
    };
    let embedder = match Embedder::from_env() {
        Ok(e) => e,
        Err(err) => return error(format!("{err:#}")),
    };
    let Some(id) = start_job(&state, "Индексация корпуса".to_string(), None) else {
        return error("идёт другая индексация — дождитесь её окончания");
    };
    let cfg = RagConfig::from_env();
    spawn_job(&state, move |progress| async move { rag::index(&cfg, &embedder, &strategies, &*progress).await });
    Json(serde_json::json!({ "job": id }))
}

async fn remove_document(State(state): State<AppState>, Query(q): Query<SourceQuery>) -> Json<serde_json::Value> {
    if state.rag_job.lock().unwrap().running {
        return error("идёт индексация — удалите документ после её окончания");
    }
    blocking(move |cfg| {
        let removed = rag::remove_document(cfg, &q.source)?;
        // В вебе документы показываются по имени файла, не по пути.
        let file_name = q.source.rsplit('/').next().unwrap_or(&q.source);
        let summary = removed.describe().replacen(&q.source, file_name, 1);
        Ok(serde_json::json!({ "removed": removed, "summary": summary }))
    })
    .await
}
