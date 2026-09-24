//! MCP-сервер обработки документов (PDF), доступный по Streamable HTTP.
//!
//! Запуск: `cargo run -p documents-mcp` — сервер слушает
//! `http://127.0.0.1:8093/mcp` (адрес меняется переменной
//! `DOCUMENTS_MCP_ADDR`). Три инструмента — три звена одной цепочки:
//! `pdf_to_markdown` получает данные, `summarize_markdown` их обрабатывает,
//! `save_summary` сохраняет результат. Цепочку собирает модель: вызывает их
//! по очереди, передавая идентификатор из ответа одного шага следующему;
//! данные сверяются по SHA-256 (см. [`store`], [`steps`]).

mod markdown;
mod steps;
mod store;
mod summarize;

use std::sync::Arc;

use anyhow::Context;
use rmcp::{
    handler::server::wrapper::{Json, Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ServerHandler,
};
use serde::{Deserialize, Serialize};

use steps::{Converted, Saved, Summarized};
use store::Store;
use summarize::{Options, Summarizer, DEFAULT_MAX_WORDS};

const DEFAULT_ADDR: &str = "127.0.0.1:8093";
const DEFAULT_DIR: &str = "documents";
const MAX_WORDS_LIMIT: usize = 2_000;

#[derive(Debug, Deserialize, JsonSchema)]
struct ConvertParams {
    /// Путь к PDF: абсолютный, `~/…` или просто имя файла из входящих (list_pdfs).
    path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SummarizeParams {
    /// document_id из ответа pdf_to_markdown.
    document_id: String,
    /// Предельная длина краткого содержания в словах, по умолчанию 250.
    #[serde(default)]
    max_words: Option<usize>,
    /// На что обратить особое внимание: «риски», «цифры и сроки» и т. п.
    #[serde(default)]
    focus: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SaveParams {
    /// summary_id из ответа summarize_markdown.
    summary_id: String,
    /// Имя файла без каталога; по умолчанию — имя исходного PDF. Сохраняется как
    /// `<имя>.summary.md` в каталоге результатов.
    #[serde(default)]
    file_name: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct FileEntry {
    name: String,
    bytes: u64,
}

#[derive(Debug, Serialize, JsonSchema)]
struct Listing {
    /// Каталог входящих: PDF отсюда можно указывать просто по имени.
    inbox: String,
    pdfs: Vec<FileEntry>,
    /// Каталог результатов: Markdown и краткие содержания.
    out: String,
    results: Vec<FileEntry>,
    /// Чем строится краткое содержание: `llm (<модель>)` или `extractive`.
    summarizer: String,
}

#[derive(Clone)]
struct PipelineServer {
    store: Store,
    summarizer: Arc<Summarizer>,
}

#[tool_router]
impl PipelineServer {
    #[tool(
        name = "list_pdfs",
        description = "Что лежит во входящих (PDF, которые можно обработать, указав просто имя файла) и в \
            каталоге результатов (Markdown и краткие содержания). Полезен, когда пользователь говорит \
            «обработай PDF» без пути.",
        annotations(read_only_hint = true)
    )]
    async fn list_pdfs(&self) -> Result<Json<Listing>, String> {
        let pdfs = list_dir(&self.store.inbox(), |name| name.to_lowercase().ends_with(".pdf"));
        let results = list_dir(&self.store.out(), |name| name.ends_with(".md"));
        Ok(Json(Listing {
            inbox: self.store.inbox().display().to_string(),
            pdfs,
            out: self.store.out().display().to_string(),
            results,
            summarizer: self.summarizer.describe(),
        }))
    }

    #[tool(
        name = "pdf_to_markdown",
        description = "Шаг 1 конвейера: извлекает текст из PDF и превращает его в Markdown (заголовки, \
            абзацы, списки; колонтитулы и номера страниц убираются). Пишет <имя>.md в каталог результатов \
            и возвращает document_id — передайте его в summarize_markdown. Сканы без текстового слоя \
            не поддерживаются."
    )]
    async fn pdf_to_markdown(&self, Parameters(p): Parameters<ConvertParams>) -> Result<Json<Converted>, String> {
        steps::pdf_to_markdown(&self.store, &p.path).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "summarize_markdown",
        description = "Шаг 2 конвейера: краткое содержание Markdown-документа по его document_id (из \
            pdf_to_markdown): суть одним предложением и ключевые пункты. Возвращает текст и summary_id — \
            передайте его в save_summary. Сам текст в аргументы не переносите: шаги связаны \
            идентификаторами, и input_sha256 этого шага равен sha256 предыдущего."
    )]
    async fn summarize_markdown(&self, Parameters(p): Parameters<SummarizeParams>) -> Result<Json<Summarized>, String> {
        let options = options(p.max_words, p.focus);
        steps::summarize_markdown(&self.store, &self.summarizer, &p.document_id, &options)
            .await
            .map(Json)
            .map_err(error_text)
    }

    #[tool(
        name = "save_summary",
        description = "Шаг 3 конвейера: сохраняет краткое содержание по summary_id (из summarize_markdown) \
            в файл <имя>.summary.md в каталоге результатов — с шапкой, откуда оно получено, и хэшами всех \
            звеньев. После записи файл перечитывается и сверяется с шагом 2."
    )]
    async fn save_summary(&self, Parameters(p): Parameters<SaveParams>) -> Result<Json<Saved>, String> {
        steps::save_summary(&self.store, &p.summary_id, p.file_name.as_deref())
            .await
            .map(Json)
            .map_err(error_text)
    }
}

#[tool_handler]
impl ServerHandler for PipelineServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("documents-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Конвейер обработки PDF из трёх инструментов, которые вызываются по очереди: \
                 1) pdf_to_markdown(path) → document_id; 2) summarize_markdown(document_id) → summary_id; \
                 3) save_summary(summary_id) → путь к файлу. Каждый следующий шаг получает \
                 идентификатор из ответа предыдущего, а не текст; в плане назовите все три вызова и что \
                 куда передаётся. Без пути к файлу начните с list_pdfs.",
            )
    }
}

fn options(max_words: Option<usize>, focus: Option<String>) -> Options {
    Options { max_words: max_words.unwrap_or(DEFAULT_MAX_WORDS).clamp(20, MAX_WORDS_LIMIT), focus }
}

fn list_dir(dir: &std::path::Path, keep: impl Fn(&str) -> bool) -> Vec<FileEntry> {
    let mut entries: Vec<FileEntry> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let meta = e.metadata().ok()?;
            (meta.is_file() && keep(&name)).then_some(FileEntry { name, bytes: meta.len() })
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

fn error_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("DOCUMENTS_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let dir = std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| DEFAULT_DIR.into());
    let store = Store::open(&dir)?;
    let summarizer = Arc::new(Summarizer::from_env());
    eprintln!("documents-mcp: входящие {}, результаты {}", store.inbox().display(), store.out().display());
    match summarizer.as_ref() {
        Summarizer::Llm(_) => eprintln!("documents-mcp: краткое содержание — {}", summarizer.describe()),
        Summarizer::Extractive => eprintln!(
            "documents-mcp: LLM_API_URL/LLM_API_KEY не заданы — краткое содержание будет извлекающим (без модели)"
        ),
    }

    let service = StreamableHttpService::new(
        move || Ok(PipelineServer { store: store.clone(), summarizer: summarizer.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("не удалось занять адрес {addr}"))?;

    eprintln!("documents-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
