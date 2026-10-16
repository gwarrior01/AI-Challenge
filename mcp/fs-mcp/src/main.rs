//! MCP-сервер файлов проекта: ассистент читает файлы, ищет по ним, создаёт и
//! правит их. Правки не пишутся на диск сразу — они копятся в оверлее
//! (`pending_changes` — unified diff), а на диск их переносит
//! `apply_changes`, который вызывает код ассистента после подтверждения
//! пользователя, а не модель. Подробно — `workspace.rs`.
//!
//! Запуск по умолчанию — stdio: клиент сам поднимает сервер дочерним
//! процессом (`cargo run -q -p fs-mcp`), и тот работает с файлами своей
//! рабочей директории (или `FS_MCP_ROOT`). С флагом `--http` — Streamable
//! HTTP на `127.0.0.1:8097/mcp` (`FS_MCP_ADDR`); оверлей тогда общий для всех
//! клиентов сервера.

mod workspace;

use std::path::PathBuf;
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
    ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};

use workspace::{ApplyResult, FileList, FileText, ListQuery, Pending, SearchQuery, SearchResult, Workspace, WriteResult};

const DEFAULT_ADDR: &str = "127.0.0.1:8097";
const DEFAULT_FILES_LIMIT: usize = 300;
const MAX_FILES_LIMIT: usize = 2_000;
const DEFAULT_FILE_LINES: usize = 400;
const MAX_FILE_LINES: usize = 2_000;
const DEFAULT_RESULTS: usize = 50;
const MAX_RESULTS: usize = 300;
const MAX_CONTEXT: usize = 5;

#[derive(Debug, Deserialize, JsonSchema)]
struct ListParams {
    /// Каталог относительно корня проекта (`core/src`); по умолчанию — весь проект.
    #[serde(default)]
    path: Option<String>,
    /// Шаблон: `*.rs`, `*.md,*.toml` (через запятую); с `/` — по пути (`mcp/*/Cargo.toml`, `docs/**/*.md`).
    #[serde(default)]
    pattern: Option<String>,
    /// Свернуть пути до этой глубины и посчитать файлы в каталогах — обзор структуры (1 — верхний уровень).
    #[serde(default)]
    depth: Option<usize>,
    /// Сколько строк вернуть, по умолчанию 300.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadParams {
    /// Файл относительно корня проекта (`core/src/lib.rs`).
    path: String,
    /// С какой строки (с 1), по умолчанию с первой.
    #[serde(default)]
    start_line: Option<usize>,
    /// Сколько строк вернуть, по умолчанию 400.
    #[serde(default)]
    max_lines: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    /// Что искать: подстрока (по умолчанию) или регулярное выражение (`regex: true`).
    query: String,
    /// `query` — регулярное выражение (синтаксис Rust regex: `fn \w+\(`, `McpManager::(load|new)`).
    #[serde(default)]
    regex: bool,
    /// Учитывать регистр. По умолчанию — нет.
    #[serde(default)]
    case_sensitive: bool,
    /// Искать только в этом каталоге или файле.
    #[serde(default)]
    path: Option<String>,
    /// Только в файлах по шаблону: `*.rs`, `*.md,*.toml`, `web/**/*.rs`.
    #[serde(default)]
    pattern: Option<String>,
    /// Сколько строк контекста до и после совпадения (0–5), по умолчанию 0.
    #[serde(default)]
    context: Option<usize>,
    /// Сколько совпадений вернуть со строками, по умолчанию 50 (счётчики по файлам — всегда полные).
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteParams {
    /// Файл относительно корня проекта; каталоги создадутся сами.
    path: String,
    /// Полное новое содержимое файла.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditParams {
    /// Файл относительно корня проекта.
    path: String,
    /// Фрагмент, который заменить: символ в символ как в файле (с отступами), без номеров строк.
    /// Должен встречаться ровно один раз — иначе добавь соседние строки или `replace_all`.
    old_text: String,
    /// Чем заменить (пусто — удалить фрагмент).
    new_text: String,
    /// Заменить все вхождения `old_text`, а не единственное.
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PathsParams {
    /// Файлы или каталоги; пусто — все изменения.
    #[serde(default)]
    paths: Option<Vec<String>>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct Discarded {
    discarded: Vec<String>,
}

#[derive(Clone)]
struct FsServer {
    ws: Arc<Workspace>,
}

#[tool_router]
impl FsServer {
    #[tool(
        name = "list_files",
        description = "Файлы проекта (с учётом .gitignore): весь список, каталог (`path`), по шаблону \
            (`pattern`: `*.rs`, `docs/**/*.md`) или обзор структуры — пути, свёрнутые до глубины `depth`, с \
            числом файлов. Новые файлы из ожидающих изменений тоже видны.",
        annotations(read_only_hint = true)
    )]
    async fn list_files(&self, Parameters(p): Parameters<ListParams>) -> Result<Json<FileList>, String> {
        let q = ListQuery {
            path: p.path,
            pattern: p.pattern,
            depth: p.depth,
            limit: p.limit.unwrap_or(DEFAULT_FILES_LIMIT).clamp(1, MAX_FILES_LIMIT),
        };
        self.ws.list(&q).map(Json).map_err(error_text)
    }

    #[tool(
        name = "read_file",
        description = "Текст файла проекта целиком или с `start_line` по `max_lines` строк (по умолчанию 400). \
            Если в файле есть ожидающие изменения — показан текст с ними (`pending: true`).",
        annotations(read_only_hint = true)
    )]
    async fn read_file(&self, Parameters(p): Parameters<ReadParams>) -> Result<Json<FileText>, String> {
        let max = p.max_lines.unwrap_or(DEFAULT_FILE_LINES).clamp(1, MAX_FILE_LINES);
        self.ws.read(&p.path, p.start_line.unwrap_or(1), max).map(Json).map_err(error_text)
    }

    #[tool(
        name = "search",
        description = "Поиск по содержимому файлов проекта: подстрока или регулярное выражение, во всём \
            проекте, в каталоге (`path`) или по шаблону файлов (`pattern`). Возвращает число совпадений в \
            каждом файле (полностью) и сами строки `path:line` с контекстом (первые `max_results`). \
            Учитывает ожидающие изменения.",
        annotations(read_only_hint = true)
    )]
    async fn search(&self, Parameters(p): Parameters<SearchParams>) -> Result<Json<SearchResult>, String> {
        let q = SearchQuery {
            query: p.query,
            regex: p.regex,
            case_sensitive: p.case_sensitive,
            path: p.path,
            pattern: p.pattern,
            context: p.context.unwrap_or(0).min(MAX_CONTEXT),
            max_results: p.max_results.unwrap_or(DEFAULT_RESULTS).clamp(1, MAX_RESULTS),
        };
        self.ws.search(&q).map(Json).map_err(error_text)
    }

    #[tool(
        name = "write_file",
        description = "Создать файл или заменить его содержимое целиком. Изменение попадает в ожидающие \
            (на диск — после подтверждения пользователя); возвращает дифф. Для правки части существующего \
            файла — edit_file.",
        annotations(destructive_hint = false, idempotent_hint = true)
    )]
    async fn write_file(&self, Parameters(p): Parameters<WriteParams>) -> Result<Json<WriteResult>, String> {
        self.ws.write(&p.path, &p.content).map(Json).map_err(error_text)
    }

    #[tool(
        name = "edit_file",
        description = "Точечная правка файла: заменить фрагмент `old_text` (точно как в файле, встречается \
            один раз) на `new_text`. Изменение попадает в ожидающие (на диск — после подтверждения \
            пользователя); возвращает дифф файла. Несколько правок одного файла накапливаются.",
        annotations(destructive_hint = false)
    )]
    async fn edit_file(&self, Parameters(p): Parameters<EditParams>) -> Result<Json<WriteResult>, String> {
        self.ws.edit(&p.path, &p.old_text, &p.new_text, p.replace_all).map(Json).map_err(error_text)
    }

    #[tool(
        name = "pending_changes",
        description = "Все ожидающие изменения: какие файлы созданы или изменены, сколько строк, и unified \
            diff в формате git.",
        annotations(read_only_hint = true)
    )]
    async fn pending_changes(&self) -> Result<Json<Pending>, String> {
        Ok(Json(self.ws.pending()))
    }

    #[tool(
        name = "discard_changes",
        description = "Отменить ожидающие изменения файлов (`paths`) или все — файлы на диске не меняются.",
        annotations(destructive_hint = false)
    )]
    async fn discard_changes(&self, Parameters(p): Parameters<PathsParams>) -> Result<Json<Discarded>, String> {
        self.ws.discard(p.paths.as_deref()).map(|discarded| Json(Discarded { discarded })).map_err(error_text)
    }

    #[tool(
        name = "apply_changes",
        description = "Записать ожидающие изменения на диск (все или `paths`). Файл, изменившийся на диске \
            после первой правки, не перезаписывается — он в `conflicts`. Вызывается по подтверждению \
            пользователя.",
        annotations(destructive_hint = true)
    )]
    async fn apply_changes(&self, Parameters(p): Parameters<PathsParams>) -> Result<Json<ApplyResult>, String> {
        self.ws.apply(p.paths.as_deref()).map(Json).map_err(error_text)
    }
}

#[tool_handler]
impl ServerHandler for FsServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("fs-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Файлы проекта. Где что лежит — list_files (depth=1 или 2 — обзор, pattern — по имени); где \
                 используется имя или API — search (pattern=\"*.rs\" сужает); текст — read_file. Изменения — \
                 edit_file (точечно) и write_file (новый файл или целиком): они копятся как ожидающие, \
                 pending_changes показывает дифф, на диск их пишет apply_changes после подтверждения \
                 пользователя. Пути — относительно корня проекта; .git, target, .env и базы закрыты.",
            )
    }
}

fn error_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = match std::env::var("FS_MCP_ROOT") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => std::env::current_dir().context("не удалось узнать рабочую директорию")?,
    };
    let ws = Arc::new(Workspace::open(&dir)?);
    // В режиме stdio stdout — канал протокола: всё служебное только в stderr.
    eprintln!("fs-mcp: корень {}", ws.root().display());

    if !std::env::args().any(|a| a == "--http") {
        let service = FsServer { ws }.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        return Ok(());
    }

    let addr = std::env::var("FS_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let service = StreamableHttpService::new(
        move || Ok(FsServer { ws: ws.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener =
        tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("не удалось занять адрес {addr}"))?;
    eprintln!("fs-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
