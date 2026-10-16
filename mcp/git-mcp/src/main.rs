//! MCP-сервер git: ассистент разработчика узнаёт из него, в каком состоянии
//! проект — текущую ветку, изменённые файлы, список файлов, текст файла, дифф
//! (в том числе ветки против ветки), журнал.
//! Только чтение: ничего, что меняет репозиторий, у сервера нет.
//!
//! Запуск по умолчанию — stdio: клиент сам поднимает сервер дочерним
//! процессом (`mcp.example.json`: `cargo run -q -p git-mcp`), и тот работает с
//! репозиторием своей рабочей директории (или `GIT_MCP_REPO`). С флагом
//! `--http` — Streamable HTTP на `127.0.0.1:8094/mcp` (`GIT_MCP_ADDR`), как
//! остальные свои серверы.

mod git;

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

use git::{BranchInfo, Commit, Diff, DiffQuery, FileList, FileQuery, FileText, FilesQuery, Repo, Status};

const DEFAULT_ADDR: &str = "127.0.0.1:8094";
const DEFAULT_FILES_LIMIT: usize = 300;
const MAX_FILES_LIMIT: usize = 2_000;
const DEFAULT_DIFF_CHARS: usize = 15_000;
const MAX_DIFF_CHARS: usize = 19_000;
const DEFAULT_LOG: usize = 10;
const DEFAULT_FILE_LINES: usize = 400;
const MAX_FILE_LINES: usize = 2_000;
const MAX_LOG: usize = 200;

#[derive(Debug, Deserialize, JsonSchema)]
struct FilesParams {
    /// Каталог или файл относительно корня репозитория (`core/src`); по умолчанию — весь репозиторий.
    #[serde(default)]
    path: Option<String>,
    /// Шаблон имени: `*.rs`, `Cargo.toml`; с `/` — по всему пути (`mcp/*/Cargo.toml`).
    #[serde(default)]
    pattern: Option<String>,
    /// Свернуть пути до этой глубины и посчитать файлы в каждом каталоге — обзор структуры
    /// (`depth: 1` — верхний уровень, `2` — на уровень глубже).
    #[serde(default)]
    depth: Option<usize>,
    /// Включить неотслеживаемые файлы (не игнорируемые .gitignore). По умолчанию — только отслеживаемые.
    #[serde(default)]
    untracked: bool,
    /// Сколько строк вернуть, по умолчанию 300.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiffParams {
    /// `true` — только то, что добавлено в индекс (`git add`), то есть войдёт в коммит.
    #[serde(default)]
    staged: bool,
    /// Ветка или коммит (`main`): показать всё, что изменено с момента ответвления от него —
    /// коммиты текущей ветки вместе с незакоммиченным. Без него — незакоммиченные изменения.
    #[serde(default)]
    base: Option<String>,
    /// Ветка или коммит, который сравнивается с `base` (`feature`): только закоммиченное в нём с
    /// момента ответвления от `base`, без рабочей копии — ревью ветки целиком. Только вместе с `base`.
    #[serde(default)]
    head: Option<String>,
    /// Ограничить дифф каталогом или файлом относительно корня репозитория.
    #[serde(default)]
    path: Option<String>,
    /// Только список файлов с числом строк, без самого диффа.
    #[serde(default)]
    stat_only: bool,
    /// Предел длины диффа в символах, по умолчанию 15000.
    #[serde(default)]
    max_chars: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadParams {
    /// Файл относительно корня репозитория (`core/src/lib.rs`).
    path: String,
    /// Ветка или коммит, из которого читать (`main`); по умолчанию — рабочая копия.
    #[serde(default, rename = "ref")]
    git_ref: Option<String>,
    /// С какой строки (с 1), по умолчанию с первой.
    #[serde(default)]
    start_line: Option<usize>,
    /// Сколько строк вернуть, по умолчанию 400.
    #[serde(default)]
    max_lines: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LogParams {
    /// Сколько последних коммитов, по умолчанию 10, не больше 200.
    #[serde(default)]
    n: Option<usize>,
    /// Только коммиты, затронувшие этот каталог или файл.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct Log {
    commits: Vec<Commit>,
}

#[derive(Clone)]
struct GitServer {
    repo: Arc<Repo>,
}

#[tool_router]
impl GitServer {
    #[tool(
        name = "git_branch",
        description = "Текущая git-ветка проекта: имя (или detached HEAD), коммит HEAD и последний коммит, \
            отслеживаемая ветка и насколько текущая впереди/позади неё, сколько файлов изменено, все \
            локальные ветки.",
        annotations(read_only_hint = true)
    )]
    async fn git_branch(&self) -> Result<Json<BranchInfo>, String> {
        git::branch(&self.repo).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "git_status",
        description = "Незакоммиченные изменения: какие файлы изменены, добавлены в индекс, удалены, \
            переименованы, не отслеживаются или в конфликте.",
        annotations(read_only_hint = true)
    )]
    async fn git_status(&self) -> Result<Json<Status>, String> {
        git::status(&self.repo).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "list_files",
        description = "Файлы проекта под git (то, что не игнорируется): весь список, каталог (`path`), \
            по шаблону (`pattern`: `*.rs`) или обзор структуры — пути, свёрнутые до глубины `depth`, с \
            числом файлов в каждом каталоге.",
        annotations(read_only_hint = true)
    )]
    async fn list_files(&self, Parameters(p): Parameters<FilesParams>) -> Result<Json<FileList>, String> {
        let query = FilesQuery {
            path: p.path,
            pattern: p.pattern,
            depth: p.depth,
            untracked: p.untracked,
            limit: p.limit.unwrap_or(DEFAULT_FILES_LIMIT).clamp(1, MAX_FILES_LIMIT),
        };
        git::files(&self.repo, &query).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "git_diff",
        description = "Дифф изменений: по умолчанию незакоммиченные (рабочая копия), `staged` — то, что в \
            индексе, `base` — всё, что сделано в ветке с момента ответвления от `base` (`main`), `base` + `head` — \
            закоммиченное в ветке `head` относительно `base`. \
            Возвращает файлы с числом добавленных/удалённых строк и сам дифф (обрезается по `max_chars`; \
            для большого диффа сначала `stat_only`, затем `path`).",
        annotations(read_only_hint = true)
    )]
    async fn git_diff(&self, Parameters(p): Parameters<DiffParams>) -> Result<Json<Diff>, String> {
        let query = DiffQuery {
            staged: p.staged,
            base: p.base,
            head: p.head,
            path: p.path,
            stat_only: p.stat_only,
            max_chars: p.max_chars.unwrap_or(DEFAULT_DIFF_CHARS).clamp(500, MAX_DIFF_CHARS),
        };
        git::diff(&self.repo, &query).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "read_file",
        description = "Текст файла проекта: из рабочей копии или из ветки/коммита (`ref`), целиком или с \
            `start_line` по `max_lines` строк. Только текстовые файлы внутри репозитория.",
        annotations(read_only_hint = true)
    )]
    async fn read_file(&self, Parameters(p): Parameters<ReadParams>) -> Result<Json<FileText>, String> {
        let query = FileQuery {
            path: p.path,
            git_ref: p.git_ref,
            start_line: p.start_line.unwrap_or(1),
            max_lines: p.max_lines.unwrap_or(DEFAULT_FILE_LINES).clamp(1, MAX_FILE_LINES),
        };
        git::read_file(&self.repo, &query).await.map(Json).map_err(error_text)
    }

    #[tool(
        name = "git_log",
        description = "Последние коммиты: хэш, заголовок, автор, дата — всего репозитория или одного \
            каталога/файла (`path`).",
        annotations(read_only_hint = true)
    )]
    async fn git_log(&self, Parameters(p): Parameters<LogParams>) -> Result<Json<Log>, String> {
        let n = p.n.unwrap_or(DEFAULT_LOG).clamp(1, MAX_LOG);
        git::log(&self.repo, n, p.path.as_deref()).await.map(|commits| Json(Log { commits })).map_err(error_text)
    }
}

#[tool_handler]
impl ServerHandler for GitServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("git-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Состояние git-репозитория проекта, только чтение. Ветка и последний коммит — git_branch; \
                 что изменено — git_status, затем git_diff (для большого диффа сначала stat_only=true, \
                 потом path); что сделано в ветке относительно main — git_diff с base=\"main\" (другая ветка — ещё head); \
                 текст файла — read_file; где что \
                 лежит — list_files (depth=1 или 2 для обзора структуры, pattern для поиска по имени); \
                 история — git_log. Пути — относительно корня репозитория.",
            )
    }
}

fn error_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = match std::env::var("GIT_MCP_REPO") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => std::env::current_dir().context("не удалось узнать рабочую директорию")?,
    };
    let repo = Arc::new(Repo::open(&dir).await?);
    // В режиме stdio stdout — канал протокола: всё служебное только в stderr.
    eprintln!("git-mcp: репозиторий {}", repo.root().display());

    if !std::env::args().any(|a| a == "--http") {
        let service = GitServer { repo }.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        return Ok(());
    }

    let addr = std::env::var("GIT_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let service = StreamableHttpService::new(
        move || Ok(GitServer { repo: repo.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener =
        tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("не удалось занять адрес {addr}"))?;
    eprintln!("git-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
