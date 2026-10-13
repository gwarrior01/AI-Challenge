//! Общее для команд ассистента: модель, свой MCP-сервер git и два своих
//! индекса — документация проекта и его код.
//!
//! **MCP.** У ассистента своя конфигурация MCP — `assistant/mcp.json`
//! (`ASSISTANT_MCP_CONFIG`), отдельная от `mcp.json` агентов: в ней только
//! сервер git (`mcp/git-mcp`, имя `ASSISTANT_GIT_SERVER`, по умолчанию `git`).
//! Подключается он при первом обращении, а не при старте.
//!
//! **Индексы.** Документация (`README.md` + `docs/`, `rag/project.db`) нужна
//! `/help` и ревью, код (`rag/code.db`) — ревью: найти, что ещё в проекте
//! опирается на изменённое. Оба целиком выводятся из файлов: перед
//! использованием они сверяются с ними по SHA-256 ([`sync_index`]), и
//! переиндексируется только изменившееся.

use std::time::Duration;

use anyhow::Result;
use llm_core::rag::{self, Embedder, RagConfig, Strategy};
use llm_core::{ChatCompletion, ChatOptions, LlmClient, McpManager, McpStatus, RequestMessage, ToolCallRecord, ToolDefinition};

/// Конфигурация MCP ассистента по умолчанию.
pub const DEFAULT_MCP_CONFIG: &str = "assistant/mcp.json";
/// Имя MCP-сервера git в ней.
pub const DEFAULT_GIT_SERVER: &str = "git";
/// Корпус документации: README и папка `docs/`.
pub const DEFAULT_DOCS: &str = "README.md,docs";
pub const DEFAULT_DOCS_DB: &str = "rag/project.db";
/// Корпус кода: все крейты workspace.
pub const DEFAULT_CODE: &str = "core/src,cli/src,tui/src,web/src,assistant/src,gateway/src,mcp";
pub const DEFAULT_CODE_DB: &str = "rag/code.db";
/// Нарезка обоих индексов: разделы Markdown по заголовкам, у кода — элементы
/// (`impl Agent › fn run`).
pub const STRATEGY: Strategy = Strategy::Structure;

/// Сколько ждать ответа модели на один запрос по умолчанию (`ASSISTANT_LLM_TIMEOUT`, секунды).
pub const DEFAULT_LLM_TIMEOUT_SEC: u64 = 180;

/// Сколько ждать сервер git, который ещё подключается.
const CONNECT_WAIT: Duration = Duration::from_secs(60);

/// Индексы — по одному на процесс: две команды подряд не индексируют одно и
/// то же наперегонки.
static INDEX_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Прогресс команды для интерфейса: этапы и вызовы MCP по мере выполнения.
pub type Progress<'a> = &'a (dyn Fn(String) + Send + Sync);

/// Модель и MCP ассистента — одни на процесс, общие для всех команд.
pub struct Assistant {
    pub client: LlmClient,
    pub mcp: McpManager,
    pub git_server: String,
}

impl Assistant {
    pub fn from_env() -> Result<Self> {
        let client = LlmClient::from_env()?;
        let mcp = McpManager::load(env_or("ASSISTANT_MCP_CONFIG", DEFAULT_MCP_CONFIG));
        Ok(Self { client, mcp, git_server: env_or("ASSISTANT_GIT_SERVER", DEFAULT_GIT_SERVER) })
    }

    /// Сервер git подключён — или почему нет (см. [`ensure_connected`]).
    pub async fn ensure_git(&self) -> Result<(), String> {
        ensure_connected(&self.mcp, &self.git_server).await
    }

    /// Вызов инструмента git кодом (не моделью). Вызов попадает в `log`;
    /// `Err` — текст ошибки (сервер вернул ошибку или инструмента нет).
    pub async fn git(
        &self,
        tool: &str,
        args: serde_json::Value,
        log: &mut Vec<ToolCallRecord>,
    ) -> Result<String, String> {
        let args = args.to_string();
        match self.mcp.call_tool(&self.git_server, tool, &args).await {
            Some(r) => {
                log.push(record(&self.git_server, tool, &args, &r.text, r.is_error));
                if r.is_error {
                    Err(r.text)
                } else {
                    Ok(r.text)
                }
            }
            None => Err(format!("у MCP-сервера «{}» нет инструмента {tool}", self.git_server)),
        }
    }

    /// Строка контекста репозитория (см. [`git_line`]) — для шапки TUI.
    pub async fn git_summary(&self) -> Result<String, String> {
        self.ensure_git().await?;
        self.git("git_branch", serde_json::json!({}), &mut Vec::new()).await.map(|json| git_line(&json))
    }
}

/// Сервер git подключён — или почему нет. Включённый, но ещё не
/// подключённый сервер подключается (или дожидается идущего подключения).
pub async fn ensure_connected(mcp: &McpManager, name: &str) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + CONNECT_WAIT;
    loop {
        let Some(server) = mcp.server(name) else {
            let detail = mcp.config_error().map(|e| format!(" ({e})")).unwrap_or_default();
            return Err(format!(
                "MCP-сервера «{name}» нет в {}{detail}",
                mcp.config_path().display()
            ));
        };
        match server.status {
            McpStatus::Connected => return Ok(()),
            McpStatus::Disabled => return Err(format!("MCP-сервер «{name}» выключен в конфигурации")),
            McpStatus::Failed { error } => {
                return Err(format!("MCP-сервер «{name}» не подключился: {}", short(&error, 200)))
            }
            McpStatus::Idle => mcp.connect(name).await,
            McpStatus::Connecting if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(250)).await
            }
            McpStatus::Connecting => return Err(format!("MCP-сервер «{name}» всё ещё подключается")),
        }
    }
}

/// Индекс документации: `ASSISTANT_DOCS` (README.md,docs) в `ASSISTANT_DOCS_DB` (rag/project.db).
pub fn docs_config() -> RagConfig {
    index_config("ASSISTANT_DOCS", DEFAULT_DOCS, "ASSISTANT_DOCS_DB", DEFAULT_DOCS_DB)
}

/// Индекс кода: `ASSISTANT_CODE` (крейты workspace) в `ASSISTANT_CODE_DB` (rag/code.db).
pub fn code_config() -> RagConfig {
    index_config("ASSISTANT_CODE", DEFAULT_CODE, "ASSISTANT_CODE_DB", DEFAULT_CODE_DB)
}

fn index_config(corpus_var: &str, corpus: &str, db_var: &str, db: &str) -> RagConfig {
    let base = std::env::current_dir().unwrap_or_else(|_| ".".into());
    let corpus: Vec<String> =
        env_or(corpus_var, corpus).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    // «Папка загрузок» у RagConfig всегда часть корпуса — здесь загрузок нет,
    // ею служит первый путь самого корпуса.
    let uploads = base.join(corpus.first().map(String::as_str).unwrap_or("docs"));
    RagConfig {
        db_path: base.join(env_or(db_var, db)),
        corpus,
        eval_path: base.join("rag/eval.json"),
        report_path: base.join("rag/compare.md"),
        uploads,
        base,
    }
}

/// Сверяет индекс с файлами — см. документацию модуля. Возвращает, что
/// изменилось (`None` — ничего).
pub async fn sync_index(cfg: &RagConfig, embedder: &Embedder) -> Result<Option<String>> {
    let _guard = INDEX_LOCK.lock().await;
    let reset = rag::reset_if_model_changed(cfg, embedder)?;
    let report = rag::index(cfg, embedder, &[STRATEGY], &|_| {}).await?;
    let mut changes: Vec<String> =
        report.new_versions.iter().map(|(source, version)| format!("{source} v{version}")).collect();
    changes.extend(report.removed.iter().map(|source| format!("{source} удалён")));
    changes.extend(report.failed.iter().map(|(source, err)| format!("{source}: {err}")));
    let name = cfg.db_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut note = match changes.len() {
        0 => None,
        n if n > 8 => Some(format!("{name}: переиндексировано файлов: {n}")),
        _ => Some(format!("{name}: переиндексировано {}", changes.join(", "))),
    };
    if reset {
        note = Some(format!("{name} пересобран моделью {} — {}", embedder.model(), note.unwrap_or_default()));
    }
    Ok(note)
}

/// Запрос к модели с пределом времени: у HTTP-клиента ядра таймаута нет, и
/// повисший ответ провайдера иначе держал бы команду бесконечно.
pub async fn chat(
    client: &LlmClient,
    wire: &[RequestMessage],
    tools: &[ToolDefinition],
    options: &ChatOptions,
) -> Result<ChatCompletion> {
    let secs: u64 = env_or("ASSISTANT_LLM_TIMEOUT", "").parse().unwrap_or(DEFAULT_LLM_TIMEOUT_SEC).max(5);
    // Отладка: `ASSISTANT_DUMP_DIR=<папка>` — каждый запрос к модели пишется туда JSON-файлом.
    if let Ok(dir) = std::env::var("ASSISTANT_DUMP_DIR") {
        let body = serde_json::json!({ "model": client.model(), "messages": wire, "tools": tools.iter().map(|t| &t.name).collect::<Vec<_>>() });
        let name = format!("request-{}.json", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0));
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(std::path::Path::new(&dir).join(name), serde_json::to_string_pretty(&body).unwrap_or_default());
    }
    match tokio::time::timeout(Duration::from_secs(secs), client.chat_with_tools(client.model(), wire, tools, false, options)).await {
        Ok(result) => result,
        Err(_) => anyhow::bail!("модель не ответила за {secs} с (предел — ASSISTANT_LLM_TIMEOUT)"),
    }
}

pub fn record(server: &str, tool: &str, arguments: &str, result: &str, is_error: bool) -> ToolCallRecord {
    ToolCallRecord {
        server: Some(server.to_string()),
        tool: tool.to_string(),
        arguments: arguments.to_string(),
        result: result.to_string(),
        is_error,
    }
}

pub fn short(s: &str, max: usize) -> String {
    rag::retrieve::short(s, max)
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).unwrap_or_else(|| default.into())
}

/// `ветка day-30 · 8eb1422 «Сообщение коммита» · изменено файлов: 3 · впереди origin/day-30 на 1`
/// из результата `git_branch`.
pub fn git_line(json: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { return short(json, 160) };
    let s = |key: &str| v.get(key).and_then(|x| x.as_str()).map(str::to_string);
    let n = |key: &str| v.get(key).and_then(|x| x.as_u64());
    let mut line = match s("branch") {
        Some(branch) => format!("ветка {branch}"),
        None => "detached HEAD".to_string(),
    };
    match v.get("last_commit").filter(|c| !c.is_null()) {
        Some(c) => {
            let hash = c.get("hash").and_then(|x| x.as_str()).unwrap_or_default();
            let subject = c.get("subject").and_then(|x| x.as_str()).unwrap_or_default();
            line.push_str(&format!(" · {hash} «{}»", short(subject, 80)));
        }
        None => line.push_str(" · коммитов ещё нет"),
    }
    match n("changed_files") {
        Some(0) => line.push_str(" · без незакоммиченных изменений"),
        Some(count) => line.push_str(&format!(" · изменено файлов: {count}")),
        None => {}
    }
    if let Some(upstream) = s("upstream") {
        match (n("ahead").unwrap_or(0), n("behind").unwrap_or(0)) {
            (0, 0) => line.push_str(&format!(" · вровень с {upstream}")),
            (ahead, 0) => line.push_str(&format!(" · впереди {upstream} на {ahead}")),
            (0, behind) => line.push_str(&format!(" · позади {upstream} на {behind}")),
            (ahead, behind) => line.push_str(&format!(" · {upstream}: +{ahead}/−{behind}")),
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_line_from_branch_result() {
        let json = r#"{"repo":"/r","branch":"day-30","head":"8eb1422","upstream":"origin/day-30","ahead":2,"behind":0,
            "last_commit":{"hash":"8eb1422","subject":"Private AI service","author":"A","date":"2026-10-11"},
            "changed_files":3,"local_branches":["main","day-30"]}"#;
        assert_eq!(
            git_line(json),
            "ветка day-30 · 8eb1422 «Private AI service» · изменено файлов: 3 · впереди origin/day-30 на 2"
        );
        let detached = r#"{"branch":null,"head":null,"last_commit":null,"changed_files":0}"#;
        assert_eq!(git_line(detached), "detached HEAD · коммитов ещё нет · без незакоммиченных изменений");
    }

    #[test]
    fn indexes_are_separate() {
        let docs = docs_config();
        assert!(docs.db_path.ends_with("rag/project.db"));
        assert!(docs.corpus.contains(&"README.md".to_string()));
        let code = code_config();
        assert!(code.db_path.ends_with("rag/code.db"));
        assert!(code.corpus.contains(&"core/src".to_string()));
        assert!(code.uploads.ends_with("core/src"));
    }

    #[tokio::test]
    async fn missing_git_server_is_explained() {
        let err = ensure_connected(&McpManager::empty(), "git").await.unwrap_err();
        assert!(err.contains("MCP-сервера «git» нет"));
    }
}
