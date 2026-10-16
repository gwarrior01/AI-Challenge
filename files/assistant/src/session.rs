//! Модель и MCP ассистента — одни на процесс.
//!
//! **MCP.** Своя конфигурация — `files/mcp.json` (`FILES_MCP_CONFIG`): сервер
//! файлов `fs` (`mcp/fs-mcp`, обязателен, имя — `FILES_FS_SERVER`) и сервер
//! git `git` (`mcp/git-mcp`, по желанию — история для changelog и ADR, имя —
//! `FILES_GIT_SERVER`). Оба — stdio: ассистент запускает их сам, и оверлей
//! изменений живёт, пока жив процесс ассистента.
//!
//! **Изменения.** Модель правит файлы только через оверлей `fs`; на диск их
//! пишет [`Session::apply`] — код, по команде пользователя (`/apply`,
//! `--apply`). Инструмента `apply_changes` у модели нет.

use std::time::Duration;

use anyhow::{bail, Result};
use llm_core::{ChatCompletion, ChatOptions, LlmClient, McpManager, McpStatus, RequestMessage, ToolCallRecord, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const DEFAULT_MCP_CONFIG: &str = "files/mcp.json";
pub const DEFAULT_FS_SERVER: &str = "fs";
pub const DEFAULT_GIT_SERVER: &str = "git";
/// Сколько ждать ответа модели на один запрос по умолчанию (`FILES_LLM_TIMEOUT`, секунды).
pub const DEFAULT_LLM_TIMEOUT_SEC: u64 = 180;

/// Инструменты, которые код оставляет себе: запись на диск — только по команде пользователя.
const CODE_ONLY_TOOLS: &[&str] = &["apply_changes"];
/// Инструменты git, которые нужны модели: история и изменения. Файлы она
/// читает через `fs` — у него оверлей.
const GIT_TOOLS: &[&str] = &["git_branch", "git_status", "git_log", "git_diff"];

const CONNECT_WAIT: Duration = Duration::from_secs(90);

/// Файл в ожидающих изменениях (`pending_changes`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PendingFile {
    pub path: String,
    pub status: String,
    pub added: usize,
    pub deleted: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Pending {
    pub files: Vec<PendingFile>,
    pub added: usize,
    pub deleted: usize,
    pub patch: String,
}

impl Pending {
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// `2 файла (+10 −3)` — для шапки и отчётов.
    pub fn summary(&self) -> String {
        format!("{} {} (+{} −{})", self.files.len(), files_word(self.files.len()), self.added, self.deleted)
    }

    /// Список файлов построчно: `  ~ docs/api.md  +3 −1`.
    pub fn listing(&self) -> String {
        self.files
            .iter()
            .map(|f| {
                let mark = if f.status == "created" { "+" } else { "~" };
                format!("  {mark} {}  +{} −{}", f.path, f.added, f.deleted)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn files_word(n: usize) -> &'static str {
    match (n % 10, n % 100) {
        (1, r) if r != 11 => "файл",
        (2..=4, r) if !(12..=14).contains(&r) => "файла",
        _ => "файлов",
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Conflict {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Applied {
    pub written: Vec<String>,
    pub conflicts: Vec<Conflict>,
}

pub struct Session {
    pub client: LlmClient,
    pub mcp: McpManager,
    pub fs_server: String,
    pub git_server: String,
}

impl Session {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client: LlmClient::from_env()?,
            mcp: McpManager::load(env_or("FILES_MCP_CONFIG", DEFAULT_MCP_CONFIG)),
            fs_server: env_or("FILES_FS_SERVER", DEFAULT_FS_SERVER),
            git_server: env_or("FILES_GIT_SERVER", DEFAULT_GIT_SERVER),
        })
    }

    /// Сервер файлов подключён — или почему нет.
    pub async fn ensure_fs(&self) -> Result<(), String> {
        ensure_connected(&self.mcp, &self.fs_server).await
    }

    pub async fn ensure_git(&self) -> Result<(), String> {
        ensure_connected(&self.mcp, &self.git_server).await
    }

    /// Инструменты для модели: `fs` без записи на диск и, если git подключён, его история.
    pub fn model_tools(&self, with_git: bool) -> Vec<ToolDefinition> {
        let fs_prefix = llm_core::mcp::qualified_tool_name(&self.fs_server, "");
        let git_prefix = llm_core::mcp::qualified_tool_name(&self.git_server, "");
        self.mcp
            .tool_definitions()
            .into_iter()
            .filter(|t| match t.name.strip_prefix(&fs_prefix) {
                Some(tool) => !CODE_ONLY_TOOLS.contains(&tool),
                None => with_git && t.name.strip_prefix(&git_prefix).is_some_and(|tool| GIT_TOOLS.contains(&tool)),
            })
            .collect()
    }

    /// Вызов инструмента `fs` кодом: текст результата или ошибка. Вызов попадает в `log`.
    pub async fn fs(&self, tool: &str, args: Value, log: &mut Vec<ToolCallRecord>) -> Result<String, String> {
        self.ensure_fs().await?;
        let args = args.to_string();
        match self.mcp.call_tool(&self.fs_server, tool, &args).await {
            Some(r) => {
                log.push(record(&self.fs_server, tool, &args, &r.text, r.is_error));
                if r.is_error {
                    Err(r.text)
                } else {
                    Ok(r.text)
                }
            }
            None => Err(format!("у MCP-сервера «{}» нет инструмента {tool}", self.fs_server)),
        }
    }

    pub async fn pending(&self, log: &mut Vec<ToolCallRecord>) -> Result<Pending, String> {
        let text = self.fs("pending_changes", json!({}), log).await?;
        serde_json::from_str(&text).map_err(|e| format!("pending_changes вернул не то: {e}"))
    }

    /// Записать изменения на диск: все или только `paths`.
    pub async fn apply(&self, paths: &[String], log: &mut Vec<ToolCallRecord>) -> Result<Applied, String> {
        let text = self.fs("apply_changes", paths_args(paths), log).await?;
        serde_json::from_str(&text).map_err(|e| format!("apply_changes вернул не то: {e}"))
    }

    pub async fn discard(&self, paths: &[String], log: &mut Vec<ToolCallRecord>) -> Result<Vec<String>, String> {
        let text = self.fs("discard_changes", paths_args(paths), log).await?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("discard_changes вернул не то: {e}"))?;
        Ok(v["discarded"].as_array().into_iter().flatten().filter_map(|p| p.as_str().map(str::to_string)).collect())
    }

    /// Строка состояния git для шапки и промпта (`ветка day-34 · 88a325c «…»`).
    pub async fn git_line(&self) -> Result<String, String> {
        self.ensure_git().await?;
        match self.mcp.call_tool(&self.git_server, "git_branch", "{}").await {
            Some(r) if !r.is_error => Ok(git_line(&r.text)),
            Some(r) => Err(r.text),
            None => Err("нет инструмента git_branch".into()),
        }
    }
}

fn paths_args(paths: &[String]) -> Value {
    if paths.is_empty() {
        json!({})
    } else {
        json!({ "paths": paths })
    }
}

/// Сервер подключён — или почему нет. Включённый, но ещё не подключённый
/// сервер подключается (или дожидается идущего подключения: первый
/// `cargo run` может собирать сервер долго).
pub async fn ensure_connected(mcp: &McpManager, name: &str) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + CONNECT_WAIT;
    loop {
        let Some(server) = mcp.server(name) else {
            let detail = mcp.config_error().map(|e| format!(" ({e})")).unwrap_or_default();
            return Err(format!("MCP-сервера «{name}» нет в {}{detail}", mcp.config_path().display()));
        };
        match server.status {
            McpStatus::Connected => return Ok(()),
            McpStatus::Disabled => return Err(format!("MCP-сервер «{name}» выключен в конфигурации")),
            McpStatus::Failed { error } => return Err(format!("MCP-сервер «{name}» не подключился: {}", short(&error, 200))),
            McpStatus::Idle => mcp.connect(name).await,
            McpStatus::Connecting if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(250)).await
            }
            McpStatus::Connecting => return Err(format!("MCP-сервер «{name}» всё ещё подключается")),
        }
    }
}

/// Запрос к модели с пределом времени (`FILES_LLM_TIMEOUT`): у HTTP-клиента
/// ядра таймаута нет.
pub async fn chat(
    client: &LlmClient,
    wire: &[RequestMessage],
    tools: &[ToolDefinition],
    options: &ChatOptions,
) -> Result<ChatCompletion> {
    let secs: u64 = env_or("FILES_LLM_TIMEOUT", "").parse().unwrap_or(DEFAULT_LLM_TIMEOUT_SEC).max(5);
    match tokio::time::timeout(Duration::from_secs(secs), client.chat_with_tools(client.model(), wire, tools, false, options)).await {
        Ok(result) => result,
        Err(_) => bail!("модель не ответила за {secs} с (предел — FILES_LLM_TIMEOUT)"),
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
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > max {
        flat.chars().take(max).collect::<String>() + "…"
    } else {
        flat
    }
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).unwrap_or_else(|| default.into())
}

/// `ветка day-34 · 88a325c «Merge branch 'day-33'» · изменено файлов: 3` из результата `git_branch`.
pub fn git_line(json: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(json) else { return short(json, 160) };
    let mut line = match v["branch"].as_str() {
        Some(branch) => format!("ветка {branch}"),
        None => "detached HEAD".to_string(),
    };
    if let (Some(hash), Some(subject)) = (v["last_commit"]["hash"].as_str(), v["last_commit"]["subject"].as_str()) {
        line.push_str(&format!(" · {hash} «{}»", short(subject, 60)));
    }
    match v["changed_files"].as_u64() {
        Some(0) => line.push_str(" · без незакоммиченных изменений"),
        Some(n) => line.push_str(&format!(" · изменено файлов: {n}")),
        None => {}
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_plurals() {
        let words: Vec<_> = [1, 2, 5, 11, 12, 21, 22, 25, 101].iter().map(|&n| files_word(n)).collect();
        assert_eq!(words, ["файл", "файла", "файлов", "файлов", "файлов", "файл", "файла", "файлов", "файл"]);
    }

    #[test]
    fn pending_parses_server_json() {
        let json = r#"{"files":[{"path":"CHANGELOG.md","status":"created","added":12,"deleted":0},
            {"path":"docs/api.md","status":"modified","added":2,"deleted":1}],"added":14,"deleted":1,"patch":"diff --git …"}"#;
        let p: Pending = serde_json::from_str(json).unwrap();
        assert_eq!(p.summary(), "2 файла (+14 −1)");
        assert_eq!(p.listing(), "  + CHANGELOG.md  +12 −0\n  ~ docs/api.md  +2 −1");
    }

    #[test]
    fn git_line_from_branch() {
        let json = r#"{"branch":"day-34","last_commit":{"hash":"88a325c","subject":"Merge branch 'day-33'"},"changed_files":0}"#;
        assert_eq!(git_line(json), "ветка day-34 · 88a325c «Merge branch 'day-33'» · без незакоммиченных изменений");
    }

    #[tokio::test]
    async fn missing_server_is_explained() {
        let err = ensure_connected(&McpManager::empty(), "fs").await.unwrap_err();
        assert!(err.contains("MCP-сервера «fs» нет"));
    }
}

/// Живой сервер `fs` через stdio, как у ассистента: поиск, правка в оверлее,
/// дифф, отказ. Собирает и запускает `fs-mcp` — поэтому `--ignored`:
/// `cargo test -p file-assistant -- --ignored`.
#[cfg(test)]
mod live {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn fs_server_over_stdio() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::env::set_current_dir(&root).unwrap();
        let s = Session {
            // Модель тесту не нужна — только клиент для структуры.
            client: {
                std::env::set_var("LLM_API_URL", "http://127.0.0.1:9");
                std::env::set_var("LLM_API_KEY", "test");
                LlmClient::from_env().unwrap()
            },
            mcp: McpManager::load(root.join(DEFAULT_MCP_CONFIG)),
            fs_server: DEFAULT_FS_SERVER.into(),
            git_server: DEFAULT_GIT_SERVER.into(),
        };
        s.ensure_fs().await.unwrap();
        let tools: Vec<String> = s.model_tools(false).into_iter().map(|t| t.name).collect();
        assert!(tools.iter().any(|t| t.ends_with("edit_file")));
        assert!(!tools.iter().any(|t| t.ends_with("apply_changes")), "{tools:?}");

        let mut log = Vec::new();
        let found = s.fs("search", json!({ "query": "McpManager::load", "pattern": "*.rs" }), &mut log).await.unwrap();
        let v: Value = serde_json::from_str(&found).unwrap();
        assert!(v["total_matches"].as_u64().unwrap() >= 2, "{found}");

        s.fs("edit_file", json!({ "path": "files/scenarios/usages.md", "old_text": "# Где", "new_text": "# Где-где" }), &mut log)
            .await
            .unwrap();
        let pending = s.pending(&mut log).await.unwrap();
        assert_eq!(pending.files[0].path, "files/scenarios/usages.md");
        assert!(pending.patch.contains("+# Где-где используется McpManager"));
        assert!(s.fs("read_file", json!({ "path": ".env" }), &mut log).await.is_err());
        assert_eq!(s.discard(&[], &mut log).await.unwrap(), vec!["files/scenarios/usages.md"]);
        assert!(s.pending(&mut log).await.unwrap().is_empty());
        assert!(std::fs::read_to_string(root.join("files/scenarios/usages.md")).unwrap().starts_with("# Где используется"));
    }
}
