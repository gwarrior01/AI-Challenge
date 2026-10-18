//! Модель и MCP ассистента — одни на процесс.
//!
//! **Модель** — общая для проекта (`LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`);
//! она ведёт обработку и разговор, а саммари пишет LM Studio, которую
//! вызывает сервер `media`.
//!
//! **MCP** — `media/mcp.json` (`MEDIA_MCP_CONFIG`), сервер `media`
//! (`mcp/media-mcp`, имя — `MEDIA_SERVER`). По умолчанию stdio: ассистент
//! запускает сервер сам, и задания живут, пока жив ассистент. Чтобы задания
//! переживали перезапуск ассистента, сервер запускают отдельно с `--http` и
//! указывают в конфигурации `url`.

use std::time::Duration;

use anyhow::{bail, Result};
use llm_core::{ChatCompletion, ChatOptions, LlmClient, McpManager, McpStatus, RequestMessage, ToolCallRecord, ToolDefinition};
use serde_json::Value;

pub const DEFAULT_MCP_CONFIG: &str = "media/mcp.json";
pub const DEFAULT_SERVER: &str = "media";
/// Сколько ждать ответа модели на один запрос по умолчанию (`MEDIA_LLM_TIMEOUT`, секунды).
pub const DEFAULT_LLM_TIMEOUT_SEC: u64 = 180;

/// Инструменты, которые код оставляет себе: ожидание заданий делает код,
/// а не модель, — она получает уже итог задания.
const CODE_ONLY_TOOLS: &[&str] = &["wait_job", "job_status"];

const CONNECT_WAIT: Duration = Duration::from_secs(90);

pub struct Session {
    pub client: LlmClient,
    pub mcp: McpManager,
    pub server: String,
}

impl Session {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client: LlmClient::from_env()?,
            mcp: McpManager::load(env_or("MEDIA_MCP_CONFIG", DEFAULT_MCP_CONFIG)),
            server: env_or("MEDIA_SERVER", DEFAULT_SERVER),
        })
    }

    /// Сервер подключён — или почему нет.
    pub async fn ensure(&self) -> Result<(), String> {
        ensure_connected(&self.mcp, &self.server).await
    }

    /// Инструменты для модели: всё `media`, кроме ожидания заданий.
    pub fn model_tools(&self) -> Vec<ToolDefinition> {
        let prefix = llm_core::mcp::qualified_tool_name(&self.server, "");
        self.mcp
            .tool_definitions()
            .into_iter()
            .filter(|t| t.name.strip_prefix(&prefix).is_some_and(|tool| !CODE_ONLY_TOOLS.contains(&tool)))
            .collect()
    }

    /// Вызов инструмента `media` кодом: текст результата или ошибка.
    pub async fn call(&self, tool: &str, args: Value) -> Result<String, String> {
        self.ensure().await?;
        match self.mcp.call_tool(&self.server, tool, &args.to_string()).await {
            Some(r) if r.is_error => Err(r.text),
            Some(r) => Ok(r.text),
            None => Err(format!("у MCP-сервера «{}» нет инструмента {tool}", self.server)),
        }
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

/// Запрос к модели с пределом времени (`MEDIA_LLM_TIMEOUT`): у HTTP-клиента
/// ядра таймаута нет.
pub async fn chat(
    client: &LlmClient,
    wire: &[RequestMessage],
    tools: &[ToolDefinition],
    options: &ChatOptions,
) -> Result<ChatCompletion> {
    let secs: u64 = env_or("MEDIA_LLM_TIMEOUT", "").parse().unwrap_or(DEFAULT_LLM_TIMEOUT_SEC).max(5);
    match tokio::time::timeout(Duration::from_secs(secs), client.chat_with_tools(client.model(), wire, tools, false, options)).await {
        Ok(result) => result,
        Err(_) => bail!("модель не ответила за {secs} с (предел — MEDIA_LLM_TIMEOUT)"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_server_is_explained() {
        let err = ensure_connected(&McpManager::empty(), "media").await.unwrap_err();
        assert!(err.contains("MCP-сервера «media» нет"));
    }
}

/// Живой сервер `media` через stdio, как у ассистента: инструменты модели без
/// ожидания заданий, `inspect` каталога. Собирает и запускает `media-mcp` —
/// поэтому `--ignored`: `cargo test -p media-assistant -- --ignored`.
#[cfg(test)]
mod live {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    #[ignore]
    async fn media_server_over_stdio() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::env::set_current_dir(&root).unwrap();
        let rec = std::env::temp_dir().join(format!("media-assistant-live-{}", std::process::id()));
        std::fs::create_dir_all(rec.join("transcript-chunks")).unwrap();
        std::fs::write(rec.join("transcript-chunks/chunk-000.mp3"), "mp3").unwrap();
        std::env::set_var("MEDIA_ROOT", &rec);
        std::env::set_var("LLM_API_URL", "http://127.0.0.1:9");
        std::env::set_var("LLM_API_KEY", "test");
        let s = Session {
            client: LlmClient::from_env().unwrap(),
            mcp: McpManager::load(root.join(DEFAULT_MCP_CONFIG)),
            server: DEFAULT_SERVER.into(),
        };
        s.ensure().await.unwrap();
        let tools: Vec<String> = s.model_tools().into_iter().map(|t| t.name).collect();
        assert!(tools.iter().any(|t| t.ends_with("__transcribe")), "{tools:?}");
        assert!(!tools.iter().any(|t| t.ends_with("wait_job")), "{tools:?}");

        let v: Value = serde_json::from_str(&s.call("inspect", json!({})).await.unwrap()).unwrap();
        assert_eq!(v["next"]["step"], "transcribe", "{v}");
        let err = s.call("split_audio", json!({})).await.unwrap_err();
        assert!(err.contains("следующий шаг"), "{err}");
    }
}
