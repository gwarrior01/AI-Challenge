//! CRM через MCP: пользователи и тикеты из `support-mcp`.
//!
//! У сервиса своя конфигурация MCP — `support/mcp.json` (`SUPPORT_MCP_CONFIG`),
//! отдельная от `mcp.json` агентов: в ней только сервер CRM (имя
//! `SUPPORT_CRM_SERVER`, по умолчанию `support`). По умолчанию он запускается
//! дочерним процессом через stdio, отдельно поднимать его не нужно.

use std::time::Duration;

use llm_core::{McpManager, McpStatus, ToolCallRecord, ToolDefinition};
use serde_json::Value;

pub const DEFAULT_MCP_CONFIG: &str = "support/mcp.json";
pub const DEFAULT_SERVER: &str = "support";

/// Сколько ждать сервер, который ещё подключается (первый `cargo run` собирает его).
const CONNECT_WAIT: Duration = Duration::from_secs(120);

/// Инструменты CRM, которые модель может вызвать сама, — дочитать то, что
/// код не подтянул заранее (история обращений, другой тикет, профиль).
const MODEL_TOOLS: [&str; 3] = ["get_ticket", "get_user", "list_user_tickets"];

pub struct Crm {
    pub mcp: McpManager,
    pub server: String,
}

/// Результат вызова инструмента CRM.
pub enum Lookup {
    Found(Value),
    /// Сервер ответил ошибкой — например, «тикет TCK-9999 не найден».
    Missing(String),
    /// До сервера не дошли: не настроен, не подключился, нет инструмента.
    Unavailable(String),
}

impl Crm {
    pub fn from_env() -> Self {
        Self {
            mcp: McpManager::load(crate::env_or("SUPPORT_MCP_CONFIG", DEFAULT_MCP_CONFIG)),
            server: crate::env_or("SUPPORT_CRM_SERVER", DEFAULT_SERVER),
        }
    }

    /// Сервер CRM подключён — или почему нет. Включённый, но ещё не
    /// подключённый сервер подключается (или дожидается идущего подключения).
    pub async fn ensure_connected(&self) -> Result<(), String> {
        let name = self.server.as_str();
        let deadline = tokio::time::Instant::now() + CONNECT_WAIT;
        loop {
            let Some(server) = self.mcp.server(name) else {
                let detail = self
                    .mcp
                    .config_error()
                    .map(|e| format!(" ({e})"))
                    .unwrap_or_default();
                return Err(format!(
                    "MCP-сервера «{name}» нет в {}{detail}",
                    self.mcp.config_path().display()
                ));
            };
            match server.status {
                McpStatus::Connected => return Ok(()),
                McpStatus::Disabled => {
                    return Err(format!("MCP-сервер «{name}» выключен в конфигурации"))
                }
                McpStatus::Failed { error } => {
                    return Err(format!(
                        "MCP-сервер «{name}» не подключился: {}",
                        crate::short(&error, 200)
                    ))
                }
                McpStatus::Idle => self.mcp.connect(name).await,
                McpStatus::Connecting if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(250)).await
                }
                McpStatus::Connecting => {
                    return Err(format!("MCP-сервер «{name}» всё ещё подключается"))
                }
            }
        }
    }

    /// Вызов инструмента CRM кодом (не моделью); вызов попадает в `log`.
    pub async fn lookup(&self, tool: &str, args: Value, log: &mut Vec<ToolCallRecord>) -> Lookup {
        if let Err(note) = self.ensure_connected().await {
            return Lookup::Unavailable(note);
        }
        let args = args.to_string();
        let Some(r) = self.mcp.call_tool(&self.server, tool, &args).await else {
            return Lookup::Unavailable(format!(
                "у MCP-сервера «{}» нет инструмента {tool}",
                self.server
            ));
        };
        log.push(crate::record(
            &self.server,
            tool,
            &args,
            &r.text,
            r.is_error,
        ));
        if r.is_error {
            return Lookup::Missing(r.text);
        }
        match serde_json::from_str(&r.text) {
            Ok(value) => Lookup::Found(value),
            Err(_) => Lookup::Unavailable(format!(
                "{tool} вернул не JSON: {}",
                crate::short(&r.text, 120)
            )),
        }
    }

    /// Определения инструментов CRM для модели (пусто, если сервер не подключён).
    pub fn model_tools(&self) -> Vec<ToolDefinition> {
        let allowed: Vec<String> = MODEL_TOOLS
            .iter()
            .map(|t| llm_core::mcp::qualified_tool_name(&self.server, t))
            .collect();
        self.mcp
            .tool_definitions()
            .into_iter()
            .filter(|t| allowed.contains(&t.name))
            .collect()
    }
}

/// Идентификатор вида `TCK-1001` / `USR-001` в тексте (без учёта регистра),
/// приведённый к верхнему регистру. Берётся последний упомянутый — в
/// «не TCK-1001, а TCK-1002» речь о втором.
pub fn find_id(text: &str, prefix: &str) -> Option<String> {
    let upper = text.to_uppercase();
    let chars: Vec<char> = upper.chars().collect();
    let pat: Vec<char> = format!("{prefix}-").chars().collect();
    let mut found = None;
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || !chars[i - 1].is_alphanumeric();
        if boundary && chars[i..].starts_with(&pat) {
            let digits: String = chars[i + pat.len()..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let end = i + pat.len() + digits.len();
            if !digits.is_empty() && chars.get(end).is_none_or(|c| !c.is_alphanumeric()) {
                found = Some(format!("{prefix}-{digits}"));
                i = end;
                continue;
            }
        }
        i += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_found_in_free_text() {
        assert_eq!(
            find_id("Тикет TCK-1001. Почему не работает вход?", "TCK").as_deref(),
            Some("TCK-1001")
        );
        assert_eq!(
            find_id("тикет tck-1002", "TCK").as_deref(),
            Some("TCK-1002")
        );
        assert_eq!(
            find_id("не TCK-1001, а TCK-1002", "TCK").as_deref(),
            Some("TCK-1002")
        );
        assert_eq!(find_id("клиент USR-001", "USR").as_deref(), Some("USR-001"));
        assert_eq!(find_id("XTCK-1001 и TCK-", "TCK"), None);
        assert_eq!(find_id("TCK-12a", "TCK"), None);
        assert_eq!(find_id("без номера", "TCK"), None);
    }

    #[tokio::test]
    async fn missing_server_is_explained() {
        let crm = Crm {
            mcp: McpManager::empty(),
            server: "support".into(),
        };
        let err = crm.ensure_connected().await.unwrap_err();
        assert!(err.contains("MCP-сервера «support» нет"));
        assert!(crm.model_tools().is_empty());
    }
}
