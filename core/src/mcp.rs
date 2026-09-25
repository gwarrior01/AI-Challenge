//! MCP-клиент (Model Context Protocol) поверх официального Rust SDK `rmcp`:
//! подключение к внешним MCP-серверам, получение списка их инструментов и
//! вызов этих инструментов моделью в цикле function calling агента (см.
//! [`crate::agent::Agent::handle_request`]).
//!
//! ## Конфигурация
//!
//! Серверы описываются в JSON-файле — `mcp.json` в текущей рабочей директории
//! или путь из [`MCP_CONFIG_ENV`]. Формат совместим с `mcpServers` у Claude
//! Desktop / Cursor, плюс поле `enabled`:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "everything": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"] },
//!     "javadocs":   { "url": "https://www.javadocs.dev/mcp", "enabled": false }
//!   }
//! }
//! ```
//!
//! - `command` + `args` (+ `env`, `cwd`) — локальный сервер, запускается
//!   дочерним процессом и общается через stdio;
//! - `url` (+ `headers`) — удалённый сервер по Streamable HTTP;
//! - `enabled: false` (или `disabled: true`, как у Cline) — сервер остаётся в
//!   конфигурации, но к нему не подключаемся и его инструменты модели не
//!   предлагаются. Переключается и из интерфейсов ([`McpManager::set_enabled`]) —
//!   тогда флаг переписывается прямо в файле, остальное содержимое не трогается.
//!
//! В `args`, `env`, `url` и `headers` подставляются переменные окружения вида
//! `${VAR}` — так токены доступа живут в `.env`/окружении, а не в файле
//! конфигурации (см. инвариант про секреты).
//!
//! ## Имена инструментов
//!
//! Модели инструмент предлагается под именем `mcp__<сервер>__<инструмент>`
//! (см. [`qualified_tool_name`]) — так инструменты разных серверов и встроенные
//! инструменты автомата задачи (`move_stage`/`update_step`) не пересекаются, а
//! по имени вызова однозначно понятно, какому серверу его отправить.

use crate::ToolDefinition;
use anyhow::{anyhow, bail, Context, Result};
use rmcp::model::{CallToolRequestParams, ClientConfig, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::ServiceExt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// Переменная окружения с путём к файлу конфигурации MCP-серверов — по аналогии
/// с `LLM_PROFILES_DIR`/`LLM_INVARIANTS_DIR`.
pub const MCP_CONFIG_ENV: &str = "LLM_MCP_CONFIG";

/// Префикс имён инструментов MCP в запросе к модели (см. [`qualified_tool_name`]).
pub const TOOL_PREFIX: &str = "mcp__";

/// Сколько ждать установки соединения (запуск процесса + `initialize` +
/// `tools/list`). С запасом: `npx -y ...` при первом запуске скачивает пакет.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// Сколько ждать ответа на один вызов инструмента, если у сервера не задан
/// `timeout_sec`.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Предел длины результата инструмента, который уходит модели: огромный ответ
/// сервера (дамп файла, страницы) иначе съел бы весь контекст одним вызовом.
const MAX_RESULT_CHARS: usize = 20_000;

/// Сколько последних строк stderr дочернего процесса хранить — их показываем
/// при сбое подключения (там обычно и написано, почему сервер не поднялся).
const STDERR_TAIL_LINES: usize = 20;

/// Путь к файлу конфигурации — `LLM_MCP_CONFIG`, если задан, иначе `mcp.json`
/// в текущей рабочей директории.
pub fn config_path() -> PathBuf {
    std::env::var(MCP_CONFIG_ENV).unwrap_or_else(|_| "mcp.json".to_string()).into()
}

/// Описание одного сервера в файле конфигурации.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Выключенный сервер остаётся в конфигурации, но к нему не подключаемся.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Синоним `enabled: false` в стиле Cline — учитывается при чтении.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled: Option<bool>,
    /// Команда запуска локального (stdio) сервера.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Адрес удалённого сервера (Streamable HTTP).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Необязательное пояснение для человека — показывается в интерфейсах.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Сколько ждать ответа на один вызов инструмента, в секундах; по
    /// умолчанию [`CALL_TIMEOUT`]. Больше — для серверов, у которых вызов
    /// сам ходит в LLM (краткое содержание длинного документа).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_sec: Option<u64>,
}

fn default_true() -> bool {
    true
}

impl McpServerConfig {
    /// Итоговый флаг с учётом обоих вариантов записи (`enabled`/`disabled`).
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.disabled != Some(true)
    }

    /// Краткое описание транспорта для интерфейсов — без подстановки `${VAR}`,
    /// чтобы значения секретов не попадали на экран.
    pub fn transport_label(&self) -> String {
        match (&self.command, &self.url) {
            (Some(command), _) => {
                let mut parts = vec![command.clone()];
                parts.extend(self.args.iter().cloned());
                format!("stdio: {}", parts.join(" "))
            }
            (None, Some(url)) => format!("http: {url}"),
            (None, None) => "не задан ни command, ни url".to_string(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct McpConfigFile {
    #[serde(default, rename = "mcpServers", alias = "servers")]
    servers: BTreeMap<String, McpServerConfig>,
}

/// Читает конфигурацию серверов. Отсутствующий файл — не ошибка, а пустой
/// список: MCP просто не используется.
pub fn load_config(path: &Path) -> Result<BTreeMap<String, McpServerConfig>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(err) => return Err(err).with_context(|| format!("не удалось прочитать {}", path.display())),
    };
    let file: McpConfigFile =
        serde_json::from_str(&raw).with_context(|| format!("не удалось разобрать {}", path.display()))?;
    Ok(file.servers)
}

/// Переписывает флаг `enabled` сервера `name` прямо в файле конфигурации,
/// сохраняя всё остальное содержимое файла как есть (в т.ч. поля, которых эта
/// программа не знает). `disabled` при этом убирается, чтобы два флага не
/// противоречили друг другу.
fn write_enabled_flag(path: &Path, name: &str, enabled: bool) -> Result<()> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("не удалось прочитать {}", path.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("не удалось разобрать {}", path.display()))?;
    let servers = ["mcpServers", "servers"]
        .into_iter()
        .find_map(|key| value.get_mut(key).filter(|v| v.is_object()).map(|_| key))
        .ok_or_else(|| anyhow!("в {} нет раздела mcpServers", path.display()))?;
    let server = value[servers]
        .get_mut(name)
        .and_then(|s| s.as_object_mut())
        .ok_or_else(|| anyhow!("MCP-сервер «{name}» не найден в {}", path.display()))?;
    server.insert("enabled".to_string(), serde_json::Value::Bool(enabled));
    server.remove("disabled");
    let text = serde_json::to_string_pretty(&value)? + "\n";
    std::fs::write(path, text).with_context(|| format!("не удалось записать {}", path.display()))
}

/// Подставляет переменные окружения вида `${VAR}`. Незаданная переменная — ошибка,
/// а не пустая строка: иначе сервер получил бы, например, пустой токен и
/// отвечал бы невнятной ошибкой авторизации.
fn expand_env(value: &str) -> Result<String> {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after.find('}').ok_or_else(|| anyhow!("незакрытая подстановка ${{ в «{value}»"))?;
        let var = &after[..end];
        let resolved =
            std::env::var(var).map_err(|_| anyhow!("не задана переменная окружения {var} (нужна для ${{{var}}})"))?;
        out.push_str(&resolved);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Имя инструмента для модели: `mcp__<сервер>__<инструмент>`, только символы
/// `[A-Za-z0-9_-]` и не длиннее 64 — ограничение OpenAI-совместимого API на
/// имена функций.
pub fn qualified_tool_name(server: &str, tool: &str) -> String {
    let sanitize = |s: &str| -> String {
        s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
    };
    let mut name = format!("{TOOL_PREFIX}{}__{}", sanitize(server), sanitize(tool));
    name.truncate(64);
    name
}

/// Инструмент MCP-сервера — как его описал сам сервер в ответе на `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    /// Имя, под которым инструмент предлагается модели (см. [`qualified_tool_name`]).
    pub qualified_name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema параметров.
    pub input_schema: serde_json::Value,
}

impl McpToolInfo {
    /// Заготовка доводов для ручного вызова из интерфейсов: объект с
    /// обязательными параметрами и значениями-заглушками по их типу (пустая
    /// строка, 0, false, [], {}) — необязательные не подставляются, чтобы
    /// вызов по умолчанию не отправил заглушку туда, где сервер обошёлся бы
    /// без значения; их список — в [`McpToolInfo::params_summary`].
    pub fn arguments_template(&self) -> serde_json::Value {
        let schema = &self.input_schema;
        let properties = schema.get("properties").and_then(|p| p.as_object());
        let mut template = serde_json::Map::new();
        for name in schema.get("required").and_then(|r| r.as_array()).into_iter().flatten().filter_map(|v| v.as_str())
        {
            let prop = properties.and_then(|p| p.get(name));
            let placeholder = match prop.and_then(|p| p.get("default")) {
                Some(default) => default.clone(),
                None => match prop.and_then(|p| p.get("type")).and_then(|t| t.as_str()) {
                    Some("number") | Some("integer") => serde_json::json!(0),
                    Some("boolean") => serde_json::json!(false),
                    Some("array") => serde_json::json!([]),
                    Some("object") => serde_json::json!({}),
                    _ => serde_json::json!(""),
                },
            };
            template.insert(name.to_string(), placeholder);
        }
        serde_json::Value::Object(template)
    }

    /// Параметры одной строкой для интерфейсов: `имя: тип*` (`*` — обязательный).
    pub fn params_summary(&self) -> String {
        let schema = &self.input_schema;
        let Some(properties) = schema.get("properties").and_then(|p| p.as_object()).filter(|p| !p.is_empty())
        else {
            return "нет".to_string();
        };
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|r| r.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        properties
            .iter()
            .map(|(name, prop)| {
                let kind = prop.get("type").and_then(|t| t.as_str()).unwrap_or("any");
                let mark = if required.contains(&name.as_str()) { "*" } else { "" };
                format!("{name}: {kind}{mark}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Состояние подключения к серверу.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpStatus {
    /// Сервер выключен в конфигурации — подключения нет и не будет.
    Disabled,
    /// Включён, но подключение ещё не начиналось.
    Idle,
    Connecting,
    Connected,
    Failed { error: String },
}

impl McpStatus {
    pub fn label(&self) -> &'static str {
        match self {
            McpStatus::Disabled => "выключен",
            McpStatus::Idle => "не подключён",
            McpStatus::Connecting => "подключение…",
            McpStatus::Connected => "подключён",
            McpStatus::Failed { .. } => "ошибка",
        }
    }
}

/// Снимок одного сервера для интерфейсов.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    pub enabled: bool,
    pub transport: String,
    pub description: Option<String>,
    pub status: McpStatus,
    /// Имя и версия сервера из ответа на `initialize` (если сервер их сообщил).
    pub server_version: Option<String>,
    /// Инструкции сервера из ответа на `initialize`: как пользоваться его
    /// инструментами вместе (порядок вызовов, что куда передаётся).
    pub instructions: Option<String>,
    pub tools: Vec<McpToolInfo>,
}

/// Результат одного вызова инструмента MCP.
#[derive(Debug, Clone)]
pub struct McpCallResult {
    pub server: String,
    pub tool: String,
    /// Текст, который уходит модели как результат вызова.
    pub text: String,
    /// Сервер пометил результат как ошибку (`isError`), или вызов не удался вовсе.
    pub is_error: bool,
}

type Client = RunningService<RoleClient, ClientConfig>;

struct ServerEntry {
    config: McpServerConfig,
    status: McpStatus,
    server_version: Option<String>,
    instructions: Option<String>,
    tools: Vec<McpToolInfo>,
    client: Option<Arc<Client>>,
    /// Растёт при каждом (пере)подключении/отключении — завершившееся
    /// подключение применяет свой результат, только если номер не сменился
    /// (иначе его уже отменили выключением или перезагрузкой конфигурации).
    generation: u64,
}

impl ServerEntry {
    fn new(config: McpServerConfig) -> Self {
        let status = if config.is_enabled() { McpStatus::Idle } else { McpStatus::Disabled };
        Self { config, status, server_version: None, instructions: None, tools: Vec::new(), client: None, generation: 0 }
    }
}

/// Реестр MCP-серверов из файла конфигурации и живых подключений к ним — один
/// на процесс, общий для всех агентов (см. [`crate::AgentManager::mcp`]).
pub struct McpManager {
    path: PathBuf,
    servers: RwLock<BTreeMap<String, ServerEntry>>,
    /// Ошибка чтения файла конфигурации — показывается в интерфейсах вместо
    /// молчаливого "серверов нет".
    config_error: RwLock<Option<String>>,
}

impl McpManager {
    /// Реестр по файлу `path`. Только читает конфигурацию — подключение
    /// запускает [`McpManager::connect_all`].
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let manager = Self { path: path.into(), servers: RwLock::new(BTreeMap::new()), config_error: RwLock::new(None) };
        manager.read_config();
        manager
    }

    /// Реестр по файлу из [`config_path`].
    pub fn from_env() -> Self {
        Self::load(config_path())
    }

    /// Пустой реестр без файла — MCP не используется (тесты и т.п.).
    pub fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            servers: RwLock::new(BTreeMap::new()),
            config_error: RwLock::new(None),
        }
    }

    pub fn config_path(&self) -> &Path {
        &self.path
    }

    pub fn config_error(&self) -> Option<String> {
        self.config_error.read().expect("реестр MCP отравлен паникой").clone()
    }

    fn read_config(&self) {
        let (configs, error) = if self.path.as_os_str().is_empty() {
            (BTreeMap::new(), None)
        } else {
            match load_config(&self.path) {
                Ok(configs) => (configs, None),
                Err(err) => (BTreeMap::new(), Some(format!("{err:#}"))),
            }
        };
        *self.config_error.write().expect("реестр MCP отравлен паникой") = error;
        let mut servers = self.servers.write().expect("реестр MCP отравлен паникой");
        // Отбрасывание ServerEntry закрывает и подключение (drop RunningService
        // отменяет сервис и завершает дочерний процесс).
        servers.clear();
        for (name, config) in configs {
            servers.insert(name, ServerEntry::new(config));
        }
    }

    /// Снимок всех серверов (в порядке имён) — синхронный, годится для
    /// отрисовки интерфейса на каждом кадре.
    pub fn servers(&self) -> Vec<McpServerInfo> {
        let servers = self.servers.read().expect("реестр MCP отравлен паникой");
        servers
            .iter()
            .map(|(name, entry)| McpServerInfo {
                name: name.clone(),
                enabled: entry.config.is_enabled(),
                transport: entry.config.transport_label(),
                description: entry.config.description.clone(),
                status: entry.status.clone(),
                server_version: entry.server_version.clone(),
                instructions: entry.instructions.clone(),
                tools: entry.tools.clone(),
            })
            .collect()
    }

    pub fn server(&self, name: &str) -> Option<McpServerInfo> {
        self.servers().into_iter().find(|s| s.name == name)
    }

    /// Подключается ко всем включённым серверам параллельно и ждёт, пока каждое
    /// подключение не завершится (успехом или ошибкой, см. [`McpServerInfo::status`]).
    pub async fn connect_all(&self) {
        let names: Vec<String> = {
            let servers = self.servers.read().expect("реестр MCP отравлен паникой");
            servers.iter().filter(|(_, e)| e.config.is_enabled()).map(|(n, _)| n.clone()).collect()
        };
        futures_util::future::join_all(names.iter().map(|name| self.connect(name))).await;
    }

    /// (Пере)подключается к серверу `name`: закрывает прежнее подключение,
    /// запускает сервер, выполняет `initialize` и `tools/list`. Ошибка
    /// подключения не возвращается, а сохраняется в статусе сервера — её
    /// показывают интерфейсы.
    pub async fn connect(&self, name: &str) {
        let (config, generation) = {
            let mut servers = self.servers.write().expect("реестр MCP отравлен паникой");
            let Some(entry) = servers.get_mut(name) else { return };
            if !entry.config.is_enabled() {
                return;
            }
            entry.generation += 1;
            entry.client = None;
            entry.tools.clear();
            entry.server_version = None;
            entry.instructions = None;
            entry.status = McpStatus::Connecting;
            (entry.config.clone(), entry.generation)
        };

        let result = tokio::time::timeout(CONNECT_TIMEOUT, connect_server(name, &config))
            .await
            .unwrap_or_else(|_| Err(anyhow!("сервер не ответил за {} с", CONNECT_TIMEOUT.as_secs())));

        let mut servers = self.servers.write().expect("реестр MCP отравлен паникой");
        let Some(entry) = servers.get_mut(name) else { return };
        if entry.generation != generation {
            return;
        }
        match result {
            Ok(connected) => {
                entry.status = McpStatus::Connected;
                entry.server_version = connected.server_version;
                entry.instructions = connected.instructions;
                entry.tools = connected.tools;
                entry.client = Some(Arc::new(connected.client));
            }
            Err(err) => entry.status = McpStatus::Failed { error: format!("{err:#}") },
        }
    }

    /// Включает/выключает сервер: флаг сохраняется в файле конфигурации
    /// (переживает перезапуск), выключение сразу закрывает подключение,
    /// включение — подключается.
    pub async fn set_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        if !self.servers.read().expect("реестр MCP отравлен паникой").contains_key(name) {
            bail!("MCP-сервер «{name}» не найден в {}", self.path.display());
        }
        write_enabled_flag(&self.path, name, enabled)?;
        {
            let mut servers = self.servers.write().expect("реестр MCP отравлен паникой");
            let entry = servers.get_mut(name).expect("наличие проверено выше");
            entry.config.enabled = enabled;
            entry.config.disabled = None;
            entry.generation += 1;
            entry.client = None;
            entry.tools.clear();
            entry.server_version = None;
            entry.status = if enabled { McpStatus::Idle } else { McpStatus::Disabled };
        }
        if enabled {
            self.connect(name).await;
        }
        Ok(())
    }

    /// Перечитывает файл конфигурации (закрывая все подключения) и заново
    /// подключается ко всем включённым серверам.
    pub async fn reload(&self) {
        self.read_config();
        self.connect_all().await;
    }

    /// Инструменты всех подключённых серверов в формате function calling.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let servers = self.servers.read().expect("реестр MCP отравлен паникой");
        servers
            .iter()
            .filter(|(_, entry)| entry.client.is_some())
            .flat_map(|(server, entry)| {
                entry.tools.iter().map(move |tool| ToolDefinition {
                    name: tool.qualified_name.clone(),
                    description: format!(
                        "[MCP-сервер «{server}»] {}",
                        tool.description.as_deref().or(tool.title.as_deref()).unwrap_or(&tool.name)
                    ),
                    parameters: tool.input_schema.clone(),
                })
            })
            .collect()
    }

    /// Карта подключённых серверов для системного промпта: какой сервер за что
    /// отвечает, его инструменты и его собственные инструкции из `initialize`.
    /// Описание одного инструмента говорит, ЧТО он делает; как связывать
    /// инструменты разных серверов в один флоу, модель узнаёт отсюда — иначе
    /// инструкции серверов (порядок шагов, какой идентификатор куда
    /// передавать) до неё вообще не доходят. `None` — серверов нет.
    pub fn routing_block(&self) -> Option<String> {
        let servers = self.servers.read().expect("реестр MCP отравлен паникой");
        let connected: Vec<(&String, &ServerEntry)> =
            servers.iter().filter(|(_, entry)| entry.client.is_some() && !entry.tools.is_empty()).collect();
        if connected.is_empty() {
            return None;
        }
        let mut block = String::from(
            "Подключённые MCP-серверы. Инструмент сервера <s> называется mcp__<s>__<инструмент>; \
             выбирай сервер по тому, за что он отвечает, а не по похожему названию инструмента.\n",
        );
        for (name, entry) in &connected {
            block.push_str(&format!("\n### {name}\n"));
            if let Some(description) = &entry.config.description {
                block.push_str(&format!("{description}\n"));
            }
            let tools: Vec<&str> = entry.tools.iter().map(|t| t.name.as_str()).collect();
            block.push_str(&format!("Инструменты: {}\n", tools.join(", ")));
            if let Some(instructions) = &entry.instructions {
                block.push_str(&format!("Инструкции сервера: {instructions}\n"));
            }
        }
        if connected.len() > 1 {
            block.push_str(
                "\nЕсли просьба затрагивает несколько серверов, это один флоу: сначала разложи её на шаги \
                 и для каждого шага выбери сервер; вызывай инструменты по одному, по порядку зависимостей — \
                 идентификаторы (pid, document_id, summary_id, job_id …) бери только из ответов уже \
                 выполненных вызовов, не придумывай их. Независимые шаги можно вызвать в одном ответе. \
                 Если вызов вернул ошибку, исправь аргументы или шаг, а не пропускай его молча. \
                 В конце перечисли, что сделано на каждом сервере.\n",
            );
        }
        Some(block)
    }

    /// Находит сервер и исходное имя инструмента по имени, под которым его
    /// вызвала модель. `None` — это не инструмент MCP (или сервер отключён).
    fn resolve(&self, qualified_name: &str) -> Option<(String, String, Arc<Client>, Duration)> {
        if !qualified_name.starts_with(TOOL_PREFIX) {
            return None;
        }
        let servers = self.servers.read().expect("реестр MCP отравлен паникой");
        servers.iter().find_map(|(server, entry)| {
            let client = entry.client.clone()?;
            let tool = entry.tools.iter().find(|t| t.qualified_name == qualified_name)?;
            let timeout = entry.config.timeout_sec.filter(|&s| s > 0).map(Duration::from_secs).unwrap_or(CALL_TIMEOUT);
            Some((server.clone(), tool.name.clone(), client, timeout))
        })
    }

    /// Ручной вызов инструмента `tool` сервера `server` (из интерфейсов, без
    /// модели) — то же, что [`McpManager::call`], но по исходным именам.
    /// `None` — сервер не подключён или такого инструмента у него нет.
    pub async fn call_tool(&self, server: &str, tool: &str, arguments: &str) -> Option<McpCallResult> {
        self.call(&qualified_tool_name(server, tool), arguments).await
    }

    /// `true`, если имя — инструмент одного из подключённых серверов.
    pub fn is_mcp_tool(&self, qualified_name: &str) -> bool {
        self.resolve(qualified_name).is_some()
    }

    /// Сервер и исходное имя инструмента по имени, под которым его вызвала
    /// модель; `None` — это не инструмент подключённого MCP-сервера.
    pub fn tool_origin(&self, qualified_name: &str) -> Option<(String, String)> {
        self.resolve(qualified_name).map(|(server, tool, _, _)| (server, tool))
    }

    /// Вызывает инструмент по имени, под которым его вызвала модель, с доводами
    /// как их прислала модель (сырой JSON-текст). Никогда не возвращает ошибку:
    /// сбой вызова — это тоже результат для модели (`is_error`), а не повод
    /// обрывать весь обмен.
    pub async fn call(&self, qualified_name: &str, arguments: &str) -> Option<McpCallResult> {
        let (server, tool, client, timeout) = self.resolve(qualified_name)?;
        let fail = |text: String| McpCallResult { server: server.clone(), tool: tool.clone(), text, is_error: true };

        let arguments = if arguments.trim().is_empty() { "{}" } else { arguments };
        let args = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(serde_json::Value::Object(map)) => map,
            _ => return Some(fail("доводы не разобраны: ожидался JSON-объект".to_string())),
        };
        let params = CallToolRequestParams::new(tool.clone()).with_arguments(args);
        let result = match tokio::time::timeout(timeout, client.call_tool(params)).await {
            Err(_) => return Some(fail(format!("сервер не ответил за {} с", timeout.as_secs()))),
            Ok(Err(err)) => return Some(fail(format!("ошибка вызова: {err}"))),
            Ok(Ok(result)) => result,
        };

        let mut parts: Vec<String> = result
            .content
            .iter()
            .map(|block| match block.as_text() {
                Some(text) => text.text.clone(),
                None => serde_json::to_string(block).unwrap_or_default(),
            })
            .collect();
        if parts.is_empty() {
            if let Some(structured) = &result.structured_content {
                parts.push(structured.to_string());
            }
        }
        let mut text = parts.join("\n");
        if text.chars().count() > MAX_RESULT_CHARS {
            text = text.chars().take(MAX_RESULT_CHARS).collect::<String>() + "\n…[результат обрезан]";
        }
        Some(McpCallResult { server, tool, text, is_error: result.is_error.unwrap_or(false) })
    }
}

#[cfg(test)]
impl McpManager {
    /// Подключает к реестру сервер `name` через уже готовый транспорт (в тестах —
    /// MCP-сервер в том же процессе, см. [`test_support`]), минуя файл конфигурации.
    pub(crate) async fn attach_for_tests(&self, name: &str, transport: tokio::io::DuplexStream) {
        let client = client_config().serve(transport).await.expect("подключение к тестовому серверу");
        let tools = client
            .list_all_tools()
            .await
            .expect("tools/list тестового сервера")
            .into_iter()
            .map(|tool| McpToolInfo {
                qualified_name: qualified_tool_name(name, &tool.name),
                name: tool.name.to_string(),
                title: None,
                description: tool.description.as_ref().map(|d| d.to_string()),
                input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
            })
            .collect();
        let mut entry = ServerEntry::new(McpServerConfig { enabled: true, ..Default::default() });
        entry.status = McpStatus::Connected;
        entry.instructions = server_instructions(&client);
        entry.tools = tools;
        entry.client = Some(Arc::new(client));
        self.servers.write().unwrap().insert(name.to_string(), entry);
    }
}

/// MCP-сервер для тестов, живущий в том же процессе: инструмент `add`
/// складывает `a` и `b`, `fail` всегда отвечает ошибкой (`isError`).
#[cfg(test)]
pub(crate) mod test_support {
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
        PaginatedRequestParams, Tool,
    };
    use rmcp::service::RequestContext;
    use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt};

    struct Calculator;

    impl ServerHandler for Calculator {
        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let schema = serde_json::json!({
                "type": "object",
                "properties": { "a": {"type": "number"}, "b": {"type": "number"} },
                "required": ["a", "b"]
            });
            let serde_json::Value::Object(schema) = schema else { unreachable!() };
            Ok(ListToolsResult::with_all_items(vec![
                Tool::new("add", "Сложить два числа", schema.clone()),
                Tool::new("fail", "Всегда ошибка", schema),
            ]))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            let args = request.arguments.unwrap_or_default();
            let num = |k: &str| args.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let result = match request.name.as_ref() {
                "add" => CallToolResult::success(vec![ContentBlock::text(format!("{}", num("a") + num("b")))]),
                _ => CallToolResult::error(vec![ContentBlock::text("что-то сломалось")]),
            };
            Ok(CallToolResponse::Complete(result))
        }
    }

    /// Запускает сервер и возвращает конец канала для клиента.
    pub(crate) fn spawn_calculator() -> tokio::io::DuplexStream {
        spawn(Calculator)
    }

    fn spawn<S: ServerHandler>(server: S) -> tokio::io::DuplexStream {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            if let Ok(service) = server.serve(server_io).await {
                let _ = service.waiting().await;
            }
        });
        client_io
    }

    /// Журнал вызовов, общий для нескольких фейковых серверов: по нему видно,
    /// в какой сервер и в каком порядке ушёл каждый вызов.
    pub(crate) type Journal = std::sync::Arc<std::sync::Mutex<Vec<(String, String, serde_json::Value)>>>;

    /// Ответ фейкового инструмента: `Ok` — текст результата, `Err` — `isError`.
    pub(crate) type Handler =
        std::sync::Arc<dyn Fn(&str, &serde_json::Value) -> Result<String, String> + Send + Sync>;

    /// Фейковый сервер для сценариев: инструменты без схем параметров,
    /// инструкции для `initialize` и ответы из `handler`; каждый вызов
    /// пишется в `journal` под именем `name`.
    struct Scripted {
        name: String,
        instructions: String,
        tools: Vec<(String, String)>,
        handler: Handler,
        journal: Journal,
    }

    impl ServerHandler for Scripted {
        fn get_info(&self) -> rmcp::model::ServerConfig {
            rmcp::model::ServerConfig::new(rmcp::model::ServerCapabilities::builder().enable_tools().build())
                .with_instructions(self.instructions.clone())
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let serde_json::Value::Object(schema) = serde_json::json!({ "type": "object" }) else { unreachable!() };
            let tools = self
                .tools
                .iter()
                .map(|(name, description)| Tool::new(name.clone(), description.clone(), schema.clone()))
                .collect();
            Ok(ListToolsResult::with_all_items(tools))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            let args = serde_json::Value::Object(request.arguments.unwrap_or_default());
            let tool = request.name.to_string();
            self.journal.lock().unwrap().push((self.name.clone(), tool.clone(), args.clone()));
            let result = match (self.handler)(&tool, &args) {
                Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
                Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
            };
            Ok(CallToolResponse::Complete(result))
        }
    }

    pub(crate) fn spawn_scripted(
        name: &str,
        instructions: &str,
        tools: &[(&str, &str)],
        handler: Handler,
        journal: Journal,
    ) -> tokio::io::DuplexStream {
        spawn(Scripted {
            name: name.to_string(),
            instructions: instructions.to_string(),
            tools: tools.iter().map(|(n, d)| (n.to_string(), d.to_string())).collect(),
            handler,
            journal,
        })
    }
}

struct Connected {
    client: Client,
    server_version: Option<String>,
    instructions: Option<String>,
    tools: Vec<McpToolInfo>,
}

fn client_config() -> ClientConfig {
    let mut info = ClientConfig::default();
    info.client_info = Implementation::new("llm-agent", env!("CARGO_PKG_VERSION"));
    info
}

/// Устанавливает соединение с сервером и получает список его инструментов.
async fn connect_server(name: &str, config: &McpServerConfig) -> Result<Connected> {
    let client = match (&config.command, &config.url) {
        (Some(command), _) => {
            let args = config.args.iter().map(|a| expand_env(a)).collect::<Result<Vec<_>>>()?;
            let env = config
                .env
                .iter()
                .map(|(k, v)| Ok((k.clone(), expand_env(v)?)))
                .collect::<Result<Vec<_>>>()?;
            let cwd = config.cwd.clone();
            let cmd = tokio::process::Command::new(command).configure(|cmd| {
                cmd.args(&args).envs(env);
                if let Some(cwd) = &cwd {
                    cmd.current_dir(cwd);
                }
            });
            // stderr дочернего процесса нельзя оставлять унаследованным — он
            // писал бы прямо поверх TUI; последние строки храним для сообщения
            // об ошибке подключения.
            let (transport, stderr) = TokioChildProcess::builder(cmd)
                .stderr(std::process::Stdio::piped())
                .spawn()
                .with_context(|| format!("не удалось запустить «{command}»"))?;
            let tail = Arc::new(Mutex::new(VecDeque::new()));
            if let Some(stderr) = stderr {
                tokio::spawn(collect_stderr(stderr, tail.clone()));
            }
            match client_config().serve(transport).await {
                Ok(client) => client,
                Err(err) => {
                    // Дать процессу договорить в stderr, прежде чем читать хвост.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    let tail: Vec<String> = tail.lock().expect("stderr MCP отравлен").iter().cloned().collect();
                    if tail.is_empty() {
                        bail!("ошибка инициализации: {err}");
                    }
                    bail!("ошибка инициализации: {err}\nstderr сервера:\n{}", tail.join("\n"));
                }
            }
        }
        (None, Some(url)) => {
            let url = expand_env(url)?;
            let mut transport_config =
                rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url);
            let mut headers = std::collections::HashMap::new();
            for (key, value) in &config.headers {
                let value = expand_env(value)?;
                if key.eq_ignore_ascii_case("authorization") {
                    // rmcp сам добавляет префикс "Bearer " к auth_header.
                    let token = value.strip_prefix("Bearer ").unwrap_or(&value).to_string();
                    transport_config = transport_config.auth_header(token);
                    continue;
                }
                let name = http::HeaderName::try_from(key.as_str())
                    .with_context(|| format!("некорректное имя заголовка «{key}»"))?;
                let value = http::HeaderValue::try_from(value)
                    .with_context(|| format!("некорректное значение заголовка «{key}»"))?;
                headers.insert(name, value);
            }
            transport_config.custom_headers = headers;
            let transport = StreamableHttpClientTransport::from_config(transport_config);
            client_config().serve(transport).await.map_err(|err| anyhow!("ошибка инициализации: {err}"))?
        }
        (None, None) => bail!("у сервера «{name}» не задан ни command, ни url"),
    };

    let server_version = client.peer_info().and_then(|info| {
        let imp = info.server_info.as_ref()?;
        Some(format!("{} {}", imp.name, imp.version))
    });
    let instructions = server_instructions(&client);
    let tools = client
        .list_all_tools()
        .await
        .map_err(|err| anyhow!("не удалось получить список инструментов: {err}"))?
        .into_iter()
        .map(|tool| McpToolInfo {
            qualified_name: qualified_tool_name(name, &tool.name),
            name: tool.name.to_string(),
            title: tool.title.clone(),
            description: tool.description.as_ref().map(|d| d.to_string()),
            input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
        })
        .collect();
    Ok(Connected { client, server_version, instructions, tools })
}

fn server_instructions(client: &Client) -> Option<String> {
    let info = client.peer_info()?;
    let text = info.instructions.as_deref()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

async fn collect_stderr(stderr: tokio::process::ChildStderr, tail: Arc<Mutex<VecDeque<String>>>) {
    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let mut tail = tail.lock().expect("stderr MCP отравлен");
        if tail.len() == STDERR_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_names_are_api_safe() {
        assert_eq!(qualified_tool_name("fs", "read_file"), "mcp__fs__read_file");
        assert_eq!(qualified_tool_name("my server", "a.b/c"), "mcp__my_server__a_b_c");
        assert!(qualified_tool_name("s", &"x".repeat(100)).len() <= 64);
    }

    #[test]
    fn expands_env_placeholders() {
        std::env::set_var("MCP_TEST_TOKEN", "abc");
        assert_eq!(expand_env("Bearer ${MCP_TEST_TOKEN}!").unwrap(), "Bearer abc!");
        assert_eq!(expand_env("plain").unwrap(), "plain");
        assert!(expand_env("${MCP_TEST_UNSET_VARIABLE}").is_err());
    }

    #[test]
    fn reads_config_and_toggles_enabled_in_file() {
        let path = std::env::temp_dir().join(format!("mcp-test-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"mcpServers":{"a":{"command":"echo","custom":1},"b":{"url":"http://x","disabled":true}},"other":true}"#,
        )
        .unwrap();
        let manager = McpManager::load(&path);
        let servers = manager.servers();
        assert_eq!(servers.len(), 2);
        assert_eq!(servers[0].status, McpStatus::Idle);
        assert_eq!(servers[1].status, McpStatus::Disabled);

        write_enabled_flag(&path, "a", false).unwrap();
        let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["mcpServers"]["a"]["enabled"], false);
        assert_eq!(value["mcpServers"]["a"]["custom"], 1, "незнакомые поля сохраняются");
        assert_eq!(value["other"], true);
        assert!(!load_config(&path).unwrap()["a"].is_enabled());
        std::fs::remove_file(&path).unwrap();
    }

    #[tokio::test]
    async fn lists_and_calls_tools_of_connected_server() {
        let manager = McpManager::empty();
        manager.attach_for_tests("calc", test_support::spawn_calculator()).await;

        let defs = manager.tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["mcp__calc__add", "mcp__calc__fail"]);
        assert!(defs[0].description.contains("calc"));

        let ok = manager.call("mcp__calc__add", r#"{"a":2,"b":40}"#).await.unwrap();
        assert_eq!((ok.server.as_str(), ok.tool.as_str(), ok.text.as_str(), ok.is_error), ("calc", "add", "42", false));

        let failed = manager.call("mcp__calc__fail", "{}").await.unwrap();
        assert!(failed.is_error);
        let bad_args = manager.call("mcp__calc__add", "не json").await.unwrap();
        assert!(bad_args.is_error);

        assert!(manager.call("move_stage", "{}").await.is_none(), "не MCP-инструмент");
        let direct = manager.call_tool("calc", "add", r#"{"a":1,"b":2}"#).await.unwrap();
        assert_eq!(direct.text, "3");
        let add = &manager.servers()[0].tools[0];
        assert_eq!(add.arguments_template(), serde_json::json!({ "a": 0, "b": 0 }));
        assert!(manager.call("mcp__calc__nope", "{}").await.is_none());
    }

    #[test]
    fn missing_config_file_means_no_servers() {
        let manager = McpManager::load("/nonexistent/mcp.json");
        assert!(manager.servers().is_empty());
        assert!(manager.config_error().is_none());
    }
}
