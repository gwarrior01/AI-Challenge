//! Что умеет выполнять задание. Планировщик не знает ни о JVM, ни о
//! конкретных API: у задания есть тип действия и параметры, а результат —
//! произвольный JSON, из которого потом берутся метрики (см. [`crate::metrics`]).
//!
//! - `reminder` — напоминание: создаёт событие с текстом;
//! - `http` — HTTP-запрос: статус, время ответа и тело (JSON или текст);
//! - `mcp_tool` — вызов инструмента другого MCP-сервера (Streamable HTTP):
//!   так по расписанию можно запускать что угодно, что уже умеет какой-то
//!   сервер, не добавляя сюда кода.
//!
//! Выполнение произвольных команд оболочки намеренно не поддерживается:
//! задания создаёт модель, и «запускай любую команду по расписанию» было бы
//! дырой, а не инструментом.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use rmcp::model::{CallToolRequestParams, ClientConfig, Implementation};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};

/// Сколько ждать HTTP-ответа по умолчанию и максимум.
const DEFAULT_HTTP_TIMEOUT_SEC: u64 = 10;
const MAX_HTTP_TIMEOUT_SEC: u64 = 60;
/// Сколько ждать подключения к MCP-серверу и ответа инструмента.
const MCP_TIMEOUT: Duration = Duration::from_secs(120);
/// Предел текста в результате: тело страницы или длинный ответ инструмента
/// хранится в каждой строке `runs`.
const MAX_TEXT_CHARS: usize = 4_000;

pub const ACTIONS: &[&str] = &["reminder", "http", "mcp_tool"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReminderParams {
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpParams {
    url: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Option<Value>,
    #[serde(default)]
    timeout_sec: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpToolParams {
    /// Имя сервера из файла конфигурации MCP (`mcp.json`).
    #[serde(default)]
    server: Option<String>,
    /// Или адрес сервера напрямую.
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    tool: String,
    #[serde(default)]
    arguments: Map<String, Value>,
}

/// Результат действия: JSON для `runs` и, если нужно, событие для агента.
#[derive(Debug)]
pub struct ActionOutput {
    pub result: Value,
    pub event: Option<String>,
}

#[derive(Clone)]
pub struct Executor {
    http: reqwest::Client,
    /// Файл конфигурации MCP-серверов клиента — по нему задания `mcp_tool`
    /// находят сервер по имени.
    mcp_config: PathBuf,
    /// Свой адрес: вызывать по расписанию собственные инструменты нельзя —
    /// задание, создающее задания, размножалось бы без конца.
    own_addr: String,
}

impl Executor {
    pub fn new(mcp_config: PathBuf, own_addr: String) -> Self {
        Self { http: reqwest::Client::new(), mcp_config, own_addr }
    }

    /// Проверяет параметры при создании задания, чтобы ошибка пришла модели
    /// сразу, а не в первом запуске.
    pub fn validate(&self, action: &str, params: &Value) -> Result<()> {
        match action {
            "reminder" => {
                let p: ReminderParams = parse(action, params)?;
                if p.text.trim().is_empty() {
                    bail!("пустой текст напоминания");
                }
            }
            "http" => {
                let p: HttpParams = parse(action, params)?;
                method(&p)?;
                let url = reqwest::Url::parse(&p.url).with_context(|| format!("некорректный url «{}»", p.url))?;
                if !matches!(url.scheme(), "http" | "https") {
                    bail!("url должен начинаться с http:// или https://");
                }
            }
            "mcp_tool" => {
                let p: McpToolParams = parse(action, params)?;
                self.mcp_endpoint(&p)?;
            }
            other => bail!("неизвестное действие «{other}»: ожидается одно из {}", ACTIONS.join(", ")),
        }
        Ok(())
    }

    pub async fn execute(&self, action: &str, params: &Value) -> Result<ActionOutput> {
        match action {
            "reminder" => {
                let p: ReminderParams = parse(action, params)?;
                Ok(ActionOutput { result: json!({ "text": p.text }), event: Some(p.text) })
            }
            "http" => self.http(parse(action, params)?).await,
            "mcp_tool" => self.mcp_tool(parse(action, params)?).await,
            other => bail!("неизвестное действие «{other}»"),
        }
    }

    async fn http(&self, p: HttpParams) -> Result<ActionOutput> {
        let timeout = p.timeout_sec.unwrap_or(DEFAULT_HTTP_TIMEOUT_SEC).clamp(1, MAX_HTTP_TIMEOUT_SEC);
        let mut request = self.http.request(method(&p)?, &p.url).timeout(Duration::from_secs(timeout));
        for (key, value) in &p.headers {
            request = request.header(key, expand_env(value)?);
        }
        if let Some(body) = &p.body {
            request = request.json(body);
        }
        let started = Instant::now();
        let response = request.send().await.map_err(|e| anyhow!("запрос не выполнен: {e}"))?;
        let status = response.status();
        let text = response.text().await.map_err(|e| anyhow!("не удалось прочитать ответ: {e}"))?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        // Статус не 2xx — не сбой запуска, а его результат: у health-check
        // «DOWN»/503 и есть то, что нужно видеть в сводке.
        let result = json!({
            "status": status.as_u16(),
            "ok": status.is_success(),
            "latency_ms": (latency_ms * 10.0).round() / 10.0,
            "body": json_or_text(&text),
        });
        Ok(ActionOutput { result, event: None })
    }

    async fn mcp_tool(&self, p: McpToolParams) -> Result<ActionOutput> {
        let (url, headers) = self.mcp_endpoint(&p)?;
        let call = async {
            let mut config = StreamableHttpClientTransportConfig::with_uri(url.clone());
            let mut custom = std::collections::HashMap::new();
            for (key, value) in &headers {
                if key.eq_ignore_ascii_case("authorization") {
                    // rmcp сам добавляет префикс "Bearer " к auth_header.
                    config = config.auth_header(value.strip_prefix("Bearer ").unwrap_or(value).to_string());
                    continue;
                }
                custom.insert(
                    http::HeaderName::try_from(key.as_str()).with_context(|| format!("некорректный заголовок «{key}»"))?,
                    http::HeaderValue::try_from(value.as_str())
                        .with_context(|| format!("некорректное значение заголовка «{key}»"))?,
                );
            }
            config.custom_headers = custom;
            let mut info = ClientConfig::default();
            info.client_info = Implementation::new("scheduler-mcp", env!("CARGO_PKG_VERSION"));
            let client = info
                .serve(StreamableHttpClientTransport::from_config(config))
                .await
                .map_err(|e| anyhow!("не удалось подключиться к {url}: {e}"))?;
            let params = CallToolRequestParams::new(p.tool.clone()).with_arguments(p.arguments.clone());
            let result = client.call_tool(params).await;
            let _ = client.cancel().await;
            result.map_err(|e| anyhow!("ошибка вызова {}: {e}", p.tool))
        };
        let result = tokio::time::timeout(MCP_TIMEOUT, call)
            .await
            .map_err(|_| anyhow!("{} не ответил за {} с", p.tool, MCP_TIMEOUT.as_secs()))??;

        let text = result
            .content
            .iter()
            .map(|block| match block.as_text() {
                Some(t) => t.text.clone(),
                None => serde_json::to_string(block).unwrap_or_default(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if result.is_error.unwrap_or(false) {
            bail!("{} вернул ошибку: {}", p.tool, truncate(&text));
        }
        let value = match result.structured_content {
            Some(structured) => structured,
            None => json_or_text(&text),
        };
        Ok(ActionOutput { result: value, event: None })
    }

    /// Адрес и заголовки сервера для `mcp_tool`: по имени из конфигурации или напрямую.
    fn mcp_endpoint(&self, p: &McpToolParams) -> Result<(String, BTreeMap<String, String>)> {
        if p.tool.trim().is_empty() {
            bail!("не задан tool");
        }
        let (url, headers) = match (&p.server, &p.url) {
            (Some(_), Some(_)) => bail!("задайте либо server, либо url"),
            (None, Some(url)) => (url.clone(), p.headers.clone()),
            (Some(name), None) => {
                let (url, mut headers) = self.server_from_config(name)?;
                headers.extend(p.headers.clone());
                (url, headers)
            }
            (None, None) => bail!("задайте server (имя из {}) или url", self.mcp_config.display()),
        };
        let url = expand_env(&url)?;
        let target = reqwest::Url::parse(&url).with_context(|| format!("некорректный url «{url}»"))?;
        let own = format!("{}:{}", target.host_str().unwrap_or_default(), target.port_or_known_default().unwrap_or(0));
        if own == self.own_addr || own.replace("localhost", "127.0.0.1") == self.own_addr {
            bail!("планировщик не вызывает по расписанию собственные инструменты");
        }
        let headers = headers.into_iter().map(|(k, v)| Ok((k, expand_env(&v)?))).collect::<Result<_>>()?;
        Ok((url, headers))
    }

    fn server_from_config(&self, name: &str) -> Result<(String, BTreeMap<String, String>)> {
        #[derive(Deserialize)]
        struct Config {
            #[serde(rename = "mcpServers", default)]
            servers: BTreeMap<String, Entry>,
        }
        #[derive(Deserialize)]
        struct Entry {
            url: Option<String>,
            #[serde(default)]
            headers: BTreeMap<String, String>,
        }
        let path = &self.mcp_config;
        let text = std::fs::read_to_string(path).with_context(|| format!("не удалось прочитать {}", path.display()))?;
        let config: Config = serde_json::from_str(&text).with_context(|| format!("некорректный {}", path.display()))?;
        let names = || config.servers.keys().cloned().collect::<Vec<_>>().join(", ");
        let entry = config
            .servers
            .get(name)
            .with_context(|| format!("сервера «{name}» нет в {}; есть: {}", path.display(), names()))?;
        let url = entry.url.clone().with_context(|| {
            format!("сервер «{name}» запускается командой (stdio), а по расписанию вызываются только серверы с url")
        })?;
        Ok((url, entry.headers.clone()))
    }
}

fn parse<T: for<'de> Deserialize<'de>>(action: &str, params: &Value) -> Result<T> {
    serde_json::from_value(params.clone()).with_context(|| format!("некорректные params для «{action}»"))
}

fn method(p: &HttpParams) -> Result<reqwest::Method> {
    match p.method.as_deref().unwrap_or("GET").to_ascii_uppercase().as_str() {
        "GET" => Ok(reqwest::Method::GET),
        "HEAD" => Ok(reqwest::Method::HEAD),
        "POST" => Ok(reqwest::Method::POST),
        other => bail!("метод {other} не поддерживается: только GET, HEAD, POST"),
    }
}

fn json_or_text(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(truncate(text)))
}

fn truncate(text: &str) -> String {
    if text.chars().count() <= MAX_TEXT_CHARS {
        return text.to_string();
    }
    text.chars().take(MAX_TEXT_CHARS).collect::<String>() + "…"
}

/// Подстановка `${VAR}` из окружения — как в конфигурации MCP-клиента: токены
/// живут в окружении, а не в параметрах задания.
fn expand_env(value: &str) -> Result<String> {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after.find('}').ok_or_else(|| anyhow!("незакрытая подстановка ${{ в «{value}»"))?;
        let var = &after[..end];
        out.push_str(&std::env::var(var).map_err(|_| anyhow!("не задана переменная окружения {var}"))?);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executor(config: &str) -> Executor {
        let path = std::env::temp_dir().join(format!("scheduler-mcp-test-{}-{}.json", std::process::id(), config.len()));
        std::fs::write(&path, config).unwrap();
        Executor::new(path, "127.0.0.1:8092".into())
    }

    #[test]
    fn validates_params_per_action() {
        let ex = executor(r#"{"mcpServers":{"prof":{"url":"http://127.0.0.1:8091/mcp"},"npx":{"command":"npx"}}}"#);
        assert!(ex.validate("reminder", &json!({"text": "позвонить"})).is_ok());
        assert!(ex.validate("reminder", &json!({"text": " "})).is_err());
        assert!(ex.validate("reminder", &json!({"txt": "x"})).is_err(), "лишние поля — ошибка");
        assert!(ex.validate("http", &json!({"url": "https://example.com"})).is_ok());
        assert!(ex.validate("http", &json!({"url": "ftp://example.com"})).is_err());
        assert!(ex.validate("http", &json!({"url": "https://example.com", "method": "DELETE"})).is_err());
        assert!(ex.validate("mcp_tool", &json!({"server": "prof", "tool": "java_list_processes"})).is_ok());
        assert!(ex.validate("mcp_tool", &json!({"server": "npx", "tool": "x"})).is_err(), "stdio-сервер");
        assert!(ex.validate("mcp_tool", &json!({"server": "nope", "tool": "x"})).is_err());
        assert!(ex.validate("mcp_tool", &json!({"url": "http://localhost:8092/mcp", "tool": "list_jobs"})).is_err());
        assert!(ex.validate("shell", &json!({})).is_err());
    }

    #[tokio::test]
    async fn reminder_produces_an_event() {
        let ex = executor("{}");
        let out = ex.execute("reminder", &json!({"text": "проверить отчёт"})).await.unwrap();
        assert_eq!(out.event.as_deref(), Some("проверить отчёт"));
        assert_eq!(out.result, json!({"text": "проверить отчёт"}));
    }
}
