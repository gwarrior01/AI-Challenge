//! Саммари транскрипта: шаблон промпта (`summary-prompt.md`, или свой —
//! `MEDIA_SUMMARY_TEMPLATE`) с полным транскриптом вместо `<TRANSCRIPT>`,
//! модель — LM Studio (`MEDIA_LMSTUDIO_URL`, OpenAI-совместимый API).
//!
//! Транскрипт не обрезается; другой LLM при сбое не подставляется; короткого
//! предела на ответ нет (`MEDIA_SUMMARY_TIMEOUT`, по умолчанию 2 часа).

use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::media::env_or;

pub const DEFAULT_URL: &str = "http://192.168.0.44:1234";
const BUILTIN_TEMPLATE: &str = include_str!("summary-prompt.md");
const PLACEHOLDER: &str = "<TRANSCRIPT>";

#[derive(Debug, Clone)]
pub struct Config {
    pub url: String,
    pub timeout: Duration,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            url: env_or("MEDIA_LMSTUDIO_URL", DEFAULT_URL).trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(env_or("MEDIA_SUMMARY_TIMEOUT", "7200").parse().unwrap_or(7200).max(1)),
        }
    }
}

/// Шаблон: свой файл из `MEDIA_SUMMARY_TEMPLATE` или встроенный.
pub fn template() -> Result<String> {
    match std::env::var("MEDIA_SUMMARY_TEMPLATE").ok().filter(|p| !p.trim().is_empty()) {
        Some(path) => {
            let t = std::fs::read_to_string(path.trim()).with_context(|| format!("шаблон {path} не читается"))?;
            if !t.contains(PLACEHOLDER) {
                bail!("в шаблоне {path} нет {PLACEHOLDER}");
            }
            Ok(t)
        }
        None => Ok(BUILTIN_TEMPLATE.to_string()),
    }
}

/// Промпт с полным транскриптом.
pub fn prompt(transcript_path: &Path) -> Result<String> {
    let transcript = std::fs::read_to_string(transcript_path).with_context(|| format!("{} не читается", transcript_path.display()))?;
    Ok(template()?.replacen(PLACEHOLDER, transcript.trim_end(), 1))
}

/// Модели LM Studio.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Models {
    pub all: Vec<String>,
    /// Пригодные для чата — без эмбеддингов.
    pub chat: Vec<String>,
    /// Пригодные для чата и уже загруженные в память; `None` — LM Studio не сообщила.
    pub loaded: Option<Vec<String>>,
}

impl Models {
    /// Модель без выбора пользователя: единственная пригодная или единственная загруженная.
    pub fn obvious(&self) -> Option<&String> {
        match (self.chat.as_slice(), self.loaded.as_deref()) {
            ([one], _) => Some(one),
            (_, Some([one])) => Some(one),
            _ => None,
        }
    }
}

/// Модели LM Studio: `/v1/models` (OpenAI-совместимый список) и, если есть,
/// `/api/v0/models` — родной API LM Studio, где видно, какие загружены.
pub async fn models(cfg: &Config) -> Result<Models> {
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(30)).timeout(Duration::from_secs(60)).build()?;
    let body = get_json(&client, &format!("{}/v1/models", cfg.url)).await?;
    let all: Vec<String> = body["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_string)).collect();
    let chat: Vec<String> = all.iter().filter(|id| !id.to_lowercase().contains("embed")).cloned().collect();
    let loaded = get_json(&client, &format!("{}/api/v0/models", cfg.url)).await.ok().map(|v| {
        v["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["state"] == "loaded")
            .filter_map(|m| m["id"].as_str())
            .filter(|id| chat.iter().any(|c| c == id))
            .map(str::to_string)
            .collect()
    });
    Ok(Models { all, chat, loaded })
}

async fn get_json(client: &reqwest::Client, url: &str) -> Result<Value> {
    let resp = client.get(url).send().await.with_context(|| format!("LM Studio недоступна ({url})"))?;
    let status = resp.status();
    let body: Value = resp.json().await.with_context(|| format!("{url}: ответ не JSON"))?;
    if !status.is_success() {
        bail!("{url}: HTTP {status} — {body}");
    }
    Ok(body)
}

/// Ответ модели на промпт — только текст ассистента.
pub async fn complete(cfg: &Config, model: &str, prompt: &str) -> Result<String> {
    let url = format!("{}/v1/chat/completions", cfg.url);
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(30)).timeout(cfg.timeout).build()?;
    let req = json!({ "model": model, "messages": [{ "role": "user", "content": prompt }], "stream": false });
    let resp = client.post(&url).json(&req).send().await.with_context(|| format!("LM Studio недоступна ({url})"))?;
    let status = resp.status();
    let text = resp.text().await.context("не удалось прочитать ответ LM Studio")?;
    if !status.is_success() {
        let short: String = text.chars().take(500).collect();
        bail!("LM Studio: HTTP {status} — {short}");
    }
    let body: Value = serde_json::from_str(&text).context("LM Studio: ответ не JSON")?;
    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    let answer = strip_fence(&strip_think(content));
    if answer.is_empty() {
        bail!("LM Studio вернула пустой ответ");
    }
    Ok(answer)
}

/// Рассуждения модели (`<think>…</think>`) — не часть ответа.
pub fn strip_think(text: &str) -> String {
    let t = text.trim();
    match (t.find("<think>"), t.find("</think>")) {
        (Some(0), Some(end)) => t[end + "</think>".len()..].trim().to_string(),
        _ => t.to_string(),
    }
}

/// Ответ целиком в блоке кода (```markdown … ```) — снять обёртку: это не часть саммари.
pub fn strip_fence(text: &str) -> String {
    let t = text.trim();
    let Some(rest) = t.strip_prefix("```") else { return t.to_string() };
    let Some((lang, body)) = rest.split_once('\n') else { return t.to_string() };
    if !matches!(lang.trim(), "" | "markdown" | "md") {
        return t.to_string();
    }
    match body.trim_end().strip_suffix("```") {
        Some(inner) if !inner.contains("```") => inner.trim().to_string(),
        _ => t.to_string(),
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Поддельная LM Studio: модели и ответ, в котором видно начало промпта.
    /// Загружена первая из моделей.
    pub async fn fake_lmstudio(models: &'static [&'static str]) -> String {
        let app = axum::Router::new()
            .route(
                "/v1/models",
                axum::routing::get(move || async move {
                    axum::Json(json!({ "data": models.iter().map(|m| json!({ "id": m })).collect::<Vec<_>>() }))
                }),
            )
            .route(
                "/api/v0/models",
                axum::routing::get(move || async move {
                    let data: Vec<Value> = models
                        .iter()
                        .enumerate()
                        .map(|(i, m)| json!({ "id": m, "state": if i == 0 { "loaded" } else { "not-loaded" } }))
                        .collect();
                    axum::Json(json!({ "data": data }))
                }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(|axum::Json(req): axum::Json<Value>| async move {
                    let prompt = req["messages"][0]["content"].as_str().unwrap_or_default().to_string();
                    let has = prompt.contains("Расшифровка:\n\nтекст встречи");
                    axum::Json(json!({ "choices": [{ "message": {
                        "content": format!("<think>думаю</think>\nКраткое резюме\n\nмодель {}, транскрипт подставлен: {has}", req["model"].as_str().unwrap_or("?"))
                    } }] }))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[test]
    fn template_takes_whole_transcript() {
        let dir = std::env::temp_dir().join(format!("media-mcp-prompt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("t.md"), "первая\nвторая\n").unwrap();
        let p = prompt(&dir.join("t.md")).unwrap();
        assert!(p.starts_with("Тебе дана расшифровка"));
        assert!(p.ends_with("Расшифровка:\n\nпервая\nвторая\n"), "{p}");
        assert!(p.contains("Формат — Markdown") && p.contains("## Ключевые мысли"), "{p}");
        assert!(!p.contains(PLACEHOLDER));
    }

    #[test]
    fn obvious_model_needs_no_question() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let m = Models { all: s(&["a", "b"]), chat: s(&["a", "b"]), loaded: Some(s(&["b"])) };
        assert_eq!(m.obvious().map(String::as_str), Some("b"), "одна загружена — она");
        let m = Models { loaded: None, ..m };
        assert_eq!(m.obvious(), None, "две и неизвестно, какая загружена — спросить");
        let m = Models { all: s(&["a"]), chat: s(&["a"]), loaded: Some(vec![]) };
        assert_eq!(m.obvious().map(String::as_str), Some("a"), "единственная пригодная");
        let m = Models { all: s(&["a", "b"]), chat: s(&["a", "b"]), loaded: Some(s(&["a", "b"])) };
        assert_eq!(m.obvious(), None, "две загружены — спросить");
    }

    #[test]
    fn code_fence_around_answer_is_removed() {
        assert_eq!(strip_fence("```markdown\n## Краткое резюме\n\nТекст\n```"), "## Краткое резюме\n\nТекст");
        assert_eq!(strip_fence("```\n- пункт\n```\n"), "- пункт");
        let code = "```rust\nfn main() {}\n```";
        assert_eq!(strip_fence(code), code, "чужой язык — не обёртка");
        assert_eq!(strip_fence("## Итог\n\n```\nx\n```"), "## Итог\n\n```\nx\n```");
    }

    #[test]
    fn reasoning_is_not_saved() {
        assert_eq!(strip_think("<think>a\nb</think>\n\nОтвет"), "Ответ");
        assert_eq!(strip_think("  Ответ <think>x</think>"), "Ответ <think>x</think>");
    }

    #[tokio::test]
    async fn lists_models_and_completes() {
        let url = fake_lmstudio(&["text-embedding-nomic", "qwen3-30b", "gpt-oss-20b"]).await;
        let cfg = Config { url, timeout: Duration::from_secs(10) };
        let m = models(&cfg).await.unwrap();
        assert_eq!(m.all.len(), 3);
        assert_eq!(m.chat, ["qwen3-30b", "gpt-oss-20b"]);
        // Загружен только эмбеддинг — чат-модель не очевидна.
        assert_eq!(m.loaded.as_deref(), Some(&[][..]));
        assert_eq!(m.obvious(), None);
        let answer = complete(&cfg, "qwen3-30b", "Расшифровка:\n\nтекст встречи").await.unwrap();
        assert_eq!(answer, "Краткое резюме\n\nмодель qwen3-30b, транскрипт подставлен: true");
    }
}
