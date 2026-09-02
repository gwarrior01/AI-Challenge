//! Минимальный клиент для общения с LLM через OpenAI-совместимый HTTP API.
//! Адрес и ключ API берутся из переменных окружения LLM_API_URL / LLM_API_KEY.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

/// Количество токенов, потраченных на запрос (приходит от LLM API).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

/// Результат обращения к LLM: текст ответа, метрики токенов и сырые JSON запроса/ответа
/// (для отладочного просмотра того, что реально было отправлено модели и получено от неё).
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    pub content: String,
    pub usage: Option<Usage>,
    pub request_json: String,
    pub response_json: String,
}

#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl LlmClient {
    /// Читает LLM_API_URL, LLM_API_KEY и (опционально) LLM_MODEL из переменных окружения.
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("LLM_API_URL").context(
            "не задана переменная окружения LLM_API_URL (например https://api.openai.com/v1)",
        )?;
        let api_key = std::env::var("LLM_API_KEY")
            .context("не задана переменная окружения LLM_API_KEY")?;
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Ok(Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
        })
    }

    /// Название используемой модели.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Отправляет список сообщений в LLM и возвращает текст ответа вместе с расходом токенов.
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatCompletion> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest { model: &self.model, messages };
        let request_json =
            serde_json::to_string_pretty(&body).context("не удалось сериализовать запрос")?;

        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("ошибка запроса к LLM API")?;

        let status = response.status();
        let raw = response
            .text()
            .await
            .context("не удалось прочитать тело ответа")?;

        if !status.is_success() {
            bail!("LLM API вернул ошибку {status}: {raw}");
        }

        let parsed: ChatResponse = serde_json::from_str(&raw)
            .with_context(|| format!("не удалось разобрать ответ LLM: {raw}"))?;

        let usage = parsed.usage;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("LLM не вернул ни одного варианта ответа")?;

        let response_json = pretty_print_json(&raw);

        Ok(ChatCompletion { content, usage, request_json, response_json })
    }

    /// Упрощённый вызов: один текстовый запрос -> один ответ (текст + токены).
    pub async fn ask(&self, prompt: &str) -> Result<ChatCompletion> {
        let messages = [ChatMessage::user(prompt)];
        self.chat(&messages).await
    }
}

/// Пытается красиво отформатировать JSON; если строка не является валидным JSON, возвращает как есть.
fn pretty_print_json(raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw)
        .and_then(|value| serde_json::to_string_pretty(&value))
        .unwrap_or_else(|_| raw.to_string())
}
