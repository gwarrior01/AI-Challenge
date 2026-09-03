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
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    stop: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a serde_json::Value>,
    /// Включает/выключает режим рассуждений ("thinking") у гибридных моделей
    /// (Qwen3, DeepSeek, GLM и т.п.), которые поддерживают это поле в OpenAI-совместимом API.
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
}

/// Необязательные настройки генерации ответа. Пустые/`None` значения означают отсутствие
/// ограничения — поведение по умолчанию соответствует обычному общению без надстроек.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// Максимальная длина ответа в токенах.
    pub max_tokens: Option<u32>,
    /// Стоп-последовательности, после которых генерация обрывается.
    pub stop: Vec<String>,
    /// Температура сэмплирования (обычно 0.0–2.0): выше — разнообразнее и менее предсказуемо.
    pub temperature: Option<f32>,
    /// Nucleus sampling (обычно 0.0–1.0): доля вероятностной массы токенов-кандидатов.
    pub top_p: Option<f32>,
    /// Значение поля `response_format` для OpenAI-совместимого API (например,
    /// `{"type":"json_schema","json_schema":{...}}`), заставляющее модель отвечать
    /// в заданном JSON-формате.
    pub response_format: Option<serde_json::Value>,
    /// Явно включить (`Some(true)`) или выключить (`Some(false)`) режим рассуждений.
    /// `None` — не переопределять, поведение модели по умолчанию.
    pub reasoning: Option<bool>,
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
        self.chat_with_options(messages, &ChatOptions::default()).await
    }

    /// То же самое, что [`LlmClient::chat`], но с настройками генерации: максимальная длина
    /// ответа, стоп-последовательности, температура, top_p, формат ответа (`response_format`)
    /// и включение/выключение режима рассуждений (`enable_thinking`).
    pub async fn chat_with_options(
        &self,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<ChatCompletion> {
        self.chat_with_model(&self.model, messages, options).await
    }

    /// То же самое, что [`LlmClient::chat_with_options`], но с явным указанием модели —
    /// не той, что задана клиенту по умолчанию (`LLM_MODEL`). Полезно, когда для отдельных
    /// запросов (например, для анализа) нужна другая модель того же API.
    pub async fn chat_with_model(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<ChatCompletion> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest {
            model,
            messages,
            max_tokens: options.max_tokens,
            stop: &options.stop,
            temperature: options.temperature,
            top_p: options.top_p,
            response_format: options.response_format.as_ref(),
            enable_thinking: options.reasoning,
        };
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
