//! Клиент эмбеддингов: `POST {base}/embeddings` OpenAI-совместимого API
//! (OpenAI, LM Studio, Ollama, …).
//!
//! Модель задаётся только явно — `LLM_EMBEDDING_MODEL`: векторы разных
//! моделей несравнимы, и молча подставить модель по умолчанию нельзя. Адрес
//! и ключ — `LLM_EMBEDDING_API_URL`/`LLM_EMBEDDING_API_KEY`, если не заданы —
//! те же, что для чата (`LLM_API_URL`/`LLM_API_KEY`).

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

const DEFAULT_BATCH: usize = 32;

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    batch: usize,
    doc_prefix: String,
    query_prefix: String,
}

/// Векторы (нормированные к длине 1, по порядку входных текстов) и расход
/// токенов, если сервер его сообщил.
pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
    usage: Option<UsageWire>,
}

#[derive(Deserialize)]
struct Item {
    embedding: Vec<f32>,
    index: Option<usize>,
}

#[derive(Deserialize)]
struct UsageWire {
    prompt_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

impl Embedder {
    pub fn from_env() -> Result<Self> {
        let model = env("LLM_EMBEDDING_MODEL").context(
            "не задана переменная окружения LLM_EMBEDDING_MODEL — модель эмбеддингов (например text-embedding-bge-m3)",
        )?;
        let base_url = env("LLM_EMBEDDING_API_URL").or_else(|| env("LLM_API_URL")).context(
            "не задан адрес API эмбеддингов: LLM_EMBEDDING_API_URL (или LLM_API_URL), например http://localhost:1234/v1",
        )?;
        let api_key = env("LLM_EMBEDDING_API_KEY").or_else(|| env("LLM_API_KEY"));
        let batch = env("LLM_EMBEDDING_BATCH").and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(DEFAULT_BATCH);
        Ok(Self::new(&base_url, api_key, &model, batch)
            .with_prefixes(env("LLM_EMBEDDING_DOC_PREFIX").unwrap_or_default(), env("LLM_EMBEDDING_QUERY_PREFIX").unwrap_or_default()))
    }

    pub fn new(base_url: &str, api_key: Option<String>, model: &str, batch: usize) -> Self {
        Self {
            // Первый запрос к локальному серверу может ждать загрузки модели.
            http: reqwest::Client::builder().timeout(Duration::from_secs(300)).build().unwrap_or_default(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
            batch: batch.max(1),
            doc_prefix: String::new(),
            query_prefix: String::new(),
        }
    }

    /// Префиксы для моделей, которые их ждут (nomic: `search_document: ` /
    /// `search_query: `; e5: `passage: ` / `query: `). bge-m3 без префиксов.
    pub fn with_prefixes(mut self, doc: String, query: String) -> Self {
        self.doc_prefix = doc;
        self.query_prefix = query;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn doc_prefix(&self) -> &str {
        &self.doc_prefix
    }

    /// Эмбеддинги фрагментов документов (с префиксом документа), одним
    /// запросом — разбивать на пачки по [`Embedder::batch`] должен вызывающий.
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Embedded> {
        let input: Vec<String> = texts.iter().map(|t| format!("{}{t}", self.doc_prefix)).collect();
        self.request(&input).await
    }

    /// Эмбеддинг поискового запроса (с префиксом запроса).
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embedded = self.request(&[format!("{}{query}", self.query_prefix)]).await?;
        embedded.vectors.into_iter().next().context("сервер эмбеддингов не вернул вектор")
    }

    async fn request(&self, input: &[String]) -> Result<Embedded> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({ "model": self.model, "input": input });
        let mut request = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.with_context(|| format!("нет связи с сервером эмбеддингов {url}"))?;
        let status = response.status();
        let raw = response.text().await.context("не удалось прочитать ответ сервера эмбеддингов")?;
        if !status.is_success() {
            bail!("сервер эмбеддингов ({}) вернул ошибку {status}: {}", self.model, raw.trim());
        }
        let parsed: Response = serde_json::from_str(&raw)
            .with_context(|| format!("не удалось разобрать ответ сервера эмбеддингов: {}", preview(&raw)))?;
        if parsed.data.len() != input.len() {
            bail!("сервер эмбеддингов вернул {} векторов на {} текстов", parsed.data.len(), input.len());
        }
        let mut items = parsed.data;
        // Порядок — по полю index, если оно есть.
        if items.iter().all(|i| i.index.is_some()) {
            items.sort_by_key(|i| i.index);
        }
        let dim = items[0].embedding.len();
        if dim == 0 || items.iter().any(|i| i.embedding.len() != dim) {
            bail!("сервер эмбеддингов вернул векторы разной длины");
        }
        let vectors = items.into_iter().map(|i| normalize(i.embedding)).collect();
        // LM Studio присылает usage с нулями — это «не сообщил», а не ноль токенов.
        let tokens = parsed.usage.and_then(|u| u.prompt_tokens.or(u.total_tokens)).filter(|&t| t > 0);
        Ok(Embedded { vectors, tokens })
    }
}

fn preview(raw: &str) -> String {
    match raw.char_indices().nth(300) {
        Some((i, _)) => format!("{}…", &raw[..i]),
        None => raw.to_string(),
    }
}

/// Вектор длины 1 — косинусная близость становится скалярным произведением.
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Фиктивный сервер эмбеддингов: вектор текста — мешок хэшированных
    //! триграмм символов (похожие тексты — близкие векторы), так что поиск
    //! в тестах осмыслен без настоящей модели.

    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    pub const DIM: usize = 256;

    pub fn fake_vector(text: &str) -> Vec<f32> {
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        let mut v = vec![0f32; DIM];
        for w in chars.windows(3) {
            let mut h: u64 = 1469598103934665603;
            for c in w {
                h = (h ^ *c as u64).wrapping_mul(1099511628211);
            }
            v[(h % DIM as u64) as usize] += 1.0;
        }
        v
    }

    /// Запускает сервер; возвращает base_url и журнал запросов (число
    /// текстов в каждом).
    pub async fn serve() -> (String, Arc<Mutex<Vec<usize>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { return };
                let seen = seen.clone();
                tokio::spawn(async move {
                    let mut raw = Vec::new();
                    let mut buf = [0u8; 8192];
                    let body_start = loop {
                        let n = socket.read(&mut buf).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        raw.extend_from_slice(&buf[..n]);
                        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                            break pos + 4;
                        }
                    };
                    let headers = String::from_utf8_lossy(&raw[..body_start]).to_lowercase();
                    let length: usize = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    while raw.len() < body_start + length {
                        let n = socket.read(&mut buf).await.unwrap();
                        raw.extend_from_slice(&buf[..n]);
                    }
                    let request: serde_json::Value = serde_json::from_slice(&raw[body_start..body_start + length]).unwrap();
                    let input: Vec<String> =
                        request["input"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
                    seen.lock().unwrap().push(input.len());
                    // В обратном порядке — клиент должен разложить по index.
                    let data: Vec<serde_json::Value> = input
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(i, t)| serde_json::json!({ "index": i, "embedding": fake_vector(t) }))
                        .collect();
                    let tokens: usize = input.iter().map(|t| t.chars().count() / 4 + 1).sum();
                    let body = serde_json::json!({ "data": data, "usage": { "prompt_tokens": tokens, "total_tokens": tokens } })
                        .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (base_url, log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embeds_in_input_order_and_normalizes() {
        let (url, log) = test_support::serve().await;
        let embedder = Embedder::new(&url, None, "test-embed", 8);
        let texts = vec!["первый текст".to_string(), "второй текст".to_string(), "совсем другое".to_string()];
        let embedded = embedder.embed_documents(&texts).await.unwrap();
        assert_eq!(embedded.vectors.len(), 3);
        for (v, t) in embedded.vectors.iter().zip(&texts) {
            assert!((dot(v, v) - 1.0).abs() < 1e-4);
            assert!((dot(v, &normalize(test_support::fake_vector(t))) - 1.0).abs() < 1e-4, "порядок по index");
        }
        assert!(embedded.tokens.is_some());
        assert_eq!(*log.lock().unwrap(), [3]);
    }

    #[tokio::test]
    async fn server_error_is_reported() {
        let embedder = Embedder::new("http://127.0.0.1:9", None, "m", 8);
        let err = embedder.embed_query("x").await.unwrap_err();
        assert!(format!("{err:#}").contains("нет связи с сервером эмбеддингов"), "{err:#}");
    }
}
