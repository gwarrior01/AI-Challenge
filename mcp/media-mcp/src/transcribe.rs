//! Транскрипция: каждый MP3-чанк отправляется в сервис распознавания
//! (`POST <MEDIA_TRANSCRIBE_URL>/v1/transcribe`, multipart `file` + `language`,
//! ответ — `text/plain`) и сохраняется рядом как `chunk-NNN.txt`.
//!
//! Правила скилла: транскрибируются **все** чанки, готовые TXT
//! перезаписываются; одновременно — не больше `MEDIA_TRANSCRIBE_PARALLEL`
//! запросов (по умолчанию и максимум 2); на запрос — 30 с на соединение,
//! 1800 с на ответ, ещё 2 попытки через 5 с при любой ошибке.
//! `output-transcript.md` собирается, только если в этом запуске все чанки
//! расшифрованы и ни один TXT не пустой, — склейкой байт в байт по порядку.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;

use crate::jobs::{Failure, JobCtx};
use crate::media::env_or;
use crate::recording::TRANSCRIPT;

pub const DEFAULT_URL: &str = "http://192.168.0.44:8000";
const MAX_PARALLEL: usize = 2;

#[derive(Debug, Clone)]
pub struct Config {
    pub url: String,
    pub language: String,
    pub parallel: usize,
    pub connect_timeout: Duration,
    pub max_time: Duration,
    pub retries: usize,
    pub retry_delay: Duration,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            url: env_or("MEDIA_TRANSCRIBE_URL", DEFAULT_URL).trim_end_matches('/').to_string(),
            language: env_or("MEDIA_LANGUAGE", "ru"),
            parallel: env_or("MEDIA_TRANSCRIBE_PARALLEL", "2").parse().unwrap_or(MAX_PARALLEL).clamp(1, MAX_PARALLEL),
            connect_timeout: Duration::from_secs(30),
            max_time: Duration::from_secs(env_or("MEDIA_TRANSCRIBE_MAX_TIME", "1800").parse().unwrap_or(1800).max(1)),
            retries: 2,
            retry_delay: Duration::from_secs(5),
        }
    }

    pub fn endpoint(&self) -> String {
        format!("{}/v1/transcribe", self.url)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkResult {
    pub chunk: String,
    pub txt: String,
    pub ok: bool,
    /// Символов в ответе.
    pub chars: usize,
    pub attempts: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Транскрибировать все чанки и, если все удались, собрать транскрипт.
pub async fn run(cfg: Config, dir: PathBuf, chunks: Vec<PathBuf>, ctx: JobCtx) -> Result<serde_json::Value, Failure> {
    let client = reqwest::Client::builder().connect_timeout(cfg.connect_timeout).timeout(cfg.max_time).build().map_err(anyhow::Error::from)?;
    let total = chunks.len();
    let done = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new(Semaphore::new(cfg.parallel));
    ctx.progress(format!("чанки: 0 из {total} готово"));

    let mut set = tokio::task::JoinSet::new();
    for (i, chunk) in chunks.iter().cloned().enumerate() {
        let (cfg, client, gate, ctx) = (cfg.clone(), client.clone(), gate.clone(), ctx.clone());
        let (done, failed) = (done.clone(), failed.clone());
        set.spawn(async move {
            let _permit = gate.acquire_owned().await.expect("семафор не закрывается");
            let result = transcribe_chunk(&cfg, &client, &chunk).await;
            let d = done.fetch_add(1, Ordering::SeqCst) + 1;
            if !result.ok {
                failed.fetch_add(1, Ordering::SeqCst);
            }
            let f = failed.load(Ordering::SeqCst);
            let errors = if f > 0 { format!(", ошибок {f}") } else { String::new() };
            ctx.progress(format!("чанки: {d} из {total} готово{errors}"));
            (i, result)
        });
    }
    let mut results: Vec<Option<ChunkResult>> = vec![None; total];
    while let Some(joined) = set.join_next().await {
        let (i, r) = joined.map_err(|e| anyhow::anyhow!("транскрипция чанка аварийно завершилась: {e}"))?;
        results[i] = Some(r);
    }
    let results: Vec<ChunkResult> = results.into_iter().flatten().collect();

    let bad: Vec<String> = results
        .iter()
        .filter(|r| !r.ok || r.chars == 0)
        .map(|r| match &r.error {
            Some(e) => format!("{}: {e}", r.chunk),
            None => format!("{}: пустой ответ", r.chunk),
        })
        .collect();
    let details = json!({ "endpoint": cfg.endpoint(), "language": cfg.language, "chunks": results });
    if !bad.is_empty() {
        return Err(Failure::with(
            format!(
                "транскрипция неполная: {} из {total} чанков без текста — {TRANSCRIPT} не создан; MP3-чанки и удачные TXT \
                 сохранены, при следующем запуске все чанки транскрибируются заново. {}",
                bad.len(),
                bad.join("; ")
            ),
            details,
        ));
    }

    let txts: Vec<PathBuf> = chunks.iter().map(|c| c.with_extension("txt")).collect();
    let bytes = build_transcript(&dir, &txts).map_err(|e| Failure::with(format!("{e:#}"), details.clone()))?;
    let mut details = details;
    details["created"] = json!(TRANSCRIPT);
    details["transcript_bytes"] = json!(bytes);
    details["overwritten_txt"] = json!("все TXT чанков записаны заново");
    Ok(details)
}

async fn transcribe_chunk(cfg: &Config, client: &reqwest::Client, chunk: &Path) -> ChunkResult {
    let name = chunk.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let txt = chunk.with_extension("txt");
    let mut attempts = 0;
    let mut last_error = String::new();
    while attempts <= cfg.retries {
        if attempts > 0 {
            tokio::time::sleep(cfg.retry_delay).await;
        }
        attempts += 1;
        match request(cfg, client, chunk, &name).await {
            Ok(text) => {
                // Сначала во временный файл: оборванная запись не оставит полупустой TXT.
                let written = write_atomic(&txt, text.as_bytes());
                return ChunkResult {
                    chunk: name,
                    txt: file_name(&txt),
                    ok: written.is_ok(),
                    chars: text.chars().count(),
                    attempts,
                    error: written.err().map(|e| format!("{e:#}")),
                };
            }
            Err(e) => last_error = format!("{e:#}"),
        }
    }
    ChunkResult { chunk: name, txt: file_name(&txt), ok: false, chars: 0, attempts, error: Some(last_error) }
}

async fn request(cfg: &Config, client: &reqwest::Client, chunk: &Path, name: &str) -> Result<String> {
    let bytes = tokio::fs::read(chunk).await.with_context(|| format!("не удалось прочитать {name}"))?;
    let part = reqwest::multipart::Part::bytes(bytes).file_name(name.to_string()).mime_str("audio/mpeg")?;
    let form = reqwest::multipart::Form::new().part("file", part).text("language", cfg.language.clone());
    let resp = client
        .post(cfg.endpoint())
        .header(reqwest::header::ACCEPT, "text/plain")
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("запрос к {}", cfg.endpoint()))?;
    let status = resp.status();
    let body = resp.text().await.context("не удалось прочитать ответ")?;
    if !status.is_success() {
        let body: String = body.chars().take(300).collect();
        bail!("HTTP {status}: {}", body.trim());
    }
    Ok(body)
}

/// Склеить TXT по порядку в `output-transcript.md` без каких-либо добавок.
pub fn build_transcript(dir: &Path, txts: &[PathBuf]) -> Result<usize> {
    let mut all = Vec::new();
    for t in txts {
        all.extend(std::fs::read(t).with_context(|| format!("не удалось прочитать {}", t.display()))?);
    }
    if all.is_empty() {
        bail!("транскрипт получился пустым");
    }
    write_atomic(&dir.join(TRANSCRIPT), &all)?;
    Ok(all.len())
}

/// Записать через временный файл и переименование.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_file_name(format!(".{}.partial", file_name(path)));
    std::fs::write(&tmp, bytes).with_context(|| format!("не удалось записать {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("не удалось записать {}", path.display()))
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::jobs::{JobState, Jobs};
    use axum::extract::Multipart;
    use std::sync::atomic::AtomicUsize;

    /// Поддельный сервис распознавания: текст по имени файла; `chunk-001` всегда
    /// падает, если `fail` = true. Считает одновременные запросы.
    pub async fn fake_transcriber(fail: bool) -> (String, Arc<AtomicUsize>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let now = Arc::new(AtomicUsize::new(0));
        let (p, n) = (peak.clone(), now.clone());
        let app = axum::Router::new().route(
            "/v1/transcribe",
            axum::routing::post(move |mut form: Multipart| {
                let (p, n) = (p.clone(), n.clone());
                async move {
                    let cur = n.fetch_add(1, Ordering::SeqCst) + 1;
                    p.fetch_max(cur, Ordering::SeqCst);
                    let mut file = String::new();
                    let mut lang = String::new();
                    while let Ok(Some(field)) = form.next_field().await {
                        match field.name() {
                            Some("file") => file = field.file_name().unwrap_or_default().to_string(),
                            Some("language") => lang = field.text().await.unwrap_or_default(),
                            _ => {}
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    n.fetch_sub(1, Ordering::SeqCst);
                    if fail && file == "chunk-001.mp3" {
                        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "модель упала".to_string());
                    }
                    (axum::http::StatusCode::OK, format!("[{lang}] текст {file}\n"))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), peak)
    }

    pub fn config(url: &str) -> Config {
        Config {
            url: url.into(),
            language: "ru".into(),
            parallel: 2,
            connect_timeout: Duration::from_secs(5),
            max_time: Duration::from_secs(10),
            retries: 2,
            retry_delay: Duration::from_millis(10),
        }
    }

    fn chunk_dir(name: &str, n: usize) -> (PathBuf, Vec<PathBuf>) {
        let dir = std::env::temp_dir().join(format!("media-mcp-tr-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("transcript-chunks")).unwrap();
        let chunks: Vec<PathBuf> = (0..n)
            .map(|i| {
                let p = dir.join(format!("transcript-chunks/chunk-{i:03}.mp3"));
                std::fs::write(&p, b"mp3").unwrap();
                p
            })
            .collect();
        (dir, chunks)
    }

    async fn run_job(cfg: Config, dir: &Path, chunks: Vec<PathBuf>) -> crate::jobs::JobView {
        let jobs = Jobs::new();
        let d = dir.to_path_buf();
        jobs.start("transcribe", dir, "x", move |ctx| run(cfg, d, chunks, ctx)).unwrap();
        jobs.wait("j1", Duration::from_secs(30)).await.unwrap()
    }

    #[tokio::test]
    async fn all_chunks_are_transcribed_and_joined() {
        let (url, peak) = fake_transcriber(false).await;
        let (dir, chunks) = chunk_dir("ok", 5);
        // Старый TXT от прошлого запуска — перезаписывается.
        std::fs::write(dir.join("transcript-chunks/chunk-002.txt"), "старое").unwrap();
        let v = run_job(config(&url), &dir, chunks).await;
        assert_eq!(v.state, JobState::Succeeded, "{v:?}");
        let text = std::fs::read_to_string(dir.join(TRANSCRIPT)).unwrap();
        assert_eq!(
            text,
            "[ru] текст chunk-000.mp3\n[ru] текст chunk-001.mp3\n[ru] текст chunk-002.mp3\n[ru] текст chunk-003.mp3\n[ru] текст chunk-004.mp3\n"
        );
        assert!(peak.load(Ordering::SeqCst) <= 2, "не больше двух запросов одновременно");
        assert_eq!(v.progress, "готово");
    }

    #[tokio::test]
    async fn failed_chunk_blocks_the_transcript() {
        let (url, _) = fake_transcriber(true).await;
        let (dir, chunks) = chunk_dir("fail", 3);
        std::fs::write(dir.join("transcript-chunks/chunk-001.txt"), "старое с прошлого раза").unwrap();
        let v = run_job(config(&url), &dir, chunks).await;
        assert_eq!(v.state, JobState::Failed);
        assert!(v.error.as_deref().unwrap().contains("chunk-001.mp3: HTTP 500"), "{v:?}");
        assert!(!dir.join(TRANSCRIPT).exists(), "старый TXT не делает транскрипцию полной");
        let r = &v.result.unwrap()["chunks"];
        assert_eq!(r[1]["attempts"], 3, "две повторные попытки");
        assert_eq!(r[0]["ok"], true);
        assert!(dir.join("transcript-chunks/chunk-000.txt").exists(), "удачные TXT сохранены");
    }

    #[tokio::test]
    async fn unreachable_service_is_reported() {
        let (dir, chunks) = chunk_dir("down", 1);
        let mut cfg = config("http://127.0.0.1:9");
        cfg.retries = 0;
        let v = run_job(cfg, &dir, chunks).await;
        assert_eq!(v.state, JobState::Failed);
        assert!(v.error.unwrap().contains("запрос к http://127.0.0.1:9/v1/transcribe"));
    }
}
