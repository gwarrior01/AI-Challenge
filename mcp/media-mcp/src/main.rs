//! MCP-сервер обработки записей встреч: видео (WebM/MP4) → MP3 → чанки →
//! транскрипт → саммари. Правила скилла `convert-summarize` живут в коде:
//!
//! - **чекпоинты** — `inspect` смотрит каталог от самого позднего артефакта к
//!   раннему и называет следующий шаг; шаги вне очереди отказывают
//!   (`recording.rs`, `steps.rs`);
//! - **ожидание** — долгие шаги идут заданиями: сервер сам держит процесс
//!   `ffmpeg` или запросы и решает по коду выхода; в каталоге одновременно
//!   идёт не больше одного задания (`jobs.rs`);
//! - **транскрипция** — всегда все чанки, не больше двух запросов разом,
//!   транскрипт только из полного набора (`transcribe.rs`);
//! - **саммари** — LM Studio, шаблон из скилла, без обрезки (`summary.rs`).
//!
//! Корень с записями — `MEDIA_ROOT` (по умолчанию рабочая директория).
//! Запуск по умолчанию — stdio; с `--http` — Streamable HTTP на
//! `127.0.0.1:8098/mcp` (`MEDIA_MCP_ADDR`): тогда задания общие для всех
//! клиентов и переживают их переподключение.

mod jobs;
mod media;
mod recording;
mod steps;
mod summary;
mod transcribe;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ServerHandler, ServiceExt,
};
use serde::Deserialize;
use serde_json::{json, Value};

use recording::Root;
use steps::Media;

const DEFAULT_ADDR: &str = "127.0.0.1:8098";
const DEFAULT_WAIT_SEC: u64 = 60;
/// Дольше одного вызова `wait_job` не ждёт: у клиентов MCP свой таймаут на вызов.
const MAX_WAIT_SEC: u64 = 600;
const DEFAULT_LIST_DEPTH: usize = 3;

#[derive(Debug, Deserialize, JsonSchema)]
struct DirParams {
    /// Каталог записи относительно корня (`2026-10-15`); по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InspectParams {
    /// Каталог записи относительно корня; по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
    /// Выбор пользователя, если исходников несколько (`next.step` = `choose_source`): имя WebM или MP4 из `next.choices`.
    #[serde(default)]
    source: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SourceParams {
    /// Каталог записи относительно корня; по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
    /// Исходник из `next.args.source` (нужен, только если их несколько).
    #[serde(default)]
    source: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ExtractParams {
    /// Каталог записи относительно корня; по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
    /// Из чего извлекать: `output.mp4` или исходный MP4 — как в `next.args.source`.
    #[serde(default)]
    source: Option<String>,
    /// WebM, с которым сверять `output.mp4`, — как в `next.args.webm` (нужен, если WebM несколько).
    #[serde(default)]
    webm: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SummarizeParams {
    /// Каталог записи относительно корня; по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
    /// Модель LM Studio из `list_llm_models` (`chat_models`); без неё — `obvious` (единственная пригодная или загруженная).
    #[serde(default)]
    model: Option<String>,
    /// Перезаписать готовый `output-summary.md` — только если пользователь явно попросил.
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProbeParams {
    /// Каталог записи относительно корня; по умолчанию — сам корень.
    #[serde(default)]
    dir: Option<String>,
    /// Файл каталога записи (`meeting.webm`, `output.mp4`, `transcript-chunks/chunk-000.mp3`).
    file: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WaitParams {
    /// Номер задания (`j3`).
    id: String,
    /// Сколько секунд ждать завершения, по умолчанию 60, не больше 600. Вышло время — `state: running`, вызови снова.
    #[serde(default)]
    seconds: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct JobParams {
    /// Номер задания; без него — все задания сервера.
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListParams {
    /// С какого каталога искать (от корня); по умолчанию — корень.
    #[serde(default)]
    dir: Option<String>,
    /// Глубина поиска, по умолчанию 3.
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Clone)]
struct MediaServer {
    media: Arc<Media>,
}

fn out(v: Result<Value, String>) -> Result<String, String> {
    v.map(|v| v.to_string())
}

fn ser<T: serde::Serialize>(v: &T) -> Result<String, String> {
    serde_json::to_string(v).map_err(|e| e.to_string())
}

#[tool_router]
impl MediaServer {
    #[tool(
        name = "list_recordings",
        description = "Каталоги с записями под корнем: где лежат видео, MP3, чанки, транскрипт или саммари, и докуда \
            дошла обработка (`stage`: source, mp4, mp3, chunks, transcript, summary).",
        annotations(read_only_hint = true)
    )]
    async fn list_recordings(&self, Parameters(p): Parameters<ListParams>) -> Result<String, String> {
        let root = &self.media.root;
        let start = root.recording(p.dir.as_deref()).map_err(|e| format!("{e:#}"))?;
        let found = root.recordings(&start, p.depth.unwrap_or(DEFAULT_LIST_DEPTH).min(6));
        out(Ok(json!({ "root": root.path().display().to_string(), "recordings": found })))
    }

    #[tool(
        name = "inspect",
        description = "Состояние каталога записи и СЛЕДУЮЩИЙ ШАГ (`next`): чекпоинты от позднего к раннему — \
            output-transcript.md, transcript-chunks/chunk-*.mp3, output.mp3, output.mp4 (сверка длительности с WebM), \
            исходные MP4 и WebM; идущее задание. `next.step`: tool-шаг (`next.tool` + `next.args`), `choose_source` \
            (спросить пользователя, варианты в `next.choices`), `ask_summary`, `done`, `wait_job`, `nothing`.",
        annotations(read_only_hint = true)
    )]
    async fn inspect(&self, Parameters(p): Parameters<InspectParams>) -> Result<String, String> {
        let (_, i) = self.media.inspect(p.dir.as_deref(), p.source.as_deref()).await.map_err(|e| format!("{e:#}"))?;
        ser(&i)
    }

    #[tool(
        name = "probe",
        description = "Длительность медиафайла каталога записи (ffprobe): секунды как есть и округлённые.",
        annotations(read_only_hint = true)
    )]
    async fn probe(&self, Parameters(p): Parameters<ProbeParams>) -> Result<String, String> {
        out(self.media.probe(p.dir.as_deref(), &p.file).await)
    }

    #[tool(
        name = "convert_to_mp4",
        description = "Задание: WebM → output.mp4 (libx264 crf 20, aac 192k), затем сверка длительности WebM и MP4 в \
            целых секундах. Выполняется, только когда inspect называет этот шаг; негодный прежний output.mp4 \
            переносится в diagnostics/. Возвращает номер задания — дождись его через wait_job.",
        annotations(destructive_hint = false)
    )]
    async fn convert_to_mp4(&self, Parameters(p): Parameters<SourceParams>) -> Result<String, String> {
        out(self.media.convert_to_mp4(p.dir.as_deref(), p.source.as_deref()).await)
    }

    #[tool(
        name = "extract_mp3",
        description = "Задание: звук из output.mp4 (сверенного с WebM) или исходного MP4 → output.mp3. Готовый \
            output.mp3 не перезаписывается. Только когда inspect называет этот шаг. Дождись через wait_job.",
        annotations(destructive_hint = false)
    )]
    async fn extract_mp3(&self, Parameters(p): Parameters<ExtractParams>) -> Result<String, String> {
        out(self.media.extract_mp3(p.dir.as_deref(), p.source.as_deref(), p.webm.as_deref()).await)
    }

    #[tool(
        name = "split_audio",
        description = "Задание: output.mp3 → transcript-chunks/chunk-NNN.mp3 по 300 секунд. Готовые чанки заново не \
            режутся. Только когда inspect называет этот шаг. Дождись через wait_job.",
        annotations(destructive_hint = false)
    )]
    async fn split_audio(&self, Parameters(p): Parameters<DirParams>) -> Result<String, String> {
        out(self.media.split_audio(p.dir.as_deref()).await)
    }

    #[tool(
        name = "transcribe",
        description = "Задание: транскрибирует ВСЕ MP3-чанки (готовые TXT перезаписываются; не больше двух запросов \
            одновременно, с повторами) и, если все удались, собирает output-transcript.md склейкой TXT по порядку. \
            Только когда inspect называет этот шаг. Дождись через wait_job.",
        annotations(destructive_hint = false)
    )]
    async fn transcribe(&self, Parameters(p): Parameters<DirParams>) -> Result<String, String> {
        out(self.media.transcribe(p.dir.as_deref()).await)
    }

    #[tool(
        name = "list_llm_models",
        description = "Модели LM Studio: все (`models`), пригодные для чата (`chat_models`, без эмбеддингов), \
            загруженные в память (`loaded_chat_models`) и `obvious` — модель, которую summarize возьмёт сам без \
            выбора (единственная пригодная или единственная загруженная); `null` — нужно спросить пользователя.",
        annotations(read_only_hint = true)
    )]
    async fn list_llm_models(&self) -> Result<String, String> {
        out(self.media.models().await)
    }

    #[tool(
        name = "summarize",
        description = "Задание: саммари транскрипта через LM Studio по шаблону книжного клуба → output-summary.md \
            (только ответ модели). Только когда пользователь согласился на саммари. Готовое саммари не \
            перезаписывается без force. Дождись через wait_job.",
        annotations(destructive_hint = false)
    )]
    async fn summarize(&self, Parameters(p): Parameters<SummarizeParams>) -> Result<String, String> {
        out(self.media.summarize(p.dir.as_deref(), p.model.as_deref(), p.force).await)
    }

    #[tool(
        name = "write_summary_prompt",
        description = "Записать output-summary-prompt.md — шаблон саммари с полным транскриптом — без вызова LM \
            Studio. Для случая, когда пользователь отказался от саммари.",
        annotations(destructive_hint = false, idempotent_hint = true)
    )]
    async fn write_summary_prompt(&self, Parameters(p): Parameters<DirParams>) -> Result<String, String> {
        out(self.media.write_summary_prompt(p.dir.as_deref()).await)
    }

    #[tool(
        name = "wait_job",
        description = "Ждать завершения задания (не дольше `seconds`). Возвращает задание: `state` running / \
            succeeded / failed, `progress`, `result` или `error`. Пока running — вызывай снова: долгая работа — \
            не сбой, задание не перезапускай.",
        annotations(read_only_hint = true)
    )]
    async fn wait_job(&self, Parameters(p): Parameters<WaitParams>) -> Result<String, String> {
        let secs = p.seconds.unwrap_or(DEFAULT_WAIT_SEC).clamp(1, MAX_WAIT_SEC);
        match self.media.jobs.wait(&p.id, Duration::from_secs(secs)).await {
            Some(job) => ser(&job),
            None => Err(format!("задания «{}» нет", p.id)),
        }
    }

    #[tool(
        name = "job_status",
        description = "Задание по номеру или все задания сервера — без ожидания.",
        annotations(read_only_hint = true)
    )]
    async fn job_status(&self, Parameters(p): Parameters<JobParams>) -> Result<String, String> {
        match p.id {
            Some(id) => self.media.jobs.get(&id).map(|j| ser(&j)).unwrap_or_else(|| Err(format!("задания «{id}» нет"))),
            None => ser(&json!({ "jobs": self.media.jobs.list() })),
        }
    }
}

#[tool_handler]
impl ServerHandler for MediaServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("media-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Обработка записи встречи: видео → output.mp4 → output.mp3 → чанки → output-transcript.md → саммари. \
                 Всегда начинай с inspect(dir): он сам находит последний годный чекпоинт и называет следующий шаг \
                 (next.tool + next.args) — выполняй его. Шаги convert_to_mp4, extract_mp3, split_audio, transcribe и \
                 summarize — задания: ответ сразу, итог — через wait_job(id); долгое выполнение не ошибка, не \
                 перезапускай. После задания снова inspect. choose_source — спроси пользователя, какой файл; \
                 ask_summary — спроси, делать ли саммари: да → summarize (модель из list_llm_models), нет → \
                 write_summary_prompt. Пути — от корня MEDIA_ROOT.",
            )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = match std::env::var("MEDIA_ROOT") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => std::env::current_dir().context("не удалось узнать рабочую директорию")?,
    };
    let media = Arc::new(Media::from_env(Root::open(&dir)?));
    // В режиме stdio stdout — канал протокола: всё служебное только в stderr.
    eprintln!(
        "media-mcp: корень {} · транскрибатор {} · LM Studio {}",
        media.root.path().display(),
        media.transcribe.url,
        media.lm.url
    );

    if !std::env::args().any(|a| a == "--http") {
        let service = MediaServer { media }.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        return Ok(());
    }

    let addr = std::env::var("MEDIA_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let service = StreamableHttpService::new(
        move || Ok(MediaServer { media: media.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener =
        tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("не удалось занять адрес {addr}"))?;
    eprintln!("media-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
