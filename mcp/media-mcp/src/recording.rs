//! Каталог записи и его чекпоинты — правила скилла `convert-summarize` в коде.
//!
//! ```text
//! webm / mp4 → output.mp4 → output.mp3 → transcript-chunks/chunk-NNN.mp3
//!            → transcript-chunks/chunk-NNN.txt → output-transcript.md
//!            → output-summary.md  или  output-summary-prompt.md
//! ```
//!
//! [`inspect`] смотрит чекпоинты от самого позднего к раннему и называет
//! следующий шаг (`next`). Инструменты, которые что-то делают, сверяются с
//! ним и отказываются, если сейчас не их очередь: порядок обеспечивает код,
//! а не память модели. TXT чанков в выборе не участвуют — это не чекпоинт.

use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::jobs::JobView;
use crate::media;

pub const MP4: &str = "output.mp4";
pub const MP3: &str = "output.mp3";
pub const CHUNKS_DIR: &str = "transcript-chunks";
pub const TRANSCRIPT: &str = "output-transcript.md";
pub const SUMMARY: &str = "output-summary.md";
pub const SUMMARY_PROMPT: &str = "output-summary-prompt.md";
/// Сюда переезжают негодные артефакты перед повторной работой: на них можно
/// посмотреть, но в выбор чекпоинта они не попадают.
pub const DIAGNOSTICS_DIR: &str = "diagnostics";

/// Корень, внутри которого лежат записи (`MEDIA_ROOT`). Пути вне него закрыты.
pub struct Root {
    root: PathBuf,
}

impl Root {
    pub fn open(dir: &Path) -> Result<Self> {
        let root = dir.canonicalize().with_context(|| format!("корень {} не найден", dir.display()))?;
        if !root.is_dir() {
            bail!("корень {} — не каталог", root.display());
        }
        Ok(Self { root })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Каталог записи: путь от корня (`2026-10-15`, по умолчанию сам корень)
    /// или абсолютный внутри корня.
    pub fn recording(&self, dir: Option<&str>) -> Result<PathBuf> {
        let dir = dir.map(str::trim).filter(|d| !d.is_empty()).unwrap_or(".");
        let rel = Path::new(dir);
        if rel.components().any(|c| c == Component::ParentDir) {
            bail!("путь «{dir}» с `..` — каталоги записей только внутри корня {}", self.root.display());
        }
        let full = if rel.is_absolute() { rel.to_path_buf() } else { self.root.join(rel) };
        let full = full.canonicalize().with_context(|| format!("каталога «{dir}» нет (корень {})", self.root.display()))?;
        if !full.starts_with(&self.root) {
            bail!("каталог «{dir}» вне корня {}", self.root.display());
        }
        if !full.is_dir() {
            bail!("«{dir}» — не каталог");
        }
        Ok(full)
    }

    /// Путь от корня для ответов: `.` — сам корень.
    pub fn label(&self, path: &Path) -> String {
        match path.strip_prefix(&self.root) {
            Ok(p) if p.as_os_str().is_empty() => ".".into(),
            Ok(p) => p.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }

    /// Каталоги с записями до глубины `depth`: где есть видео, MP3, чанки или
    /// транскрипт, и докуда дошла обработка (по файлам, без `ffprobe`).
    pub fn recordings(&self, start: &Path, depth: usize) -> Vec<RecordingBrief> {
        let mut out = Vec::new();
        self.walk(start, depth, &mut out);
        out
    }

    fn walk(&self, dir: &Path, depth: usize, out: &mut Vec<RecordingBrief>) {
        let brief = brief(dir);
        if let Some(stage) = brief {
            out.push(RecordingBrief { dir: self.label(dir), stage });
        }
        if depth == 0 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut subdirs: Vec<PathBuf> = entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                !name.starts_with('.') && ![CHUNKS_DIR, DIAGNOSTICS_DIR, "target", "node_modules"].contains(&name.as_str())
            })
            .map(|e| e.path())
            .collect();
        subdirs.sort();
        for sub in subdirs {
            self.walk(&sub, depth - 1, out);
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RecordingBrief {
    pub dir: String,
    /// Самое позднее, что есть: `summary`, `transcript`, `chunks`, `mp3`, `mp4`, `source`.
    pub stage: &'static str,
}

fn brief(dir: &Path) -> Option<&'static str> {
    if file_state(&dir.join(SUMMARY)) == FileState::Valid {
        return Some("summary");
    }
    if file_state(&dir.join(TRANSCRIPT)) == FileState::Valid {
        return Some("transcript");
    }
    if scan_chunks(dir).state == SetState::Valid {
        return Some("chunks");
    }
    if file_state(&dir.join(MP3)) == FileState::Valid {
        return Some("mp3");
    }
    if file_state(&dir.join(MP4)) == FileState::Valid {
        return Some("mp4");
    }
    let s = sources(dir);
    (!s.webm.is_empty() || !s.mp4.is_empty()).then_some("source")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FileState {
    Missing,
    /// Обычный непустой файл — годится как чекпоинт.
    Valid,
    Empty,
    NotAFile,
}

pub fn file_state(path: &Path) -> FileState {
    match std::fs::metadata(path) {
        Err(_) => FileState::Missing,
        Ok(m) if !m.is_file() => FileState::NotAFile,
        Ok(m) if m.len() == 0 => FileState::Empty,
        Ok(_) => FileState::Valid,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SetState {
    Missing,
    Valid,
    Invalid,
}

/// MP3-чанки `transcript-chunks/chunk-*.mp3`.
#[derive(Debug, Clone, Serialize)]
pub struct Chunks {
    pub state: SetState,
    /// Имена по порядку номеров.
    pub files: Vec<String>,
    /// Почему набор негоден: пустой чанк, чужое имя.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<String>,
}

impl Chunks {
    pub fn paths(&self, dir: &Path) -> Vec<PathBuf> {
        self.files.iter().map(|f| dir.join(CHUNKS_DIR).join(f)).collect()
    }
}

/// Номер чанка из `chunk-007.mp3`; `None` — имя не по соглашению `chunk-NNN.mp3`.
pub fn chunk_number(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("chunk-")?.strip_suffix(".mp3")?;
    (digits.len() >= 3 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok()).flatten()
}

pub fn scan_chunks(dir: &Path) -> Chunks {
    let mut found: Vec<(u32, String)> = Vec::new();
    let mut problems = Vec::new();
    let mut any = false;
    if let Ok(entries) = std::fs::read_dir(dir.join(CHUNKS_DIR)) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // Шаблон поиска — chunk-*.mp3; TXT не смотрим вовсе.
            if !(name.starts_with("chunk-") && name.ends_with(".mp3")) {
                continue;
            }
            any = true;
            match chunk_number(&name) {
                None => problems.push(format!("{name}: имя не по соглашению chunk-NNN.mp3")),
                Some(n) => {
                    match file_state(&e.path()) {
                        FileState::Valid => {}
                        FileState::Empty => problems.push(format!("{name}: пустой")),
                        _ => problems.push(format!("{name}: не обычный файл")),
                    }
                    found.push((n, name));
                }
            }
        }
    }
    found.sort();
    let state = match (any, problems.is_empty()) {
        (false, _) => SetState::Missing,
        (true, true) => SetState::Valid,
        (true, false) => SetState::Invalid,
    };
    problems.sort();
    Chunks { state, files: found.into_iter().map(|(_, n)| n).collect(), problems }
}

/// Исходные видео в самом каталоге (без подкаталогов): обычные непустые файлы.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Sources {
    pub webm: Vec<String>,
    /// Без `output.mp4` — это чекпоинт, а не исходник.
    pub mp4: Vec<String>,
}

pub fn sources(dir: &Path) -> Sources {
    let mut s = Sources::default();
    let Ok(entries) = std::fs::read_dir(dir) else { return s };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || file_state(&e.path()) != FileState::Valid {
            continue;
        }
        let lower = name.to_lowercase();
        if lower.ends_with(".webm") {
            s.webm.push(name);
        } else if lower.ends_with(".mp4") && name != MP4 {
            s.mp4.push(name);
        }
    }
    s.webm.sort();
    s.mp4.sort();
    s
}

/// Проверка `output.mp4` по исходному WebM.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Mp4Check {
    Missing,
    Empty,
    NotAFile,
    /// WebM рядом нет — непустой `output.mp4` переиспользуется без сверки.
    Unverified,
    Valid { webm: String, webm_seconds: i64, mp4_seconds: i64 },
    Mismatch { webm: String, webm_seconds: i64, mp4_seconds: i64 },
    /// `ffprobe` не смог прочитать один из файлов (например, конвертацию прервали).
    Unreadable { webm: String, error: String },
    /// WebM несколько — неясно, из какого сделан `output.mp4`.
    Ambiguous { webm: Vec<String> },
    /// В каталоге идёт задание — файл, возможно, ещё пишется; не проверялся.
    Busy,
}

/// Сверить `output.mp4` с WebM: `chosen` — выбранный пользователем исходник.
pub async fn check_mp4(dir: &Path, webms: &[String], chosen: Option<&str>) -> Mp4Check {
    match file_state(&dir.join(MP4)) {
        FileState::Missing => return Mp4Check::Missing,
        FileState::Empty => return Mp4Check::Empty,
        FileState::NotAFile => return Mp4Check::NotAFile,
        FileState::Valid => {}
    }
    let webm = match (webms, chosen.filter(|c| webms.iter().any(|w| w == c))) {
        ([], _) => return Mp4Check::Unverified,
        (_, Some(c)) => c.to_string(),
        ([one], None) => one.clone(),
        (many, None) => return Mp4Check::Ambiguous { webm: many.to_vec() },
    };
    match durations(&dir.join(&webm), &dir.join(MP4)).await {
        Ok((w, m)) if w == m => Mp4Check::Valid { webm, webm_seconds: w, mp4_seconds: m },
        Ok((w, m)) => Mp4Check::Mismatch { webm, webm_seconds: w, mp4_seconds: m },
        Err(e) => Mp4Check::Unreadable { webm, error: format!("{e:#}") },
    }
}

/// Длительности WebM и MP4 в целых секундах.
pub async fn durations(webm: &Path, mp4: &Path) -> Result<(i64, i64)> {
    let w = media::probe(webm).await?;
    let m = media::probe(mp4).await?;
    Ok((w.seconds, m.seconds))
}

/// Следующий шаг обработки.
#[derive(Debug, Clone, Serialize)]
pub struct Next {
    /// `wait_job`, `transcribe`, `split_audio`, `extract_mp3`, `convert_to_mp4`,
    /// `choose_source`, `ask_summary`, `done`, `nothing`.
    pub step: &'static str,
    /// Инструмент и доводы для этого шага (если шаг — вызов инструмента).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<&'static str>,
    #[serde(skip_serializing_if = "Value::is_null")]
    pub args: Value,
    /// Почему этот шаг — для отчёта пользователю.
    pub why: String,
    /// Из чего выбирать (`choose_source`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
}

impl Next {
    fn tool(step: &'static str, args: Value, why: impl Into<String>) -> Self {
        Self { step, tool: Some(step), args, why: why.into(), choices: vec![] }
    }

    fn say(step: &'static str, why: impl Into<String>) -> Self {
        Self { step, tool: None, args: Value::Null, why: why.into(), choices: vec![] }
    }

    fn choose(choices: Vec<String>, why: impl Into<String>) -> Self {
        Self { step: "choose_source", tool: None, args: Value::Null, why: why.into(), choices }
    }

    /// Доводы шага как строка (`source`), если есть.
    pub fn arg(&self, name: &str) -> Option<String> {
        self.args[name].as_str().map(str::to_string)
    }
}

/// Состояние каталога записи.
#[derive(Debug, Clone, Serialize)]
pub struct Inspection {
    pub root: String,
    pub dir: String,
    pub transcript: FileState,
    pub chunks: Chunks,
    pub mp3: FileState,
    pub mp4: Mp4Check,
    pub sources: Sources,
    pub summary: FileState,
    pub summary_prompt: FileState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<JobView>,
    pub next: Next,
}

/// Разобрать каталог и выбрать следующий шаг. `source` — выбор пользователя,
/// когда исходников несколько (WebM для сверки или конвертации, MP4 для
/// извлечения звука).
pub async fn inspect(root: &Root, dir: &Path, source: Option<&str>, job: Option<JobView>) -> Result<Inspection> {
    let sources = sources(dir);
    if let Some(s) = source {
        if !sources.webm.iter().chain(&sources.mp4).any(|n| n == s) {
            bail!(
                "«{s}» — не исходник в этом каталоге; есть WebM: {}, MP4: {}",
                list_or_none(&sources.webm),
                list_or_none(&sources.mp4)
            );
        }
    }
    let transcript = file_state(&dir.join(TRANSCRIPT));
    let chunks = scan_chunks(dir);
    let mp3 = file_state(&dir.join(MP3));
    // ffprobe нужен только тогда, когда до output.mp4 дело дошло, и не во время задания.
    let need_mp4 = job.is_none() && transcript != FileState::Valid && chunks.state != SetState::Valid && mp3 != FileState::Valid;
    let mp4 = if job.is_some() {
        Mp4Check::Busy
    } else if need_mp4 {
        check_mp4(dir, &sources.webm, source).await
    } else {
        match file_state(&dir.join(MP4)) {
            FileState::Missing => Mp4Check::Missing,
            FileState::Empty => Mp4Check::Empty,
            FileState::NotAFile => Mp4Check::NotAFile,
            FileState::Valid => Mp4Check::Unverified,
        }
    };
    let summary = file_state(&dir.join(SUMMARY));
    let summary_prompt = file_state(&dir.join(SUMMARY_PROMPT));
    let next = match &job {
        Some(j) => Next::say("wait_job", format!("идёт задание {} ({}): {}", j.id, j.kind, j.progress)),
        None => next_step(transcript, &chunks, mp3, &mp4, &sources, source, summary),
    };
    Ok(Inspection {
        root: root.path().display().to_string(),
        dir: root.label(dir),
        transcript,
        chunks,
        mp3,
        mp4,
        sources,
        summary,
        summary_prompt,
        job,
        next,
    })
}

/// Step 0 скилла: от самого позднего чекпоинта к раннему.
fn next_step(
    transcript: FileState,
    chunks: &Chunks,
    mp3: FileState,
    mp4: &Mp4Check,
    sources: &Sources,
    source: Option<&str>,
    summary: FileState,
) -> Next {
    if transcript == FileState::Valid {
        return if summary == FileState::Valid {
            Next::say("done", format!("{TRANSCRIPT} и {SUMMARY} уже есть — саммари переиспользуется; заново — только по явной просьбе (summarize с force)"))
        } else {
            Next::say("ask_summary", format!("{TRANSCRIPT} готов — спросить пользователя, делать ли саммари"))
        };
    }
    if chunks.state == SetState::Valid {
        return Next::tool(
            "transcribe",
            json!({}),
            format!("есть {} MP3-чанков — транскрибировать все (готовые TXT перезапишутся)", chunks.files.len()),
        );
    }
    let bad_chunks = if chunks.state == SetState::Invalid {
        format!("; набор чанков негоден ({}) — нарезка переложит его в {DIAGNOSTICS_DIR}/", chunks.problems.join(", "))
    } else {
        String::new()
    };
    if mp3 == FileState::Valid {
        return Next::tool("split_audio", json!({}), format!("{MP3} есть — нарезать на чанки{bad_chunks}"));
    }
    match mp4 {
        Mp4Check::Valid { webm, webm_seconds, .. } => {
            return Next::tool(
                "extract_mp3",
                json!({ "source": MP4, "webm": webm }),
                format!("{MP4} совпал с {webm} по длительности ({webm_seconds} с) — извлечь звук"),
            )
        }
        Mp4Check::Unverified => {
            return Next::tool("extract_mp3", json!({ "source": MP4 }), format!("{MP4} есть, WebM рядом нет — извлечь звук"))
        }
        Mp4Check::Mismatch { webm, webm_seconds, mp4_seconds } => {
            return Next::tool(
                "convert_to_mp4",
                json!({ "source": webm }),
                format!(
                    "{MP4} негоден: длительность WebM {webm_seconds} с, MP4 {mp4_seconds} с — пересоздать из {webm} \
                     (старый {MP4} переедет в {DIAGNOSTICS_DIR}/)"
                ),
            )
        }
        Mp4Check::Unreadable { webm, error } => {
            return Next::tool(
                "convert_to_mp4",
                json!({ "source": webm }),
                format!("{MP4} не читается ({error}) — пересоздать из {webm} (старый переедет в {DIAGNOSTICS_DIR}/)"),
            )
        }
        Mp4Check::Ambiguous { webm } => {
            return Next::choose(webm.clone(), format!("есть {MP4} и несколько WebM — какой из них исходник для сверки?"))
        }
        Mp4Check::Missing | Mp4Check::Empty | Mp4Check::NotAFile | Mp4Check::Busy => {}
    }
    match (sources.mp4.as_slice(), source.filter(|s| sources.mp4.iter().any(|m| m == s))) {
        (_, Some(s)) => return Next::tool("extract_mp3", json!({ "source": s }), format!("исходный MP4 {s} — извлечь звук")),
        ([one], None) => return Next::tool("extract_mp3", json!({ "source": one }), format!("исходный MP4 {one} — извлечь звук")),
        ([], None) => {}
        (many, None) => return Next::choose(many.to_vec(), "несколько исходных MP4 — какой обработать?"),
    }
    match (sources.webm.as_slice(), source.filter(|s| sources.webm.iter().any(|w| w == s))) {
        (_, Some(s)) => Next::tool("convert_to_mp4", json!({ "source": s }), format!("только WebM — конвертировать {s} в {MP4}")),
        ([one], None) => Next::tool("convert_to_mp4", json!({ "source": one }), format!("только WebM — конвертировать {one} в {MP4}")),
        ([], None) => Next::say("nothing", "нет ни транскрипта, ни чанков, ни MP3, ни видео (.mp4, .webm) — обрабатывать нечего"),
        (many, None) => Next::choose(many.to_vec(), "несколько WebM — какой обработать?"),
    }
}

fn list_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "нет".into()
    } else {
        items.join(", ")
    }
}

/// Убрать файл или каталог в `diagnostics/` под именем с отметкой времени.
/// Возвращает новый путь от каталога записи.
pub fn to_diagnostics(dir: &Path, name: &str) -> Result<String> {
    let from = dir.join(name);
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let base = Path::new(name).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| name.into());
    let target_name = match base.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => format!("{stem}-{stamp}.{ext}"),
        _ => format!("{base}-{stamp}"),
    };
    std::fs::create_dir_all(dir.join(DIAGNOSTICS_DIR))?;
    let rel = format!("{DIAGNOSTICS_DIR}/{target_name}");
    std::fs::rename(&from, dir.join(&rel)).with_context(|| format!("не удалось перенести {name} в {rel}"))?;
    Ok(rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("media-mcp-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn put(dir: &Path, name: &str, text: &str) {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    async fn next(dir: &Path, source: Option<&str>) -> Next {
        let root = Root::open(dir).unwrap();
        inspect(&root, &root.recording(None).unwrap(), source, None).await.unwrap().next
    }

    #[test]
    fn sandbox_keeps_recordings_inside_root() {
        let dir = tmp("sandbox");
        std::fs::create_dir_all(dir.join("2026-10-15")).unwrap();
        let root = Root::open(&dir).unwrap();
        assert_eq!(root.label(&root.recording(Some("2026-10-15")).unwrap()), "2026-10-15");
        assert_eq!(root.label(&root.recording(None).unwrap()), ".");
        assert!(root.recording(Some("../")).unwrap_err().to_string().contains(".."));
        assert!(root.recording(Some("/")).unwrap_err().to_string().contains("вне корня"));
        assert!(root.recording(Some("нет")).is_err());
    }

    #[test]
    fn chunk_names_and_set_validity() {
        assert_eq!(chunk_number("chunk-007.mp3"), Some(7));
        assert_eq!(chunk_number("chunk-1000.mp3"), Some(1000));
        assert_eq!(chunk_number("chunk-7.mp3"), None);
        assert_eq!(chunk_number("chunk-abc.mp3"), None);

        let dir = tmp("chunks");
        assert_eq!(scan_chunks(&dir).state, SetState::Missing);
        put(&dir, "transcript-chunks/chunk-001.mp3", "b");
        put(&dir, "transcript-chunks/chunk-000.mp3", "a");
        // TXT не влияют ни на что.
        put(&dir, "transcript-chunks/chunk-000.txt", "");
        let c = scan_chunks(&dir);
        assert_eq!((c.state, c.files.clone()), (SetState::Valid, vec!["chunk-000.mp3".to_string(), "chunk-001.mp3".into()]));
        put(&dir, "transcript-chunks/chunk-002.mp3", "");
        put(&dir, "transcript-chunks/chunk-x.mp3", "x");
        let c = scan_chunks(&dir);
        assert_eq!(c.state, SetState::Invalid);
        assert_eq!(c.problems, ["chunk-002.mp3: пустой", "chunk-x.mp3: имя не по соглашению chunk-NNN.mp3"]);
    }

    #[tokio::test]
    async fn latest_checkpoint_wins() {
        let dir = tmp("order");
        assert_eq!(next(&dir, None).await.step, "nothing");

        put(&dir, "a.webm", "w");
        let n = next(&dir, None).await;
        assert_eq!((n.step, n.arg("source").as_deref()), ("convert_to_mp4", Some("a.webm")));

        put(&dir, "b.webm", "w");
        let n = next(&dir, None).await;
        assert_eq!((n.step, n.choices.clone()), ("choose_source", vec!["a.webm".to_string(), "b.webm".into()]));
        assert_eq!(next(&dir, Some("b.webm")).await.arg("source").as_deref(), Some("b.webm"));

        // Исходный MP4 важнее WebM; output.mp4 исходником не считается.
        put(&dir, "talk.mp4", "m");
        let n = next(&dir, None).await;
        assert_eq!((n.step, n.arg("source").as_deref()), ("extract_mp3", Some("talk.mp4")));

        put(&dir, "output.mp3", "");
        assert_eq!(next(&dir, None).await.step, "extract_mp3", "пустой MP3 — не чекпоинт");
        put(&dir, "output.mp3", "audio");
        assert_eq!(next(&dir, None).await.step, "split_audio");

        put(&dir, "transcript-chunks/chunk-000.mp3", "c");
        put(&dir, "transcript-chunks/chunk-000.txt", "старый текст");
        assert_eq!(next(&dir, None).await.step, "transcribe", "готовые TXT не пропускают транскрипцию");

        put(&dir, "output-transcript.md", "");
        assert_eq!(next(&dir, None).await.step, "transcribe");
        put(&dir, "output-transcript.md", "текст");
        assert_eq!(next(&dir, None).await.step, "ask_summary");
        put(&dir, "output-summary.md", "саммари");
        assert_eq!(next(&dir, None).await.step, "done");
    }

    #[tokio::test]
    async fn output_mp4_without_webm_is_reused() {
        let dir = tmp("mp4-alone");
        put(&dir, "output.mp4", "m");
        let root = Root::open(&dir).unwrap();
        let i = inspect(&root, &dir, None, None).await.unwrap();
        assert!(matches!(i.mp4, Mp4Check::Unverified));
        assert_eq!((i.next.step, i.next.arg("source").as_deref()), ("extract_mp3", Some("output.mp4")));
        assert!(inspect(&root, &dir, Some("чужой.webm"), None).await.is_err());
    }

    #[test]
    fn diagnostics_keep_old_artifacts() {
        let dir = tmp("diag");
        put(&dir, "output.mp4", "broken");
        let rel = to_diagnostics(&dir, "output.mp4").unwrap();
        assert!(rel.starts_with("diagnostics/output-") && rel.ends_with(".mp4"), "{rel}");
        assert!(!dir.join("output.mp4").exists());
        assert_eq!(std::fs::read_to_string(dir.join(rel)).unwrap(), "broken");
    }

    #[test]
    fn recordings_are_found_with_their_stage() {
        let dir = tmp("list");
        put(&dir, "2026-10-01/output-transcript.md", "t");
        put(&dir, "2026-10-15/meeting.webm", "w");
        put(&dir, "2026-10-15/transcript-chunks/chunk-000.mp3", "c");
        put(&dir, "notes/readme.txt", "x");
        let root = Root::open(&dir).unwrap();
        let found: Vec<(String, &str)> = root.recordings(root.path(), 3).into_iter().map(|r| (r.dir, r.stage)).collect();
        assert_eq!(found, [("2026-10-01".to_string(), "transcript"), ("2026-10-15".to_string(), "chunks")]);
    }
}
