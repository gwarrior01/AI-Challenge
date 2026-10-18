//! Шаги обработки записи. Каждый сначала спрашивает [`recording::inspect`],
//! чья сейчас очередь, и отказывается, если не его: из негодного `output.mp4`
//! звук не извлечь, при готовых чанках MP3 заново не нарезать, при готовом
//! транскрипте не транскрибировать. Долгие шаги запускаются заданиями
//! ([`crate::jobs`]); ответ — номер задания, итог — в `wait_job`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::jobs::{Failure, JobCtx, JobView, Jobs};
use crate::media;
use crate::recording::{self, file_state, FileState, Inspection, Root, SetState, CHUNKS_DIR, MP3, MP4, SUMMARY, SUMMARY_PROMPT, TRANSCRIPT};
use crate::summary;
use crate::transcribe::{self, write_atomic};

/// Длина чанка, секунды (`MEDIA_CHUNK_SECONDS`).
const DEFAULT_CHUNK_SECONDS: u32 = 300;

pub struct Media {
    pub root: Root,
    pub jobs: Arc<Jobs>,
    pub transcribe: transcribe::Config,
    pub lm: summary::Config,
    pub chunk_seconds: u32,
}

/// Ответ инструмента, запустившего задание.
fn started(job: JobView) -> Value {
    json!({
        "job": job,
        "note": "Задание идёт в фоне сервера. Дождись его: wait_job(id) — вызов вернётся, когда задание закончится \
                 (или по таймауту — тогда снова wait_job). Пока оно идёт, этот каталог не трогай.",
    })
}

impl Media {
    pub fn from_env(root: Root) -> Self {
        Self {
            root,
            jobs: Jobs::new(),
            transcribe: transcribe::Config::from_env(),
            lm: summary::Config::from_env(),
            chunk_seconds: media::env_or("MEDIA_CHUNK_SECONDS", "").parse().unwrap_or(DEFAULT_CHUNK_SECONDS).max(1),
        }
    }

    pub async fn inspect(&self, dir: Option<&str>, source: Option<&str>) -> Result<(PathBuf, Inspection)> {
        let path = self.root.recording(dir)?;
        let job = self.jobs.running_in(&path);
        let i = recording::inspect(&self.root, &path, source, job).await?;
        Ok((path, i))
    }

    /// Состояние каталога, если сейчас очередь шага `step`; иначе — почему нет.
    async fn expect(&self, dir: Option<&str>, source: Option<&str>, step: &str) -> Result<(PathBuf, Inspection), String> {
        let (path, i) = self.inspect(dir, source).await.map_err(text)?;
        if i.next.step != step {
            let choices = if i.next.choices.is_empty() { String::new() } else { format!(" Варианты: {}.", i.next.choices.join(", ")) };
            return Err(format!(
                "{step} сейчас не выполняется: следующий шаг для «{}» — {} ({}).{choices}",
                i.dir, i.next.step, i.next.why
            ));
        }
        Ok((path, i))
    }

    /// WebM → `output.mp4` и сверка длительностей.
    pub async fn convert_to_mp4(&self, dir: Option<&str>, source: Option<&str>) -> Result<Value, String> {
        let (path, i) = self.expect(dir, source, "convert_to_mp4").await?;
        let webm = i.next.arg("source").expect("у convert_to_mp4 всегда есть source");
        if let Some(s) = source.filter(|s| *s != webm) {
            return Err(format!("сейчас конвертируется {webm}, а не {s}"));
        }
        // Негодный или недописанный output.mp4 — не затирать, а убрать для разбора.
        let moved = if file_state(&path.join(MP4)) != FileState::Missing {
            Some(recording::to_diagnostics(&path, MP4).map_err(text)?)
        } else {
            None
        };
        let label = i.dir.clone();
        let job = self
            .jobs
            .start("convert_to_mp4", &path.clone(), &label, move |ctx| convert_job(path, webm, moved, ctx))
            .map_err(|e| e.to_string())?;
        Ok(started(job))
    }

    /// `output.mp4` или исходный MP4 → `output.mp3`.
    pub async fn extract_mp3(&self, dir: Option<&str>, source: Option<&str>, webm: Option<&str>) -> Result<Value, String> {
        // Выбор для inspect — исходник: WebM для сверки output.mp4 или исходный MP4.
        let choice = webm.or(source.filter(|s| *s != MP4));
        let (path, i) = self.expect(dir, choice, "extract_mp3").await?;
        let input = i.next.arg("source").expect("у extract_mp3 всегда есть source");
        if let Some(s) = source.filter(|s| *s != input) {
            return Err(format!("звук сейчас извлекается из {input}, а не из {s} ({})", i.next.why));
        }
        let label = i.dir.clone();
        let job = self.jobs.start("extract_mp3", &path.clone(), &label, move |ctx| extract_job(path, input, ctx)).map_err(|e| e.to_string())?;
        Ok(started(job))
    }

    /// `output.mp3` → `transcript-chunks/chunk-NNN.mp3`.
    pub async fn split_audio(&self, dir: Option<&str>) -> Result<Value, String> {
        let (path, i) = self.expect(dir, None, "split_audio").await?;
        // Негодный набор чанков не смешивать с новым.
        let moved = if i.chunks.state == SetState::Invalid { Some(recording::to_diagnostics(&path, CHUNKS_DIR).map_err(text)?) } else { None };
        let seconds = self.chunk_seconds;
        let job = self
            .jobs
            .start("split_audio", &path.clone(), &i.dir, move |ctx| split_job(path, seconds, moved, ctx))
            .map_err(|e| e.to_string())?;
        Ok(started(job))
    }

    /// Все MP3-чанки → TXT → `output-transcript.md`.
    pub async fn transcribe(&self, dir: Option<&str>) -> Result<Value, String> {
        let (path, i) = self.expect(dir, None, "transcribe").await?;
        let chunks = i.chunks.paths(&path);
        let cfg = self.transcribe.clone();
        let job = self
            .jobs
            .start("transcribe", &path.clone(), &i.dir, move |ctx| transcribe::run(cfg, path, chunks, ctx))
            .map_err(|e| e.to_string())?;
        Ok(started(job))
    }

    pub async fn models(&self) -> Result<Value, String> {
        let m = summary::models(&self.lm).await.map_err(text)?;
        Ok(json!({
            "url": self.lm.url,
            "models": m.all,
            "chat_models": m.chat,
            "loaded_chat_models": m.loaded,
            "obvious": m.obvious(),
        }))
    }

    /// Саммари через LM Studio → `output-summary.md`.
    pub async fn summarize(&self, dir: Option<&str>, model: Option<&str>, force: bool) -> Result<Value, String> {
        let (path, i) = self.inspect(dir, None).await.map_err(text)?;
        if let Some(j) = &i.job {
            return Err(format!("в каталоге идёт задание {} ({}) — дождись его", j.id, j.kind));
        }
        if i.transcript != FileState::Valid {
            return Err(format!("{TRANSCRIPT} ещё нет — саммари не из чего; следующий шаг — {} ({})", i.next.step, i.next.why));
        }
        if i.summary == FileState::Valid && !force {
            return Err(format!("{SUMMARY} уже есть и переиспользуется; перегенерировать — force=true, только по явной просьбе пользователя"));
        }
        let models = summary::models(&self.lm).await.map_err(text)?;
        let model = match model.map(str::trim).filter(|m| !m.is_empty()) {
            Some(m) if models.chat.iter().any(|c| c == m) => m.to_string(),
            Some(m) => return Err(format!("в LM Studio нет модели «{m}»; есть: {}", models.chat.join(", "))),
            None if models.chat.is_empty() => return Err(format!("в LM Studio ({}) нет моделей для чата", self.lm.url)),
            None => match models.obvious() {
                Some(m) => m.clone(),
                None => {
                    let loaded = models.loaded.as_ref().filter(|l| !l.is_empty()).map(|l| format!(" (загружены: {})", l.join(", ")));
                    return Err(format!(
                        "в LM Studio несколько моделей — спроси пользователя, какую взять: {}{}",
                        models.chat.join(", "),
                        loaded.unwrap_or_default()
                    ));
                }
            },
        };
        let cfg = self.lm.clone();
        let job = self
            .jobs
            .start("summarize", &path.clone(), &i.dir, move |ctx| summarize_job(cfg, path, model, ctx))
            .map_err(|e| e.to_string())?;
        Ok(started(job))
    }

    /// Промпт для саммари в файл — без вызова LM Studio.
    pub async fn write_summary_prompt(&self, dir: Option<&str>) -> Result<Value, String> {
        let (path, i) = self.inspect(dir, None).await.map_err(text)?;
        if let Some(j) = &i.job {
            return Err(format!("в каталоге идёт задание {} ({}) — дождись его", j.id, j.kind));
        }
        if i.transcript != FileState::Valid {
            return Err(format!("{TRANSCRIPT} ещё нет — промпт не из чего; следующий шаг — {} ({})", i.next.step, i.next.why));
        }
        let prompt = summary::prompt(&path.join(TRANSCRIPT)).map_err(text)?;
        write_atomic(&path.join(SUMMARY_PROMPT), prompt.as_bytes()).map_err(text)?;
        Ok(json!({ "created": SUMMARY_PROMPT, "chars": prompt.chars().count(), "lm_studio_called": false }))
    }

    /// Длительность файла каталога записи.
    pub async fn probe(&self, dir: Option<&str>, file: &str) -> Result<Value, String> {
        let path = self.root.recording(dir).map_err(text)?;
        let name = Path::new(file);
        if name.components().count() != 1 && !file.starts_with(&format!("{CHUNKS_DIR}/")) || file.contains("..") {
            return Err(format!("«{file}» — нужен файл самого каталога записи или {CHUNKS_DIR}/…"));
        }
        let d = media::probe(&path.join(name)).await.map_err(text)?;
        Ok(json!({ "file": file, "duration": d.raw, "seconds": d.seconds, "clock": media::clock(d.raw) }))
    }
}

fn text(e: anyhow::Error) -> String {
    format!("{e:#}")
}

fn os(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

async fn convert_job(dir: PathBuf, webm: String, moved: Option<String>, ctx: JobCtx) -> Result<Value, Failure> {
    let src = dir.join(&webm);
    let total = media::probe(&src).await.ok().map(|d| d.raw);
    let mut args = vec![OsString::from("-i"), src.clone().into_os_string()];
    args.extend(os(&["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-c:a", "aac", "-b:a", "192k"]));
    args.push(dir.join(MP4).into_os_string());
    let base = json!({ "source": webm, "moved_to_diagnostics": moved });
    if let Err(e) = media::ffmpeg(args, total, &ctx).await {
        let partial = file_state(&dir.join(MP4)) != FileState::Missing;
        return Err(Failure::with(
            format!("конвертация не удалась: {e:#}. Исходник сохранён{}", if partial { ", недописанный output.mp4 оставлен для разбора" } else { "" }),
            base,
        ));
    }
    // Только после выхода ffmpeg с кодом 0.
    if file_state(&dir.join(MP4)) != FileState::Valid {
        return Err(Failure::with("ffmpeg завершился успешно, но output.mp4 пустой или отсутствует", base));
    }
    ctx.progress("сверяю длительность…");
    let (w, m) = recording::durations(&src, &dir.join(MP4))
        .await
        .map_err(|e| Failure::with(format!("проверка длительности не удалась: {e:#}"), base.clone()))?;
    let mut details = base;
    details["webm_seconds"] = json!(w);
    details["mp4_seconds"] = json!(m);
    if w != m {
        details["duration_check"] = json!("MISMATCH");
        return Err(Failure::with(
            format!("WebM/MP4 duration mismatch: WebM={w}s, MP4={m}s — output.mp4 негоден, оба файла сохранены, к MP3 не переходить"),
            details,
        ));
    }
    details["duration_check"] = json!("OK");
    details["created"] = json!(MP4);
    Ok(details)
}

async fn extract_job(dir: PathBuf, input: String, ctx: JobCtx) -> Result<Value, Failure> {
    let src = dir.join(&input);
    let total = media::probe(&src).await.ok().map(|d| d.raw);
    // Во временный файл: прерванное извлечение не оставит «годный» по размеру output.mp3.
    let partial = dir.join(".output.mp3.partial");
    let _ = std::fs::remove_file(&partial);
    let mut args = vec![OsString::from("-i"), src.into_os_string()];
    args.extend(os(&["-vn", "-f", "mp3"]));
    args.push(partial.clone().into_os_string());
    if let Err(e) = media::ffmpeg(args, total, &ctx).await {
        let _ = std::fs::remove_file(&partial);
        return Err(Failure::with(format!("извлечение звука не удалось: {e:#}; {input} сохранён"), json!({ "source": input })));
    }
    if file_state(&partial) != FileState::Valid {
        return Err(Failure::with("ffmpeg завершился успешно, но MP3 пустой", json!({ "source": input })));
    }
    std::fs::rename(&partial, dir.join(MP3)).context("не удалось сохранить output.mp3")?;
    Ok(json!({ "source": input, "created": MP3 }))
}

async fn split_job(dir: PathBuf, seconds: u32, moved: Option<String>, ctx: JobCtx) -> Result<Value, Failure> {
    let total = media::probe(&dir.join(MP3)).await.ok().map(|d| d.raw);
    let tmp = dir.join(".transcript-chunks.partial");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).context("не удалось создать каталог для нарезки")?;
    let mut args = vec![OsString::from("-i"), dir.join(MP3).into_os_string()];
    let seg = seconds.to_string();
    args.extend(os(&["-f", "segment", "-segment_time", &seg, "-reset_timestamps", "1", "-c", "copy"]));
    args.push(tmp.join("chunk-%03d.mp3").into_os_string());
    let details = json!({ "source": MP3, "segment_seconds": seconds, "moved_to_diagnostics": moved });
    if let Err(e) = media::ffmpeg(args, total, &ctx).await {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(Failure::with(format!("нарезка не удалась: {e:#}; output.mp3 сохранён"), details));
    }
    let mut made: Vec<String> = std::fs::read_dir(&tmp)
        .context("каталог нарезки пропал")?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| recording::chunk_number(n).is_some())
        .collect();
    made.sort();
    if made.is_empty() || made.iter().any(|n| file_state(&tmp.join(n)) != FileState::Valid) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(Failure::with("нарезка не дала непустых чанков", details));
    }
    let chunks = dir.join(CHUNKS_DIR);
    std::fs::create_dir_all(&chunks).context("не удалось создать transcript-chunks")?;
    for n in &made {
        std::fs::rename(tmp.join(n), chunks.join(n)).with_context(|| format!("не удалось перенести {n}"))?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let mut details = details;
    details["chunks"] = json!(made.len());
    details["created"] = json!(format!("{CHUNKS_DIR}/chunk-*.mp3"));
    Ok(details)
}

async fn summarize_job(cfg: summary::Config, dir: PathBuf, model: String, ctx: JobCtx) -> Result<Value, Failure> {
    let prompt = summary::prompt(&dir.join(TRANSCRIPT))?;
    ctx.progress(format!("LM Studio · {model} думает над транскриптом ({} символов)…", prompt.chars().count()));
    let answer = summary::complete(&cfg, &model, &prompt)
        .await
        .map_err(|e| Failure::with(format!("{e:#}; {TRANSCRIPT} сохранён, другой LLM не подставлялся"), json!({ "model": model })))?;
    write_atomic(&dir.join(SUMMARY), answer.as_bytes())?;
    Ok(json!({ "created": SUMMARY, "model": model, "chars": answer.chars().count() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobState;
    use std::process::Command;
    use std::time::Duration;

    fn have_ffmpeg() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("media-mcp-steps-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Короткое WebM-видео со звуком, сгенерированное самим ffmpeg.
    fn make_webm(path: &Path, seconds: u32) {
        let s = seconds.to_string();
        let ok = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", &format!("testsrc=size=160x120:rate=10:duration={s}")])
            .args(["-f", "lavfi", "-i", &format!("sine=frequency=440:duration={s}")])
            .args(["-c:v", "libvpx", "-b:v", "50k", "-c:a", "libopus", "-shortest"])
            .arg(path)
            .status()
            .unwrap()
            .success();
        assert!(ok, "не удалось сгенерировать WebM");
    }

    fn media(dir: &Path, transcriber: &str, lm: &str) -> Media {
        Media {
            root: Root::open(dir).unwrap(),
            jobs: Jobs::new(),
            transcribe: crate::transcribe::tests::config(transcriber),
            lm: summary::Config { url: lm.into(), timeout: Duration::from_secs(10) },
            chunk_seconds: 2,
        }
    }

    async fn wait(m: &Media, started: Value) -> JobView {
        let id = started["job"]["id"].as_str().unwrap().to_string();
        m.jobs.wait(&id, Duration::from_secs(120)).await.unwrap()
    }

    /// Вся цепочка на настоящем ffmpeg и поддельных сервисах: WebM → MP4
    /// (сверка) → MP3 → чанки → транскрипт → саммари; шаги вне очереди — отказ.
    #[tokio::test]
    async fn whole_pipeline() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg не найден — тест пропущен");
            return;
        }
        let (tr, _) = crate::transcribe::tests::fake_transcriber(false).await;
        let lm = crate::summary::tests::fake_lmstudio(&["qwen3-30b"]).await;
        let dir = tmp("pipeline");
        std::fs::create_dir_all(dir.join("rec")).unwrap();
        make_webm(&dir.join("rec/meeting.webm"), 5);
        let m = media(&dir, &tr, &lm);
        let rec = Some("rec");

        assert!(m.extract_mp3(rec, None, None).await.unwrap_err().contains("следующий шаг для «rec» — convert_to_mp4"));
        let job = m.convert_to_mp4(rec, None).await.unwrap();
        assert_eq!(job["job"]["state"], "running");
        // Пока идёт конвертация — ни второй конвертации, ни извлечения.
        let busy = m.convert_to_mp4(rec, None).await.unwrap_err();
        assert!(busy.contains("wait_job"), "{busy}");
        let done = wait(&m, job).await;
        assert_eq!(done.state, JobState::Succeeded, "{done:?}");
        let r = done.result.unwrap();
        assert_eq!((r["duration_check"].as_str(), r["webm_seconds"].as_i64()), (Some("OK"), Some(5)));

        // Теперь output.mp4 сверяется с WebM и годится.
        let (_, i) = m.inspect(rec, None).await.unwrap();
        assert_eq!(i.next.step, "extract_mp3");
        assert!(matches!(i.mp4, recording::Mp4Check::Valid { webm_seconds: 5, mp4_seconds: 5, .. }));
        assert_eq!(wait(&m, m.extract_mp3(rec, None, None).await.unwrap()).await.state, JobState::Succeeded);
        assert!(!dir.join("rec/.output.mp3.partial").exists());

        let split = wait(&m, m.split_audio(rec).await.unwrap()).await;
        assert_eq!(split.state, JobState::Succeeded, "{split:?}");
        assert!(split.result.unwrap()["chunks"].as_u64().unwrap() >= 2);
        assert!(m.split_audio(rec).await.unwrap_err().contains("следующий шаг для «rec» — transcribe"));

        let tr = wait(&m, m.transcribe(rec).await.unwrap()).await;
        assert_eq!(tr.state, JobState::Succeeded, "{tr:?}");
        assert!(std::fs::read_to_string(dir.join("rec/output-transcript.md")).unwrap().starts_with("[ru] текст chunk-000.mp3\n"));
        assert_eq!(m.inspect(rec, None).await.unwrap().1.next.step, "ask_summary");

        let p = m.write_summary_prompt(rec).await.unwrap();
        assert_eq!(p["lm_studio_called"], false);
        let s = wait(&m, m.summarize(rec, None, false).await.unwrap()).await;
        assert_eq!(s.state, JobState::Succeeded, "{s:?}");
        assert!(std::fs::read_to_string(dir.join("rec/output-summary.md")).unwrap().starts_with("Краткое резюме"));
        assert!(m.summarize(rec, None, false).await.unwrap_err().contains("force=true"));
        assert_eq!(m.inspect(rec, None).await.unwrap().1.next.step, "done");
    }

    /// `output.mp4` не совпал с WebM по длительности: в MP3 не идём, старый
    /// файл уезжает в diagnostics/, MP4 пересоздаётся.
    #[tokio::test]
    async fn mismatched_mp4_is_regenerated() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg не найден — тест пропущен");
            return;
        }
        let dir = tmp("mismatch");
        make_webm(&dir.join("talk.webm"), 4);
        make_webm(&dir.join("short.webm"), 2);
        let m = media(&dir, "http://127.0.0.1:9", "http://127.0.0.1:9");
        // Чужой output.mp4: сделан из short.webm.
        let ok = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(dir.join("short.webm"))
            .arg(dir.join("output.mp4"))
            .status()
            .unwrap()
            .success();
        assert!(ok);

        let (_, i) = m.inspect(None, None).await.unwrap();
        assert_eq!((i.next.step, i.next.choices.len()), ("choose_source", 2), "два WebM — спросить");
        let (_, i) = m.inspect(None, Some("talk.webm")).await.unwrap();
        assert!(matches!(i.mp4, recording::Mp4Check::Mismatch { webm_seconds: 4, mp4_seconds: 2, .. }), "{:?}", i.mp4);
        assert_eq!(i.next.step, "convert_to_mp4");
        assert!(m.extract_mp3(None, None, Some("talk.webm")).await.unwrap_err().contains("convert_to_mp4"));

        let done = wait(&m, m.convert_to_mp4(None, Some("talk.webm")).await.unwrap()).await;
        assert_eq!(done.state, JobState::Succeeded, "{done:?}");
        let r = done.result.unwrap();
        assert!(r["moved_to_diagnostics"].as_str().unwrap().starts_with("diagnostics/output-"));
        assert_eq!(r["mp4_seconds"], 4);
        assert!(dir.join("talk.webm").exists() && dir.join("short.webm").exists());
    }
}
