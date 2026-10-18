//! `ffprobe` и `ffmpeg` — дочерние процессы сервера. Завершение — только
//! выход процесса: код 0 — успех, иначе ошибка с последними строками stderr.
//! Ход работы `ffmpeg` берётся из `-progress pipe:1` (`out_time_us`).

use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::jobs::JobCtx;

/// Сколько последних строк stderr `ffmpeg` показывать при ошибке.
const STDERR_TAIL: usize = 15;

pub fn ffmpeg_bin() -> String {
    env_or("MEDIA_FFMPEG", "ffmpeg")
}

pub fn ffprobe_bin() -> String {
    env_or("MEDIA_FFPROBE", "ffprobe")
}

/// Длительность файла: как есть и в целых секундах — округление как у
/// `printf "%.0f"` (половина — к чётному).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Duration {
    pub raw: f64,
    pub seconds: i64,
}

pub async fn probe(path: &Path) -> Result<Duration> {
    let out = Command::new(ffprobe_bin())
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(path)
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .with_context(|| format!("не удалось запустить {}", ffprobe_bin()))?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !out.status.success() {
        bail!("ffprobe {name}: код {} — {}", code(out.status), String::from_utf8_lossy(&out.stderr).trim());
    }
    parse_duration(&String::from_utf8_lossy(&out.stdout)).with_context(|| format!("ffprobe {name}: длительность не число"))
}

pub fn parse_duration(text: &str) -> Result<Duration> {
    let raw: f64 = text.trim().parse().context("пустой или нечисловой ответ")?;
    if !raw.is_finite() || raw < 0.0 {
        bail!("недопустимая длительность {raw}");
    }
    Ok(Duration { raw, seconds: raw.round_ties_even() as i64 })
}

/// Запустить `ffmpeg` и дождаться его выхода. `total` — длительность
/// входа (для процентов хода работы).
pub async fn ffmpeg(args: Vec<OsString>, total: Option<f64>, ctx: &JobCtx) -> Result<()> {
    let mut child = Command::new(ffmpeg_bin())
        // -nostdin: ffmpeg не ждёт ответа с клавиатуры; -n: не перезаписывать
        // существующий выход молча — перед запуском выход убран кодом.
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-n"])
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Сервер остановили — ffmpeg не остаётся сиротой.
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("не удалось запустить {}", ffmpeg_bin()))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let progress_ctx = ctx.clone();
    let progress = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(us) = line.strip_prefix("out_time_us=").and_then(|v| v.trim().parse::<i64>().ok()) {
                progress_ctx.progress(progress_text(us.max(0) as f64 / 1e6, total));
            }
        }
    });
    let errors = tokio::spawn(async move {
        let mut tail = VecDeque::new();
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if tail.len() == STDERR_TAIL {
                tail.pop_front();
            }
            tail.push_back(line);
        }
        tail.into_iter().collect::<Vec<_>>().join("\n")
    });

    // Ждём именно этот процесс — сколько бы он ни шёл.
    let status = child.wait().await.context("ffmpeg: не удалось дождаться процесса")?;
    let _ = progress.await;
    let stderr = errors.await.unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg завершился с кодом {}{}", code(status), if stderr.is_empty() { String::new() } else { format!(": {stderr}") });
    }
    Ok(())
}

fn code(status: std::process::ExitStatus) -> String {
    status.code().map(|c| c.to_string()).unwrap_or_else(|| "— остановлен сигналом".into())
}

/// `12:01 из 58:10 (21%)`.
pub fn progress_text(done: f64, total: Option<f64>) -> String {
    match total.filter(|t| *t > 0.0) {
        Some(t) => format!("{} из {} ({}%)", clock(done), clock(t), ((done / t * 100.0).clamp(0.0, 100.0)) as u32),
        None => format!("обработано {}", clock(done)),
    }
}

/// `1:02:03` или `58:10`.
pub fn clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).unwrap_or_else(|| default.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_round_like_printf() {
        assert_eq!(parse_duration("3605.437000\n").unwrap().seconds, 3605);
        assert_eq!(parse_duration("12.5").unwrap().seconds, 12);
        assert_eq!(parse_duration("13.5").unwrap().seconds, 14);
        assert_eq!(parse_duration("0.6").unwrap().seconds, 1);
        assert!(parse_duration("N/A").is_err());
        assert!(parse_duration("").is_err());
    }

    #[test]
    fn progress_is_readable() {
        assert_eq!(progress_text(721.0, Some(3490.0)), "12:01 из 58:10 (20%)");
        assert_eq!(progress_text(3723.0, None), "обработано 1:02:03");
    }
}
