//! Диалоги сохраняются, чтобы было видно, что и почему делал ассистент:
//!
//! ```text
//! media/runs/<UTC-время>-<каталог записи>/
//!   dialog.md   — реплики и ответы (с замечаниями)
//!   tools.json  — все вызовы инструментов: доводы и результаты (у заданий — итог)
//!   run.json    — каталог, модель, температура, реплики, шаги, токены, задания
//! ```
//!
//! Один диалог — один каталог: после каждой реплики файлы переписываются.
//! Каталог — `MEDIA_RUNS_DIR` (по умолчанию `media/runs`, в `.gitignore`);
//! хранятся последние `MEDIA_RUNS_KEEP` (по умолчанию 50).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::dialog::Dialog;
use crate::session::env_or;

pub const DEFAULT_RUNS_DIR: &str = "media/runs";
pub const DEFAULT_KEEP: usize = 50;

pub fn runs_dir() -> PathBuf {
    PathBuf::from(env_or("MEDIA_RUNS_DIR", DEFAULT_RUNS_DIR))
}

/// Сохранить диалог: в `existing` (каталог этого же диалога) или в новый каталог под `root`.
pub fn save(root: &Path, existing: Option<&Path>, d: &Dialog, model: &str) -> Result<PathBuf> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|x| x.as_secs()).unwrap_or(0);
    let dir = match existing {
        Some(dir) => dir.to_path_buf(),
        None => {
            let label = slug(&d.dir);
            let mut dir = root.join(format!("{}-{label}", stamp(now)));
            let mut n = 2;
            while dir.exists() {
                dir = root.join(format!("{}-{label}-{n}", stamp(now)));
                n += 1;
            }
            dir
        }
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("не удалось создать {}", dir.display()))?;

    let mut md = format!("# Запись `{}`\n", d.dir);
    for t in &d.turns {
        md.push_str(&format!("\n## Вы\n\n{}\n\n## Ассистент\n\n{}\n", t.user, t.reply));
        for note in &t.notes {
            md.push_str(&format!("\n> ⚠ {note}\n"));
        }
    }
    write(&dir, "dialog.md", &md)?;
    write(&dir, "tools.json", &serde_json::to_string_pretty(&d.tool_calls)?)?;
    let jobs: Vec<Value> = d
        .tool_calls
        .iter()
        .filter_map(|c| {
            let v: Value = serde_json::from_str(&c.result).ok()?;
            let job = v.get("job")?;
            Some(json!({ "tool": c.tool, "id": job["id"], "state": job["state"], "elapsed_sec": job["elapsed_sec"], "error": job["error"] }))
        })
        .collect();
    let started = existing.and_then(read_started).unwrap_or_else(|| iso(now));
    let meta = json!({
        "started_utc": started,
        "updated_utc": iso(now),
        "dir": d.dir,
        "model": model,
        "temperature": d.temperature,
        "turns": d.turns.len(),
        "rounds": d.turns.iter().map(|t| t.rounds).sum::<usize>(),
        "tool_calls": d.tool_calls.len(),
        "jobs": jobs,
        "usage": d.usage,
    });
    write(&dir, "run.json", &serde_json::to_string_pretty(&meta)?)?;
    if existing.is_none() {
        let keep: usize = env_or("MEDIA_RUNS_KEEP", "").parse().unwrap_or(DEFAULT_KEEP).max(1);
        // Не удалось убрать старые — не повод терять новый диалог.
        let _ = prune(root, keep);
    }
    Ok(dir)
}

fn read_started(dir: &Path) -> Option<String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).ok()?).ok()?;
    v["started_utc"].as_str().map(str::to_string)
}

/// Оставить в `root` только `keep` последних диалогов (каталоги `ГГГГММДД-ЧЧММСС-…` с `run.json`).
pub fn prune(root: &Path, keep: usize) -> Result<usize> {
    let mut runs: Vec<PathBuf> = std::fs::read_dir(root)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("run.json").is_file())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(is_run_name))
        .collect();
    runs.sort();
    let excess = runs.len().saturating_sub(keep);
    for dir in &runs[..excess] {
        std::fs::remove_dir_all(dir).with_context(|| format!("не удалось удалить {}", dir.display()))?;
    }
    Ok(excess)
}

fn is_run_name(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() > 16 && b[..8].iter().all(u8::is_ascii_digit) && b[8] == b'-' && b[9..15].iter().all(u8::is_ascii_digit) && b[15] == b'-'
}

fn write(dir: &Path, name: &str, text: &str) -> Result<()> {
    std::fs::write(dir.join(name), text).with_context(|| format!("не удалось записать {}", dir.join(name).display()))
}

/// Каталог записи для имени: `2026-10-15-bookclub`; корень — `root`.
fn slug(dir: &str) -> String {
    let s: String = dir
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
        .chars()
        .take(40)
        .collect();
    if s.is_empty() {
        "root".into()
    } else {
        s.trim_end_matches('-').to_string()
    }
}

/// Дата и время UTC из секунд Unix (алгоритм Говарда Хиннанта).
fn civil(secs: u64) -> (i64, u32, u32, u64, u64, u64) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month, day, rem / 3600, rem % 3600 / 60, rem % 60)
}

fn stamp(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

fn iso(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialog::Turn;
    use crate::session::record;

    #[test]
    fn names_and_dates() {
        assert_eq!(iso(1_792_160_701), "2026-10-16T14:25:01Z");
        assert_eq!(slug("2026-10-15/Book club"), "2026-10-15-book-club");
        assert_eq!(slug("."), "root");
    }

    #[test]
    fn dialog_is_saved_and_updated_in_place() {
        let root = std::env::temp_dir().join(format!("media-assistant-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut d = Dialog::new("2026-10-15");
        d.turns.push(Turn { user: "обработай".into(), reply: "Сделать саммари? (да/нет)".into(), rounds: 6, usage: None, notes: vec![] });
        d.tool_calls.push(record(
            "media",
            "transcribe",
            r#"{"dir":"2026-10-15"}"#,
            r#"{"job":{"id":"j4","kind":"transcribe","state":"succeeded","elapsed_sec":1520}}"#,
            false,
        ));
        let dir = save(&root, None, &d, "qwen3").unwrap();
        assert!(dir.file_name().unwrap().to_string_lossy().ends_with("-2026-10-15"));
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(meta["jobs"][0]["elapsed_sec"], 1520);

        d.turns.push(Turn { user: "да".into(), reply: "## Итог".into(), rounds: 3, usage: None, notes: vec!["заметка".into()] });
        assert_eq!(save(&root, Some(&dir), &d, "qwen3").unwrap(), dir);
        let md = std::fs::read_to_string(dir.join("dialog.md")).unwrap();
        assert!(md.contains("## Вы\n\nда\n\n## Ассистент\n\n## Итог\n\n> ⚠ заметка"), "{md}");
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!((meta["turns"].as_u64(), meta["rounds"].as_u64()), (Some(2), Some(9)));
    }
}
