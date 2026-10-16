//! Запуски сохраняются, чтобы результат можно было проверить и повторить:
//!
//! ```text
//! files/runs/<UTC-время>-<сценарий|цель>/
//!   goal.md        — цель (и сценарий, если из него)
//!   report.md      — отчёт модели
//!   changes.patch  — дифф ожидающих изменений: git apply files/runs/<id>/changes.patch
//!   tools.json     — все вызовы инструментов с доводами и результатами
//!   run.json       — модель, температура, шаги, токены, файлы; после /apply — что записано
//! ```
//!
//! Каталог — `FILES_RUNS_DIR` (по умолчанию `files/runs`, в `.gitignore`).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::json;

use crate::session::{env_or, Applied};
use crate::task::Outcome;

pub const DEFAULT_RUNS_DIR: &str = "files/runs";

pub fn runs_dir() -> PathBuf {
    PathBuf::from(env_or("FILES_RUNS_DIR", DEFAULT_RUNS_DIR))
}

/// Сохранить запуск в `root`; возвращает его каталог.
pub fn save(root: &Path, outcome: &Outcome) -> Result<PathBuf> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let label = outcome.scenario.clone().unwrap_or_else(|| slug(&outcome.goal));
    let mut dir = root.join(format!("{}-{label}", stamp(now)));
    // Два запуска в одну секунду — второй с суффиксом.
    let mut n = 2;
    while dir.exists() {
        dir = root.join(format!("{}-{label}-{n}", stamp(now)));
        n += 1;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("не удалось создать {}", dir.display()))?;

    let scenario = outcome.scenario.as_deref().map(|s| format!("Сценарий: `{s}`\n\n")).unwrap_or_default();
    write(&dir, "goal.md", &format!("# Цель\n\n{scenario}{}\n", outcome.goal))?;
    let mut report = outcome.report.clone();
    for note in &outcome.notes {
        report.push_str(&format!("\n\n> ⚠ {note}"));
    }
    write(&dir, "report.md", &format!("{report}\n"))?;
    if !outcome.pending.is_empty() {
        write(&dir, "changes.patch", &outcome.pending.patch)?;
    }
    write(&dir, "tools.json", &serde_json::to_string_pretty(&outcome.tool_calls)?)?;
    let meta = json!({
        "started_utc": iso(now),
        "goal": outcome.goal,
        "scenario": outcome.scenario,
        "model": outcome.model,
        "temperature": outcome.temperature,
        "rounds": outcome.rounds,
        "hit_limit": outcome.hit_limit,
        "tool_calls": outcome.tool_calls.len(),
        "touched_files": outcome.touched_files(),
        "changes": outcome.pending.files,
        "usage": outcome.usage,
        "notes": outcome.notes,
        "applied": null,
    });
    write(&dir, "run.json", &serde_json::to_string_pretty(&meta)?)?;
    Ok(dir)
}

/// Отметить в `run.json`, что из изменений записано на диск.
pub fn mark_applied(dir: &Path, applied: &Applied) -> Result<()> {
    let path = dir.join("run.json");
    let mut meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let conflicts: Vec<_> = applied.conflicts.iter().map(|c| json!({ "path": c.path, "reason": c.reason })).collect();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    meta["applied"] = json!({ "at_utc": iso(now), "written": applied.written, "conflicts": conflicts });
    write(dir, "run.json", &serde_json::to_string_pretty(&meta)?)
}

fn write(dir: &Path, name: &str, text: &str) -> Result<()> {
    std::fs::write(dir.join(name), text).with_context(|| format!("не удалось записать {}", dir.join(name).display()))
}

/// Начало цели для имени каталога: `найди-все-места-где`.
fn slug(goal: &str) -> String {
    let words: Vec<String> = goal
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(5)
        .map(|w| w.to_lowercase())
        .collect();
    let s = words.join("-");
    let s: String = s.chars().take(40).collect();
    if s.is_empty() {
        "task".into()
    } else {
        s.trim_end_matches('-').to_string()
    }
}

/// Дата и время UTC из секунд Unix: (год, месяц, день, час, минута, секунда).
fn civil(secs: u64) -> (i64, u32, u32, u64, u64, u64) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Алгоритм Говарда Хиннанта: дни от 1970-01-01 → григорианская дата.
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

/// `20261016-142501` — для имени каталога.
fn stamp(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// `2026-10-16T14:25:01Z`.
fn iso(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Pending, PendingFile};

    #[test]
    fn dates_and_slugs() {
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(1_792_160_701), "2026-10-16T14:25:01Z");
        assert_eq!(stamp(951_782_400), "20000229-000000");
        assert_eq!(slug("Найди все места, где используется `McpManager`!"), "найди-все-места-где-используется");
        assert_eq!(slug("???"), "task");
    }

    #[test]
    fn saves_a_run() {
        let root = std::env::temp_dir().join(format!("file-assistant-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let outcome = Outcome {
            goal: "Сгенерируй CHANGELOG".into(),
            scenario: Some("changelog".into()),
            report: "## Итог\nГотово.".into(),
            pending: Pending {
                files: vec![PendingFile { path: "CHANGELOG.md".into(), status: "created".into(), added: 3, deleted: 0 }],
                added: 3,
                deleted: 0,
                patch: "diff --git a/CHANGELOG.md b/CHANGELOG.md\n".into(),
            },
            tool_calls: vec![],
            rounds: 4,
            hit_limit: false,
            model: "m".into(),
            temperature: 0.0,
            usage: None,
            notes: vec!["git недоступен".into()],
        };
        let dir = save(&root, &outcome).unwrap();
        assert!(dir.file_name().unwrap().to_string_lossy().ends_with("-changelog"));
        assert!(std::fs::read_to_string(dir.join("changes.patch")).unwrap().starts_with("diff --git"));
        assert!(std::fs::read_to_string(dir.join("report.md")).unwrap().contains("> ⚠ git недоступен"));
        // Второй запуск в ту же секунду — другой каталог.
        assert_ne!(save(&root, &outcome).unwrap(), dir);

        let applied = Applied { written: vec!["CHANGELOG.md".into()], conflicts: vec![] };
        mark_applied(&dir, &applied).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(meta["applied"]["written"][0], "CHANGELOG.md");
        assert_eq!(meta["scenario"], "changelog");
    }
}
