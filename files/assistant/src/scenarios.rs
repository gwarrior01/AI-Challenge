//! Готовые цели — `files/scenarios/<имя>.md` (`FILES_SCENARIOS_DIR`): первая
//! строка `# Название`, дальше — сама цель, как её написал бы человек.
//! Один и тот же сценарий можно запускать сколько угодно раз —
//! `file-assistant run --scenario <имя>` или `/scenario <имя>` в TUI.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::session::env_or;

pub const DEFAULT_DIR: &str = "files/scenarios";

#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    /// Имя файла без `.md`.
    pub name: String,
    pub title: String,
    pub goal: String,
}

pub fn dir() -> PathBuf {
    PathBuf::from(env_or("FILES_SCENARIOS_DIR", DEFAULT_DIR))
}

pub fn parse(name: &str, text: &str) -> Scenario {
    let text = text.trim();
    let (title, goal) = match text.split_once('\n') {
        Some((first, rest)) if first.starts_with('#') => (first.trim_start_matches('#').trim().to_string(), rest.trim()),
        _ => (name.to_string(), text),
    };
    Scenario { name: name.to_string(), title, goal: goal.to_string() }
}

pub fn list_in(dir: &Path) -> Result<Vec<Scenario>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).with_context(|| format!("нет каталога сценариев {}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "md") {
            let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            out.push(parse(&name, &std::fs::read_to_string(&path)?));
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn list() -> Result<Vec<Scenario>> {
    list_in(&dir())
}

pub fn find(name: &str) -> Result<Scenario> {
    let all = list()?;
    match all.iter().find(|s| s.name == name) {
        Some(s) => Ok(s.clone()),
        None => {
            let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
            bail!("нет сценария «{name}» — есть: {}", names.join(", "))
        }
    }
}

/// `  usages      — Где используется McpManager` построчно.
pub fn listing(all: &[Scenario]) -> String {
    let width = all.iter().map(|s| s.name.chars().count()).max().unwrap_or(0);
    all.iter().map(|s| format!("  {:width$}  — {}", s.name, s.title)).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_title_and_goal() {
        let s = parse("usages", "# Где используется McpManager\n\nНайди все места…\nФайлы не меняй.\n");
        assert_eq!(s.title, "Где используется McpManager");
        assert_eq!(s.goal, "Найди все места…\nФайлы не меняй.");
        assert_eq!(parse("x", "просто цель").title, "x");
    }

    /// Сценарии репозитория читаются и у каждого есть цель.
    #[test]
    fn repository_scenarios_load() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../scenarios");
        let all = list_in(&dir).unwrap();
        assert!(all.len() >= 3, "{all:?}");
        assert!(all.iter().all(|s| !s.goal.is_empty() && s.title != s.name));
        assert!(listing(&all).contains(" — "));
    }
}
