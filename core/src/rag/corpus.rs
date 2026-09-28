//! Корпус документов: какие файлы индексировать и как получить их текст.
//!
//! Корпус — список путей (файлы и папки) относительно рабочей директории,
//! `RAG_CORPUS` через запятую. Папки обходятся рекурсивно; берутся `.md`,
//! `.txt`, `.rs` и `.pdf`, скрытые папки, `target` и `testdata` пропускаются.
//! PDF превращается в Markdown тем же конвертером, что у `documents-mcp`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use super::chunking::Kind;

/// Корпус по умолчанию пуст: в индексе только то, что добавлено явно
/// (загрузка в вебе, команда `add`) — папка загрузок индексируется всегда.
pub const DEFAULT_CORPUS: &str = "";

/// Корпус, на котором сравнивались стратегии (день 21): README, ядро и
/// собственные MCP-серверы, файлы инвариантов и профилей, входящие PDF.
/// Включается через `RAG_CORPUS`.
pub const PROJECT_CORPUS: &str = "README.md,core/src,mcp,invariants,profiles,documents/inbox";

const SKIP_DIRS: &[&str] = &["target", "testdata", "node_modules"];

/// Файл корпуса: `source` — путь относительно рабочей директории (через `/`),
/// он же идентификатор документа.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub source: String,
    pub path: PathBuf,
    pub kind: Kind,
    pub is_pdf: bool,
}

/// Содержимое файла, готовое к разбиению.
pub struct Loaded {
    pub text: String,
    pub title: String,
    /// SHA-256 байт файла — по нему видно, что документ изменился.
    pub sha256: String,
    pub bytes: u64,
    pub mtime: i64,
}

fn kind_of(path: &Path) -> Option<(Kind, bool)> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "md" | "markdown" => Some((Kind::Markdown, false)),
        "txt" => Some((Kind::Text, false)),
        "rs" => Some((Kind::Rust, false)),
        "pdf" => Some((Kind::Markdown, true)),
        _ => None,
    }
}

/// Файлы корпуса по списку путей. Несуществующий путь — не ошибка
/// (например, пустая папка входящих), он попадает в `missing`.
pub fn scan(base: &Path, roots: &[String]) -> (Vec<SourceFile>, Vec<String>) {
    let mut files = Vec::new();
    let mut missing = Vec::new();
    for root in roots {
        let path = base.join(root);
        if path.is_dir() {
            walk(base, &path, &mut files);
        } else if path.is_file() {
            push_file(base, &path, &mut files);
        } else {
            missing.push(root.clone());
        }
    }
    files.sort_by(|a, b| a.source.cmp(&b.source));
    files.dedup_by(|a, b| a.source == b.source);
    (files, missing)
}

fn walk(base: &Path, dir: &Path, files: &mut Vec<SourceFile>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                walk(base, &path, files);
            }
        } else {
            push_file(base, &path, files);
        }
    }
}

fn push_file(base: &Path, path: &Path, files: &mut Vec<SourceFile>) {
    let Some((kind, is_pdf)) = kind_of(path) else { return };
    let source = path.strip_prefix(base).unwrap_or(path).to_string_lossy().replace('\\', "/");
    files.push(SourceFile { source, path: path.to_path_buf(), kind, is_pdf });
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

pub async fn load(file: &SourceFile) -> Result<Loaded> {
    let bytes = tokio::fs::read(&file.path).await.with_context(|| format!("не удалось прочитать {}", file.source))?;
    let sha256 = sha256_hex(&bytes);
    let meta = std::fs::metadata(&file.path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let file_name = file.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| file.source.clone());
    let size = bytes.len() as u64;

    let (text, title) = if file.is_pdf {
        let (pages, _) = crate::pdf::extract_pages(&file.path, bytes).await?;
        if pages.iter().all(|p| p.trim().is_empty()) {
            bail!("{}: в PDF нет текстового слоя", file.source);
        }
        let stem = file.path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let converted = crate::pdf::pages_to_markdown(&pages, &stem);
        let title = converted.title.unwrap_or(file_name);
        (converted.markdown, title)
    } else {
        let text = String::from_utf8(bytes).with_context(|| format!("{}: файл не в UTF-8", file.source))?;
        let title = title_of(&text, file.kind).unwrap_or(file_name);
        (text, title)
    };
    Ok(Loaded { text, title, sha256, bytes: size, mtime })
}

/// Заголовок: первый `# ` в Markdown, первая строка `//!` в Rust.
fn title_of(text: &str, kind: Kind) -> Option<String> {
    let line = match kind {
        Kind::Markdown => text.lines().find_map(|l| l.strip_prefix("# ")),
        Kind::Rust => text.lines().find_map(|l| l.strip_prefix("//!")).filter(|l| !l.trim().is_empty()),
        Kind::Text => text.lines().find(|l| !l.trim().is_empty()),
    }?;
    let line = line.trim();
    Some(match line.char_indices().nth(100) {
        Some((i, _)) => format!("{}…", &line[..i]),
        None => line.to_string(),
    })
}

/// Версия файла в git: последний коммит, который его менял, и признак
/// «в рабочей копии есть незакоммиченные изменения». Вне репозитория (или
/// для игнорируемого файла) коммита нет.
#[derive(Debug, Clone, Default)]
pub struct GitInfo {
    pub commit: Option<String>,
    pub dirty: bool,
}

/// Изменённые и неотслеживаемые файлы рабочей копии (`git status`); `None`,
/// если это не репозиторий или git не установлен.
pub fn git_dirty_set(base: &Path) -> Option<std::collections::HashSet<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(base)
        .args(["status", "--porcelain", "--untracked-files=all", "--no-renames"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    // Пути в выводе — от корня репозитория, а source — от рабочей директории.
    let prefix = git_prefix(base).unwrap_or_default();
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.get(3..))
            .map(|p| p.trim_matches('"').to_string())
            .filter_map(|p| p.strip_prefix(&prefix).map(str::to_string))
            .collect(),
    )
}

fn git_prefix(base: &Path) -> Option<String> {
    let out = std::process::Command::new("git").arg("-C").arg(base).args(["rev-parse", "--show-prefix"]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn git_info(base: &Path, source: &str, dirty: Option<&std::collections::HashSet<String>>) -> GitInfo {
    let Some(dirty) = dirty else { return GitInfo::default() };
    let commit = std::process::Command::new("git")
        .arg("-C")
        .arg(base)
        .args(["log", "-1", "--format=%H", "--"])
        .arg(source)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|c| !c.is_empty());
    GitInfo { commit, dirty: dirty.contains(source) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_filters_extensions_and_skips_dirs() {
        let dir = std::env::temp_dir().join(format!("rag-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("docs/target")).unwrap();
        std::fs::create_dir_all(dir.join("docs/.hidden")).unwrap();
        std::fs::write(dir.join("docs/a.md"), "# Заголовок A\n\nтекст").unwrap();
        std::fs::write(dir.join("docs/b.rs"), "//! Модуль B.\nfn x() {}").unwrap();
        std::fs::write(dir.join("docs/c.java"), "class C {}").unwrap();
        std::fs::write(dir.join("docs/target/d.md"), "нет").unwrap();
        std::fs::write(dir.join("docs/.hidden/e.md"), "нет").unwrap();
        std::fs::write(dir.join("README.md"), "# R").unwrap();

        let roots = vec!["README.md".to_string(), "docs".to_string(), "nope".to_string()];
        let (files, missing) = scan(&dir, &roots);
        let sources: Vec<&str> = files.iter().map(|f| f.source.as_str()).collect();
        assert_eq!(sources, ["README.md", "docs/a.md", "docs/b.rs"]);
        assert_eq!(missing, ["nope"]);
        assert_eq!(files[2].kind, Kind::Rust);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn titles() {
        assert_eq!(title_of("текст\n# Главный\n## Второй", Kind::Markdown).as_deref(), Some("Главный"));
        assert_eq!(title_of("//! Клиент MCP.\n//! ещё", Kind::Rust).as_deref(), Some("Клиент MCP."));
        assert_eq!(title_of("fn x() {}", Kind::Rust), None);
    }
}
