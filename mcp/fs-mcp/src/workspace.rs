//! Файлы проекта: чтение, поиск и изменения через оверлей.
//!
//! **Песочница.** Все пути — относительно корня проекта; `..`, абсолютные
//! пути вне корня и симлинки наружу отклоняются ([`Workspace::resolve`]).
//! Закрыто всегда, и на чтение, и на запись: `.git/`, `target/`,
//! `node_modules/`, `.env` и `.env.*` (кроме `.env.example`), базы и ключи
//! (`*.db`, `*.pem`, …) — см. [`is_denied`]. Обход каталогов учитывает
//! `.gitignore`.
//!
//! **Оверлей.** `write`/`edit` не трогают диск: новое содержимое ложится в
//! набор ожидающих изменений вместе с тем, что было на диске в момент первой
//! правки. Чтение и поиск идут через оверлей — модель видит свои правки.
//! [`Workspace::pending`] — unified diff всех изменений (годится для
//! `git apply`), [`Workspace::apply`] — запись на диск, но только если файл на
//! диске с тех пор не изменился (иначе — конфликт, файл не трогается),
//! [`Workspace::discard`] — отказ от изменений. Удалять файлы нельзя вовсе.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use globset::{Glob, GlobBuilder, GlobSet, GlobSetBuilder};
use rmcp::schemars::{self, JsonSchema};
use serde::Serialize;

/// Больше этого файл не читается и не пишется.
pub const MAX_FILE_BYTES: usize = 1_000_000;
/// Сколько файлов может быть в оверлее одновременно.
pub const MAX_PENDING_FILES: usize = 60;
/// Длина строки совпадения в выдаче поиска.
const MAX_MATCH_CHARS: usize = 300;
/// Сколько файлов со счётчиками совпадений возвращает поиск.
const MAX_MATCH_FILES: usize = 200;

const DENY_DIRS: &[&str] = &[".git", "target", "node_modules"];
const DENY_EXT: &[&str] = &["db", "db-shm", "db-wal", "db-journal", "sqlite", "sqlite3", "pem", "key", "p12"];

/// Путь закрыт для сервера — см. документацию модуля.
pub fn is_denied(rel: &str) -> bool {
    let mut parts = rel.split('/');
    let name = rel.rsplit('/').next().unwrap_or(rel);
    let env = name == ".env" || (name.starts_with(".env.") && name != ".env.example");
    let ext = name.rsplit_once('.').is_some_and(|(stem, ext)| !stem.is_empty() && DENY_EXT.contains(&ext));
    parts.any(|part| DENY_DIRS.contains(&part)) || env || ext
}

/// Путь от модели → нормализованный путь относительно корня (`core/src/lib.rs`).
/// Пустая строка — сам корень.
pub fn normalize(root: &Path, raw: &str) -> Result<String> {
    let raw = raw.trim();
    let raw = match Path::new(raw).strip_prefix(root) {
        Ok(inside) => inside.to_string_lossy().into_owned(),
        Err(_) if raw.starts_with('/') => bail!("путь «{raw}» вне проекта — пути относительно его корня"),
        Err(_) => raw.to_string(),
    };
    let mut parts = Vec::new();
    for part in raw.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => bail!("путь «{raw}» выходит за корень проекта"),
            p => parts.push(p),
        }
    }
    Ok(parts.join("/"))
}

#[derive(Debug, Clone)]
struct Change {
    /// Что было на диске в момент первой правки; `None` — файла не было.
    original: Option<String>,
    content: String,
}

pub struct Workspace {
    root: PathBuf,
    overlay: Mutex<BTreeMap<String, Change>>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FileText {
    pub path: String,
    /// Номер первой возвращённой строки (с 1).
    pub start_line: usize,
    /// Номер последней возвращённой строки.
    pub end_line: usize,
    pub total_lines: usize,
    /// Строки файла как есть, без номеров.
    pub text: String,
    /// Возвращены не все строки — остальное через `start_line`.
    pub truncated: bool,
    /// В файле есть ещё не записанные на диск изменения: показан текст с ними.
    pub pending: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct DirEntry {
    /// Каталог (с `/` на конце) или файл.
    pub path: String,
    /// Сколько файлов внутри (у файла — 1).
    pub files: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FileList {
    /// Сколько файлов подошло всего.
    pub total: usize,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    /// С `depth` — пути, свёрнутые до этой глубины, с числом файлов.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tree: Vec<DirEntry>,
}

pub struct ListQuery {
    pub path: Option<String>,
    pub pattern: Option<String>,
    pub depth: Option<usize>,
    pub limit: usize,
}

pub struct SearchQuery {
    pub query: String,
    pub regex: bool,
    pub case_sensitive: bool,
    pub path: Option<String>,
    pub pattern: Option<String>,
    pub context: usize,
    pub max_results: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Match {
    pub path: String,
    pub line: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub before: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub after: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FileCount {
    pub path: String,
    pub matches: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResult {
    /// Совпадений всего (во всех файлах, даже если `matches` обрезан).
    pub total_matches: usize,
    /// Файлы с совпадениями и сколько их в каждом — полная картина, даже когда `matches` обрезан.
    pub files: Vec<FileCount>,
    /// Первые совпадения со строкой и контекстом.
    pub matches: Vec<Match>,
    pub truncated: bool,
    pub files_scanned: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct WriteResult {
    pub path: String,
    /// Файла раньше не было.
    pub created: bool,
    /// Сколько замен сделано (`edit_file`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacements: Option<usize>,
    pub lines: usize,
    /// Всего строк добавлено и удалено в этом файле относительно диска.
    pub added: usize,
    pub deleted: usize,
    /// Дифф файла относительно диска (обрезан до нескольких тысяч символов).
    pub diff: String,
    /// Изменения пока только в оверлее, на диск не записаны.
    pub note: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct PendingFile {
    pub path: String,
    /// `created` или `modified`.
    pub status: String,
    pub added: usize,
    pub deleted: usize,
}

#[derive(Debug, Default, Serialize, JsonSchema)]
pub struct Pending {
    pub files: Vec<PendingFile>,
    pub added: usize,
    pub deleted: usize,
    /// Unified diff всех изменений в формате git (`git apply`).
    pub patch: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Conflict {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ApplyResult {
    pub written: Vec<String>,
    /// Файлы, изменившиеся на диске после первой правки, — не записаны, остаются в оверлее.
    pub conflicts: Vec<Conflict>,
}

impl Workspace {
    pub fn open(root: &Path) -> Result<Self> {
        let root = root.canonicalize().with_context(|| format!("нет каталога {}", root.display()))?;
        if !root.is_dir() {
            bail!("{} — не каталог", root.display());
        }
        Ok(Self { root, overlay: Mutex::new(BTreeMap::new()) })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Путь от модели → (нормализованный относительный, абсолютный); закрытые
    /// пути и выход за корень (в том числе симлинком) — ошибка.
    pub fn resolve(&self, raw: &str) -> Result<(String, PathBuf)> {
        let rel = normalize(&self.root, raw)?;
        if is_denied(&rel) {
            bail!("«{rel}» закрыт для ассистента (служебные каталоги, .env, базы и ключи)");
        }
        let abs = self.root.join(&rel);
        // Ближайший существующий предок должен остаться внутри корня после разворота симлинков.
        let mut probe = abs.as_path();
        loop {
            if probe.exists() {
                let real = probe.canonicalize()?;
                if !real.starts_with(&self.root) {
                    bail!("«{rel}» ведёт за пределы проекта (симлинк)");
                }
                break;
            }
            match probe.parent() {
                Some(parent) => probe = parent,
                None => break,
            }
        }
        Ok((rel, abs))
    }

    fn disk_text(rel: &str, abs: &Path) -> Result<Option<String>> {
        if !abs.exists() {
            return Ok(None);
        }
        if abs.is_dir() {
            bail!("«{rel}» — каталог, а не файл (list_files покажет, что в нём)");
        }
        let size = abs.metadata()?.len() as usize;
        if size > MAX_FILE_BYTES {
            bail!("«{rel}» слишком большой ({size} байт, предел {MAX_FILE_BYTES})");
        }
        let bytes = std::fs::read(abs).with_context(|| format!("не удалось прочитать {rel}"))?;
        if bytes.contains(&0) {
            bail!("«{rel}» — двоичный файл");
        }
        String::from_utf8(bytes).map(Some).map_err(|_| anyhow::anyhow!("«{rel}» не в UTF-8 — похоже на двоичный файл"))
    }

    /// Текущий текст файла — с изменениями из оверлея; `None` — файла нет.
    fn current(&self, rel: &str, abs: &Path) -> Result<(Option<String>, bool)> {
        if let Some(change) = self.overlay.lock().unwrap().get(rel) {
            return Ok((Some(change.content.clone()), true));
        }
        Ok((Self::disk_text(rel, abs)?, false))
    }

    pub fn read(&self, raw: &str, start_line: usize, max_lines: usize) -> Result<FileText> {
        let (rel, abs) = self.resolve(raw)?;
        let (text, pending) = self.current(&rel, &abs)?;
        let Some(text) = text else { bail!("файла «{rel}» нет (list_files или search помогут найти нужный)") };
        let lines: Vec<&str> = text.lines().collect();
        let start = start_line.max(1);
        if start > lines.len().max(1) {
            bail!("в «{rel}» всего {} строк", lines.len());
        }
        let end = (start - 1 + max_lines).min(lines.len());
        Ok(FileText {
            path: rel,
            start_line: start,
            end_line: end,
            total_lines: lines.len(),
            text: lines.get(start - 1..end).unwrap_or_default().join("\n"),
            truncated: end < lines.len(),
            pending,
        })
    }

    /// Все файлы под `base` (относительный путь, пустой — корень), с учётом
    /// `.gitignore` и закрытых путей, плюс новые файлы из оверлея. Отсортированы.
    fn walk(&self, base: &str) -> Result<Vec<String>> {
        let start = self.root.join(base);
        let mut out = Vec::new();
        if start.is_file() {
            out.push(base.to_string());
        } else if start.is_dir() {
            let walker = ignore::WalkBuilder::new(&start)
                .hidden(false)
                .git_global(false)
                .require_git(false)
                .filter_entry(|e| !DENY_DIRS.contains(&e.file_name().to_string_lossy().as_ref()))
                .build();
            for entry in walker.flatten() {
                if !entry.file_type().is_some_and(|t| t.is_file()) {
                    continue;
                }
                let Ok(rel) = entry.path().strip_prefix(&self.root) else { continue };
                let rel = rel.to_string_lossy().replace('\\', "/");
                if !is_denied(&rel) {
                    out.push(rel);
                }
            }
        } else if !self.overlay.lock().unwrap().keys().any(|k| under(k, base)) {
            bail!("«{base}» не найден");
        }
        for (path, change) in self.overlay.lock().unwrap().iter() {
            if change.original.is_none() && under(path, base) && !out.contains(path) {
                out.push(path.clone());
            }
        }
        out.sort();
        Ok(out)
    }

    pub fn list(&self, q: &ListQuery) -> Result<FileList> {
        let base = match &q.path {
            Some(p) => self.resolve(p)?.0,
            None => String::new(),
        };
        let filter = q.pattern.as_deref().map(Pattern::new).transpose()?;
        let files: Vec<String> =
            self.walk(&base)?.into_iter().filter(|f| filter.as_ref().is_none_or(|p| p.matches(f))).collect();
        let total = files.len();
        let Some(depth) = q.depth.filter(|d| *d > 0) else {
            let truncated = total > q.limit;
            return Ok(FileList { total, truncated, files: files.into_iter().take(q.limit).collect(), tree: Vec::new() });
        };
        // Свернуть до глубины `depth` под `base`.
        let skip = if base.is_empty() { 0 } else { base.split('/').count() };
        let mut tree: BTreeMap<String, usize> = BTreeMap::new();
        for f in &files {
            let parts: Vec<&str> = f.split('/').collect();
            let key = if parts.len() > skip + depth {
                format!("{}/", parts[..skip + depth].join("/"))
            } else {
                f.clone()
            };
            *tree.entry(key).or_default() += 1;
        }
        let truncated = tree.len() > q.limit;
        let tree = tree.into_iter().take(q.limit).map(|(path, files)| DirEntry { path, files }).collect();
        Ok(FileList { total, truncated, files: Vec::new(), tree })
    }

    pub fn search(&self, q: &SearchQuery) -> Result<SearchResult> {
        if q.query.is_empty() {
            bail!("пустой запрос поиска");
        }
        let pattern = if q.regex { q.query.clone() } else { regex::escape(&q.query) };
        let re = regex::RegexBuilder::new(&pattern)
            .case_insensitive(!q.case_sensitive)
            .size_limit(1 << 20)
            .build()
            .with_context(|| format!("неверное регулярное выражение «{}»", q.query))?;
        let base = match &q.path {
            Some(p) => self.resolve(p)?.0,
            None => String::new(),
        };
        let filter = q.pattern.as_deref().map(Pattern::new).transpose()?;
        let mut result =
            SearchResult { total_matches: 0, files: Vec::new(), matches: Vec::new(), truncated: false, files_scanned: 0 };
        for rel in self.walk(&base)? {
            if filter.as_ref().is_some_and(|p| !p.matches(&rel)) {
                continue;
            }
            let abs = self.root.join(&rel);
            // Двоичные и слишком большие файлы поиск просто пропускает.
            let Ok((Some(text), _)) = self.current(&rel, &abs) else { continue };
            result.files_scanned += 1;
            let lines: Vec<&str> = text.lines().collect();
            let mut count = 0;
            for (i, line) in lines.iter().enumerate() {
                if !re.is_match(line) {
                    continue;
                }
                count += 1;
                if result.matches.len() >= q.max_results {
                    result.truncated = true;
                    continue;
                }
                let ctx = |range: std::ops::Range<usize>| lines[range].iter().map(|l| cut(l)).collect();
                result.matches.push(Match {
                    path: rel.clone(),
                    line: i + 1,
                    text: cut(line),
                    before: ctx(i.saturating_sub(q.context)..i),
                    after: ctx(i + 1..(i + 1 + q.context).min(lines.len())),
                });
            }
            if count > 0 {
                result.total_matches += count;
                if result.files.len() < MAX_MATCH_FILES {
                    result.files.push(FileCount { path: rel, matches: count });
                } else {
                    result.truncated = true;
                }
            }
        }
        Ok(result)
    }

    /// Положить новое содержимое файла в оверлей.
    fn stage(&self, rel: &str, abs: &Path, content: String) -> Result<(bool, Change)> {
        if content.len() > MAX_FILE_BYTES {
            bail!("слишком большое содержимое ({} байт, предел {MAX_FILE_BYTES})", content.len());
        }
        let mut overlay = self.overlay.lock().unwrap();
        let change = match overlay.get(rel) {
            Some(existing) => Change { original: existing.original.clone(), content },
            None => {
                if overlay.len() >= MAX_PENDING_FILES {
                    bail!("уже изменено {MAX_PENDING_FILES} файлов — сначала примените или отмените изменения");
                }
                Change { original: Self::disk_text(rel, abs)?, content }
            }
        };
        let created = change.original.is_none();
        if change.original.as_deref() == Some(change.content.as_str()) {
            // Правка вернула файл к тому, что на диске, — изменений больше нет.
            overlay.remove(rel);
        } else {
            overlay.insert(rel.to_string(), change.clone());
        }
        Ok((created, change))
    }

    fn write_result(rel: String, created: bool, change: &Change, replacements: Option<usize>) -> WriteResult {
        let (patch, added, deleted) = file_patch(&rel, change.original.as_deref(), &change.content);
        WriteResult {
            lines: change.content.lines().count(),
            diff: cut_chars(&patch, 4000),
            path: rel,
            created,
            replacements,
            added,
            deleted,
            note: "изменение в оверлее: на диск запишется после подтверждения пользователя".into(),
        }
    }

    pub fn write(&self, raw: &str, content: &str) -> Result<WriteResult> {
        let (rel, abs) = self.resolve(raw)?;
        if rel.is_empty() || abs.is_dir() {
            bail!("«{raw}» — каталог, нужен путь к файлу");
        }
        let (created, change) = self.stage(&rel, &abs, content.to_string())?;
        Ok(Self::write_result(rel, created, &change, None))
    }

    pub fn edit(&self, raw: &str, old_text: &str, new_text: &str, replace_all: bool) -> Result<WriteResult> {
        let (rel, abs) = self.resolve(raw)?;
        if old_text.is_empty() {
            bail!("old_text пустой — новый файл создаётся write_file");
        }
        let (text, _) = self.current(&rel, &abs)?;
        let Some(text) = text else { bail!("файла «{rel}» нет — новый файл создаётся write_file") };
        let positions: Vec<usize> = text.match_indices(old_text).map(|(i, _)| i).collect();
        match positions.len() {
            0 => bail!(
                "в «{rel}» нет такого фрагмента old_text — он должен совпадать с текстом файла символ в символ, \
                 с пробелами и отступами; перечитай нужное место read_file"
            ),
            n if n > 1 && !replace_all => {
                let lines: Vec<String> = positions.iter().map(|&i| (text[..i].lines().count().max(1)).to_string()).collect();
                bail!(
                    "фрагмент old_text встречается в «{rel}» {n} раз (строки {}) — добавь в него соседние строки, \
                     чтобы он стал уникальным, или replace_all=true, если заменить нужно все",
                    lines.join(", ")
                )
            }
            _ => {}
        }
        let content = text.replace(old_text, new_text);
        let (created, change) = self.stage(&rel, &abs, content)?;
        Ok(Self::write_result(rel, created, &change, Some(positions.len())))
    }

    pub fn pending(&self) -> Pending {
        let overlay = self.overlay.lock().unwrap();
        let mut out = Pending::default();
        for (path, change) in overlay.iter() {
            let (patch, added, deleted) = file_patch(path, change.original.as_deref(), &change.content);
            let status = if change.original.is_none() { "created" } else { "modified" };
            out.files.push(PendingFile { path: path.clone(), status: status.into(), added, deleted });
            out.added += added;
            out.deleted += deleted;
            out.patch.push_str(&patch);
        }
        out
    }

    /// Пути из `paths` (нормализованные), которые есть в оверлее; `None` — все.
    fn selected(&self, paths: Option<&[String]>) -> Result<Vec<String>> {
        let overlay = self.overlay.lock().unwrap();
        match paths {
            None => Ok(overlay.keys().cloned().collect()),
            Some(paths) => {
                let mut out = Vec::new();
                for p in paths {
                    let rel = normalize(&self.root, p)?;
                    let hits: Vec<String> = overlay.keys().filter(|k| under(k, &rel)).cloned().collect();
                    if hits.is_empty() {
                        bail!("у «{rel}» нет ожидающих изменений");
                    }
                    out.extend(hits);
                }
                out.dedup();
                Ok(out)
            }
        }
    }

    pub fn apply(&self, paths: Option<&[String]>) -> Result<ApplyResult> {
        let mut result = ApplyResult { written: Vec::new(), conflicts: Vec::new() };
        for rel in self.selected(paths)? {
            let (_, abs) = self.resolve(&rel)?;
            let change = self.overlay.lock().unwrap().get(&rel).cloned();
            let Some(change) = change else { continue };
            let now = match Self::disk_text(&rel, &abs) {
                Ok(now) => now,
                Err(err) => {
                    result.conflicts.push(Conflict { path: rel, reason: format!("{err:#}") });
                    continue;
                }
            };
            if now != change.original {
                let reason = match now {
                    None => "файл удалили с диска после первой правки".to_string(),
                    Some(_) if change.original.is_none() => "файл появился на диске после первой правки".to_string(),
                    Some(_) => "файл изменился на диске после первой правки".to_string(),
                };
                result.conflicts.push(Conflict { path: rel, reason });
                continue;
            }
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent).with_context(|| format!("не удалось создать каталог для {rel}"))?;
            }
            std::fs::write(&abs, &change.content).with_context(|| format!("не удалось записать {rel}"))?;
            self.overlay.lock().unwrap().remove(&rel);
            result.written.push(rel);
        }
        Ok(result)
    }

    pub fn discard(&self, paths: Option<&[String]>) -> Result<Vec<String>> {
        let selected = self.selected(paths)?;
        let mut overlay = self.overlay.lock().unwrap();
        for rel in &selected {
            overlay.remove(rel);
        }
        Ok(selected)
    }
}

/// `path` — это `base` или лежит под ним (`base` пустой — корень).
fn under(path: &str, base: &str) -> bool {
    base.is_empty() || path == base || path.strip_prefix(base).is_some_and(|rest| rest.starts_with('/'))
}

/// Шаблоны через запятую: `*.rs`, `*.md,*.toml`; с `/` — по всему пути
/// (`mcp/*/Cargo.toml`, `docs/**/*.md`), без — по имени файла.
struct Pattern {
    by_name: GlobSet,
    by_path: GlobSet,
}

impl Pattern {
    fn new(raw: &str) -> Result<Self> {
        let (mut by_name, mut by_path) = (GlobSetBuilder::new(), GlobSetBuilder::new());
        for p in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if p.contains('/') {
                let glob = GlobBuilder::new(p).literal_separator(true).build();
                by_path.add(glob.with_context(|| format!("неверный шаблон «{p}»"))?);
            } else {
                by_name.add(Glob::new(p).with_context(|| format!("неверный шаблон «{p}»"))?);
            }
        }
        Ok(Self { by_name: by_name.build()?, by_path: by_path.build()? })
    }

    fn matches(&self, rel: &str) -> bool {
        let name = rel.rsplit('/').next().unwrap_or(rel);
        self.by_name.is_match(name) || self.by_path.is_match(rel)
    }
}

/// Дифф одного файла в формате git и число добавленных/удалённых строк.
fn file_patch(path: &str, original: Option<&str>, content: &str) -> (String, usize, usize) {
    let old = original.unwrap_or("");
    let diff = similar::TextDiff::from_lines(old, content);
    let (mut added, mut deleted) = (0, 0);
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => added += 1,
            similar::ChangeTag::Delete => deleted += 1,
            similar::ChangeTag::Equal => {}
        }
    }
    let from = if original.is_some() { format!("a/{path}") } else { "/dev/null".to_string() };
    let body = diff.unified_diff().context_radius(3).header(&from, &format!("b/{path}")).to_string();
    let mut patch = format!("diff --git a/{path} b/{path}\n");
    if original.is_none() {
        patch.push_str("new file mode 100644\n");
    }
    patch.push_str(&body);
    if !patch.ends_with('\n') {
        patch.push('\n');
    }
    (patch, added, deleted)
}

fn cut(line: &str) -> String {
    cut_chars(line.trim_end(), MAX_MATCH_CHARS)
}

fn cut_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(name: &str) -> (PathBuf, Workspace) {
        let dir = std::env::temp_dir().join(format!("fs-mcp-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("docs")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "use crate::api::Client;\n\nfn main() {\n    Client::new();\n}\n").unwrap();
        std::fs::write(dir.join("src/api.rs"), "pub struct Client;\nimpl Client {\n    pub fn new() -> Self { Client }\n}\n").unwrap();
        std::fs::write(dir.join("docs/api.md"), "# API\n\nКлиент: `Client`.\n").unwrap();
        std::fs::write(dir.join(".env"), "LLM_API_KEY=secret\n").unwrap();
        let ws = Workspace::open(&dir).unwrap();
        (dir, ws)
    }

    fn search(ws: &Workspace, query: &str) -> SearchResult {
        ws.search(&SearchQuery {
            query: query.into(),
            regex: false,
            case_sensitive: true,
            path: None,
            pattern: None,
            context: 0,
            max_results: 50,
        })
        .unwrap()
    }

    #[test]
    fn denies_secrets_and_escapes() {
        assert!(is_denied(".env"));
        assert!(is_denied("web/.env.local"));
        assert!(!is_denied(".env.example"));
        assert!(is_denied("target/debug/x"));
        assert!(is_denied(".git/config"));
        assert!(is_denied("agents.db"));
        assert!(is_denied("scheduler.db-wal"));
        assert!(!is_denied("docs/db.md"));
        let (dir, ws) = sandbox("deny");
        assert!(ws.read(".env", 1, 10).is_err());
        assert!(ws.read("../etc/passwd", 1, 10).is_err());
        assert!(ws.read("/etc/passwd", 1, 10).is_err());
        assert!(ws.write(".env", "x").is_err());
        // Абсолютный путь внутри проекта — можно.
        assert_eq!(ws.read(&ws.root().join("src/api.rs").to_string_lossy(), 1, 1).unwrap().path, "src/api.rs");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", dir.join("etc")).unwrap();
            assert!(ws.read("etc/hosts", 1, 1).unwrap_err().to_string().contains("симлинк"));
        }
        let listed = ws.list(&ListQuery { path: None, pattern: None, depth: None, limit: 100 }).unwrap();
        assert!(!listed.files.iter().any(|f| f == ".env"));
    }

    #[test]
    fn reads_lists_and_searches() {
        let (_dir, ws) = sandbox("read");
        let f = ws.read("src/main.rs", 3, 2).unwrap();
        assert_eq!((f.start_line, f.end_line, f.total_lines, f.truncated), (3, 4, 5, true));
        assert_eq!(f.text, "fn main() {\n    Client::new();");

        let rs = ws.list(&ListQuery { path: None, pattern: Some("*.rs".into()), depth: None, limit: 100 }).unwrap();
        assert_eq!(rs.files, vec!["src/api.rs", "src/main.rs"]);
        let tree = ws.list(&ListQuery { path: None, pattern: None, depth: Some(1), limit: 100 }).unwrap();
        let paths: Vec<_> = tree.tree.iter().map(|d| (d.path.as_str(), d.files)).collect();
        assert_eq!(paths, vec![("docs/", 1), ("src/", 2)]);

        let found = search(&ws, "Client");
        assert_eq!(found.total_matches, 6);
        assert_eq!(found.files.iter().map(|f| f.matches).collect::<Vec<_>>(), vec![1, 3, 2]);
        let regex = ws
            .search(&SearchQuery {
                query: r"Client::\w+".into(),
                regex: true,
                case_sensitive: true,
                path: Some("src".into()),
                pattern: Some("*.rs".into()),
                context: 1,
                max_results: 1,
            })
            .unwrap();
        assert_eq!(regex.matches[0].line, 4);
        assert_eq!(regex.matches[0].before, vec!["fn main() {"]);
        assert!(!regex.truncated);
    }

    #[test]
    fn edits_go_to_overlay_until_applied() {
        let (dir, ws) = sandbox("overlay");
        // Фрагмент встречается дважды — без replace_all нельзя.
        let err = ws.edit("src/api.rs", "Client", "Api", false).unwrap_err().to_string();
        assert!(err.contains("3 раз"), "{err}");
        assert!(ws.edit("src/api.rs", "нет такого", "x", false).is_err());

        let r = ws.edit("src/api.rs", "pub fn new()", "pub fn create()", false).unwrap();
        assert_eq!((r.added, r.deleted, r.replacements), (1, 1, Some(1)));
        assert!(r.diff.contains("+    pub fn create()"));
        // На диске по-прежнему старое, а чтение и поиск видят новое.
        assert!(std::fs::read_to_string(dir.join("src/api.rs")).unwrap().contains("pub fn new()"));
        assert!(ws.read("src/api.rs", 1, 10).unwrap().pending);
        assert_eq!(search(&ws, "create").total_matches, 1);

        let w = ws.write("docs/CHANGELOG.md", "# Changelog\n\n- create()\n").unwrap();
        assert!(w.created);
        assert!(ws.list(&ListQuery { path: Some("docs".into()), pattern: None, depth: None, limit: 10 })
            .unwrap()
            .files
            .contains(&"docs/CHANGELOG.md".to_string()));

        let pending = ws.pending();
        assert_eq!(pending.files.len(), 2);
        assert_eq!(pending.files[0].status, "created");
        assert_eq!(pending.files[1].path, "src/api.rs");
        assert!(pending.patch.contains("--- /dev/null\n+++ b/docs/CHANGELOG.md"));
        assert!(pending.patch.contains("--- a/src/api.rs\n+++ b/src/api.rs"));

        // Правка обратно к диску — изменения исчезают.
        ws.edit("src/api.rs", "pub fn create()", "pub fn new()", false).unwrap();
        assert_eq!(ws.pending().files.len(), 1);

        let applied = ws.apply(None).unwrap();
        assert_eq!(applied.written, vec!["docs/CHANGELOG.md"]);
        assert!(std::fs::read_to_string(dir.join("docs/CHANGELOG.md")).unwrap().starts_with("# Changelog"));
        assert!(ws.pending().files.is_empty());
    }

    #[test]
    fn conflicting_files_are_not_overwritten() {
        let (dir, ws) = sandbox("conflict");
        ws.edit("docs/api.md", "# API", "# API клиента", false).unwrap();
        ws.edit("src/main.rs", "fn main()", "pub fn main()", false).unwrap();
        std::fs::write(dir.join("docs/api.md"), "# Кто-то поправил\n").unwrap();
        let applied = ws.apply(None).unwrap();
        assert_eq!(applied.written, vec!["src/main.rs"]);
        assert_eq!(applied.conflicts[0].path, "docs/api.md");
        assert_eq!(std::fs::read_to_string(dir.join("docs/api.md")).unwrap(), "# Кто-то поправил\n");
        // Конфликтный файл остаётся в оверлее — от него можно отказаться.
        assert_eq!(ws.discard(Some(&["docs".into()])).unwrap(), vec!["docs/api.md"]);
        assert!(ws.pending().files.is_empty());
        assert!(ws.discard(Some(&["src/main.rs".into()])).is_err());
    }

    /// Патч из `pending` применяется настоящим `git apply`.
    #[test]
    fn patch_applies_with_git() {
        let (dir, ws) = sandbox("gitapply");
        ws.edit("src/main.rs", "    Client::new();\n", "    let _c = Client::new();\n", false).unwrap();
        ws.write("NOTES.md", "без перевода строки в конце").unwrap();
        let patch = dir.join("changes.patch");
        std::fs::write(&patch, ws.pending().patch).unwrap();
        let out = std::process::Command::new("git").arg("-C").arg(&dir).args(["apply", "--check", "changes.patch"]).output();
        match out {
            Ok(out) => assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr)),
            Err(_) => eprintln!("git не найден — проверка git apply пропущена"),
        }
    }
}
