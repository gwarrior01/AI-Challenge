//! Чтение состояния git-репозитория: только команды, которые ничего не
//! меняют (`symbolic-ref`, `status`, `ls-files`, `diff`, `log`, …). Git
//! запускается напрямую, без оболочки; ссылки и пути от модели проверяются
//! ([`check_ref`]) и отделяются `--`, чтобы их нельзя было выдать за ключ.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rmcp::schemars::{self, JsonSchema};
use serde::Serialize;

/// Репозиторий, с которым работает сервер.
pub struct Repo {
    root: PathBuf,
}

impl Repo {
    /// Корень репозитория, в котором лежит `dir`.
    pub async fn open(dir: &Path) -> Result<Self> {
        let out = run(dir, &["rev-parse", "--show-toplevel"])
            .await
            .with_context(|| format!("{} — не git-репозиторий", dir.display()))?;
        Ok(Self { root: PathBuf::from(out.trim()) })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    async fn git(&self, args: &[&str]) -> Result<String> {
        run(&self.root, args).await
    }

    /// Вывод команды без концевых пробелов или `None`, если она не удалась
    /// (нет upstream, нет ни одного коммита и т. п. — это не ошибка).
    async fn git_opt(&self, args: &[&str]) -> Option<String> {
        self.git(args).await.ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    }
}

async fn run(dir: &Path, args: &[&str]) -> Result<String> {
    let out = tokio::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.quotepath=off", "-c", "color.ui=never"])
        .args(args)
        .env("GIT_PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .context("не удалось запустить git — он установлен и есть в PATH?")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("git {} — {}", args.join(" "), stderr.trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Имя ветки, тега или коммита от модели: не ключ (`-…`) и без символов,
/// которых в ссылках не бывает.
pub fn check_ref(r: &str) -> Result<()> {
    let ok = !r.is_empty()
        && !r.starts_with('-')
        && !r.contains("..")
        && r.chars().all(|c| c.is_alphanumeric() || "._/-~^@{}".contains(c));
    if !ok {
        bail!("«{r}» не похоже на имя ветки или коммита");
    }
    Ok(())
}

/// Путь от модели — относительно корня репозитория, без выхода за него.
pub fn check_path(p: &str) -> Result<String> {
    let p = p.trim().trim_start_matches("./").trim_end_matches('/');
    if p.starts_with('/') || p.split('/').any(|part| part == "..") {
        bail!("путь «{p}» должен быть внутри репозитория, относительно его корня");
    }
    Ok(p.to_string())
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Commit {
    pub hash: String,
    pub subject: String,
    pub author: String,
    /// Дата в RFC 3339.
    pub date: String,
}

const LOG_FORMAT: &str = "--format=%h%x1f%s%x1f%an%x1f%aI";

fn parse_commits(out: &str) -> Vec<Commit> {
    out.lines()
        .filter_map(|line| {
            let mut parts = line.split('\x1f');
            Some(Commit {
                hash: parts.next()?.to_string(),
                subject: parts.next()?.to_string(),
                author: parts.next()?.to_string(),
                date: parts.next()?.to_string(),
            })
        })
        .collect()
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct BranchInfo {
    /// Корень репозитория.
    pub repo: String,
    /// Текущая ветка; `null` — HEAD отсоединён (detached).
    pub branch: Option<String>,
    /// Коммит, на котором стоит HEAD (короткий хэш); `null` — коммитов ещё нет.
    pub head: Option<String>,
    /// Ветка, которую отслеживает текущая (`origin/main`), если она задана.
    pub upstream: Option<String>,
    /// На сколько коммитов текущая ветка впереди upstream.
    pub ahead: Option<u32>,
    /// На сколько коммитов позади upstream.
    pub behind: Option<u32>,
    pub last_commit: Option<Commit>,
    /// Файлов с незакоммиченными изменениями (включая неотслеживаемые).
    pub changed_files: usize,
    /// Все локальные ветки.
    pub local_branches: Vec<String>,
}

pub async fn branch(repo: &Repo) -> Result<BranchInfo> {
    let branch = repo.git_opt(&["symbolic-ref", "--short", "-q", "HEAD"]).await;
    let head = repo.git_opt(&["rev-parse", "--short", "HEAD"]).await;
    let upstream = repo.git_opt(&["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]).await;
    let (behind, ahead) = match &upstream {
        Some(_) => match repo.git_opt(&["rev-list", "--left-right", "--count", "@{u}...HEAD"]).await {
            Some(counts) => {
                let mut it = counts.split_whitespace().map(|n| n.parse::<u32>().ok());
                (it.next().flatten(), it.next().flatten())
            }
            None => (None, None),
        },
        None => (None, None),
    };
    let last_commit = match head {
        Some(_) => parse_commits(&repo.git(&["log", "-1", LOG_FORMAT]).await?).into_iter().next(),
        None => None,
    };
    let changed_files = status_entries(repo).await?.len();
    let local_branches = repo
        .git(&["for-each-ref", "--format=%(refname:short)", "refs/heads"])
        .await?
        .lines()
        .map(str::to_string)
        .collect();
    Ok(BranchInfo {
        repo: repo.root().display().to_string(),
        branch,
        head,
        upstream,
        ahead,
        behind,
        last_commit,
        changed_files,
        local_branches,
    })
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StatusEntry {
    pub path: String,
    /// Прежний путь у переименованного или скопированного файла.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Состояние в индексе (то, что уйдёт в коммит): `M`, `A`, `D`, `R`, `C`, `U`, `?` или пробел.
    pub index: String,
    /// Состояние в рабочей копии относительно индекса — те же буквы.
    pub worktree: String,
    /// Человекочитаемо: «изменён», «добавлен», «удалён», «переименован», «не отслеживается», «конфликт».
    pub state: String,
}

fn describe_state(x: char, y: char) -> &'static str {
    match (x, y) {
        ('?', '?') => "не отслеживается",
        ('U', _) | (_, 'U') | ('A', 'A') | ('D', 'D') => "конфликт",
        ('R', _) => "переименован",
        ('C', _) => "скопирован",
        ('A', _) => "добавлен",
        ('D', _) | (_, 'D') => "удалён",
        _ => "изменён",
    }
}

async fn status_entries(repo: &Repo) -> Result<Vec<StatusEntry>> {
    let out = repo.git(&["status", "--porcelain=v1", "-z", "--untracked-files=all"]).await?;
    let mut parts = out.split('\0').filter(|s| !s.is_empty());
    let mut entries = Vec::new();
    while let Some(entry) = parts.next() {
        let mut chars = entry.chars();
        let (Some(x), Some(y)) = (chars.next(), chars.next()) else { continue };
        let path = entry.get(3..).unwrap_or_default().to_string();
        // У переименования следом идёт прежний путь отдельной записью.
        let from = if matches!(x, 'R' | 'C') { parts.next().map(str::to_string) } else { None };
        entries.push(StatusEntry {
            path,
            from,
            index: x.to_string(),
            worktree: y.to_string(),
            state: describe_state(x, y).to_string(),
        });
    }
    Ok(entries)
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Status {
    pub branch: Option<String>,
    /// Изменения, добавленные в индекс (`git add`) — войдут в следующий коммит.
    pub staged: usize,
    /// Изменения отслеживаемых файлов, ещё не добавленные в индекс.
    pub unstaged: usize,
    pub untracked: usize,
    pub conflicts: usize,
    pub clean: bool,
    pub files: Vec<StatusEntry>,
}

pub async fn status(repo: &Repo) -> Result<Status> {
    let files = status_entries(repo).await?;
    let count = |f: &dyn Fn(&StatusEntry) -> bool| files.iter().filter(|e| f(e)).count();
    Ok(Status {
        branch: repo.git_opt(&["symbolic-ref", "--short", "-q", "HEAD"]).await,
        staged: count(&|e| !matches!(e.index.as_str(), " " | "?")),
        unstaged: count(&|e| !matches!(e.worktree.as_str(), " " | "?")),
        untracked: count(&|e| e.index == "?"),
        conflicts: count(&|e| e.state == "конфликт"),
        clean: files.is_empty(),
        files,
    })
}

/// Совпадение имени с шаблоном `*.rs`, `main.?s`: `*` — любые символы, `?` — один.
pub fn wildcard(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    let (mut pi, mut ni) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ni;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
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
    /// Список обрезан по `limit`.
    pub truncated: bool,
    /// Пути файлов (без `depth`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    /// С `depth` — пути, свёрнутые до этой глубины, с числом файлов.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tree: Vec<DirEntry>,
}

pub struct FilesQuery {
    pub path: Option<String>,
    pub pattern: Option<String>,
    pub depth: Option<usize>,
    pub untracked: bool,
    pub limit: usize,
}

pub async fn files(repo: &Repo, q: &FilesQuery) -> Result<FileList> {
    let mut args = vec!["ls-files", "-z", "--cached"];
    if q.untracked {
        args.extend(["--others", "--exclude-standard"]);
    }
    let path = q.path.as_deref().map(check_path).transpose()?.filter(|p| !p.is_empty());
    if let Some(path) = &path {
        args.extend(["--", path.as_str()]);
    }
    let out = repo.git(&args).await?;
    let mut all: Vec<&str> = out.split('\0').filter(|s| !s.is_empty()).collect();
    all.sort_unstable();
    all.dedup();
    if let Some(pattern) = q.pattern.as_deref().filter(|p| !p.is_empty()) {
        // Шаблон с `/` — по всему пути, без — по имени файла.
        all.retain(|f| {
            let subject = if pattern.contains('/') { f } else { f.rsplit('/').next().unwrap_or(f) };
            wildcard(pattern, subject)
        });
    }
    let total = all.len();
    match q.depth {
        Some(depth) => {
            let depth = depth.max(1);
            let mut tree: Vec<DirEntry> = Vec::new();
            for f in &all {
                let parts: Vec<&str> = f.split('/').collect();
                let key =
                    if parts.len() > depth { format!("{}/", parts[..depth].join("/")) } else { f.to_string() };
                match tree.last_mut() {
                    Some(last) if last.path == key => last.files += 1,
                    _ => tree.push(DirEntry { path: key, files: 1 }),
                }
            }
            let truncated = tree.len() > q.limit;
            tree.truncate(q.limit);
            Ok(FileList { total, truncated, files: Vec::new(), tree })
        }
        None => {
            let truncated = total > q.limit;
            let files = all.into_iter().take(q.limit).map(str::to_string).collect();
            Ok(FileList { total, truncated, files, tree: Vec::new() })
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FileStat {
    pub path: String,
    /// Добавлено строк; `null` у двоичного файла.
    pub added: Option<u32>,
    pub deleted: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Diff {
    /// С чем сравнивали: «рабочая копия ↔ индекс», «индекс ↔ HEAD», «рабочая копия ↔ main (точка ответвления 1a2b3c4)».
    pub compared: String,
    pub files: Vec<FileStat>,
    pub added: u32,
    pub deleted: u32,
    /// Сам дифф (unified); пусто при `stat_only`.
    pub patch: String,
    /// Дифф обрезан по `max_chars`.
    pub truncated: bool,
    /// Новые неотслеживаемые файлы — в дифф они не входят, только перечислены.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub untracked: Vec<String>,
}

pub struct DiffQuery {
    pub staged: bool,
    pub base: Option<String>,
    /// Ветка или коммит, который сравнивается с `base` (ревью ветки целиком):
    /// только закоммиченное в `head` с точки ответвления от `base`. Без него —
    /// текущая рабочая копия.
    pub head: Option<String>,
    pub path: Option<String>,
    pub stat_only: bool,
    pub max_chars: usize,
}

pub async fn diff(repo: &Repo, q: &DiffQuery) -> Result<Diff> {
    let mut range: Vec<String> = Vec::new();
    let mut compared = if q.staged { "индекс ↔ HEAD".to_string() } else { "рабочая копия ↔ индекс".to_string() };
    if q.staged {
        range.push("--cached".into());
    }
    let head = q.head.as_deref().filter(|h| !h.is_empty());
    if let Some(head) = head {
        check_ref(head)?;
        if q.staged {
            bail!("head сравнивает закоммиченное — вместе со staged он не имеет смысла");
        }
        let Some(base) = q.base.as_deref().filter(|b| !b.is_empty()) else {
            bail!("head — только вместе с base: с чем сравнивать ветку «{head}»?");
        };
        check_ref(base)?;
        let fork = repo
            .git(&["merge-base", base, head])
            .await
            .with_context(|| format!("не найдена общая точка «{base}» и «{head}»"))?;
        let fork = fork.trim().to_string();
        let short: String = fork.chars().take(7).collect();
        compared = format!("{head} ↔ {base} (точка ответвления {short})");
        range.push(fork);
        range.push(head.to_string());
    } else if let Some(base) = q.base.as_deref().filter(|b| !b.is_empty()) {
        check_ref(base)?;
        // Всё, что сделано с момента ответвления от base: коммиты ветки и
        // незакоммиченное (с `staged` — только то, что в индексе).
        let fork = repo
            .git(&["merge-base", base, "HEAD"])
            .await
            .with_context(|| format!("не найдена общая точка «{base}» и HEAD"))?;
        let fork = fork.trim().to_string();
        let short: String = fork.chars().take(7).collect();
        let side = if q.staged { "индекс" } else { "рабочая копия" };
        compared = format!("{side} ↔ {base} (точка ответвления {short})");
        range.push(fork);
    }
    let path = q.path.as_deref().map(check_path).transpose()?.filter(|p| !p.is_empty());
    let mut tail: Vec<String> = Vec::new();
    if let Some(path) = &path {
        tail.extend(["--".to_string(), path.clone()]);
    }
    let args = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> = ["diff", "--no-ext-diff", "--no-color"].iter().map(|s| s.to_string()).collect();
        a.extend(extra.iter().map(|s| s.to_string()));
        a.extend(range.iter().cloned());
        a.extend(tail.iter().cloned());
        a
    };
    let numstat_args = args(&["--numstat", "-z", "-M"]);
    let numstat = repo.git(&numstat_args.iter().map(String::as_str).collect::<Vec<_>>()).await?;
    let files = parse_numstat(&numstat);
    let added = files.iter().filter_map(|f| f.added).sum();
    let deleted = files.iter().filter_map(|f| f.deleted).sum();

    let (patch, truncated) = if q.stat_only {
        (String::new(), false)
    } else {
        let patch_args = args(&["-M"]);
        let full = repo.git(&patch_args.iter().map(String::as_str).collect::<Vec<_>>()).await?;
        cut(&full, q.max_chars)
    };

    let untracked = if q.staged || head.is_some() {
        Vec::new()
    } else {
        status_entries(repo)
            .await?
            .into_iter()
            .filter(|e| e.index == "?")
            .map(|e| e.path)
            .filter(|p| path.as_deref().is_none_or(|dir| p == dir || p.starts_with(&format!("{dir}/"))))
            .collect()
    };
    Ok(Diff { compared, files, added, deleted, patch, truncated, untracked })
}

/// `--numstat -z`: `added\tdeleted\tpath\0`, у переименования —
/// `added\tdeleted\t\0from\0to\0`.
fn parse_numstat(out: &str) -> Vec<FileStat> {
    let mut parts = out.split('\0');
    let mut files = Vec::new();
    while let Some(entry) = parts.next() {
        if entry.trim().is_empty() {
            continue;
        }
        let mut cols = entry.splitn(3, '\t');
        let (Some(a), Some(d), Some(path)) = (cols.next(), cols.next(), cols.next()) else { continue };
        let path = if path.is_empty() {
            let from = parts.next().unwrap_or_default();
            let to = parts.next().unwrap_or_default();
            format!("{from} → {to}")
        } else {
            path.to_string()
        };
        let a = a.trim_start_matches('\n');
        files.push(FileStat { path, added: a.parse().ok(), deleted: d.parse().ok() });
    }
    files
}

/// Первые `max` символов по границе строки.
fn cut(text: &str, max: usize) -> (String, bool) {
    if text.chars().count() <= max {
        return (text.to_string(), false);
    }
    let mut end = text.char_indices().nth(max).map(|(i, _)| i).unwrap_or(text.len());
    if let Some(nl) = text[..end].rfind('\n') {
        end = nl + 1;
    }
    (format!("{}… [дифф обрезан — сузьте path или запросите stat_only]", &text[..end]), true)
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FileText {
    pub path: String,
    /// Ветка или коммит, из которого прочитан файл; `null` — рабочая копия.
    #[serde(rename = "ref")]
    pub git_ref: Option<String>,
    pub total_lines: usize,
    /// Номера возвращённых строк (с 1, включительно).
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    /// Файл не поместился в `max_lines` — остальное по `start_line`.
    pub truncated: bool,
}

pub struct FileQuery {
    pub path: String,
    pub git_ref: Option<String>,
    pub start_line: usize,
    pub max_lines: usize,
}

/// Текст файла: из рабочей копии или, с `git_ref`, из ветки/коммита
/// (`git show ref:path`). Только внутри репозитория, только текст.
pub async fn read_file(repo: &Repo, q: &FileQuery) -> Result<FileText> {
    let path = check_path(&q.path)?;
    if path.is_empty() {
        bail!("укажите путь к файлу относительно корня репозитория");
    }
    let git_ref = q.git_ref.as_deref().map(str::trim).filter(|r| !r.is_empty());
    let content = match git_ref {
        Some(r) => {
            check_ref(r)?;
            repo.git(&["show", &format!("{r}:{path}")]).await?
        }
        None => {
            // Симлинк не должен выводить за пределы репозитория.
            let full = tokio::fs::canonicalize(repo.root.join(&path))
                .await
                .with_context(|| format!("файла «{path}» нет в рабочей копии"))?;
            let root = tokio::fs::canonicalize(&repo.root).await?;
            if !full.starts_with(&root) {
                bail!("путь «{path}» ведёт за пределы репозитория");
            }
            let bytes = tokio::fs::read(&full).await.with_context(|| format!("не удалось прочитать «{path}»"))?;
            if bytes.contains(&0) {
                bail!("«{path}» — двоичный файл");
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    if content.contains('\0') {
        bail!("«{path}» — двоичный файл");
    }
    Ok(slice_lines(path, git_ref.map(str::to_string), &content, q.start_line, q.max_lines))
}

fn slice_lines(path: String, git_ref: Option<String>, content: &str, start: usize, max: usize) -> FileText {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = start.max(1).min(total.max(1));
    let end = (start + max - 1).min(total);
    let text = if total == 0 { String::new() } else { lines[start - 1..end].join("\n") };
    FileText { path, git_ref, total_lines: total, start_line: start, end_line: end, text, truncated: end < total }
}

pub async fn log(repo: &Repo, n: usize, path: Option<&str>) -> Result<Vec<Commit>> {
    let n = format!("-{n}");
    let mut args = vec!["log", n.as_str(), LOG_FORMAT];
    let path = path.map(check_path).transpose()?.filter(|p| !p.is_empty());
    if let Some(path) = &path {
        args.extend(["--", path.as_str()]);
    }
    // Нет ни одного коммита — пустая история, а не ошибка.
    match repo.git_opt(&["rev-parse", "HEAD"]).await {
        Some(_) => Ok(parse_commits(&repo.git(&args).await?)),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matches() {
        assert!(wildcard("*.rs", "main.rs"));
        assert!(!wildcard("*.rs", "main.rsx"));
        assert!(wildcard("core/*/mod.rs", "core/src/rag/mod.rs"));
        assert!(wildcard("ma?n.*", "main.rs"));
        assert!(wildcard("*", ""));
    }

    #[test]
    fn rejects_refs_and_paths_that_look_like_options() {
        assert!(check_ref("main").is_ok());
        assert!(check_ref("origin/day-30").is_ok());
        assert!(check_ref("HEAD~3").is_ok());
        assert!(check_ref("--output=/tmp/x").is_err());
        assert!(check_ref("a b").is_err());
        assert!(check_ref("main..dev").is_err());
        assert_eq!(check_path("./core/src/").unwrap(), "core/src");
        assert!(check_path("../etc").is_err());
        assert!(check_path("/etc/passwd").is_err());
    }

    #[test]
    fn parses_numstat_with_renames() {
        let out = "3\t1\tREADME.md\0-\t-\tlogo.png\0\n2\t0\t\0old.rs\0new.rs\0";
        let files = parse_numstat(out);
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].path, "README.md");
        assert_eq!(files[0].added, Some(3));
        assert_eq!(files[1].added, None);
        assert_eq!(files[2].path, "old.rs → new.rs");
    }

    #[test]
    fn cuts_on_line_boundary() {
        let (text, truncated) = cut("abc\ndef\nghi\n", 6);
        assert!(truncated);
        assert!(text.starts_with("abc\n…"));
        assert_eq!(cut("abc", 10), ("abc".to_string(), false));
    }

    async fn git_in(dir: &Path, args: &[&str]) {
        let ok = tokio::process::Command::new("git").arg("-C").arg(dir).args(args).output().await.unwrap();
        assert!(ok.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&ok.stderr));
    }

    /// Живой репозиторий во временной папке: ветка, статус, файлы, дифф, журнал.
    #[tokio::test]
    async fn reads_a_real_repository() {
        let dir = std::env::temp_dir().join(format!("git-mcp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        git_in(&dir, &["init", "-q", "-b", "main"]).await;
        git_in(&dir, &["config", "user.email", "t@example.com"]).await;
        git_in(&dir, &["config", "user.name", "Test"]).await;
        std::fs::write(dir.join("README.md"), "# Проект\n").unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
        git_in(&dir, &["add", "."]).await;
        git_in(&dir, &["commit", "-q", "-m", "Первый коммит"]).await;
        git_in(&dir, &["checkout", "-q", "-b", "feature"]).await;
        std::fs::write(dir.join("src/main.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "заметка\n").unwrap();

        let repo = Repo::open(&dir.join("src")).await.unwrap();
        let b = branch(&repo).await.unwrap();
        assert_eq!(b.branch.as_deref(), Some("feature"));
        assert_eq!(b.last_commit.as_ref().unwrap().subject, "Первый коммит");
        assert_eq!(b.changed_files, 2);
        assert_eq!(b.local_branches, vec!["feature", "main"]);

        let s = status(&repo).await.unwrap();
        assert_eq!((s.unstaged, s.untracked, s.staged), (1, 1, 0));

        let q = FilesQuery { path: None, pattern: Some("*.rs".into()), depth: None, untracked: false, limit: 100 };
        assert_eq!(files(&repo, &q).await.unwrap().files, vec!["src/main.rs"]);
        let q = FilesQuery { path: None, pattern: None, depth: Some(1), untracked: true, limit: 100 };
        let tree = files(&repo, &q).await.unwrap().tree;
        let paths: Vec<&str> = tree.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, vec!["README.md", "notes.txt", "src/"]);

        let q = DiffQuery { staged: false, base: Some("main".into()), head: None, path: None, stat_only: false, max_chars: 10_000 };
        let d = diff(&repo, &q).await.unwrap();
        assert_eq!(d.files.len(), 1);
        assert_eq!((d.added, d.deleted), (3, 1));
        assert!(d.patch.contains("println!"));
        assert_eq!(d.untracked, vec!["notes.txt"]);

        assert_eq!(log(&repo, 5, None).await.unwrap().len(), 1);

        // Ветка против ветки: только закоммиченное в head, без рабочей копии.
        git_in(&dir, &["commit", "-q", "-am", "Привет"]).await;
        std::fs::write(dir.join("src/main.rs"), "fn main() { todo!() }\n").unwrap();
        let q = DiffQuery { staged: false, base: Some("main".into()), head: Some("feature".into()), path: None, stat_only: false, max_chars: 10_000 };
        let d = diff(&repo, &q).await.unwrap();
        assert!(d.compared.starts_with("feature ↔ main"));
        assert!(d.patch.contains("println!") && !d.patch.contains("todo!"));
        assert!(d.untracked.is_empty());
        let q = DiffQuery { staged: false, base: None, head: Some("feature".into()), path: None, stat_only: true, max_chars: 10_000 };
        assert!(diff(&repo, &q).await.is_err());

        // Файл из рабочей копии и из ветки, по строкам.
        let q = FileQuery { path: "src/main.rs".into(), git_ref: None, start_line: 1, max_lines: 10 };
        assert_eq!(read_file(&repo, &q).await.unwrap().text, "fn main() { todo!() }");
        let q = FileQuery { path: "src/main.rs".into(), git_ref: Some("feature".into()), start_line: 2, max_lines: 1 };
        let f = read_file(&repo, &q).await.unwrap();
        assert_eq!((f.text.as_str(), f.start_line, f.end_line, f.total_lines, f.truncated), ("    println!(\"hi\");", 2, 2, 3, true));
        let q = FileQuery { path: "../etc/passwd".into(), git_ref: None, start_line: 1, max_lines: 10 };
        assert!(read_file(&repo, &q).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
