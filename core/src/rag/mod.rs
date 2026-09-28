//! Индексация документов для RAG: корпус → чанки (три стратегии) →
//! эмбеддинги → локальный индекс в SQLite с метаданными и версиями.
//!
//! - [`corpus`] — какие файлы индексировать и их текст (PDF — через
//!   [`crate::pdf`]);
//! - [`chunking`] — стратегии `fixed`, `structure`, `sentence`, `parent`;
//! - [`embed`] — клиент `/embeddings` (модель — `LLM_EMBEDDING_MODEL`);
//! - [`store`] — индекс: документы, версии, чанки, векторы.
//!
//! **Версии.** Индексация сравнивает SHA-256 файла с действующей версией
//! документа: изменился — появляется версия N+1, прежняя становится
//! `superseded` и уходит из поиска, но остаётся в индексе (поиск по ней —
//! `--version N`), пока её не удалит `prune`. Файл пропал из корпуса —
//! версия `removed`. У каждой версии — хэш содержимого, коммит git, который
//! последним менял файл, и признак незакоммиченных изменений; у каждого
//! чанка — версия документа и модель, которой получен его вектор. Векторы
//! хранятся по ключу «модель + текст чанка», поэтому в новой версии заново
//! эмбеддятся только изменившиеся куски.
//!
//! Все интерфейсы (TUI, CLI) работают через [`run_command`] — одни и те же
//! команды с одним и тем же результатом.

pub mod browse;
pub mod chunking;
pub mod corpus;
pub mod embed;
pub mod store;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

pub use chunking::{ChunkParams, Kind, ParamSpec, Strategy};
pub use embed::Embedder;
use store::{ChunkHit, ChunkRow, NewVersion, Store};

/// Куда писать о ходе долгой операции (индексации, сравнения).
pub type Progress<'a> = &'a (dyn Fn(String) + Send + Sync);

/// Пути индекса и корпуса.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Рабочая директория — от неё считаются пути корпуса (`source`).
    pub base: PathBuf,
    pub db_path: PathBuf,
    pub corpus: Vec<String>,
    pub eval_path: PathBuf,
    pub report_path: PathBuf,
    /// Папка загруженных документов (веб, команда `add` для файла вне
    /// корпуса) — всегда часть корпуса, иначе полная индексация сочла бы
    /// загрузки пропавшими.
    pub uploads: PathBuf,
}

impl RagConfig {
    /// `RAG_DB_PATH` (rag/index.db), `RAG_CORPUS` (пути через запятую; по
    /// умолчанию пусто — только загрузки, см. [`corpus::DEFAULT_CORPUS`]), `RAG_EVAL` (rag/eval.json), `RAG_REPORT`
    /// (rag/compare.md), `RAG_UPLOADS` (rag/uploads).
    pub fn from_env() -> Self {
        let var = |name: &str, default: &str| {
            std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
        };
        let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            db_path: base.join(var("RAG_DB_PATH", "rag/index.db")),
            corpus: var("RAG_CORPUS", corpus::DEFAULT_CORPUS)
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            eval_path: base.join(var("RAG_EVAL", "rag/eval.json")),
            report_path: base.join(var("RAG_REPORT", "rag/compare.md")),
            uploads: base.join(var("RAG_UPLOADS", "rag/uploads")),
            base,
        }
    }

    /// Пути корпуса вместе с папкой загрузок.
    fn roots(&self) -> Vec<String> {
        let mut roots = self.corpus.clone();
        let uploads = self.source_of(&self.uploads);
        if !roots.contains(&uploads) {
            roots.push(uploads);
        }
        roots
    }

    /// Путь относительно рабочей директории (через `/`) — `source` документа.
    fn source_of(&self, path: &std::path::Path) -> String {
        path.strip_prefix(&self.base).unwrap_or(path).to_string_lossy().replace('\\', "/")
    }
}

const META_MODEL: &str = "embedding_model";
const META_DIM: &str = "embedding_dim";
const META_INDEXED_AT: &str = "indexed_at";

/// Итог индексации одной стратегии.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyRun {
    pub strategy: String,
    /// Версий документов, разбитых в этот раз.
    pub versions: usize,
    pub chunks: usize,
    /// Текстов, отправленных в модель, и взятых из кэша векторов.
    pub embedded: usize,
    pub cached: usize,
    pub tokens: Option<u64>,
    pub embed_ms: u64,
    pub at: i64,
    /// Из разбитых версий — сколько уже были нарезаны этой стратегией с
    /// другими параметрами (прежняя нарезка заменена).
    #[serde(default)]
    pub rechunked: usize,
    /// Параметры нарезки одной строкой — когда разбивался один документ.
    #[serde(default)]
    pub params: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct IndexReport {
    pub files: usize,
    pub new_versions: Vec<(String, i64)>,
    pub unchanged: usize,
    pub removed: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub missing_roots: Vec<String>,
    pub runs: Vec<StrategyRun>,
    /// Индексировался один документ (загрузка, `add`), а не весь корпус.
    pub single: bool,
}

impl IndexReport {
    pub fn describe(&self) -> String {
        let mut out = if self.single {
            match self.new_versions.first() {
                Some((source, version)) => format!("{source}: новая версия v{version}"),
                None => "Документ не изменился с действующей версии".to_string(),
            }
        } else {
            format!(
                "Документов в корпусе: {} · новых версий: {} · без изменений: {}",
                self.files,
                self.new_versions.len(),
                self.unchanged
            )
        };
        let listed = if self.single { &[][..] } else { &self.new_versions[..] };
        for (source, version) in listed {
            out.push_str(&format!("\n  + {source} → v{version}"));
        }
        for source in &self.removed {
            out.push_str(&format!("\n  − {source}: нет в корпусе, версия снята с поиска"));
        }
        for (source, err) in &self.failed {
            out.push_str(&format!("\n  ! {source}: {err}"));
        }
        if !self.missing_roots.is_empty() {
            out.push_str(&format!("\n  (нет путей корпуса: {})", self.missing_roots.join(", ")));
        }
        for run in &self.runs {
            if run.versions == 0 {
                out.push_str(&format!("\n{}: без изменений", run.strategy));
                continue;
            }
            let tokens = run.tokens.map(|t| format!(", {t} токенов")).unwrap_or_default();
            let params = run.params.as_ref().map(|p| format!(" ({p})")).unwrap_or_default();
            let rechunked = if run.rechunked > 0 { format!(", перенарезано с новыми параметрами: {}", run.rechunked) } else { String::new() };
            out.push_str(&format!(
                "\n{}{params}: {} версий → {} чанков{rechunked}; эмбеддингов {} (из кэша {}) за {}{tokens}",
                run.strategy,
                run.versions,
                run.chunks,
                run.embedded,
                run.cached,
                fmt_ms(run.embed_ms)
            ));
        }
        out
    }
}

/// Ключ вектора: модель, префикс и текст — одинаковый текст той же модели
/// эмбеддится один раз.
fn vector_key(model: &str, prefix: &str, text: &str) -> String {
    corpus::sha256_hex(format!("{model}\n{prefix}{text}").as_bytes())
}

/// Индекс собран той же моделью, что задана сейчас: векторы разных
/// моделей сравнивать нельзя.
fn check_model(store: &Store, embedder: &Embedder) -> Result<()> {
    if let Some(indexed) = store.meta(META_MODEL)? {
        if indexed != embedder.model() && store.count("chunks")? > 0 {
            bail!(
                "индекс построен моделью «{indexed}», а сейчас задана «{}»: векторы разных моделей несравнимы. \
                 Верните LLM_EMBEDDING_MODEL={indexed} или очистите индекс командой `reset` и проиндексируйте заново",
                embedder.model()
            );
        }
    }
    Ok(())
}

struct LoadedDoc {
    file: corpus::SourceFile,
    text: String,
    version_id: i64,
    version: i64,
}

/// Читает файл и сверяет его с действующей версией: изменился — новая
/// версия. Ошибка чтения попадает в отчёт, а не прерывает индексацию.
async fn load_version(
    store: &mut Store,
    cfg: &RagConfig,
    file: &corpus::SourceFile,
    dirty: Option<&std::collections::HashSet<String>>,
    report: &mut IndexReport,
) -> Result<Option<LoadedDoc>> {
    let loaded = match corpus::load(file).await {
        Ok(l) => l,
        Err(err) => {
            report.failed.push((file.source.clone(), format!("{err:#}")));
            return Ok(None);
        }
    };
    let git = corpus::git_info(&cfg.base, &file.source, dirty);
    let version = match store.current_version(&file.source)? {
        Some(v) if v.sha256 == loaded.sha256 => {
            report.unchanged += 1;
            if v.git_commit != git.commit || v.git_dirty != git.dirty {
                store.refresh_git(v.version_id, git.commit.as_deref(), git.dirty)?;
            }
            v
        }
        _ => {
            let v = store.add_version(&NewVersion {
                source: &file.source,
                title: &loaded.title,
                kind: file.kind.name(),
                sha256: &loaded.sha256,
                git_commit: git.commit.as_deref(),
                git_dirty: git.dirty,
                bytes: loaded.bytes as i64,
                mtime: loaded.mtime,
                text: &loaded.text,
            })?;
            report.new_versions.push((file.source.clone(), v.version));
            v
        }
    };
    Ok(Some(LoadedDoc { file: file.clone(), text: loaded.text, version_id: version.version_id, version: version.version }))
}

/// Индексация всего корпуса: новые и изменённые документы получают новую
/// версию, для каждой версии без чанков стратегии — разбиение и эмбеддинги.
/// Можно прерывать: векторы сохраняются пачками, чанки версии — целиком,
/// так что следующий запуск продолжит с того же места.
pub async fn index(cfg: &RagConfig, embedder: &Embedder, strategies: &[Strategy], progress: Progress<'_>) -> Result<IndexReport> {
    let mut store = Store::open(&cfg.db_path)?;
    check_model(&store, embedder)?;

    let (files, missing_roots) = corpus::scan(&cfg.base, &cfg.roots());
    if files.is_empty() {
        bail!(
            "индексировать нечего: в {} нет файлов (.md, .txt, .rs, .pdf){} — добавьте документ командой `add` или загрузкой",
            cfg.source_of(&cfg.uploads),
            if cfg.corpus.is_empty() { String::new() } else { format!(", в RAG_CORPUS ({}) тоже", cfg.corpus.join(", ")) }
        );
    }
    // Пустая папка загрузок — не «недостающий путь корпуса».
    let uploads = cfg.source_of(&cfg.uploads);
    let missing_roots = missing_roots.into_iter().filter(|r| *r != uploads).collect();
    let mut report = IndexReport { files: files.len(), missing_roots, ..Default::default() };
    let dirty = corpus::git_dirty_set(&cfg.base);

    let mut docs = Vec::new();
    for (i, file) in files.iter().enumerate() {
        progress(format!("Чтение {}/{}: {}", i + 1, files.len(), file.source));
        docs.extend(load_version(&mut store, cfg, file, dirty.as_ref(), &mut report).await?);
    }
    let present: Vec<String> = files.iter().map(|f| f.source.clone()).collect();
    report.removed = store.mark_removed(&present)?;

    report.runs = embed_versions(&mut store, embedder, &docs, strategies, None, true, progress).await?;
    store.set_meta(META_INDEXED_AT, &store::now_secs().to_string())?;
    Ok(report)
}

/// Индексация одного документа корпуса (путь `source` относительно рабочей
/// директории) — остальные документы не трогаются. Так векторизуется
/// загруженный файл. `params` — параметры нарезки: если версия уже
/// нарезана стратегией с другими, нарезка заменяется.
pub async fn index_file(
    cfg: &RagConfig,
    embedder: &Embedder,
    source: &str,
    strategies: &[Strategy],
    params: &ChunkParams,
    progress: Progress<'_>,
) -> Result<IndexReport> {
    for &s in strategies {
        params.validate(s).map_err(|e| anyhow!("{}: {e}", s.name()))?;
    }
    let mut store = Store::open(&cfg.db_path)?;
    check_model(&store, embedder)?;
    let path = cfg.base.join(source);
    let (files, _) = corpus::scan(&cfg.base, &[source.to_string()]);
    let file = files
        .into_iter()
        .next()
        .with_context(|| format!("{}: нет такого файла или формат не поддерживается (.md, .txt, .rs, .pdf)", path.display()))?;
    let mut report = IndexReport { files: 1, single: true, ..Default::default() };
    progress(format!("Чтение {}", file.source));
    let dirty = corpus::git_dirty_set(&cfg.base);
    let Some(doc) = load_version(&mut store, cfg, &file, dirty.as_ref(), &mut report).await? else {
        let (_, err) = report.failed.pop().unwrap_or_default();
        bail!("{}: {err}", file.source);
    };
    report.runs =
        embed_versions(&mut store, embedder, std::slice::from_ref(&doc), strategies, Some(params), false, progress).await?;
    store.set_meta(META_INDEXED_AT, &store::now_secs().to_string())?;
    Ok(report)
}

/// Параметры стратегии из JSON нарезки (недостающие — по умолчанию).
fn parse_params(json: &str) -> ChunkParams {
    serde_json::from_str(json).unwrap_or_default()
}

/// Разбиение и эмбеддинги для версий, у которых нет нарезки стратегией.
///
/// `explicit` — параметры заданы явно (загрузка, `add`): нарезка с другими
/// параметрами заменяется. `None` (полная индексация) — существующие
/// нарезки не трогаются; новая версия документа режется теми же
/// стратегиями и параметрами, что и его прошлая версия, — стратегии,
/// которых у документа не было, не добавляются. Все запрошенные стратегии
/// получает только документ, которого в индексе ещё не было.
///
/// `full` — разбирается весь корпус: итог стратегии запоминается как
/// полная сборка (её время и объём сравниваются в отчёте).
async fn embed_versions(
    store: &mut Store,
    embedder: &Embedder,
    docs: &[LoadedDoc],
    strategies: &[Strategy],
    explicit: Option<&ChunkParams>,
    full: bool,
    progress: Progress<'_>,
) -> Result<Vec<StrategyRun>> {
    let model = embedder.model().to_string();
    // Стратегии каждого документа до этой индексации: полная индексация
    // режет документ только ими (новый документ — всеми запрошенными).
    let mut had: HashMap<String, Vec<String>> = HashMap::new();
    if explicit.is_none() {
        for doc in docs {
            had.insert(doc.file.source.clone(), store.document_strategies(&doc.file.source)?);
        }
    }
    let mut runs = Vec::new();
    for &strategy in strategies {
        let mut run = StrategyRun { strategy: strategy.name().to_string(), at: store::now_secs(), ..Default::default() };
        // Версии, которые надо (пере)нарезать: версия, параметры (JSON), чанки.
        let mut pending: Vec<(i64, String, Vec<ChunkRow>)> = Vec::new();
        for doc in docs {
            // Нарезка без записи параметров (индекс до их появления) —
            // параметрами по умолчанию.
            let existing = match store.chunking_params(doc.version_id, strategy.name())? {
                Some(json) => Some(parse_params(&json)),
                None if store.has_chunks(doc.version_id, strategy.name())? => Some(ChunkParams::default()),
                None => None,
            };
            let params = match (explicit, existing) {
                (Some(p), Some(e)) if p.of(strategy) == e.of(strategy) => continue,
                (Some(p), existing) => {
                    run.rechunked += existing.is_some() as usize;
                    *p
                }
                (None, Some(_)) => continue,
                (None, None)
                    if had
                        .get(&doc.file.source)
                        .is_some_and(|h| !h.is_empty() && !h.iter().any(|s| s == strategy.name())) =>
                {
                    continue
                }
                (None, None) => store
                    .latest_chunking_params(&doc.file.source, strategy.name())?
                    .map(|json| parse_params(&json))
                    .unwrap_or_default(),
            };
            if docs.len() == 1 {
                run.params = Some(params.describe(strategy));
            }
            let json = serde_json::to_string(&params.of(strategy))?;
            pending.push((doc.version_id, json, chunk_rows(doc, strategy, &params, &model, embedder.doc_prefix())));
        }
        run.versions = pending.len();
        run.chunks = pending.iter().map(|(_, _, rows)| rows.len()).sum();

        // Уникальные тексты без вектора в кэше.
        let mut to_embed: Vec<(String, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for row in pending.iter().flat_map(|(_, _, rows)| rows) {
            if !seen.insert(row.vector_key.clone()) {
                continue;
            }
            if store.has_vector(&row.vector_key)? {
                run.cached += 1;
            } else {
                to_embed.push((row.vector_key.clone(), row.text.clone()));
            }
        }

        let started = Instant::now();
        let mut tokens: Option<u64> = None;
        for (n, batch) in to_embed.chunks(embedder.batch()).enumerate() {
            progress(format!(
                "{}: эмбеддинги {}/{} ({})",
                strategy.name(),
                (n * embedder.batch() + batch.len()).min(to_embed.len()),
                to_embed.len(),
                model
            ));
            let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
            let embedded = embedder.embed_documents(&texts).await?;
            if let Some(t) = embedded.tokens {
                *tokens.get_or_insert(0) += t;
            }
            let dim = embedded.vectors[0].len();
            match store.meta(META_DIM)?.and_then(|d| d.parse::<usize>().ok()) {
                Some(indexed) if indexed != dim && store.count("vectors")? > 0 => {
                    bail!("модель вернула векторы длины {dim}, а в индексе — {indexed}: очистите индекс командой `reset`")
                }
                _ => store.set_meta(META_DIM, &dim.to_string())?,
            }
            store.set_meta(META_MODEL, &model)?;
            let items: Vec<(String, Vec<f32>)> = batch.iter().map(|(k, _)| k.clone()).zip(embedded.vectors).collect();
            store.put_vectors(&model, &items)?;
            run.embedded += items.len();
        }
        run.embed_ms = started.elapsed().as_millis() as u64;
        run.tokens = tokens;

        for (version_id, json, rows) in &pending {
            store.insert_chunks(*version_id, strategy.name(), json, rows)?;
        }
        if run.versions > 0 {
            store.set_meta(&format!("run.{}", strategy.name()), &serde_json::to_string(&run)?)?;
        }
        if full && run.versions == docs.len() {
            store.set_meta(&format!("build.{}", strategy.name()), &serde_json::to_string(&run)?)?;
        }
        runs.push(run);
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// Загрузка и удаление документов

/// Имя загружаемого файла без пути и опасных символов; расширение — из
/// поддерживаемых.
pub fn safe_upload_name(name: &str) -> Result<String> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    let cleaned: String = base
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .collect();
    let cleaned = cleaned.trim_start_matches('.').to_string();
    let ext = cleaned.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
    if !matches!(ext.as_str(), "md" | "markdown" | "txt" | "rs" | "pdf") {
        bail!("«{name}»: поддерживаются .md, .txt, .rs и .pdf");
    }
    if cleaned.len() <= ext.len() + 1 {
        bail!("«{name}»: пустое имя файла");
    }
    Ok(cleaned)
}

/// Предел размера загружаемого файла.
pub const MAX_UPLOAD_BYTES: usize = 50 * 1024 * 1024;

/// Сохраняет загруженный файл в папку загрузок (файл с тем же именем
/// заменяется — при индексации это станет новой версией документа) и
/// возвращает его `source`.
pub fn save_upload(cfg: &RagConfig, name: &str, bytes: &[u8]) -> Result<String> {
    if bytes.is_empty() {
        bail!("файл пустой");
    }
    if bytes.len() > MAX_UPLOAD_BYTES {
        bail!("файл больше {} МБ", MAX_UPLOAD_BYTES / 1024 / 1024);
    }
    let name = safe_upload_name(name)?;
    std::fs::create_dir_all(&cfg.uploads).with_context(|| format!("не удалось создать {}", cfg.uploads.display()))?;
    let path = cfg.uploads.join(&name);
    std::fs::write(&path, bytes).with_context(|| format!("не удалось записать {}", path.display()))?;
    Ok(cfg.source_of(&path))
}

/// Документ по пути вне индекса: внутри корпуса индексируется на месте,
/// иначе копируется в папку загрузок (чтобы полная индексация его не
/// потеряла). Возвращает `source`.
pub fn add_path(cfg: &RagConfig, path: &str) -> Result<String> {
    let expanded = match path.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|_| PathBuf::from(path)),
        None => PathBuf::from(path),
    };
    let full = if expanded.is_absolute() { expanded } else { cfg.base.join(expanded) };
    let full = full.canonicalize().with_context(|| format!("нет файла {path}"))?;
    if !full.is_file() {
        bail!("{path} — не файл");
    }
    let base = cfg.base.canonicalize().unwrap_or_else(|_| cfg.base.clone());
    if let Ok(rel) = full.strip_prefix(&base) {
        let source = rel.to_string_lossy().replace('\\', "/");
        let in_corpus = cfg.roots().iter().any(|root| source == *root || source.starts_with(&format!("{}/", root.trim_end_matches('/'))));
        if in_corpus {
            return Ok(source);
        }
    }
    let name = full.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let bytes = std::fs::read(&full).with_context(|| format!("не удалось прочитать {path}"))?;
    save_upload(cfg, &name, &bytes)
}

/// Итог удаления документа из индекса.
#[derive(Debug, Serialize)]
pub struct Removed {
    pub source: String,
    pub versions: usize,
    pub chunks: usize,
    pub vectors: usize,
    /// Файл был загружен и удалён вместе с документом.
    pub file_deleted: bool,
    /// Файл остался в корпусе — следующая полная индексация вернёт документ.
    pub still_in_corpus: bool,
}

impl Removed {
    pub fn describe(&self) -> String {
        let mut out = format!(
            "{} удалён из индекса: версий {}, чанков {}, векторов {}.",
            self.source, self.versions, self.chunks, self.vectors
        );
        if self.file_deleted {
            out.push_str(" Загруженный файл удалён.");
        }
        if self.still_in_corpus {
            out.push_str(" Файл остался в корпусе — при следующей полной индексации документ вернётся.");
        }
        out
    }
}

/// Удаляет документ со всеми версиями, чанками и ставшими ничьими векторами;
/// загруженный файл (из папки загрузок) удаляется тоже.
pub fn remove_document(cfg: &RagConfig, source: &str) -> Result<Removed> {
    let mut store = Store::open(&cfg.db_path)?;
    let (versions, chunks, vectors) =
        store.delete_document(source)?.ok_or_else(|| anyhow!("документа «{source}» нет в индексе (путь — как в списке docs)"))?;
    let path = cfg.base.join(source);
    let uploaded = path.starts_with(&cfg.uploads);
    let file_deleted = uploaded && std::fs::remove_file(&path).is_ok();
    let still_in_corpus = !uploaded && path.is_file();
    Ok(Removed { source: source.to_string(), versions, chunks, vectors, file_deleted, still_in_corpus })
}

fn chunk_rows(doc: &LoadedDoc, strategy: Strategy, params: &ChunkParams, model: &str, prefix: &str) -> Vec<ChunkRow> {
    let text = &doc.text;
    chunking::chunk(text, doc.file.kind, strategy, params)
        .into_iter()
        .enumerate()
        .map(|(ord, piece)| {
            let chunk_text = text[piece.start..piece.end].to_string();
            let char_len = chunk_text.chars().count();
            ChunkRow {
                chunk_id: format!("{}@v{}/{}/{ord:04}", doc.file.source, doc.version, strategy.name()),
                version_id: doc.version_id,
                strategy: strategy.name().to_string(),
                ord: ord as i64,
                section: piece.section.clone(),
                start_line: chunking::line_of(text, piece.start) as i64,
                end_line: chunking::line_of(text, piece.end) as i64,
                start_byte: piece.start as i64,
                end_byte: piece.end as i64,
                char_len: char_len as i64,
                // Грубая оценка: ~4 символа на токен.
                tokens_est: char_len.div_ceil(4) as i64,
                content_sha256: corpus::sha256_hex(chunk_text.as_bytes()),
                vector_key: vector_key(model, prefix, &chunk_text),
                embedding_model: model.to_string(),
                clean_end: chunking::clean_end(text, &piece),
                splits_code: chunking::splits_code(text, doc.file.kind, &piece),
                text: chunk_text,
                context_start_byte: piece.parent.map(|(a, _)| a as i64),
                context_end_byte: piece.parent.map(|(_, b)| b as i64),
                context_start_line: piece.parent.map(|(a, _)| chunking::line_of(text, a) as i64),
                context_end_line: piece.parent.map(|(_, b)| chunking::line_of(text, b) as i64),
                context: piece.parent.map(|(a, b)| text[a..b].to_string()),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Поиск

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub score: f32,
    pub hit: ChunkHit,
}

/// Топ-`k` по близости. Parent-child: у одного родителя остаётся лучший
/// ребёнок — в выдаче `k` разных абзацев, а не один абзац несколько раз.
fn rank(set: &[(ChunkHit, Vec<f32>)], query: &[f32], k: usize) -> Vec<SearchHit> {
    let mut scored: Vec<(f32, usize)> = set.iter().enumerate().map(|(i, (_, v))| (embed::dot(query, v), i)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut seen_parents = std::collections::HashSet::new();
    scored
        .into_iter()
        .filter(|&(_, i)| {
            let c = &set[i].0.chunk;
            match c.context_start_byte.zip(c.context_end_byte) {
                Some(range) => seen_parents.insert((c.version_id, range)),
                None => true,
            }
        })
        .take(k)
        .map(|(score, i)| SearchHit { score, hit: set[i].0.clone() })
        .collect()
}

/// Стратегии, которыми нарезан хотя бы один действующий документ, — в
/// порядке [`Strategy::ALL`].
pub fn indexed_strategies(cfg: &RagConfig) -> Result<Vec<Strategy>> {
    let indexed = Store::open(&cfg.db_path)?.strategies()?;
    Ok(Strategy::ALL.into_iter().filter(|s| indexed.iter().any(|i| i == s.name())).collect())
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub strategies: Vec<Strategy>,
    pub k: usize,
    pub source: Option<String>,
    pub version: Option<i64>,
}

/// Поиск по косинусной близости (векторы нормированы — это скалярное
/// произведение) перебором всех чанков стратегии: на нескольких тысячах
/// чанков это миллисекунды.
pub async fn search(cfg: &RagConfig, embedder: &Embedder, query: &str, opts: &SearchOptions) -> Result<Vec<(Strategy, Vec<SearchHit>)>> {
    let store = Store::open(&cfg.db_path)?;
    check_model(&store, embedder)?;
    if opts.version.is_some() && opts.source.is_none() {
        bail!("поиск по версии — только в одном документе: укажите --source <путь> вместе с --version");
    }
    let qvec = embedder.embed_query(query).await?;
    let mut result = Vec::new();
    for &strategy in &opts.strategies {
        let set = store.search_set(strategy.name(), opts.source.as_deref(), opts.version)?;
        result.push((strategy, rank(&set, &qvec, opts.k)));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Сравнение стратегий

#[derive(Debug, Clone, Deserialize)]
pub struct EvalQuestion {
    pub question: String,
    /// Документ, где ответ (путь в корпусе); вопрос пропускается, если его
    /// нет в индексе.
    pub source: String,
    /// Фрагмент ответа: чанк найден, если содержит фрагмент целиком (без
    /// учёта регистра и пробелов) — из любого документа: тот же факт в
    /// doc-комментарии кода — тоже правильный ответ.
    pub expect: String,
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

const EVAL_K: usize = 5;

#[derive(Debug, Clone)]
struct QuestionResult {
    /// Место первого подходящего чанка в выдаче (с 1), если он в топ-5.
    rank: Option<usize>,
    top: Option<SearchHit>,
}

pub async fn compare(cfg: &RagConfig, embedder: &Embedder, progress: Progress<'_>) -> Result<String> {
    let store = Store::open(&cfg.db_path)?;
    check_model(&store, embedder)?;
    let raw = std::fs::read_to_string(&cfg.eval_path)
        .with_context(|| format!("нет набора контрольных вопросов {}", cfg.eval_path.display()))?;
    let questions: Vec<EvalQuestion> =
        serde_json::from_str(&raw).with_context(|| format!("не удалось разобрать {}", cfg.eval_path.display()))?;
    let indexed: Vec<String> = store.strategies()?;
    let strategies: Vec<Strategy> = Strategy::ALL.into_iter().filter(|s| indexed.iter().any(|i| i == s.name())).collect();
    if strategies.is_empty() {
        bail!("индекс пуст — сначала проиндексируйте корпус командой `index`");
    }
    let sources: std::collections::HashSet<String> =
        store.latest_versions()?.into_iter().filter(|v| v.status == "current").map(|v| v.source).collect();

    // Вопросы, чей документ есть в индексе; вектор каждого — один раз.
    let mut asked = Vec::new();
    let mut skipped = Vec::new();
    for (i, q) in questions.iter().enumerate() {
        if !sources.contains(&q.source) {
            skipped.push(q);
            continue;
        }
        progress(format!("Сравнение: вопрос {}/{}", i + 1, questions.len()));
        asked.push((q, embedder.embed_query(&q.question).await?));
    }
    if asked.is_empty() {
        bail!("ни один документ из {} не проиндексирован", cfg.eval_path.display());
    }

    let mut results: HashMap<Strategy, Vec<QuestionResult>> = HashMap::new();
    for &strategy in &strategies {
        let set = store.search_set(strategy.name(), None, None)?;
        let per_question = asked
            .iter()
            .map(|(q, qvec)| {
                let hits = rank(&set, qvec, EVAL_K);
                let expect = normalize_ws(&q.expect);
                let rank = hits
                    .iter()
                    // Найдено то, что уйдёт в контекст модели: у parent-child — родитель.
                    .position(|h| normalize_ws(h.hit.chunk.context_text()).contains(&expect))
                    .map(|p| p + 1);
                QuestionResult { rank, top: hits.into_iter().next() }
            })
            .collect();
        results.insert(strategy, per_question);
    }

    let mut stats = HashMap::new();
    let mut runs = HashMap::new();
    for &s in &strategies {
        stats.insert(s, store.stats(s.name())?);
        if let Some(run) = store.meta(&format!("build.{}", s.name()))?.and_then(|r| serde_json::from_str::<StrategyRun>(&r).ok()) {
            runs.insert(s, run);
        }
    }
    let model = store.meta(META_MODEL)?.unwrap_or_default();
    let dim = store.meta(META_DIM)?.unwrap_or_default();
    let versions = store.latest_versions()?.into_iter().filter(|v| v.status == "current").collect::<Vec<_>>();

    let report = compare_report(&CompareData {
        strategies: &strategies,
        stats: &stats,
        runs: &runs,
        asked: &asked.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
        skipped: &skipped,
        results: &results,
        model: &model,
        dim: &dim,
        versions: &versions,
    });
    if let Some(dir) = cfg.report_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&cfg.report_path, &report.markdown)
        .with_context(|| format!("не удалось записать {}", cfg.report_path.display()))?;
    Ok(format!("{}\nОтчёт: {}", report.text, cfg.report_path.strip_prefix(&cfg.base).unwrap_or(&cfg.report_path).display()))
}

struct CompareData<'a> {
    strategies: &'a [Strategy],
    stats: &'a HashMap<Strategy, store::StrategyStats>,
    runs: &'a HashMap<Strategy, StrategyRun>,
    asked: &'a [&'a EvalQuestion],
    skipped: &'a [&'a EvalQuestion],
    results: &'a HashMap<Strategy, Vec<QuestionResult>>,
    model: &'a str,
    dim: &'a str,
    versions: &'a [store::VersionRow],
}

struct Report {
    markdown: String,
    text: String,
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn pct(n: usize, of: usize) -> String {
    if of == 0 {
        "—".into()
    } else {
        format!("{:.0}%", n as f64 * 100.0 / of as f64)
    }
}

fn retrieval_metrics(results: &[QuestionResult]) -> (usize, usize, usize, f64) {
    let hit = |k: usize| results.iter().filter(|r| r.rank.is_some_and(|x| x <= k)).count();
    let mrr = results.iter().map(|r| r.rank.map(|x| 1.0 / x as f64).unwrap_or(0.0)).sum::<f64>() / results.len().max(1) as f64;
    (hit(1), hit(3), hit(5), mrr)
}

fn compare_report(d: &CompareData) -> Report {
    let n = d.asked.len();
    let names: Vec<&str> = d.strategies.iter().map(|s| s.name()).collect();

    let mut stat_rows = Vec::new();
    for s in d.strategies {
        let st = &d.stats[s];
        let mut lengths = st.lengths.clone();
        lengths.sort_unstable();
        let run = d.runs.get(s);
        stat_rows.push(vec![
            s.name().to_string(),
            st.chunks.to_string(),
            format!(
                "{} / {} / {} / {}",
                lengths.first().unwrap_or(&0),
                percentile(&lengths, 0.5),
                percentile(&lengths, 0.95),
                lengths.last().unwrap_or(&0)
            ),
            format!("≈{}", st.tokens_est / st.chunks.max(1)),
            pct(st.text_dirty_end, st.text_chunks),
            pct(st.text_split_fence, st.text_chunks),
            pct(st.code_split, st.code_chunks),
            pct(st.with_section, st.chunks),
            run.map(|r| format!("{} / {}", r.embedded, fmt_ms(r.embed_ms))).unwrap_or_else(|| "—".into()),
        ]);
    }
    let stat_headers = [
        "стратегия",
        "чанков",
        "символов min/медиана/p95/max",
        "токенов на чанк",
        "текст: обрыв фразы",
        "текст: разрез ```",
        "код: разрез элемента",
        "с разделом",
        "эмбеддингов / время",
    ];

    let mut quality_rows = Vec::new();
    for s in d.strategies {
        let (h1, h3, h5, mrr) = retrieval_metrics(&d.results[s]);
        quality_rows.push(vec![
            s.name().to_string(),
            format!("{h1}/{n} ({})", pct(h1, n)),
            format!("{h3}/{n} ({})", pct(h3, n)),
            format!("{h5}/{n} ({})", pct(h5, n)),
            format!("{mrr:.2}"),
        ]);
    }
    let quality_headers = ["стратегия", "hit@1", "hit@3", "hit@5", "MRR@5"];

    let mark = |r: &QuestionResult| match r.rank {
        Some(x) => format!("✓{x}"),
        None => "✗".into(),
    };
    let mut question_rows = Vec::new();
    for (i, q) in d.asked.iter().enumerate() {
        let mut row = vec![(i + 1).to_string(), q.question.clone(), format!("`{}`", q.source)];
        for s in d.strategies {
            row.push(mark(&d.results[s][i]));
        }
        question_rows.push(row);
    }
    let mut question_headers = vec!["#", "вопрос", "где ответ"];
    question_headers.extend(names.iter());

    // Вопросы, где стратегии разошлись: лучший результат против худшего.
    let mut examples = String::new();
    for (i, q) in d.asked.iter().enumerate() {
        let ranks: Vec<Option<usize>> = d.strategies.iter().map(|s| d.results[s][i].rank).collect();
        let score = |r: &Option<usize>| r.map(|x| 10 - x).unwrap_or(0);
        let (best, worst) = (ranks.iter().map(score).max().unwrap_or(0), ranks.iter().map(score).min().unwrap_or(0));
        if best == worst || examples.matches("\n### ").count() >= 4 {
            continue;
        }
        examples.push_str(&format!("\n### {}. {}\n\nОтвет: «{}» в `{}`.\n\n", i + 1, q.question, q.expect, q.source));
        for s in d.strategies {
            let r = &d.results[s][i];
            match &r.top {
                Some(top) => examples.push_str(&format!(
                    "- **{}** ({}): топ-1 — `{}` · {} · строки {}–{} · score {:.3}\n  > {}\n",
                    s.name(),
                    mark(r),
                    top.hit.version.source,
                    top.hit.chunk.section.as_deref().unwrap_or("без раздела"),
                    top.hit.chunk.start_line,
                    top.hit.chunk.end_line,
                    top.score,
                    preview_line(&top.hit.chunk.text, 160)
                )),
                None => examples.push_str(&format!("- **{}**: пусто\n", s.name())),
            }
        }
    }

    let multi_version = d.versions.iter().filter(|v| v.version > 1).count();
    let mut md = String::new();
    md.push_str("# Сравнение стратегий chunking\n\n");
    md.push_str(&format!(
        "Сформировано {} · модель эмбеддингов `{}` (размерность {}) · документов в индексе: {} (из них с версией > 1: {}).\n\n",
        fmt_time(store::now_secs()),
        d.model,
        d.dim,
        d.versions.len(),
        multi_version
    ));
    md.push_str("## Стратегии\n\n");
    for s in d.strategies {
        md.push_str(&format!("- **{}** — {}.\n", s.name(), s.describe()));
    }
    md.push_str("\n## Чанки\n\n");
    md.push_str(&md_table(&stat_headers, &stat_rows));
    md.push_str(
        "\nДоли — среди чанков текста (Markdown, PDF) и кода Rust отдельно. «Обрыв фразы» — чанк текста кончается не на конце \
         предложения, абзаца или блока. «Разрез ```» — в чанке нечётное число ```` ``` ````. «Разрез элемента» — в чанке кода \
         не сходятся `{` и `}`. \
         «Эмбеддингов / время» — последняя полная сборка стратегии, с учётом кэша векторов.\n",
    );
    md.push_str(&format!(
        "\n## Поиск: {n} контрольных вопросов\n\nЧанк считается найденным, если содержит фрагмент ответа целиком (из любого документа — \
         тот же факт в doc-комментарии кода тоже ответ); у `parent` проверяется родитель — он уходит в контекст модели. \
         Крупный чанк чаще вмещает фрагмент целиком, мелкий — точнее по смыслу: \
         смотрите вместе с размерами выше. Вопросы — `{}`, «где ответ» — документ, по которому вопрос составлен.\n\n",
        "rag/eval.json"
    ));
    md.push_str(&md_table(&quality_headers, &quality_rows));
    md.push('\n');
    md.push_str(&md_table(&question_headers, &question_rows));
    if !d.skipped.is_empty() {
        md.push_str(&format!(
            "\nПропущено (документа нет в индексе): {}.\n",
            d.skipped.iter().map(|q| format!("`{}`", q.source)).collect::<Vec<_>>().join(", ")
        ));
    }
    if !examples.is_empty() {
        md.push_str("\n## Где стратегии разошлись\n");
        md.push_str(&examples);
    }

    let mut text = String::from("Чанки\n");
    text.push_str(&text_table(&stat_headers, &stat_rows));
    text.push_str(&format!("\nПоиск по {n} контрольным вопросам\n"));
    text.push_str(&text_table(&quality_headers, &quality_rows));
    Report { markdown: md, text }
}

fn md_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = format!("| {} |\n|{}|\n", headers.join(" | "), headers.iter().map(|_| "---").collect::<Vec<_>>().join("|"));
    for row in rows {
        out.push_str(&format!("| {} |\n", row.iter().map(|c| c.replace('|', "\\|")).collect::<Vec<_>>().join(" | ")));
    }
    out
}

/// Таблица моноширинным текстом (для терминала).
fn text_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let width = |s: &str| s.chars().count();
    let mut widths: Vec<usize> = headers.iter().map(|h| width(h)).collect();
    for row in rows {
        for (i, c) in row.iter().enumerate() {
            widths[i] = widths[i].max(width(c));
        }
    }
    let line = |cells: Vec<&str>| {
        cells.iter().enumerate().map(|(i, c)| format!("{c}{}", " ".repeat(widths[i] - width(c)))).collect::<Vec<_>>().join("  ").trim_end().to_string()
    };
    let mut out = line(headers.to_vec()) + "\n";
    for row in rows {
        out.push_str(&line(row.iter().map(String::as_str).collect()));
        out.push('\n');
    }
    out
}

fn preview_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]),
        None => flat,
    }
}

fn fmt_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} мс")
    } else {
        format!("{:.1} с", ms as f64 / 1000.0)
    }
}

/// Время UTC: `2026-09-28 14:03 UTC`.
pub fn fmt_time(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Гражданская дата из числа дней (алгоритм Хиннанта).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + (month <= 2) as i64;
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02} UTC", rem / 3600, rem % 3600 / 60)
}

// ---------------------------------------------------------------------------
// Команды

pub const HELP: &str = "Команды индекса:
  status                         — сводка индекса
  index [fixed|structure|sentence|parent|all] — проиндексировать корпус (новые версии, недостающие чанки)
  add <путь> [стратегии] [параметр=значение …] — векторизовать один файл (вне корпуса — копия в папку загрузок);
                                   параметры: fixed — size, overlap; structure — min_chars, max_chars;
                                   sentence — window, stride, window_chars, code_window, code_stride
  remove <путь>                  — удалить документ из индекса (загруженный — и файл)
  search [-s стратегия] [-k N] [--source путь [--version N]] <запрос> — поиск
  docs                           — документы и их действующие версии
  versions <путь>                — все версии документа
  compare                        — сравнить стратегии, отчёт в rag/compare.md
  prune                          — удалить заменённые версии
  reset                          — очистить индекс";

/// Команда индекса — общая для всех интерфейсов. `args` — слова команды
/// (без префикса интерфейса).
pub async fn run_command(cfg: &RagConfig, args: &[&str], progress: Progress<'_>) -> Result<String> {
    match args.first().copied() {
        None | Some("status") => status(cfg),
        Some("help") => Ok(HELP.to_string()),
        Some("index") => {
            let strategies = parse_strategies(&args[1..])?;
            let embedder = Embedder::from_env()?;
            let report = index(cfg, &embedder, &strategies, progress).await?;
            Ok(format!("{}\n\n{}", report.describe(), status(cfg)?))
        }
        Some("search") => {
            let (opts, query) = parse_search(&args[1..])?;
            let opts = SearchOptions {
                strategies: if opts.strategies.is_empty() {
                    indexed_strategies(cfg)?
                } else {
                    opts.strategies
                },
                ..opts
            };
            if opts.strategies.is_empty() {
                bail!("индекс пуст — сначала проиндексируйте корпус командой `index`");
            }
            let embedder = Embedder::from_env()?;
            let results = search(cfg, &embedder, &query, &opts).await?;
            Ok(format_search(&query, &results))
        }
        Some("add") => {
            let path = args.get(1).ok_or_else(|| anyhow!("укажите файл: add <путь> [стратегии] [параметр=значение …]"))?;
            let (names, assignments): (Vec<&str>, Vec<&str>) = args[2..].iter().partition(|a| !a.contains('='));
            let strategies = parse_strategies(&names)?;
            let params = parse_params_args(&assignments)?;
            let embedder = Embedder::from_env()?;
            let source = add_path(cfg, path)?;
            let report = index_file(cfg, &embedder, &source, &strategies, &params, progress).await?;
            Ok(format!("{}\n\n{}", report.describe(), versions(cfg, &source)?))
        }
        Some("remove") => {
            let source = args.get(1).ok_or_else(|| anyhow!("укажите документ: remove <путь>"))?;
            Ok(remove_document(cfg, source)?.describe())
        }
        Some("docs") => docs(cfg),
        Some("versions") => {
            let source = args.get(1).ok_or_else(|| anyhow!("укажите документ: versions <путь>"))?;
            versions(cfg, source)
        }
        Some("compare") => {
            let embedder = Embedder::from_env()?;
            compare(cfg, &embedder, progress).await
        }
        Some("prune") => {
            let (versions, chunks, vectors) = Store::open(&cfg.db_path)?.prune()?;
            Ok(format!("Удалено: версий {versions}, чанков {chunks}, векторов {vectors}."))
        }
        Some("reset") => {
            Store::open(&cfg.db_path)?.reset()?;
            Ok("Индекс очищен.".into())
        }
        Some(other) => bail!("неизвестная команда индекса «{other}»\n{HELP}"),
    }
}

fn parse_strategies(args: &[&str]) -> Result<Vec<Strategy>> {
    if args.is_empty() || args == ["all"] {
        return Ok(Strategy::ALL.to_vec());
    }
    args.iter()
        .map(|a| Strategy::parse(a).ok_or_else(|| anyhow!("неизвестная стратегия «{a}» — есть fixed, structure, sentence, parent, all")))
        .collect()
}

/// `size=800 overlap=100` → параметры нарезки (остальные — по умолчанию).
fn parse_params_args(args: &[&str]) -> Result<ChunkParams> {
    let mut params = ChunkParams::default();
    for arg in args {
        let (name, value) = arg.split_once('=').ok_or_else(|| anyhow!("ожидается параметр=значение: «{arg}»"))?;
        let value: usize = value.trim().parse().map_err(|_| anyhow!("{name}: нужно целое число, задано «{value}»"))?;
        params.set(name.trim(), value).map_err(|e| anyhow!(e))?;
    }
    Ok(params)
}

fn parse_search(args: &[&str]) -> Result<(SearchOptions, String)> {
    let mut opts = SearchOptions { k: 0, ..Default::default() };
    let mut words = Vec::new();
    let mut i = 0;
    let value = |i: usize, flag: &str| args.get(i + 1).copied().ok_or_else(|| anyhow!("после {flag} нужно значение"));
    while i < args.len() {
        match args[i] {
            flag @ ("-s" | "--strategy") => {
                let v = value(i, flag)?;
                opts.strategies = parse_strategies(&[v])?;
                i += 2;
            }
            flag @ "-k" => {
                opts.k = value(i, flag)?.parse().map_err(|_| anyhow!("-k — число"))?;
                i += 2;
            }
            flag @ "--source" => {
                opts.source = Some(value(i, flag)?.to_string());
                i += 2;
            }
            flag @ "--version" => {
                let v = value(i, flag)?.trim_start_matches('v');
                opts.version = Some(v.parse().map_err(|_| anyhow!("--version — номер версии"))?);
                i += 2;
            }
            word => {
                words.push(word);
                i += 1;
            }
        }
    }
    if words.is_empty() {
        bail!("укажите запрос: search [-s стратегия] [-k N] <запрос>");
    }
    if opts.k == 0 {
        opts.k = if opts.strategies.len() == 1 { 5 } else { 3 };
    }
    Ok((opts, words.join(" ")))
}

fn format_search(query: &str, results: &[(Strategy, Vec<SearchHit>)]) -> String {
    let mut out = format!("Запрос: {query}");
    for (strategy, hits) in results {
        out.push_str(&format!("\n\n── {} ──", strategy.name()));
        if hits.is_empty() {
            out.push_str("\n  ничего не найдено");
        }
        for (i, h) in hits.iter().enumerate() {
            let c = &h.hit.chunk;
            let v = &h.hit.version;
            out.push_str(&format!(
                "\n{}. {:.3}  {} · {} · строки {}–{}\n   {} · v{}{} · {} симв.\n   {}",
                i + 1,
                h.score,
                v.source,
                c.section.as_deref().unwrap_or("без раздела"),
                c.start_line,
                c.end_line,
                c.chunk_id,
                v.version,
                if v.status == "current" { String::new() } else { format!(" ({})", v.status) },
                c.char_len,
                preview_line(&c.text, 220)
            ));
            if let (Some(context), Some(a), Some(b)) = (&c.context, c.context_start_line, c.context_end_line) {
                out.push_str(&format!(
                    "\n   в контекст — родитель, строки {a}–{b}, {} симв.: {}",
                    context.chars().count(),
                    preview_line(context, 220)
                ));
            }
        }
    }
    out
}

fn status(cfg: &RagConfig) -> Result<String> {
    let store = Store::open(&cfg.db_path)?;
    let db = cfg.db_path.strip_prefix(&cfg.base).unwrap_or(&cfg.db_path).display().to_string();
    let latest = store.latest_versions()?;
    let current = latest.iter().filter(|v| v.status == "current").count();
    if store.count("chunks")? == 0 {
        return Ok(format!(
            "Индекс {db} пуст. Корпус: {}.\nДобавить документ: add <путь>; проиндексировать корпус: index [fixed|structure|sentence|parent|all].",
            corpus_label(cfg)
        ));
    }
    let model = store.meta(META_MODEL)?.unwrap_or_else(|| "—".into());
    let dim = store.meta(META_DIM)?.unwrap_or_else(|| "—".into());
    let at = store.meta(META_INDEXED_AT)?.and_then(|s| s.parse().ok()).map(fmt_time).unwrap_or_else(|| "—".into());
    let mut out = format!(
        "Индекс {db} · модель {model} ({dim}) · проиндексирован {at}\nДокументов: {current} · версий в индексе: {} · векторов: {}",
        store.count("versions")?,
        store.count("vectors")?
    );
    let mut rows = Vec::new();
    for s in Strategy::ALL {
        let st = store.stats(s.name())?;
        if st.chunks == 0 {
            rows.push(vec![s.name().to_string(), "—".into(), "—".into(), "—".into()]);
            continue;
        }
        let mut lengths = st.lengths.clone();
        lengths.sort_unstable();
        rows.push(vec![
            s.name().to_string(),
            st.chunks.to_string(),
            st.documents.to_string(),
            format!("{} / {} / {}", lengths[0], percentile(&lengths, 0.5), lengths[lengths.len() - 1]),
        ]);
    }
    out.push_str("\n\n");
    out.push_str(&text_table(&["стратегия", "чанков", "документов", "символов min/медиана/max"], &rows));
    Ok(out.trim_end().to_string())
}

fn corpus_label(cfg: &RagConfig) -> String {
    let uploads = cfg.source_of(&cfg.uploads);
    if cfg.corpus.is_empty() {
        format!("только загрузки ({uploads})")
    } else {
        format!("{} + загрузки ({uploads})", cfg.corpus.join(", "))
    }
}

fn docs(cfg: &RagConfig) -> Result<String> {
    let store = Store::open(&cfg.db_path)?;
    let latest = store.latest_versions()?;
    if latest.is_empty() {
        return Ok("В индексе нет документов.".into());
    }
    let mut rows = Vec::new();
    for v in &latest {
        let counts = store.chunk_counts(v.version_id)?;
        let count = |s: Strategy| counts.iter().find(|(n, _)| n == s.name()).map(|(_, c)| c.to_string()).unwrap_or_else(|| "—".into());
        rows.push(vec![
            v.source.clone(),
            format!("v{}", v.version),
            match v.status.as_str() {
                "current" => String::new(),
                other => other.to_string(),
            },
            git_label(v),
            count(Strategy::Fixed),
            count(Strategy::Structure),
            count(Strategy::Sentence),
            count(Strategy::Parent),
        ]);
    }
    Ok(text_table(&["документ", "версия", "статус", "git", "fixed", "structure", "sentence", "parent"], &rows).trim_end().to_string())
}

fn git_label(v: &store::VersionRow) -> String {
    let commit = v.git_commit.as_deref().map(|c| c[..c.len().min(8)].to_string()).unwrap_or_else(|| "—".into());
    if v.git_dirty {
        format!("{commit}+изм.")
    } else {
        commit
    }
}

fn versions(cfg: &RagConfig, source: &str) -> Result<String> {
    let store = Store::open(&cfg.db_path)?;
    let all = store.versions(source)?;
    if all.is_empty() {
        bail!("документа «{source}» нет в индексе (путь — как в списке docs)");
    }
    let mut rows = Vec::new();
    for v in &all {
        let counts = store.chunk_counts(v.version_id)?;
        rows.push(vec![
            format!("v{}", v.version),
            v.status.clone(),
            v.sha256[..12].to_string(),
            git_label(v),
            fmt_time(v.indexed_at),
            v.bytes.to_string(),
            counts.iter().map(|(s, c)| format!("{s} {c}")).collect::<Vec<_>>().join(", "),
        ]);
    }
    Ok(format!(
        "{source} — {}\n{}",
        all[0].title,
        text_table(&["версия", "статус", "sha256", "git", "проиндексирована", "байт", "чанки"], &rows).trim_end()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg(name: &str) -> RagConfig {
        let base = std::env::temp_dir().join(format!("rag-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("docs")).unwrap();
        RagConfig {
            db_path: base.join("rag/index.db"),
            corpus: vec!["docs".into()],
            eval_path: base.join("rag/eval.json"),
            report_path: base.join("rag/compare.md"),
            uploads: base.join("rag/uploads"),
            base,
        }
    }

    /// Руководство из трёх разделов, каждый длиннее порога слияния
    /// структурной стратегии.
    fn guide() -> String {
        let filler = |topic: &str| {
            (1..=6).map(|i| format!("Пункт {i} про {topic} описан подробно, чтобы раздел был длинным.")).collect::<Vec<_>>().join(" ")
        };
        format!(
            "# Руководство\n\n## Установка\n\nСкачайте архив и распакуйте его в домашнюю папку. {}\n\n\
             ## Настройка\n\nПорт сервера задаётся переменной APP_PORT. По умолчанию сервер слушает порт 8080. {}\n\n\
             ## Журналы\n\nЖурналы пишутся в каталог logs и хранятся семь дней. {}\n",
            filler("установку"),
            filler("настройку"),
            filler("журналы")
        )
    }

    const CODE: &str = "//! Планировщик.\n\n/// Минимальный интервал между запусками.\npub const MIN_INTERVAL_SECS: u64 = 30;\n\n\
        /// Запускает задание.\npub fn run_job(id: u64) -> bool {\n    id > 0\n}\n";

    fn silent() -> impl Fn(String) + Send + Sync {
        |_| {}
    }

    #[tokio::test]
    async fn index_versions_cache_and_search() {
        let cfg = temp_cfg("index");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        std::fs::write(cfg.base.join("docs/sched.rs"), CODE).unwrap();
        let (url, log) = embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 4);
        let progress = silent();

        let report = index(&cfg, &embedder, &Strategy::ALL, &progress).await.unwrap();
        assert_eq!(report.new_versions, [("docs/guide.md".to_string(), 1), ("docs/sched.rs".to_string(), 1)]);
        assert_eq!(report.runs.len(), Strategy::ALL.len());
        assert!(report.runs.iter().all(|r| r.chunks > 0 && r.embedded + r.cached > 0));
        let requests_after_first = log.lock().unwrap().len();

        // Повторный запуск без изменений — ничего не эмбеддится.
        let report = index(&cfg, &embedder, &Strategy::ALL, &progress).await.unwrap();
        assert_eq!(report.unchanged, 2);
        assert!(report.runs.iter().all(|r| r.versions == 0));
        assert_eq!(log.lock().unwrap().len(), requests_after_first);

        // Новая версия: изменился один раздел — заново эмбеддится только он
        // (у structure; остальные разделы берутся из кэша).
        let changed = guide().replace("хранятся семь дней", "хранятся тридцать дней");
        std::fs::write(cfg.base.join("docs/guide.md"), &changed).unwrap();
        let report = index(&cfg, &embedder, &[Strategy::Structure], &progress).await.unwrap();
        assert_eq!(report.new_versions, [("docs/guide.md".to_string(), 2)]);
        let run = &report.runs[0];
        assert!(run.cached > 0 && run.embedded >= 1 && run.embedded < run.chunks, "{run:?}");

        let store = Store::open(&cfg.db_path).unwrap();
        let versions = store.versions("docs/guide.md").unwrap();
        assert_eq!(versions.iter().map(|v| (v.version, v.status.as_str())).collect::<Vec<_>>(), [(2, "current"), (1, "superseded")]);
        drop(store);

        // Поиск по действующей версии находит новый текст, по v1 — старый.
        let opts = SearchOptions { strategies: vec![Strategy::Structure], k: 1, ..Default::default() };
        let hits = search(&cfg, &embedder, "Журналы хранятся тридцать дней", &opts).await.unwrap();
        let top = &hits[0].1[0].hit;
        assert!(top.chunk.text.contains("тридцать"), "{:?}", top.chunk.text);
        assert_eq!(top.version.version, 2);
        assert_eq!(top.chunk.section.as_deref(), Some("Руководство › Журналы"));
        assert_eq!(top.chunk.chunk_id, format!("docs/guide.md@v2/structure/{:04}", top.chunk.ord));
        assert_eq!(top.chunk.embedding_model, "fake-embed");

        let old = SearchOptions { source: Some("docs/guide.md".into()), version: Some(1), ..opts.clone() };
        let hits = search(&cfg, &embedder, "Журналы хранятся семь дней", &old).await.unwrap();
        assert!(hits[0].1[0].hit.chunk.text.contains("семь дней"));
        assert_eq!(hits[0].1[0].hit.version.status, "superseded");

        // prune удаляет v1; удалённый файл снимается с поиска.
        let (v, _, _) = Store::open(&cfg.db_path).unwrap().prune().unwrap();
        assert_eq!(v, 1);
        std::fs::remove_file(cfg.base.join("docs/sched.rs")).unwrap();
        let report = index(&cfg, &embedder, &[Strategy::Fixed], &progress).await.unwrap();
        assert_eq!(report.removed, ["docs/sched.rs"]);
        let hits = search(&cfg, &embedder, "MIN_INTERVAL_SECS", &SearchOptions { strategies: vec![Strategy::Fixed], k: 5, ..Default::default() })
            .await
            .unwrap();
        assert!(hits[0].1.iter().all(|h| h.hit.version.source != "docs/sched.rs"));

        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn other_model_is_refused() {
        let cfg = temp_cfg("model");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        let (url, _) = embed::test_support::serve().await;
        let progress = silent();
        index(&cfg, &Embedder::new(&url, None, "model-a", 8), &[Strategy::Fixed], &progress).await.unwrap();
        let err = index(&cfg, &Embedder::new(&url, None, "model-b", 8), &[Strategy::Fixed], &progress).await.unwrap_err();
        assert!(err.to_string().contains("индекс построен моделью «model-a»"), "{err}");
        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn compare_writes_report() {
        let cfg = temp_cfg("compare");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        std::fs::write(cfg.base.join("docs/sched.rs"), CODE).unwrap();
        std::fs::create_dir_all(cfg.base.join("rag")).unwrap();
        std::fs::write(
            &cfg.eval_path,
            r#"[
              {"question": "По умолчанию сервер слушает порт", "source": "docs/guide.md", "expect": "порт 8080"},
              {"question": "Минимальный интервал между запусками", "source": "docs/sched.rs", "expect": "MIN_INTERVAL_SECS: u64 = 30"},
              {"question": "Чего нет", "source": "docs/none.md", "expect": "x"}
            ]"#,
        )
        .unwrap();
        let (url, _) = embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 8);
        let progress = silent();
        index(&cfg, &embedder, &Strategy::ALL, &progress).await.unwrap();
        let text = compare(&cfg, &embedder, &progress).await.unwrap();
        assert!(text.contains("hit@1") && text.contains("sentence"), "{text}");
        let md = std::fs::read_to_string(&cfg.report_path).unwrap();
        assert!(md.contains("| fixed |") && md.contains("MRR@5"), "{md}");
        assert!(md.contains("Пропущено (документа нет в индексе): `docs/none.md`"), "{md}");
        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn upload_add_remove_and_map() {
        let cfg = temp_cfg("upload");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        let (url, _) = embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 8);
        let progress = silent();
        index(&cfg, &embedder, &Strategy::ALL, &progress).await.unwrap();

        // Загрузка: файл в папке загрузок, индексируется только он.
        let source = save_upload(&cfg, "../../Отчёт за Q3.md", "# Отчёт\n\nВыручка выросла на 12%. Расходы не изменились.\n".as_bytes()).unwrap();
        assert_eq!(source, "rag/uploads/Отчёт_за_Q3.md");
        let report = index_file(&cfg, &embedder, &source, &Strategy::ALL, &ChunkParams::default(), &progress).await.unwrap();
        assert_eq!(report.new_versions, [(source.clone(), 1)]);
        assert_eq!(report.unchanged, 0);

        // Полная индексация загрузку не теряет (папка загрузок — часть корпуса).
        let report = index(&cfg, &embedder, &[Strategy::Fixed], &progress).await.unwrap();
        assert!(report.removed.is_empty(), "{:?}", report.removed);
        assert_eq!(report.unchanged, 2);

        // Повторная загрузка с другим содержимым — v2.
        save_upload(&cfg, "Отчёт за Q3.md", "# Отчёт\n\nВыручка выросла на 15%.\n".as_bytes()).unwrap();
        let report = index_file(&cfg, &embedder, &source, &[Strategy::Structure], &ChunkParams::default(), &progress).await.unwrap();
        assert_eq!(report.new_versions, [(source.clone(), 2)]);

        // Карта: текст версии и границы чанков в UTF-16.
        let map = browse::document_map(&cfg, &source, None, "structure").unwrap();
        let text = map.text.clone().unwrap();
        assert!(text.contains("15%"));
        let utf16: Vec<u16> = text.encode_utf16().collect();
        let first = &map.chunks[0];
        assert!(String::from_utf16(&utf16[first.start..first.end]).unwrap().starts_with("# Отчёт"));
        let old = browse::document_map(&cfg, &source, Some(1), "fixed").unwrap();
        assert!(old.text.unwrap().contains("12%"));
        let page = browse::chunks(&cfg, &source, Some(1), "sentence", 0, 10).unwrap();
        assert_eq!(page.version.status, "superseded");
        assert!(page.total >= 1);
        let detail = browse::chunk(&cfg, &page.items[0].chunk_id).unwrap();
        assert_eq!(detail.vector.as_ref().unwrap().dim, embed::test_support::DIM);

        // add: файл вне корпуса копируется в загрузки; внутри корпуса — на месте.
        let outside = cfg.base.join("elsewhere.txt");
        std::fs::write(&outside, "Внешний файл.").unwrap();
        assert_eq!(add_path(&cfg, outside.to_str().unwrap()).unwrap(), "rag/uploads/elsewhere.txt");
        assert_eq!(add_path(&cfg, "docs/guide.md").unwrap(), "docs/guide.md");
        assert!(safe_upload_name("virus.exe").is_err());

        // Удаление: загруженный — вместе с файлом, из корпуса — с предупреждением.
        let removed = remove_document(&cfg, &source).unwrap();
        assert!(removed.file_deleted && removed.versions == 2);
        assert!(!cfg.base.join(&source).exists());
        assert!(browse::documents(&cfg).unwrap().iter().all(|d| d.version.source != source));
        let removed = remove_document(&cfg, "docs/guide.md").unwrap();
        assert!(removed.still_in_corpus && !removed.file_deleted);
        assert!(remove_document(&cfg, "docs/guide.md").is_err());

        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn chunk_params_are_stored_replaced_and_inherited() {
        let cfg = temp_cfg("params");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        let (url, _) = embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 8);
        let progress = silent();
        let source = "docs/guide.md";
        let small = ChunkParams { size: 300, overlap: 30, ..ChunkParams::default() };

        let report = index_file(&cfg, &embedder, source, &[Strategy::Fixed], &small, &progress).await.unwrap();
        let first = &report.runs[0];
        assert_eq!(first.params.as_deref(), Some("окно 300 символов, перекрытие 30"));
        let chunks_small = first.chunks;

        // Те же параметры — ничего не делается.
        let report = index_file(&cfg, &embedder, source, &[Strategy::Fixed], &small, &progress).await.unwrap();
        assert_eq!(report.runs[0].versions, 0);

        // Другие параметры — та же версия перенарезана, старая нарезка заменена.
        let big = ChunkParams { size: 800, overlap: 0, ..ChunkParams::default() };
        let report = index_file(&cfg, &embedder, source, &[Strategy::Fixed], &big, &progress).await.unwrap();
        assert!(report.new_versions.is_empty());
        assert_eq!(report.runs[0].rechunked, 1);
        let page = browse::chunks(&cfg, source, None, "fixed", 0, 500).unwrap();
        assert!(page.total < chunks_small && page.items.iter().all(|c| c.char_len <= 800));
        assert_eq!(page.chunking.unwrap().params["size"], 800);

        // Недопустимые параметры — отказ до чтения файла.
        let bad = ChunkParams { overlap: 500, ..big };
        assert!(index_file(&cfg, &embedder, source, &[Strategy::Fixed], &bad, &progress).await.is_err());

        // Полная индексация не добавляет стратегий, которых у документа не было.
        let report = index(&cfg, &embedder, &[Strategy::Structure, Strategy::Parent], &progress).await.unwrap();
        assert!(report.runs.iter().all(|r| r.versions == 0), "{:?}", report.runs);

        // Полная индексация не перенарезает, а новая версия наследует параметры.
        let report = index(&cfg, &embedder, &[Strategy::Fixed], &progress).await.unwrap();
        assert_eq!(report.runs[0].versions, 0);
        std::fs::write(cfg.base.join("docs/guide.md"), guide().replace("семь дней", "десять дней")).unwrap();
        index(&cfg, &embedder, &[Strategy::Fixed], &progress).await.unwrap();
        let versions = browse::versions(&cfg, source).unwrap();
        assert_eq!(versions[0].version.version, 2);
        assert_eq!(versions[0].chunkings[0].params["size"], 800);

        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn parent_child_search_returns_paragraph_as_context() {
        let cfg = temp_cfg("parent");
        std::fs::write(cfg.base.join("docs/guide.md"), guide()).unwrap();
        let (url, _) = embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 8);
        let progress = silent();
        index(&cfg, &embedder, &[Strategy::Parent], &progress).await.unwrap();

        let opts = SearchOptions { strategies: vec![Strategy::Parent], k: 1, ..Default::default() };
        let hits = search(&cfg, &embedder, "По умолчанию сервер слушает порт 8080.", &opts).await.unwrap();
        let chunk = &hits[0].1[0].hit.chunk;
        // Найдено одно предложение…
        assert_eq!(chunk.text, "По умолчанию сервер слушает порт 8080.");
        // …а в контекст идёт весь абзац с заголовком раздела.
        let context = chunk.context_text();
        assert!(context.starts_with("## Настройка") && context.contains("APP_PORT") && context.contains("Пункт 6 про настройку"), "{context}");
        assert!(chunk.context_start_line.unwrap() <= chunk.start_line && chunk.end_line <= chunk.context_end_line.unwrap());

        // В выдаче — разные родители, а не несколько детей одного абзаца.
        let wide = SearchOptions { strategies: vec![Strategy::Parent], k: 3, ..Default::default() };
        let hits = search(&cfg, &embedder, "Пункт про настройку описан подробно", &wide).await.unwrap();
        let parents: std::collections::HashSet<_> =
            hits[0].1.iter().map(|h| (h.hit.chunk.context_start_byte, h.hit.chunk.context_end_byte)).collect();
        assert_eq!(parents.len(), hits[0].1.len());
        assert_eq!(hits[0].1.len(), 3);

        let map = browse::document_map(&cfg, "docs/guide.md", None, "parent").unwrap();
        assert!(map.chunks.iter().all(|c| c.parent.is_some()));
        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[test]
    fn params_args() {
        let p = parse_params_args(&["size=800", "overlap=100"]).unwrap();
        assert_eq!((p.size, p.overlap, p.window), (800, 100, chunking::SENT_WINDOW));
        assert!(parse_params_args(&["size=много"]).is_err());
        assert!(parse_params_args(&["colour=1"]).is_err());
    }

    #[test]
    fn search_args() {
        let (opts, q) = parse_search(&["-s", "structure", "--source", "README.md", "--version", "v2", "как", "дела"]).unwrap();
        assert_eq!(opts.strategies, [Strategy::Structure]);
        assert_eq!(opts.version, Some(2));
        assert_eq!(opts.k, 5);
        assert_eq!(q, "как дела");
        let (opts, _) = parse_search(&["запрос"]).unwrap();
        assert!(opts.strategies.is_empty());
        assert_eq!(opts.k, 3);
        assert!(parse_search(&["-k"]).is_err());
    }

    #[test]
    fn time_format() {
        assert_eq!(fmt_time(0), "1970-01-01 00:00 UTC");
        assert_eq!(fmt_time(1_790_000_000), "2026-09-21 14:13 UTC");
    }
}
