//! Индекс в SQLite.
//!
//! - `documents` — документ = путь в корпусе (`source`);
//! - `versions` — версии документа: номер 1, 2, 3…, SHA-256 содержимого,
//!   коммит git, время изменения, полный текст (из него нарезаны чанки —
//!   по нему строится карта документа); статус `current` (действующая),
//!   `superseded` (заменена новой) или `removed` (файл пропал из корпуса);
//! - `chunks` — чанки версии для каждой стратегии с метаданными и текстом;
//! - `chunkings` — с какими параметрами стратегии нарезана версия (JSON);
//! - `vectors` — векторы по ключу «модель + текст»: одинаковый текст не
//!   эмбеддится второй раз — ни в новой версии документа, ни при пересборке;
//! - `meta` — модель эмбеддингов индекса, размерность, итоги последней
//!   индексации по стратегиям.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

pub struct Store {
    conn: Connection,
}

#[derive(Debug, Clone, Serialize)]
pub struct VersionRow {
    pub version_id: i64,
    pub source: String,
    pub version: i64,
    pub title: String,
    pub kind: String,
    pub sha256: String,
    pub git_commit: Option<String>,
    pub git_dirty: bool,
    pub bytes: i64,
    pub mtime: i64,
    pub indexed_at: i64,
    pub status: String,
}

/// Новая версия документа — то, что о ней известно до разбиения.
pub struct NewVersion<'a> {
    pub source: &'a str,
    pub title: &'a str,
    pub kind: &'a str,
    pub sha256: &'a str,
    pub git_commit: Option<&'a str>,
    pub git_dirty: bool,
    pub bytes: i64,
    pub mtime: i64,
    /// Текст документа, который разбивается на чанки (у PDF — Markdown).
    pub text: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkRow {
    pub chunk_id: String,
    pub version_id: i64,
    pub strategy: String,
    pub ord: i64,
    pub section: Option<String>,
    pub start_line: i64,
    pub end_line: i64,
    pub start_byte: i64,
    pub end_byte: i64,
    pub char_len: i64,
    pub tokens_est: i64,
    pub content_sha256: String,
    pub vector_key: String,
    pub embedding_model: String,
    pub clean_end: bool,
    pub splits_code: bool,
    pub text: String,
    /// Parent-child: родитель — то, что уходит в контекст вместо самого
    /// чанка (диапазон, строки, текст). У остальных стратегий — `None`.
    pub context_start_byte: Option<i64>,
    pub context_end_byte: Option<i64>,
    pub context_start_line: Option<i64>,
    pub context_end_line: Option<i64>,
    pub context: Option<String>,
}

impl ChunkRow {
    /// Текст для контекста модели: родитель, если он есть, иначе сам чанк.
    pub fn context_text(&self) -> &str {
        self.context.as_deref().unwrap_or(&self.text)
    }
}

/// Чанк с метаданными документа и версии — то, что видит поиск.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkHit {
    pub chunk: ChunkRow,
    pub version: VersionRow,
}

/// Сводка по стратегии: чанки действующих версий.
#[derive(Debug, Clone, Default)]
pub struct StrategyStats {
    pub chunks: usize,
    pub documents: usize,
    pub lengths: Vec<usize>,
    pub with_section: usize,
    pub tokens_est: usize,
    /// Чанки текста (Markdown, PDF, txt): всего, кончаются посреди фразы,
    /// разрезают блок кода ```.
    pub text_chunks: usize,
    pub text_dirty_end: usize,
    pub text_split_fence: usize,
    /// Чанки кода Rust: всего и с несбалансированными `{}` (разрезан элемент).
    pub code_chunks: usize,
    pub code_split: usize,
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_to_vector(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

const VERSION_COLUMNS: &str = "v.version_id, d.source, v.version, v.title, d.kind, v.sha256, v.git_commit, v.git_dirty, \
     v.bytes, v.mtime, v.indexed_at, v.status";

fn version_from_row(row: &rusqlite::Row, at: usize) -> rusqlite::Result<VersionRow> {
    Ok(VersionRow {
        version_id: row.get(at)?,
        source: row.get(at + 1)?,
        version: row.get(at + 2)?,
        title: row.get(at + 3)?,
        kind: row.get(at + 4)?,
        sha256: row.get(at + 5)?,
        git_commit: row.get(at + 6)?,
        git_dirty: row.get(at + 7)?,
        bytes: row.get(at + 8)?,
        mtime: row.get(at + 9)?,
        indexed_at: row.get(at + 10)?,
        status: row.get(at + 11)?,
    })
}

const CHUNK_COLUMNS: &str = "c.chunk_id, c.version_id, c.strategy, c.ord, c.section, c.start_line, c.end_line, \
     c.start_byte, c.end_byte, c.char_len, c.tokens_est, c.content_sha256, c.vector_key, c.embedding_model, \
     c.clean_end, c.splits_code, c.text, c.context_start_byte, c.context_end_byte, c.context_start_line, \
     c.context_end_line, c.context";
const CHUNK_COLUMN_COUNT: usize = 22;

fn chunk_from_row(row: &rusqlite::Row) -> rusqlite::Result<ChunkRow> {
    Ok(ChunkRow {
        chunk_id: row.get(0)?,
        version_id: row.get(1)?,
        strategy: row.get(2)?,
        ord: row.get(3)?,
        section: row.get(4)?,
        start_line: row.get(5)?,
        end_line: row.get(6)?,
        start_byte: row.get(7)?,
        end_byte: row.get(8)?,
        char_len: row.get(9)?,
        tokens_est: row.get(10)?,
        content_sha256: row.get(11)?,
        vector_key: row.get(12)?,
        embedding_model: row.get(13)?,
        clean_end: row.get(14)?,
        splits_code: row.get(15)?,
        text: row.get(16)?,
        context_start_byte: row.get(17)?,
        context_end_byte: row.get(18)?,
        context_start_line: row.get(19)?,
        context_end_line: row.get(20)?,
        context: row.get(21)?,
    })
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("не удалось создать {}", dir.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("не удалось открыть индекс {}", path.display()))?;
        Self::init(conn)
    }

    #[cfg(test)]
    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS documents (
                 doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 source TEXT NOT NULL UNIQUE,
                 kind TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS versions (
                 version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                 version INTEGER NOT NULL,
                 title TEXT NOT NULL,
                 sha256 TEXT NOT NULL,
                 git_commit TEXT,
                 git_dirty INTEGER NOT NULL DEFAULT 0,
                 bytes INTEGER NOT NULL,
                 mtime INTEGER NOT NULL,
                 indexed_at INTEGER NOT NULL,
                 status TEXT NOT NULL,
                 text TEXT,
                 UNIQUE (doc_id, version)
             );
             CREATE TABLE IF NOT EXISTS chunks (
                 chunk_id TEXT PRIMARY KEY,
                 version_id INTEGER NOT NULL REFERENCES versions(version_id) ON DELETE CASCADE,
                 strategy TEXT NOT NULL,
                 ord INTEGER NOT NULL,
                 section TEXT,
                 start_line INTEGER NOT NULL,
                 end_line INTEGER NOT NULL,
                 start_byte INTEGER NOT NULL,
                 end_byte INTEGER NOT NULL,
                 char_len INTEGER NOT NULL,
                 tokens_est INTEGER NOT NULL,
                 content_sha256 TEXT NOT NULL,
                 vector_key TEXT NOT NULL,
                 embedding_model TEXT NOT NULL,
                 clean_end INTEGER NOT NULL,
                 splits_code INTEGER NOT NULL,
                 text TEXT NOT NULL,
                 context_start_byte INTEGER,
                 context_end_byte INTEGER,
                 context_start_line INTEGER,
                 context_end_line INTEGER,
                 context TEXT
             );
             CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks (version_id, strategy);
             CREATE TABLE IF NOT EXISTS chunkings (
                 version_id INTEGER NOT NULL REFERENCES versions(version_id) ON DELETE CASCADE,
                 strategy TEXT NOT NULL,
                 params TEXT NOT NULL,
                 created_at INTEGER NOT NULL,
                 PRIMARY KEY (version_id, strategy)
             );
             CREATE TABLE IF NOT EXISTS vectors (
                 vector_key TEXT PRIMARY KEY,
                 model TEXT NOT NULL,
                 dim INTEGER NOT NULL,
                 vector BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )?;
        // Индекс первой версии схемы: без текста версии (карта документа для
        // таких версий недоступна, пока документ не проиндексирован заново).
        let has_text: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('versions') WHERE name = 'text'")?
            .exists([])?;
        if !has_text {
            conn.execute("ALTER TABLE versions ADD COLUMN text TEXT", [])?;
        }
        // …и без контекста чанков (parent-child).
        for (column, ty) in [
            ("context_start_byte", "INTEGER"),
            ("context_end_byte", "INTEGER"),
            ("context_start_line", "INTEGER"),
            ("context_end_line", "INTEGER"),
            ("context", "TEXT"),
        ] {
            let exists = conn
                .prepare("SELECT 1 FROM pragma_table_info('chunks') WHERE name = ?1")?
                .exists([column])?;
            if !exists {
                conn.execute(&format!("ALTER TABLE chunks ADD COLUMN {column} {ty}"), [])?;
            }
        }
        Ok(Self { conn })
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Действующая версия документа.
    pub fn current_version(&self, source: &str) -> Result<Option<VersionRow>> {
        let sql = format!(
            "SELECT {VERSION_COLUMNS} FROM versions v JOIN documents d ON d.doc_id = v.doc_id \
             WHERE d.source = ?1 AND v.status = 'current'"
        );
        Ok(self.conn.query_row(&sql, [source], |r| version_from_row(r, 0)).optional()?)
    }

    /// Все версии документа, новые первыми.
    pub fn versions(&self, source: &str) -> Result<Vec<VersionRow>> {
        let sql = format!(
            "SELECT {VERSION_COLUMNS} FROM versions v JOIN documents d ON d.doc_id = v.doc_id \
             WHERE d.source = ?1 ORDER BY v.version DESC"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([source], |r| version_from_row(r, 0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Последние версии всех документов (действующие и удалённые из корпуса).
    pub fn latest_versions(&self) -> Result<Vec<VersionRow>> {
        let sql = format!(
            "SELECT {VERSION_COLUMNS} FROM versions v JOIN documents d ON d.doc_id = v.doc_id \
             WHERE v.version = (SELECT MAX(version) FROM versions WHERE doc_id = v.doc_id) ORDER BY d.source"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| version_from_row(r, 0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Новая действующая версия документа; прежняя действующая становится
    /// `superseded`.
    pub fn add_version(&mut self, v: &NewVersion) -> Result<VersionRow> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO documents (source, kind) VALUES (?1, ?2) ON CONFLICT(source) DO UPDATE SET kind = excluded.kind",
            params![v.source, v.kind],
        )?;
        let doc_id: i64 = tx.query_row("SELECT doc_id FROM documents WHERE source = ?1", [v.source], |r| r.get(0))?;
        tx.execute("UPDATE versions SET status = 'superseded' WHERE doc_id = ?1 AND status IN ('current', 'removed')", [doc_id])?;
        let next: i64 =
            tx.query_row("SELECT COALESCE(MAX(version), 0) + 1 FROM versions WHERE doc_id = ?1", [doc_id], |r| r.get(0))?;
        tx.execute(
            "INSERT INTO versions (doc_id, version, title, sha256, git_commit, git_dirty, bytes, mtime, indexed_at, status, text) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'current', ?10)",
            params![doc_id, next, v.title, v.sha256, v.git_commit, v.git_dirty, v.bytes, v.mtime, now_secs(), v.text],
        )?;
        tx.commit()?;
        self.current_version(v.source)?.context("версия только что записана")
    }

    /// Обновляет сведения git у действующей версии (коммит мог появиться
    /// позже, чем версия была проиндексирована, — содержимое то же).
    pub fn refresh_git(&self, version_id: i64, commit: Option<&str>, dirty: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE versions SET git_commit = ?2, git_dirty = ?3 WHERE version_id = ?1",
            params![version_id, commit, dirty],
        )?;
        Ok(())
    }

    /// Документы, которых больше нет в корпусе: их действующая версия
    /// становится `removed` и пропадает из поиска. Возвращает их пути.
    pub fn mark_removed(&self, present: &[String]) -> Result<Vec<String>> {
        let mut removed = Vec::new();
        for v in self.latest_versions()? {
            if v.status == "current" && !present.contains(&v.source) {
                self.conn.execute("UPDATE versions SET status = 'removed' WHERE version_id = ?1", [v.version_id])?;
                removed.push(v.source);
            }
        }
        Ok(removed)
    }

    pub fn has_chunks(&self, version_id: i64, strategy: &str) -> Result<bool> {
        Ok(self
            .conn
            .query_row("SELECT 1 FROM chunks WHERE version_id = ?1 AND strategy = ?2 LIMIT 1", params![version_id, strategy], |_| Ok(()))
            .optional()?
            .is_some())
    }

    pub fn has_vector(&self, key: &str) -> Result<bool> {
        Ok(self.conn.query_row("SELECT 1 FROM vectors WHERE vector_key = ?1", [key], |_| Ok(())).optional()?.is_some())
    }

    pub fn put_vectors(&mut self, model: &str, items: &[(String, Vec<f32>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO vectors (vector_key, model, dim, vector) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (key, v) in items {
                stmt.execute(params![key, model, v.len() as i64, vector_to_blob(v)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Параметры, с которыми нарезана версия стратегией (JSON); `None` —
    /// нарезки нет или она сделана до того, как индекс начал их хранить
    /// (тогда — параметрами по умолчанию).
    pub fn chunking_params(&self, version_id: i64, strategy: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT params FROM chunkings WHERE version_id = ?1 AND strategy = ?2",
                params![version_id, strategy],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Параметры последней нарезки документа этой стратегией (по любой из
    /// его версий) — новая версия наследует их при полной индексации.
    pub fn latest_chunking_params(&self, source: &str, strategy: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT k.params FROM chunkings k JOIN versions v ON v.version_id = k.version_id \
                 JOIN documents d ON d.doc_id = v.doc_id WHERE d.source = ?1 AND k.strategy = ?2 \
                 ORDER BY v.version DESC LIMIT 1",
                params![source, strategy],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Стратегии, которыми документ нарезан хоть в одной версии.
    pub fn document_strategies(&self, source: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT c.strategy FROM chunks c JOIN versions v ON v.version_id = c.version_id \
             JOIN documents d ON d.doc_id = v.doc_id WHERE d.source = ?1 \
             UNION SELECT DISTINCT k.strategy FROM chunkings k JOIN versions v ON v.version_id = k.version_id \
             JOIN documents d ON d.doc_id = v.doc_id WHERE d.source = ?1",
        )?;
        let rows = stmt.query_map([source], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Нарезки версии: стратегия → параметры (JSON).
    pub fn chunkings(&self, version_id: i64) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare("SELECT strategy, params FROM chunkings WHERE version_id = ?1 ORDER BY strategy")?;
        let rows = stmt.query_map([version_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Чанки одной версии и стратегии — одной транзакцией: либо все, либо
    /// ни одного (прерванная индексация доделает их в следующий раз).
    /// Прежняя нарезка этой версии стратегией (другими параметрами)
    /// заменяется; её векторы, ставшие ничьими, удаляются.
    pub fn insert_chunks(&mut self, version_id: i64, strategy: &str, params_json: &str, chunks: &[ChunkRow]) -> Result<()> {
        let tx = self.conn.transaction()?;
        let replaced = tx.execute("DELETE FROM chunks WHERE version_id = ?1 AND strategy = ?2", params![version_id, strategy])?;
        tx.execute(
            "INSERT OR REPLACE INTO chunkings (version_id, strategy, params, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![version_id, strategy, params_json, now_secs()],
        )?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO chunks (chunk_id, version_id, strategy, ord, section, start_line, end_line, \
                 start_byte, end_byte, char_len, tokens_est, content_sha256, vector_key, embedding_model, clean_end, \
                 splits_code, text, context_start_byte, context_end_byte, context_start_line, context_end_line, context) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22)",
            )?;
            for c in chunks {
                stmt.execute(params![
                    c.chunk_id,
                    c.version_id,
                    c.strategy,
                    c.ord,
                    c.section,
                    c.start_line,
                    c.end_line,
                    c.start_byte,
                    c.end_byte,
                    c.char_len,
                    c.tokens_est,
                    c.content_sha256,
                    c.vector_key,
                    c.embedding_model,
                    c.clean_end,
                    c.splits_code,
                    c.text,
                    c.context_start_byte,
                    c.context_end_byte,
                    c.context_start_line,
                    c.context_end_line,
                    c.context
                ])?;
            }
        }
        if replaced > 0 {
            tx.execute("DELETE FROM vectors WHERE vector_key NOT IN (SELECT vector_key FROM chunks)", [])?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Чанки стратегии с векторами: действующих версий или, если задан
    /// `version` (с `source`), — этой версии документа.
    pub fn search_set(&self, strategy: &str, source: Option<&str>, version: Option<i64>) -> Result<Vec<(ChunkHit, Vec<f32>)>> {
        let mut sql = format!(
            "SELECT {CHUNK_COLUMNS}, {VERSION_COLUMNS}, x.vector FROM chunks c \
             JOIN versions v ON v.version_id = c.version_id JOIN documents d ON d.doc_id = v.doc_id \
             JOIN vectors x ON x.vector_key = c.vector_key WHERE c.strategy = ?1"
        );
        match version {
            Some(_) => sql.push_str(" AND v.version = ?3"),
            None => sql.push_str(" AND v.status = 'current' AND ?3 IS NULL"),
        }
        sql.push_str(" AND (?2 IS NULL OR d.source = ?2)");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![strategy, source, version], |r| {
            let chunk = chunk_from_row(r)?;
            let version = version_from_row(r, CHUNK_COLUMN_COUNT)?;
            let blob: Vec<u8> = r.get(CHUNK_COLUMN_COUNT + 12)?;
            Ok((ChunkHit { chunk, version }, blob_to_vector(&blob)))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Стратегии, для которых в индексе есть чанки действующих версий.
    pub fn strategies(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT c.strategy FROM chunks c JOIN versions v ON v.version_id = c.version_id WHERE v.status = 'current'",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn stats(&self, strategy: &str) -> Result<StrategyStats> {
        let mut stmt = self.conn.prepare(
            "SELECT c.char_len, c.section IS NOT NULL, c.clean_end, c.splits_code, c.tokens_est, c.version_id, d.kind \
             FROM chunks c JOIN versions v ON v.version_id = c.version_id JOIN documents d ON d.doc_id = v.doc_id \
             WHERE v.status = 'current' AND c.strategy = ?1",
        )?;
        let mut stats = StrategyStats::default();
        let mut docs = std::collections::HashSet::new();
        let mut rows = stmt.query([strategy])?;
        while let Some(r) = rows.next()? {
            stats.chunks += 1;
            stats.lengths.push(r.get::<_, i64>(0)? as usize);
            stats.with_section += r.get::<_, bool>(1)? as usize;
            let clean_end: bool = r.get(2)?;
            let splits_code: bool = r.get(3)?;
            stats.tokens_est += r.get::<_, i64>(4)? as usize;
            docs.insert(r.get::<_, i64>(5)?);
            if r.get::<_, String>(6)? == "rust" {
                stats.code_chunks += 1;
                stats.code_split += splits_code as usize;
            } else {
                stats.text_chunks += 1;
                stats.text_dirty_end += !clean_end as usize;
                stats.text_split_fence += splits_code as usize;
            }
        }
        stats.documents = docs.len();
        Ok(stats)
    }

    /// Число чанков у версии по стратегиям.
    pub fn chunk_counts(&self, version_id: i64) -> Result<Vec<(String, usize)>> {
        let mut stmt =
            self.conn.prepare("SELECT strategy, COUNT(*) FROM chunks WHERE version_id = ?1 GROUP BY strategy ORDER BY strategy")?;
        let rows = stmt.query_map([version_id], |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as usize)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn count(&self, table: &str) -> Result<usize> {
        Ok(self.conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get::<_, i64>(0))? as usize)
    }

    /// Текст версии документа (`None` — версия проиндексирована до того, как
    /// индекс начал хранить текст).
    pub fn version_text(&self, version_id: i64) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT text FROM versions WHERE version_id = ?1", [version_id], |r| r.get(0))?)
    }

    /// Версия документа по номеру.
    pub fn version(&self, source: &str, version: i64) -> Result<Option<VersionRow>> {
        let sql = format!(
            "SELECT {VERSION_COLUMNS} FROM versions v JOIN documents d ON d.doc_id = v.doc_id \
             WHERE d.source = ?1 AND v.version = ?2"
        );
        Ok(self.conn.query_row(&sql, params![source, version], |r| version_from_row(r, 0)).optional()?)
    }

    /// Чанки версии и стратегии по порядку: страница `offset..offset+limit`
    /// и общее число.
    pub fn chunks_page(&self, version_id: i64, strategy: &str, offset: usize, limit: usize) -> Result<(Vec<ChunkRow>, usize)> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE version_id = ?1 AND strategy = ?2",
            params![version_id, strategy],
            |r| r.get(0),
        )?;
        let sql = format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks c WHERE c.version_id = ?1 AND c.strategy = ?2 ORDER BY c.ord LIMIT ?3 OFFSET ?4"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![version_id, strategy, limit as i64, offset as i64], chunk_from_row)?;
        Ok((rows.collect::<rusqlite::Result<_>>()?, total as usize))
    }

    /// Чанк по идентификатору вместе с версией и вектором (если он есть).
    pub fn chunk(&self, chunk_id: &str) -> Result<Option<(ChunkHit, Option<Vec<f32>>)>> {
        let sql = format!(
            "SELECT {CHUNK_COLUMNS}, {VERSION_COLUMNS}, x.vector FROM chunks c \
             JOIN versions v ON v.version_id = c.version_id JOIN documents d ON d.doc_id = v.doc_id \
             LEFT JOIN vectors x ON x.vector_key = c.vector_key WHERE c.chunk_id = ?1"
        );
        Ok(self
            .conn
            .query_row(&sql, [chunk_id], |r| {
                let chunk = chunk_from_row(r)?;
                let version = version_from_row(r, CHUNK_COLUMN_COUNT)?;
                let blob: Option<Vec<u8>> = r.get(CHUNK_COLUMN_COUNT + 12)?;
                Ok((ChunkHit { chunk, version }, blob.map(|b| blob_to_vector(&b))))
            })
            .optional()?)
    }

    /// Удаляет документ со всеми версиями и чанками, затем векторы, на
    /// которые больше никто не ссылается. Возвращает (версий, чанков,
    /// векторов) или `None`, если документа нет.
    pub fn delete_document(&mut self, source: &str) -> Result<Option<(usize, usize, usize)>> {
        let tx = self.conn.transaction()?;
        let Some(doc_id): Option<i64> =
            tx.query_row("SELECT doc_id FROM documents WHERE source = ?1", [source], |r| r.get(0)).optional()?
        else {
            return Ok(None);
        };
        let chunks = tx.execute(
            "DELETE FROM chunks WHERE version_id IN (SELECT version_id FROM versions WHERE doc_id = ?1)",
            [doc_id],
        )?;
        tx.execute(
            "DELETE FROM chunkings WHERE version_id IN (SELECT version_id FROM versions WHERE doc_id = ?1)",
            [doc_id],
        )?;
        let versions = tx.execute("DELETE FROM versions WHERE doc_id = ?1", [doc_id])?;
        tx.execute("DELETE FROM documents WHERE doc_id = ?1", [doc_id])?;
        let vectors = tx.execute("DELETE FROM vectors WHERE vector_key NOT IN (SELECT vector_key FROM chunks)", [])?;
        tx.commit()?;
        Ok(Some((versions, chunks, vectors)))
    }

    /// Удаляет заменённые и удалённые версии с их чанками, затем векторы,
    /// на которые больше не ссылается ни один чанк. Возвращает (версий,
    /// чанков, векторов).
    pub fn prune(&mut self) -> Result<(usize, usize, usize)> {
        let tx = self.conn.transaction()?;
        let chunks = tx.execute(
            "DELETE FROM chunks WHERE version_id IN (SELECT version_id FROM versions WHERE status != 'current')",
            [],
        )?;
        tx.execute(
            "DELETE FROM chunkings WHERE version_id IN (SELECT version_id FROM versions WHERE status != 'current')",
            [],
        )?;
        let versions = tx.execute("DELETE FROM versions WHERE status != 'current'", [])?;
        tx.execute("DELETE FROM documents WHERE doc_id NOT IN (SELECT doc_id FROM versions)", [])?;
        let vectors = tx.execute("DELETE FROM vectors WHERE vector_key NOT IN (SELECT vector_key FROM chunks)", [])?;
        tx.commit()?;
        Ok((versions, chunks, vectors))
    }

    /// Очищает индекс целиком (нужно при смене модели эмбеддингов).
    pub fn reset(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM chunks; DELETE FROM chunkings; DELETE FROM vectors; DELETE FROM versions; DELETE FROM documents; DELETE FROM meta;",
        )?;
        Ok(())
    }
}
