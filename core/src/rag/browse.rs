//! Просмотр индекса: сводка, документы, версии, чанки, карта документа —
//! структурами (для веб-интерфейса), а не текстом, как у [`super::run_command`].

use anyhow::{anyhow, Result};
use serde::Serialize;

use super::store::{ChunkRow, Store, VersionRow};
use super::{fmt_time, parse_params, percentile, ChunkParams, ParamSpec, RagConfig, Strategy, META_DIM, META_INDEXED_AT, META_MODEL};

#[derive(Debug, Serialize)]
pub struct StrategyOverview {
    pub strategy: &'static str,
    pub description: String,
    /// Параметры стратегии с диапазонами и значениями по умолчанию — для
    /// формы загрузки.
    pub params: Vec<ParamSpec>,
    pub chunks: usize,
    pub documents: usize,
    pub min_chars: usize,
    pub median_chars: usize,
    pub max_chars: usize,
}

#[derive(Debug, Serialize)]
pub struct Overview {
    pub db_path: String,
    pub model: Option<String>,
    pub dim: Option<usize>,
    pub indexed_at: Option<String>,
    pub documents: usize,
    pub versions: usize,
    pub vectors: usize,
    pub db_bytes: u64,
    pub uploads_dir: String,
    pub strategies: Vec<StrategyOverview>,
}

pub fn overview(cfg: &RagConfig) -> Result<Overview> {
    let store = Store::open(&cfg.db_path)?;
    let mut strategies = Vec::new();
    for s in Strategy::ALL {
        let st = store.stats(s.name())?;
        let mut lengths = st.lengths;
        lengths.sort_unstable();
        strategies.push(StrategyOverview {
            strategy: s.name(),
            description: s.describe(),
            params: ChunkParams::specs(s),
            chunks: st.chunks,
            documents: st.documents,
            min_chars: lengths.first().copied().unwrap_or(0),
            median_chars: percentile(&lengths, 0.5),
            max_chars: lengths.last().copied().unwrap_or(0),
        });
    }
    Ok(Overview {
        db_path: cfg.source_of(&cfg.db_path),
        model: store.meta(META_MODEL)?,
        dim: store.meta(META_DIM)?.and_then(|d| d.parse().ok()),
        indexed_at: store.meta(META_INDEXED_AT)?.and_then(|s| s.parse().ok()).map(fmt_time),
        documents: store.latest_versions()?.iter().filter(|v| v.status == "current").count(),
        versions: store.count("versions")?,
        vectors: store.count("vectors")?,
        db_bytes: std::fs::metadata(&cfg.db_path).map(|m| m.len()).unwrap_or(0),
        uploads_dir: cfg.source_of(&cfg.uploads),
        strategies,
    })
}

#[derive(Debug, Serialize)]
pub struct DocumentInfo {
    #[serde(flatten)]
    pub version: VersionRow,
    pub indexed_at_text: String,
    /// Всего версий документа в индексе.
    pub versions: usize,
    /// Чанки последней версии по стратегиям.
    pub chunks: Vec<(String, usize)>,
    /// Документ загружен (лежит в папке загрузок).
    pub uploaded: bool,
}

/// Документы индекса с их последней версией.
pub fn documents(cfg: &RagConfig) -> Result<Vec<DocumentInfo>> {
    let store = Store::open(&cfg.db_path)?;
    let uploads = format!("{}/", cfg.source_of(&cfg.uploads));
    store
        .latest_versions()?
        .into_iter()
        .map(|v| {
            Ok(DocumentInfo {
                indexed_at_text: fmt_time(v.indexed_at),
                versions: store.versions(&v.source)?.len(),
                chunks: store.chunk_counts(v.version_id)?,
                uploaded: v.source.starts_with(&uploads),
                version: v,
            })
        })
        .collect()
}

/// Нарезка версии стратегией: параметры и они же строкой.
#[derive(Debug, Serialize)]
pub struct Chunking {
    pub strategy: String,
    pub params: serde_json::Map<String, serde_json::Value>,
    pub description: String,
}

fn chunking(strategy: Strategy, params: &ChunkParams) -> Chunking {
    Chunking { strategy: strategy.name().to_string(), params: params.of(strategy), description: params.describe(strategy) }
}

/// Параметры, которыми версия нарезана стратегией; нарезка без записи
/// параметров (индекс до их появления) — по умолчанию. `None` — нарезки нет.
fn chunking_of(store: &Store, version_id: i64, strategy: Strategy) -> Result<Option<Chunking>> {
    Ok(match store.chunking_params(version_id, strategy.name())? {
        Some(json) => Some(chunking(strategy, &parse_params(&json))),
        None if store.has_chunks(version_id, strategy.name())? => Some(chunking(strategy, &ChunkParams::default())),
        None => None,
    })
}

#[derive(Debug, Serialize)]
pub struct VersionInfo {
    #[serde(flatten)]
    pub version: VersionRow,
    pub indexed_at_text: String,
    pub chunks: Vec<(String, usize)>,
    /// Нарезки версии с их параметрами.
    pub chunkings: Vec<Chunking>,
    /// Есть ли текст версии (для карты документа).
    pub has_text: bool,
}

pub fn versions(cfg: &RagConfig, source: &str) -> Result<Vec<VersionInfo>> {
    let store = Store::open(&cfg.db_path)?;
    let all = store.versions(source)?;
    if all.is_empty() {
        return Err(anyhow!("документа «{source}» нет в индексе"));
    }
    all.into_iter()
        .map(|v| {
            Ok(VersionInfo {
                indexed_at_text: fmt_time(v.indexed_at),
                chunks: store.chunk_counts(v.version_id)?,
                chunkings: Strategy::ALL
                    .into_iter()
                    .filter_map(|s| chunking_of(&store, v.version_id, s).transpose())
                    .collect::<Result<_>>()?,
                has_text: store.version_text(v.version_id)?.is_some(),
                version: v,
            })
        })
        .collect()
}

/// Версия документа: заданная номером или последняя.
fn resolve_version(store: &Store, source: &str, version: Option<i64>) -> Result<VersionRow> {
    let found = match version {
        Some(n) => store.version(source, n)?,
        None => store.versions(source)?.into_iter().next(),
    };
    found.ok_or_else(|| match version {
        Some(n) => anyhow!("у документа «{source}» нет версии v{n}"),
        None => anyhow!("документа «{source}» нет в индексе"),
    })
}

fn strategy(name: &str) -> Result<Strategy> {
    Strategy::parse(name).ok_or_else(|| anyhow!("неизвестная стратегия «{name}» — есть fixed, structure, sentence, parent"))
}

#[derive(Debug, Serialize)]
pub struct ChunkPage {
    pub version: VersionRow,
    pub strategy: &'static str,
    pub chunking: Option<Chunking>,
    pub total: usize,
    pub offset: usize,
    pub items: Vec<ChunkRow>,
}

pub fn chunks(cfg: &RagConfig, source: &str, version: Option<i64>, strategy_name: &str, offset: usize, limit: usize) -> Result<ChunkPage> {
    let s = strategy(strategy_name)?;
    let store = Store::open(&cfg.db_path)?;
    let version = resolve_version(&store, source, version)?;
    let (items, total) = store.chunks_page(version.version_id, s.name(), offset, limit.clamp(1, 500))?;
    let chunking = chunking_of(&store, version.version_id, s)?;
    Ok(ChunkPage { version, strategy: s.name(), chunking, total, offset, items })
}

#[derive(Debug, Serialize)]
pub struct VectorInfo {
    pub dim: usize,
    pub norm: f32,
    /// Первые числа вектора — показать, что он есть и какой.
    pub head: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDetail {
    pub chunk: ChunkRow,
    pub version: VersionRow,
    pub vector: Option<VectorInfo>,
}

pub fn chunk(cfg: &RagConfig, chunk_id: &str) -> Result<ChunkDetail> {
    let store = Store::open(&cfg.db_path)?;
    let (hit, vector) = store.chunk(chunk_id)?.ok_or_else(|| anyhow!("чанка «{chunk_id}» нет в индексе"))?;
    let vector = vector.map(|v| VectorInfo {
        dim: v.len(),
        norm: v.iter().map(|x| x * x).sum::<f32>().sqrt(),
        head: v.iter().take(8).copied().collect(),
    });
    Ok(ChunkDetail { chunk: hit.chunk, version: hit.version, vector })
}

/// Чанк на карте документа: границы — в единицах UTF-16 (так индексирует
/// строки JavaScript), чтобы страница могла выделить их в тексте.
#[derive(Debug, Serialize)]
pub struct MapChunk {
    pub chunk_id: String,
    pub ord: i64,
    pub section: Option<String>,
    pub start: usize,
    pub end: usize,
    pub start_line: i64,
    pub end_line: i64,
    pub char_len: i64,
    /// Parent-child: границы родителя (UTF-16) и его номер по порядку —
    /// дети одного родителя на карте одного цвета.
    pub parent: Option<(usize, usize)>,
    pub group: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct DocumentMap {
    pub version: VersionRow,
    pub strategy: &'static str,
    pub chunking: Option<Chunking>,
    /// Текст версии; `None` — версия проиндексирована до того, как индекс
    /// начал хранить текст.
    pub text: Option<String>,
    pub chunks: Vec<MapChunk>,
}

/// Текст версии и границы всех её чанков стратегии.
pub fn document_map(cfg: &RagConfig, source: &str, version: Option<i64>, strategy_name: &str) -> Result<DocumentMap> {
    let s = strategy(strategy_name)?;
    let store = Store::open(&cfg.db_path)?;
    let version = resolve_version(&store, source, version)?;
    let text = store.version_text(version.version_id)?;
    let (rows, _) = store.chunks_page(version.version_id, s.name(), 0, usize::MAX >> 1)?;
    let chunks = match &text {
        Some(text) => {
            let u16_at = |byte: i64| -> usize {
                let b = (byte as usize).min(text.len());
                if text.is_char_boundary(b) {
                    text[..b].encode_utf16().count()
                } else {
                    0
                }
            };
            let mut groups: Vec<(i64, i64)> = Vec::new();
            rows.into_iter()
                .map(|c| {
                    let range = c.context_start_byte.zip(c.context_end_byte);
                    let group = range.map(|r| match groups.iter().position(|g| *g == r) {
                        Some(i) => i,
                        None => {
                            groups.push(r);
                            groups.len() - 1
                        }
                    });
                    (c, range, group)
                })
                .map(|(c, range, group)| MapChunk {
                    parent: range.map(|(a, b)| (u16_at(a), u16_at(b))),
                    group,
                    start: u16_at(c.start_byte),
                    end: u16_at(c.end_byte),
                    chunk_id: c.chunk_id,
                    ord: c.ord,
                    section: c.section,
                    start_line: c.start_line,
                    end_line: c.end_line,
                    char_len: c.char_len,
                })
                .collect()
        }
        None => Vec::new(),
    };
    let chunking = chunking_of(&store, version.version_id, s)?;
    Ok(DocumentMap { version, strategy: s.name(), chunking, text, chunks })
}
