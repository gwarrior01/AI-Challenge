//! RAG-режим агента: вопрос → поиск релевантных чанков в индексе →
//! объединение найденного с вопросом → запрос к LLM.
//!
//! Режим включается на агента ([`crate::AgentConfig::rag`]): `None` — агент
//! отвечает только из своих знаний, как раньше; [`RagSettings`] — перед
//! каждым вопросом человека ищутся `k` ближайших чанков выбранной стратегии,
//! и в запрос уходит не голый вопрос, а фрагменты с источниками и сам вопрос
//! ([`augment_prompt`]). В истории диалога остаётся исходный вопрос: контекст
//! нужен только для ответа на него, и следующий вопрос получит свой.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use super::{indexed_strategies, search, Embedder, RagConfig, SearchOptions, Strategy};

/// Сколько чанков по умолчанию уходит в контекст.
pub const DEFAULT_K: usize = 5;
/// Порог, при котором фрагмент ещё считают релевантным, если он не задан;
/// 0 — порога нет, модель сама отбрасывает лишнее по инструкции.
pub const DEFAULT_MIN_SCORE: f32 = 0.0;

/// Настройки RAG-режима агента.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagSettings {
    /// Стратегия нарезки, по чанкам которой ищем; `None` — первая
    /// проиндексированная в порядке [`Strategy::ALL`].
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default = "default_k")]
    pub k: usize,
    /// Чанки с близостью ниже порога в контекст не попадают.
    #[serde(default)]
    pub min_score: f32,
}

fn default_k() -> usize {
    DEFAULT_K
}

impl Default for RagSettings {
    fn default() -> Self {
        Self { strategy: None, k: DEFAULT_K, min_score: DEFAULT_MIN_SCORE }
    }
}

impl RagSettings {
    /// Разбор слов команды `rag on [стратегия] [k=N] [min=0.5]`, общей для
    /// интерфейсов.
    pub fn parse_args(args: &[&str]) -> Result<Self> {
        let mut settings = Self::default();
        for arg in args {
            if let Some(v) = arg.strip_prefix("k=") {
                settings.k = v.parse().ok().filter(|&k| (1..=20).contains(&k)).ok_or_else(|| anyhow::anyhow!("k — число от 1 до 20"))?;
            } else if let Some(v) = arg.strip_prefix("min=") {
                settings.min_score =
                    v.parse().ok().filter(|m| (0.0..=1.0).contains(m)).ok_or_else(|| anyhow::anyhow!("min — число от 0 до 1"))?;
            } else if let Some(s) = Strategy::parse(arg) {
                settings.strategy = Some(s.name().to_string());
            } else {
                bail!("непонятный параметр «{arg}» — rag on [fixed|structure|sentence|parent] [k=N] [min=0.5]");
            }
        }
        Ok(settings)
    }

    /// Одной строкой — для статуса агента в интерфейсах.
    pub fn describe(&self) -> String {
        let strategy = self.strategy.as_deref().unwrap_or("авто");
        let mut s = format!("RAG: {strategy}, k={}", self.k);
        if self.min_score > 0.0 {
            s.push_str(&format!(", порог {:.2}", self.min_score));
        }
        s
    }
}

/// Найденный фрагмент — то, что ушло модели под номером `n`.
#[derive(Debug, Clone, Serialize)]
pub struct RagSource {
    pub n: usize,
    pub source: String,
    pub version: i64,
    pub section: Option<String>,
    pub start_line: i64,
    pub end_line: i64,
    pub score: f32,
    /// Текст, отданный модели (у parent-child — родительский абзац).
    pub text: String,
}

impl RagSource {
    /// `README.md › Раздел · строки 10–24`
    pub fn location(&self) -> String {
        let mut s = self.source.clone();
        if self.version > 1 {
            s.push_str(&format!(" (версия {})", self.version));
        }
        if let Some(section) = self.section.as_deref().filter(|s| !s.is_empty()) {
            s.push_str(&format!(" › {section}"));
        }
        s.push_str(&format!(" · строки {}–{}", self.start_line, self.end_line));
        s
    }
}

/// Что поиск дал для одного вопроса: фрагменты или причина, по которой их нет.
#[derive(Debug, Clone, Serialize)]
pub struct RagContext {
    pub strategy: String,
    pub sources: Vec<RagSource>,
    /// Отброшено по порогу близости.
    pub below_threshold: usize,
    /// Поиск не удался (нет индекса, сервер эмбеддингов недоступен…) — ответ
    /// тогда дан без контекста, интерфейсы об этом предупреждают.
    pub error: Option<String>,
}

impl RagContext {
    /// Строки для интерфейсов: источники по номерам или почему их нет.
    pub fn summary_lines(&self) -> Vec<String> {
        if let Some(err) = &self.error {
            return vec![format!("RAG: поиск не удался — ответ без контекста: {err}")];
        }
        if self.sources.is_empty() {
            let reason = if self.below_threshold > 0 {
                format!("все {} найденных фрагментов ниже порога близости", self.below_threshold)
            } else {
                "в индексе ничего не нашлось".to_string()
            };
            return vec![format!("RAG ({}): {reason}", self.strategy)];
        }
        let mut lines = vec![format!("RAG ({}): {} фрагм.", self.strategy, self.sources.len())];
        lines.extend(self.sources.iter().map(|s| format!("[{}] {} · {:.2}", s.n, s.location(), s.score)));
        lines
    }
}

/// Поиск по индексу для вопроса. Ошибку не пробрасывает — она уходит в
/// [`RagContext::error`], чтобы недоступный сервер эмбеддингов не оставлял
/// человека вовсе без ответа.
pub async fn retrieve(question: &str, settings: &RagSettings) -> RagContext {
    let strategy = settings.strategy.clone().unwrap_or_default();
    match try_retrieve(question, settings).await {
        Ok(ctx) => ctx,
        Err(err) => RagContext { strategy, sources: Vec::new(), below_threshold: 0, error: Some(format!("{err:#}")) },
    }
}

async fn try_retrieve(question: &str, settings: &RagSettings) -> Result<RagContext> {
    let cfg = RagConfig::from_env();
    let strategy = resolve_strategy(&cfg, settings)?;
    let embedder = Embedder::from_env()?;
    let opts = SearchOptions { strategies: vec![strategy], k: settings.k.max(1), ..SearchOptions::default() };
    let hits = search(&cfg, &embedder, question, &opts).await?.into_iter().next().map(|(_, hits)| hits).unwrap_or_default();
    let total = hits.len();
    let sources: Vec<RagSource> = hits
        .into_iter()
        .filter(|h| h.score >= settings.min_score)
        .enumerate()
        .map(|(i, h)| {
            let c = &h.hit.chunk;
            RagSource {
                n: i + 1,
                source: h.hit.version.source.clone(),
                version: h.hit.version.version,
                section: c.section.clone(),
                start_line: c.context_start_line.unwrap_or(c.start_line),
                end_line: c.context_end_line.unwrap_or(c.end_line),
                score: h.score,
                text: c.context_text().trim().to_string(),
            }
        })
        .collect();
    Ok(RagContext { strategy: strategy.name().to_string(), below_threshold: total - sources.len(), sources, error: None })
}

fn resolve_strategy(cfg: &RagConfig, settings: &RagSettings) -> Result<Strategy> {
    let indexed = indexed_strategies(cfg)?;
    match settings.strategy.as_deref() {
        Some(name) => {
            let strategy = Strategy::parse(name).ok_or_else(|| anyhow::anyhow!("неизвестная стратегия «{name}»"))?;
            if !indexed.contains(&strategy) {
                bail!("в индексе нет чанков стратегии «{name}» — проиндексируйте документы ею или выберите другую");
            }
            Ok(strategy)
        }
        None => indexed.first().copied().ok_or_else(|| anyhow::anyhow!("индекс документов пуст — добавьте документы (rag add …)")),
    }
}

/// Вопрос вместе с найденными фрагментами — то, что уходит модели вместо
/// голого вопроса. Без фрагментов (ничего не нашлось или ниже порога) модель
/// всё равно узнаёт, что база ответа не содержит, — иначе она молча ответила
/// бы из общих знаний, будто из документов. Ошибка поиска — вопрос как есть.
pub fn augment_prompt(question: &str, ctx: &RagContext) -> String {
    if ctx.error.is_some() {
        return question.to_string();
    }
    if ctx.sources.is_empty() {
        return format!(
            "[Режим RAG] В базе документов не нашлось фрагментов, относящихся к вопросу. Скажи об этом прямо; \
             если отвечаешь из общих знаний — явно пометь, что это не из документов.\n\nВопрос: {question}"
        );
    }
    let mut s = String::from(
        "[Режим RAG] Ниже — фрагменты из базы документов, найденные поиском по вопросу. Отвечай на их основе:\n\
         - опирайся на факты из фрагментов и ссылайся на них номерами в квадратных скобках — [1], [2];\n\
         - фрагменты, не относящиеся к вопросу, игнорируй;\n\
         - если во фрагментах ответа нет или он неполон — так и скажи; то, что добавляешь из общих знаний, \
         явно пометь как не из документов;\n\
         - не выдумывай источники, цифры и цитаты.\n\n",
    );
    for src in &ctx.sources {
        s.push_str(&format!("[{}] {} (близость {:.2})\n{}\n\n", src.n, src.location(), src.score, src.text));
    }
    s.push_str(&format!("Вопрос: {question}"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(n: usize) -> RagSource {
        RagSource {
            n,
            source: "README.md".into(),
            version: 1,
            section: Some("MCP".into()),
            start_line: 10,
            end_line: 12,
            score: 0.71,
            text: "адрес http://127.0.0.1:8092/mcp".into(),
        }
    }

    #[test]
    fn parses_settings() {
        let s = RagSettings::parse_args(&["structure", "k=3", "min=0.5"]).unwrap();
        assert_eq!(s, RagSettings { strategy: Some("structure".into()), k: 3, min_score: 0.5 });
        assert_eq!(RagSettings::parse_args(&[]).unwrap(), RagSettings::default());
        assert!(RagSettings::parse_args(&["k=0"]).is_err());
        assert!(RagSettings::parse_args(&["борщ"]).is_err());
    }

    #[test]
    fn prompt_carries_sources_and_question() {
        let ctx = RagContext { strategy: "fixed".into(), sources: vec![source(1)], below_threshold: 0, error: None };
        let p = augment_prompt("Какой адрес?", &ctx);
        assert!(p.contains("[1] README.md › MCP · строки 10–12"));
        assert!(p.contains("127.0.0.1:8092"));
        assert!(p.ends_with("Вопрос: Какой адрес?"));
    }

    #[test]
    fn empty_context_says_so_and_error_passes_question() {
        let empty = RagContext { strategy: "fixed".into(), sources: vec![], below_threshold: 2, error: None };
        assert!(augment_prompt("q", &empty).contains("не нашлось"));
        assert!(empty.summary_lines()[0].contains("ниже порога"));
        let failed = RagContext { error: Some("нет связи".into()), ..empty };
        assert_eq!(augment_prompt("q", &failed), "q");
    }

    #[test]
    fn old_configs_have_no_rag() {
        let s: RagSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, RagSettings::default());
    }
}
