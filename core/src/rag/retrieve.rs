//! RAG-режим агента: вопрос → поиск релевантных чанков в индексе →
//! объединение найденного с вопросом → запрос к LLM.
//!
//! Режим включается на агента ([`crate::AgentConfig::rag`]): `None` — агент
//! отвечает только из своих знаний, как раньше; [`RagSettings`] — перед
//! каждым вопросом человека ищутся `k` ближайших чанков выбранной стратегии,
//! и в запрос уходит не голый вопрос, а фрагменты с источниками и сам вопрос
//! ([`augment_prompt`]). В истории диалога остаётся исходный вопрос: контекст
//! нужен только для ответа на него, и следующий вопрос получит свой.
//!
//! Поиск идёт в два этапа ([`retrieve`]):
//!
//! ```text
//! вопрос → [переписывание] → эмбеддинг → candidates ближайших
//!        → порог близости min_score → [реранкинг] → порог реранкера rerank_min → k лучших
//! ```
//!
//! Переписывание и реранкинг — [`super::rerank`], оба выключены по умолчанию.
//!
//! Если ни один фрагмент не прошёл пороги, модель не спрашивают: ответ —
//! «не знаю» и просьба уточнить вопрос ([`super::answer::refusal`]). Иначе
//! модель отвечает блоками «Ответ / Источники / Цитаты», и код их проверяет
//! ([`super::answer`]).

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use super::rerank::{self, LlmStep, Rerank};
use super::{indexed_strategies, search_queries, Embedder, RagConfig, Strategy};

/// Сколько чанков по умолчанию уходит в контекст.
pub const DEFAULT_K: usize = 5;
/// Порог близости, ниже которого фрагмент не считают релевантным, если он не
/// задан. Ничего не прошло порог — ответ «не знаю» без модели. 0.5 — под
/// bge-m3: в `rag/compare.md` найденные ответы — 0.63–0.71.
pub const DEFAULT_MIN_SCORE: f32 = 0.5;
/// Сколько кандидатов по умолчанию берёт первый этап — поиск по эмбеддингу —
/// до фильтра и реранкинга.
pub const DEFAULT_CANDIDATES: usize = 20;
/// Верхняя граница `candidates`: реранкер `llm` читает всех кандидатов одним
/// запросом.
pub const MAX_CANDIDATES: usize = 50;

/// Параметры команды `rag on …` — одна строка для подсказок всех интерфейсов.
pub const RAG_ON_USAGE: &str =
    "rag on [fixed|structure|sentence|parent] [k=5] [n=20] [min=0.5] [rewrite] [rerank=heuristic|llm] [rmin=0.5]";

/// Настройки RAG-режима агента.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagSettings {
    /// Стратегия нарезки, по чанкам которой ищем; `None` — первая
    /// проиндексированная в порядке [`Strategy::ALL`].
    #[serde(default)]
    pub strategy: Option<String>,
    /// Сколько фрагментов уходит в контекст — топ-K после фильтра и реранкинга.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Чанки с близостью ниже порога в контекст не попадают.
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Топ-N первого этапа: сколько ближайших чанков ищется до фильтра и
    /// реранкинга (не меньше `k`).
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    /// Переписать вопрос в поисковый запрос перед поиском.
    #[serde(default)]
    pub rewrite: bool,
    #[serde(default)]
    pub rerank: Rerank,
    /// Порог оценки реранкера: у `llm` — оценка модели 0–1, у `heuristic` —
    /// близость с бонусом за термины. 0 — порога нет.
    #[serde(default)]
    pub rerank_min: f32,
}

fn default_k() -> usize {
    DEFAULT_K
}

fn default_min_score() -> f32 {
    DEFAULT_MIN_SCORE
}

fn default_candidates() -> usize {
    DEFAULT_CANDIDATES
}

impl Default for RagSettings {
    fn default() -> Self {
        Self {
            strategy: None,
            k: DEFAULT_K,
            min_score: DEFAULT_MIN_SCORE,
            candidates: DEFAULT_CANDIDATES,
            rewrite: false,
            rerank: Rerank::None,
            rerank_min: 0.0,
        }
    }
}

impl RagSettings {
    /// Разбор слов команды [`RAG_ON_USAGE`], общей для интерфейсов.
    pub fn parse_args(args: &[&str]) -> Result<Self> {
        let unit = |v: &str, name: &str| -> Result<f32> {
            v.parse().ok().filter(|m| (0.0..=1.0).contains(m)).ok_or_else(|| anyhow::anyhow!("{name} — число от 0 до 1"))
        };
        let mut settings = Self::default();
        let mut candidates = None;
        for arg in args {
            if let Some(v) = arg.strip_prefix("k=") {
                settings.k = v.parse().ok().filter(|&k| (1..=20).contains(&k)).ok_or_else(|| anyhow::anyhow!("k — число от 1 до 20"))?;
            } else if let Some(v) = arg.strip_prefix("n=") {
                candidates = Some(
                    v.parse()
                        .ok()
                        .filter(|&n| (1..=MAX_CANDIDATES).contains(&n))
                        .ok_or_else(|| anyhow::anyhow!("n — число от 1 до {MAX_CANDIDATES}"))?,
                );
            } else if let Some(v) = arg.strip_prefix("min=") {
                settings.min_score = unit(v, "min")?;
            } else if let Some(v) = arg.strip_prefix("rmin=") {
                settings.rerank_min = unit(v, "rmin")?;
            } else if let Some(v) = arg.strip_prefix("rerank=") {
                settings.rerank = Rerank::parse(v).ok_or_else(|| anyhow::anyhow!("rerank — none, heuristic или llm"))?;
            } else if *arg == "rewrite" {
                settings.rewrite = true;
            } else if let Some(s) = Strategy::parse(arg) {
                settings.strategy = Some(s.name().to_string());
            } else {
                bail!("непонятный параметр «{arg}» — {RAG_ON_USAGE}");
            }
        }
        // Без явного n кандидатов не меньше k — иначе k=30 молча урезалось бы.
        settings.candidates = match candidates {
            Some(n) if n < settings.k => bail!("n (кандидатов до фильтра) не может быть меньше k ({})", settings.k),
            Some(n) => n,
            None => DEFAULT_CANDIDATES.max(settings.k),
        };
        if settings.rerank_min > 0.0 && settings.rerank == Rerank::None {
            bail!("rmin — порог реранкера: задайте и rerank=heuristic или rerank=llm");
        }
        Ok(settings)
    }

    /// Одной строкой — для статуса агента в интерфейсах.
    pub fn describe(&self) -> String {
        let strategy = self.strategy.as_deref().unwrap_or("авто");
        let mut s = format!("RAG: {strategy}, топ-{} → k={}", self.candidates, self.k);
        if self.min_score > 0.0 {
            s.push_str(&format!(", порог {:.2}", self.min_score));
        }
        if self.rewrite {
            s.push_str(", rewrite");
        }
        if self.rerank != Rerank::None {
            s.push_str(&format!(", rerank {}", self.rerank.name()));
            if self.rerank_min > 0.0 {
                s.push_str(&format!(" ≥ {:.2}", self.rerank_min));
            }
        }
        s
    }
}

/// Найденный фрагмент — то, что ушло модели под номером `n`.
#[derive(Debug, Clone, Serialize)]
pub struct RagSource {
    pub n: usize,
    /// `README.md@v2/structure/0007` — чанк в индексе.
    pub chunk_id: String,
    pub source: String,
    /// Документ для показа — без папки загрузок (см. [`super::display_source`]).
    pub name: String,
    pub version: i64,
    pub section: Option<String>,
    pub start_line: i64,
    pub end_line: i64,
    /// Косинусная близость к запросу (лучшая из формулировок).
    pub score: f32,
    /// Оценка реранкера, если он работал.
    pub rerank_score: Option<f32>,
    /// Место среди кандидатов по близости (с 1) — до реранкинга.
    pub rank_before: usize,
    /// Текст, отданный модели (у parent-child — родительский абзац).
    pub text: String,
}

impl RagSource {
    /// `README.md › Раздел · строки 10–24`
    pub fn location(&self) -> String {
        let mut s = self.name.clone();
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
#[derive(Debug, Clone, Default, Serialize)]
pub struct RagContext {
    pub strategy: String,
    pub sources: Vec<RagSource>,
    /// Переписанный запрос, если переписывание включено и удалось.
    pub query: Option<String>,
    /// Реранкер, который действительно отработал (`none`, если выключен или
    /// упал — тогда порядок по близости).
    pub rerank: Rerank,
    /// Кандидатов нашёл первый этап (топ-N по близости).
    pub candidates: usize,
    /// Отброшено по порогу близости.
    pub below_threshold: usize,
    /// Отброшено по порогу реранкера.
    pub below_rerank: usize,
    /// Токены вспомогательных запросов к модели (переписывание, оценка).
    pub tokens: u64,
    /// Что из второго этапа не удалось — поиск при этом продолжился без него.
    pub warnings: Vec<String>,
    /// Поиск не удался (нет индекса, сервер эмбеддингов недоступен…) — ответ
    /// тогда дан без контекста, интерфейсы об этом предупреждают.
    pub error: Option<String>,
    /// Пороги, с которыми шёл поиск, — для объяснения «не знаю».
    pub min_score: f32,
    pub rerank_min: f32,
    /// Лучшая близость среди кандидатов.
    pub best_score: Option<f32>,
    /// Ничего не прошло порог — ближайшие кандидаты (до 3), подсказка, как
    /// уточнить вопрос.
    pub nearest: Vec<RagSource>,
    /// Проверка ответа модели (см. [`super::answer::check`]); `None` — модель
    /// не отвечала (отказ «не знаю» или поиск не удался).
    pub check: Option<super::answer::AnswerCheck>,
}

impl RagContext {
    /// Строки для интерфейсов: этапы поиска, источники по номерам или почему
    /// их нет.
    pub fn summary_lines(&self) -> Vec<String> {
        if let Some(err) = &self.error {
            return vec![format!("RAG: поиск не удался — ответ без контекста: {err}")];
        }
        let mut lines = Vec::new();
        if self.sources.is_empty() {
            let reason = if self.below_rerank > 0 {
                format!("все {} кандидатов ниже порога реранкера {:.2}", self.below_threshold + self.below_rerank, self.rerank_min)
            } else if self.below_threshold > 0 {
                let best = self.best_score.map(|b| format!(": лучшая {b:.2} при пороге {:.2}", self.min_score)).unwrap_or_default();
                format!("все {} найденных фрагментов ниже порога близости{best}", self.below_threshold)
            } else {
                "в индексе ничего не нашлось".to_string()
            };
            lines.push(format!("RAG ({}): {reason} → «не знаю» без запроса к модели", self.label()));
        } else {
            lines.push(format!("RAG ({}): {}", self.label(), self.stages()));
        }
        if let Some(query) = &self.query {
            lines.push(format!("запрос: {query}"));
        }
        lines.extend(self.warnings.iter().map(|w| format!("⚠ {w}")));
        lines.extend(self.sources.iter().map(|s| format!("[{}] {} · {}", s.n, s.location(), s.scores())));
        lines.extend(self.nearest.iter().map(|s| format!("≈ {} · {:.2} (ниже порога)", s.location(), s.score)));
        if let Some(check) = &self.check {
            lines.push(check.summary());
            lines.extend(check.problems.iter().map(|p| format!("⚠ {p}")));
        }
        lines
    }

    /// `structure · rewrite · rerank llm`
    pub fn label(&self) -> String {
        let mut s = self.strategy.clone();
        if self.query.is_some() {
            s.push_str(" · rewrite");
        }
        if self.rerank != Rerank::None {
            s.push_str(&format!(" · rerank {}", self.rerank.name()));
        }
        s
    }

    /// `20 кандидатов → 14 после порога → 5 фрагм.`: сколько осталось после
    /// каждого этапа, который что-то отсёк.
    pub fn stages(&self) -> String {
        let mut stages = vec![format!("{} кандидатов", self.candidates)];
        let after_threshold = self.candidates - self.below_threshold;
        if self.below_threshold > 0 {
            stages.push(format!("{after_threshold} после порога"));
        }
        if self.below_rerank > 0 {
            stages.push(format!("{} после реранкера", after_threshold - self.below_rerank));
        }
        stages.push(format!("{} фрагм.", self.sources.len()));
        stages.join(" → ")
    }
}

impl RagSource {
    /// `0.62` или `0.62 → реранк 0.85 (было #7)`.
    pub fn scores(&self) -> String {
        let mut s = format!("{:.2}", self.score);
        if let Some(r) = self.rerank_score {
            s.push_str(&format!(" → реранк {r:.2}"));
            if self.rank_before != self.n {
                s.push_str(&format!(" (было #{})", self.rank_before));
            }
        }
        s
    }
}

/// Поиск по индексу для вопроса. Ошибку не пробрасывает — она уходит в
/// [`RagContext::error`], чтобы недоступный сервер эмбеддингов не оставлял
/// человека вовсе без ответа. `llm` — модель для переписывания запроса и
/// реранкера `llm`; без неё эти шаги пропускаются с предупреждением.
pub async fn retrieve(question: &str, settings: &RagSettings, llm: Option<&LlmStep<'_>>) -> RagContext {
    let strategy = settings.strategy.clone().unwrap_or_default();
    match try_retrieve(question, settings, llm).await {
        Ok(ctx) => ctx,
        Err(err) => RagContext { strategy, error: Some(format!("{err:#}")), ..RagContext::default() },
    }
}

async fn try_retrieve(question: &str, settings: &RagSettings, llm: Option<&LlmStep<'_>>) -> Result<RagContext> {
    let cfg = RagConfig::from_env();
    let embedder = Embedder::from_env()?;
    retrieve_in(&cfg, &embedder, question, settings, llm).await
}

async fn retrieve_in(
    cfg: &RagConfig,
    embedder: &Embedder,
    question: &str,
    settings: &RagSettings,
    llm: Option<&LlmStep<'_>>,
) -> Result<RagContext> {
    let strategy = resolve_strategy(cfg, settings)?;
    let mut ctx = RagContext {
        strategy: strategy.name().to_string(),
        min_score: settings.min_score,
        rerank_min: settings.rerank_min,
        ..RagContext::default()
    };

    // Переписывание: ищем и по вопросу, и по запросу — переписанный запрос
    // может потерять то, что было в вопросе дословно.
    let mut queries = vec![question.to_string()];
    if settings.rewrite {
        match llm {
            Some(llm) => match rerank::rewrite_query(llm, question).await {
                Ok((query, tokens)) => {
                    ctx.tokens += tokens;
                    if query != question {
                        queries.push(query.clone());
                    }
                    ctx.query = Some(query);
                }
                Err(err) => ctx.warnings.push(format!("запрос не переписан — поиск по вопросу: {err:#}")),
            },
            None => ctx.warnings.push("запрос не переписан — нет модели".into()),
        }
    }

    // Первый этап: топ-N по близости, затем порог близости.
    let k = settings.k.max(1);
    let hits = search_queries(cfg, embedder, &queries, strategy, settings.candidates.max(k)).await?;
    ctx.candidates = hits.len();
    ctx.best_score = hits.iter().map(|h| h.score).reduce(f32::max);
    let nearest: Vec<RagSource> = hits.iter().take(3).enumerate().map(|(i, h)| to_source(i + 1, i, h, None)).collect();
    let kept: Vec<(usize, super::SearchHit)> =
        hits.into_iter().enumerate().filter(|(_, h)| h.score >= settings.min_score).collect();
    ctx.below_threshold = ctx.candidates - kept.len();

    // Второй этап: реранкинг оставшихся и его порог.
    let mut rerank_scores: Option<Vec<f32>> = None;
    if settings.rerank != Rerank::None && !kept.is_empty() {
        let texts: Vec<&str> = kept.iter().map(|(_, h)| h.hit.chunk.context_text()).collect();
        let cosines: Vec<f32> = kept.iter().map(|(_, h)| h.score).collect();
        match settings.rerank {
            Rerank::Heuristic => rerank_scores = Some(rerank::heuristic_scores(&queries, &cosines, &texts)),
            Rerank::Llm => match llm {
                Some(llm) => match rerank::llm_scores(llm, question, &texts).await {
                    Ok((scores, tokens)) => {
                        ctx.tokens += tokens;
                        rerank_scores = Some(scores);
                    }
                    Err(err) => ctx.warnings.push(format!("реранкер llm не сработал — порядок по близости: {err:#}")),
                },
                None => ctx.warnings.push("реранкер llm пропущен — нет модели".into()),
            },
            Rerank::None => {}
        }
    }
    let mut ranked: Vec<(usize, super::SearchHit, Option<f32>)> = match rerank_scores {
        Some(scores) => {
            ctx.rerank = settings.rerank;
            let mut ranked: Vec<_> = kept.into_iter().zip(scores).map(|((i, h), r)| (i, h, Some(r))).collect();
            // Устойчивая сортировка: при равной оценке — порядок по близости.
            ranked.sort_by(|a, b| b.2.unwrap_or(0.0).total_cmp(&a.2.unwrap_or(0.0)));
            let before = ranked.len();
            ranked.retain(|(_, _, r)| r.unwrap_or(0.0) >= settings.rerank_min);
            ctx.below_rerank = before - ranked.len();
            ranked
        }
        None => kept.into_iter().map(|(i, h)| (i, h, None)).collect(),
    };
    ranked.truncate(k);

    ctx.sources =
        ranked.into_iter().enumerate().map(|(n, (before, h, rerank_score))| to_source(n + 1, before, &h, rerank_score)).collect();
    if ctx.sources.is_empty() {
        ctx.nearest = nearest;
    }
    Ok(ctx)
}

fn to_source(n: usize, before: usize, h: &super::SearchHit, rerank_score: Option<f32>) -> RagSource {
    let c = &h.hit.chunk;
    // Текст без пустых строк по краям, а первая строка — та, с которой он
    // теперь начинается: по ней считаются строки цитат.
    let raw = c.context_text();
    let lead = &raw[..raw.len() - raw.trim_start().len()];
    RagSource {
        n,
        chunk_id: c.chunk_id.clone(),
        source: h.hit.version.source.clone(),
        name: super::display_source(&h.hit.version.source).to_string(),
        version: h.hit.version.version,
        section: c.section.clone(),
        start_line: c.context_start_line.unwrap_or(c.start_line) + lead.matches('\n').count() as i64,
        end_line: c.context_end_line.unwrap_or(c.end_line),
        score: h.score,
        rerank_score,
        rank_before: before + 1,
        text: raw.trim().to_string(),
    }
}

/// Первые `max` символов строки — для сообщений об ошибках.
pub(crate) fn short(s: &str, max: usize) -> String {
    let t: String = s.chars().take(max).collect();
    if t.len() < s.len() {
        format!("{t}…")
    } else {
        t
    }
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

/// Вопрос вместе с найденными фрагментами и правилами ответа (см.
/// [`super::answer::FORMAT_RULES`]) — то, что уходит модели вместо голого
/// вопроса. Без фрагментов модель не спрашивают вовсе (см.
/// [`super::answer::refusal`]); при ошибке поиска — вопрос как есть.
pub fn augment_prompt(question: &str, ctx: &RagContext) -> String {
    if ctx.error.is_some() || ctx.sources.is_empty() {
        return question.to_string();
    }
    let mut s = format!("{}\n\n", super::answer::FORMAT_RULES);
    for src in &ctx.sources {
        s.push_str(&format!(
            "[{}] {} · chunk {} (близость {:.2})\n{}\n\n",
            src.n,
            src.location(),
            super::display_source(&src.chunk_id),
            src.score,
            src.text
        ));
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
            chunk_id: format!("README.md@v1/structure/{n:04}"),
            source: "README.md".into(),
            name: "README.md".into(),
            version: 1,
            section: Some("MCP".into()),
            start_line: 10,
            end_line: 12,
            score: 0.71,
            rerank_score: None,
            rank_before: n,
            text: "адрес http://127.0.0.1:8092/mcp".into(),
        }
    }

    #[test]
    fn parses_settings() {
        let s = RagSettings::parse_args(&["structure", "k=3", "min=0.3"]).unwrap();
        assert_eq!(s, RagSettings { strategy: Some("structure".into()), k: 3, min_score: 0.3, ..RagSettings::default() });
        assert_eq!(RagSettings::parse_args(&["min=0"]).unwrap().min_score, 0.0);
        assert_eq!(RagSettings::parse_args(&[]).unwrap(), RagSettings::default());
        assert!(RagSettings::parse_args(&["k=0"]).is_err());
        assert!(RagSettings::parse_args(&["борщ"]).is_err());

        let s = RagSettings::parse_args(&["n=30", "k=4", "rewrite", "rerank=llm", "rmin=0.6"]).unwrap();
        assert_eq!((s.candidates, s.k, s.rewrite, s.rerank, s.rerank_min), (30, 4, true, Rerank::Llm, 0.6));
        assert_eq!(s.describe(), "RAG: авто, топ-30 → k=4, порог 0.50, rewrite, rerank llm ≥ 0.60");
        // Без n кандидатов не меньше k; явный n меньше k — ошибка.
        assert_eq!(RagSettings::parse_args(&["k=20"]).unwrap().candidates, 20);
        assert!(RagSettings::parse_args(&["n=3", "k=5"]).is_err());
        assert!(RagSettings::parse_args(&["n=51"]).is_err());
        assert!(RagSettings::parse_args(&["rerank=борщ"]).is_err());
        // Порог реранкера без реранкера ничего бы не делал.
        assert!(RagSettings::parse_args(&["rmin=0.5"]).is_err());
    }

    #[test]
    fn prompt_carries_sources_and_question() {
        let ctx = RagContext { strategy: "fixed".into(), sources: vec![source(1)], ..RagContext::default() };
        let p = augment_prompt("Какой адрес?", &ctx);
        assert!(p.contains("[1] README.md › MCP · строки 10–12 · chunk README.md@v1/structure/0001"));
        assert!(p.contains("Цитаты:"));
        assert!(p.contains("127.0.0.1:8092"));
        assert!(p.ends_with("Вопрос: Какой адрес?"));
    }

    #[test]
    fn empty_context_is_refused_and_error_passes_question() {
        let empty = RagContext { strategy: "fixed".into(), candidates: 2, below_threshold: 2, ..RagContext::default() };
        assert!(empty.summary_lines()[0].contains("ниже порога близости → «не знаю»"));
        let failed = RagContext { error: Some("нет связи".into()), ..empty };
        assert_eq!(augment_prompt("q", &failed), "q");
    }

    /// Индекс из двух документов стратегией structure поверх фиктивного
    /// сервера эмбеддингов.
    async fn indexed() -> (RagConfig, Embedder) {
        let base = std::env::temp_dir().join(format!("rag-retrieve-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(base.join("docs")).unwrap();
        let cfg = RagConfig {
            db_path: base.join("rag/index.db"),
            corpus: vec!["docs".into()],
            eval_path: base.join("rag/eval.json"),
            report_path: base.join("rag/compare.md"),
            uploads: base.join("rag/uploads"),
            base,
        };
        let section = |title: &str, body: &str| {
            let filler = (1..=6).map(|i| format!("Пункт {i} раздела описан подробно, чтобы раздел был длинным.")).collect::<Vec<_>>().join(" ");
            format!("## {title}\n\n{body} {filler}\n\n")
        };
        let guide = format!(
            "# Руководство\n\n{}{}{}",
            section("Установка", "Скачайте архив и распакуйте его."),
            section("Настройка", "Порт сервера задаётся переменной APP_PORT, по умолчанию 8080."),
            section("Журналы", "Журналы пишутся в каталог logs.")
        );
        std::fs::write(cfg.base.join("docs/guide.md"), guide).unwrap();
        let (url, _) = super::super::embed::test_support::serve().await;
        let embedder = Embedder::new(&url, None, "fake-embed", 8);
        super::super::index(&cfg, &embedder, &[Strategy::Structure], &|_| {}).await.unwrap();
        (cfg, embedder)
    }

    fn rand_suffix() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    }

    /// Фиктивный чат: на каждый запрос отдаёт `reply(текст промпта)`.
    async fn mock_chat(reply: fn(&str) -> String) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                let body_start = loop {
                    let n = socket.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                    if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&raw[..body_start]).to_lowercase();
                let length: usize =
                    headers.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                while raw.len() < body_start + length {
                    let n = socket.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                }
                let request: serde_json::Value = serde_json::from_slice(&raw[body_start..body_start + length]).unwrap();
                let prompt = request["messages"][0]["content"].as_str().unwrap_or_default();
                let body = serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": reply(prompt) } }] }).to_string();
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        base_url
    }

    /// Переписывание — в термины; оценка — 10 фрагменту с APP_PORT, 1 остальным.
    fn judge(prompt: &str) -> String {
        if prompt.contains("Перепиши вопрос") {
            return "APP_PORT server port".into();
        }
        let scores: Vec<String> = prompt
            .split("\n[")
            .skip(1)
            .filter_map(|part| {
                let (n, text) = part.split_once("]\n")?;
                let score = if text.contains("APP_PORT") { 10 } else { 1 };
                Some(format!("{{\"n\":{n},\"score\":{score}}}"))
            })
            .collect();
        format!("{{\"scores\":[{}]}}", scores.join(","))
    }

    #[tokio::test]
    async fn two_stage_search_with_rewrite_llm_rerank_and_cutoffs() {
        let (cfg, embedder) = indexed().await;
        let client = crate::LlmClient::for_tests_at(&mock_chat(judge).await);
        let llm = LlmStep { client: &client, model: "test-model", reasoning: None };
        let question = "Как поменять, где слушает сервер?";

        // Без второго этапа — все кандидаты по близости, топ-K.
        let plain = RagSettings { k: 2, min_score: 0.0, ..RagSettings::default() };
        let ctx = retrieve_in(&cfg, &embedder, question, &plain, Some(&llm)).await.unwrap();
        assert_eq!((ctx.sources.len(), ctx.rerank, ctx.query.as_deref()), (2, Rerank::None, None));
        assert!(ctx.candidates >= 3 && ctx.sources.iter().all(|s| s.rerank_score.is_none()));

        // Порог близости выше любой близости — контекст пуст.
        let strict = RagSettings { min_score: 1.0, ..plain.clone() };
        let ctx = retrieve_in(&cfg, &embedder, question, &strict, Some(&llm)).await.unwrap();
        assert!(ctx.sources.is_empty() && ctx.below_threshold == ctx.candidates);
        // Ближайшие кандидаты — для просьбы уточнить вопрос; модель не нужна.
        assert_eq!(ctx.nearest.len(), 3);
        assert_eq!(ctx.best_score, Some(ctx.nearest[0].score));
        assert!(crate::rag::answer::refusal(&ctx).unwrap().starts_with("Не знаю"));

        // Rewrite + llm с порогом: остаётся только фрагмент с ответом.
        let full = RagSettings { k: 3, rewrite: true, rerank: Rerank::Llm, rerank_min: 0.5, min_score: 0.0, ..RagSettings::default() };
        let ctx = retrieve_in(&cfg, &embedder, question, &full, Some(&llm)).await.unwrap();
        assert_eq!(ctx.query.as_deref(), Some("APP_PORT server port"));
        assert_eq!(ctx.rerank, Rerank::Llm);
        assert_eq!(ctx.sources.len(), 1, "{:?}", ctx.summary_lines());
        assert_eq!(ctx.below_rerank, ctx.candidates - 1);
        assert!(ctx.sources[0].text.contains("APP_PORT"));
        assert_eq!(ctx.sources[0].rerank_score, Some(1.0));
        assert!(ctx.warnings.is_empty());

        // Эвристика поднимает фрагмент с идентификатором из переписанного запроса.
        let heuristic = RagSettings { k: 1, rewrite: true, rerank: Rerank::Heuristic, min_score: 0.0, ..RagSettings::default() };
        let ctx = retrieve_in(&cfg, &embedder, question, &heuristic, Some(&llm)).await.unwrap();
        assert!(ctx.sources[0].text.contains("APP_PORT"), "{:?}", ctx.summary_lines());

        // Без модели rewrite и llm пропускаются с предупреждением, поиск идёт.
        let ctx = retrieve_in(&cfg, &embedder, question, &full, None).await.unwrap();
        assert_eq!((ctx.rerank, ctx.query.as_deref(), ctx.warnings.len()), (Rerank::None, None, 2));
        assert_eq!(ctx.sources.len(), 3);
        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[tokio::test]
    async fn broken_llm_rerank_falls_back_to_similarity() {
        let (cfg, embedder) = indexed().await;
        let client = crate::LlmClient::for_tests_at(&mock_chat(|_| "не знаю".into()).await);
        let llm = LlmStep { client: &client, model: "test-model", reasoning: None };
        let settings = RagSettings { k: 2, rerank: Rerank::Llm, rerank_min: 0.9, min_score: 0.0, ..RagSettings::default() };
        let ctx = retrieve_in(&cfg, &embedder, "Где пишутся журналы?", &settings, Some(&llm)).await.unwrap();
        assert_eq!((ctx.rerank, ctx.sources.len(), ctx.below_rerank), (Rerank::None, 2, 0));
        assert!(ctx.warnings[0].contains("реранкер llm не сработал"), "{:?}", ctx.warnings);
        std::fs::remove_dir_all(&cfg.base).unwrap();
    }

    #[test]
    fn old_configs_have_no_rag() {
        let s: RagSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, RagSettings::default());
        // Настройки дня 22: второй этап выключен, кандидатов — по умолчанию.
        let s: RagSettings = serde_json::from_str(r#"{"strategy":"fixed","k":3,"min_score":0.4}"#).unwrap();
        assert_eq!((s.candidates, s.rewrite, s.rerank), (DEFAULT_CANDIDATES, false, Rerank::None));
    }

    #[test]
    fn summary_shows_stages_query_and_moves() {
        let mut moved = source(1);
        moved.rerank_score = Some(0.9);
        moved.rank_before = 7;
        let ctx = RagContext {
            strategy: "structure".into(),
            sources: vec![moved, source(2)],
            query: Some("MCP scheduler address".into()),
            rerank: Rerank::Llm,
            candidates: 20,
            below_threshold: 6,
            below_rerank: 9,
            warnings: vec!["что-то".into()],
            ..RagContext::default()
        };
        let lines = ctx.summary_lines();
        assert_eq!(lines[0], "RAG (structure · rewrite · rerank llm): 20 кандидатов → 14 после порога → 5 после реранкера → 2 фрагм.");
        assert_eq!(lines[1], "запрос: MCP scheduler address");
        assert_eq!(lines[2], "⚠ что-то");
        assert!(lines[3].ends_with("· 0.71 → реранк 0.90 (было #7)"), "{}", lines[3]);
        assert!(lines[4].ends_with("· 0.71"));

        let cut = RagContext { strategy: "fixed".into(), rerank: Rerank::Heuristic, candidates: 5, below_threshold: 1, below_rerank: 4, ..RagContext::default() };
        assert!(cut.summary_lines()[0].contains("все 5 кандидатов ниже порога реранкера"));
    }
}
