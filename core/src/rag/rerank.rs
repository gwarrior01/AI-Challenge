//! Второй этап поиска RAG-режима: переписывание запроса перед поиском и
//! реранкинг найденных кандидатов после него.
//!
//! - **Переписывание** ([`rewrite_query`]) — модель превращает вопрос в
//!   поисковый запрос: ключевые термины, идентификаторы, синонимы и перевод
//!   терминов на английский (вопросы по-русски, а документы и код часто
//!   по-английски). Ищут по обеим формулировкам, близость чанка — лучшая.
//! - **Реранкинг** ([`Rerank`]) — `candidates` ближайших по эмбеддингу
//!   чанков переупорядочиваются точнее, чем умеет косинусная близость:
//!   - `heuristic` ([`heuristic_scores`]) — к близости добавляется бонус за
//!     термины запроса, найденные в тексте чанка; идентификаторы и числа весят
//!     больше слов, термины, что есть почти у всех кандидатов, — меньше;
//!   - `llm` ([`llm_scores`]) — модель одним запросом оценивает каждого
//!     кандидата по шкале 0–10 (в оценке реранкера — 0–1).
//!
//! Оба шага — вспомогательные: если модель не ответила или ответила не по
//! формату, поиск продолжается без них, а причина уходит в предупреждения
//! [`super::RagContext`].

use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{ChatMessage, ChatOptions, LlmClient};

/// Способ реранкинга кандидатов.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rerank {
    /// Порядок по косинусной близости, как нашёл поиск.
    #[default]
    None,
    /// Близость + бонус за совпадение терминов запроса.
    Heuristic,
    /// Оценка релевантности моделью.
    Llm,
}

impl Rerank {
    pub const ALL: [Rerank; 3] = [Rerank::None, Rerank::Heuristic, Rerank::Llm];

    pub fn name(self) -> &'static str {
        match self {
            Rerank::None => "none",
            Rerank::Heuristic => "heuristic",
            Rerank::Llm => "llm",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "none" | "off" | "no" => Some(Rerank::None),
            "heuristic" | "terms" => Some(Rerank::Heuristic),
            "llm" | "model" => Some(Rerank::Llm),
            _ => None,
        }
    }
}

/// Модель для вспомогательных запросов (переписывание, оценка) — та же, что
/// отвечает агенту, с его режимом рассуждений.
pub struct LlmStep<'a> {
    pub client: &'a LlmClient,
    pub model: &'a str,
    pub reasoning: Option<bool>,
}

impl LlmStep<'_> {
    /// Ответ модели без блока рассуждений и расход токенов.
    async fn ask(&self, prompt: String) -> Result<(String, u64)> {
        let options = ChatOptions { temperature: Some(0.0), reasoning: self.reasoning, ..ChatOptions::default() };
        let reply = self.client.chat_with_model(self.model, &[ChatMessage::user(prompt)], &options).await?;
        let tokens = reply.usage.map(|u| u.total_tokens as u64).unwrap_or(0);
        Ok((strip_thinking(&reply.content).trim().to_string(), tokens))
    }
}

/// Убирает `<think>…</think>`, который часть моделей пишет в content.
fn strip_thinking(s: &str) -> &str {
    match s.find("</think>") {
        Some(end) => &s[end + "</think>".len()..],
        None => s,
    }
}

// ---------------------------------------------------------------------------
// Переписывание запроса

fn rewrite_prompt(question: &str) -> String {
    format!(
        "Перепиши вопрос в поисковый запрос для векторного поиска по базе документов (документация и исходный код, \
         часто на английском). Оставь ключевые термины, имена, идентификаторы и числа; добавь синонимы и перевод \
         ключевых терминов на английский; убери вежливые и служебные слова. Не отвечай на вопрос. \
         Выведи только запрос одной строкой, без пояснений и кавычек.\n\nВопрос: {question}"
    )
}

/// Поисковый запрос по вопросу и расход токенов. Пустой ответ — ошибка.
pub async fn rewrite_query(llm: &LlmStep<'_>, question: &str) -> Result<(String, u64)> {
    let (reply, tokens) = llm.ask(rewrite_prompt(question)).await.context("переписывание запроса")?;
    let query = clean_rewrite(&reply);
    if query.is_empty() {
        bail!("модель вернула пустой запрос");
    }
    Ok((query, tokens))
}

/// Первая непустая строка ответа без префикса «Запрос:» и кавычек.
fn clean_rewrite(reply: &str) -> String {
    let line = reply.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or_default();
    let line = ["Запрос:", "Поисковый запрос:", "Query:", "Search query:"]
        .iter()
        .find_map(|p| line.strip_prefix(p))
        .unwrap_or(line);
    line.trim().trim_matches(|c| matches!(c, '"' | '«' | '»' | '`' | '\'')).trim().to_string()
}

// ---------------------------------------------------------------------------
// Эвристика: совпадение терминов

/// Сколько добавляет к близости полное совпадение терминов запроса: у bge-m3
/// близости кандидатов обычно различаются на сотые, так что это заметно, но
/// не перекрывает смысловую близость целиком.
pub const TERM_BOOST: f32 = 0.2;

/// Вес идентификатора или числа против обычного слова: `MIN_INTERVAL`, `8092`
/// в запросе почти всегда и есть то, что ищут.
const IDENT_WEIGHT: f32 = 2.5;

const STOPWORDS: &[&str] = &[
    "что", "как", "какой", "какая", "какие", "каким", "какую", "каков", "где", "когда", "почему", "зачем", "сколько",
    "если", "или", "это", "этот", "эта", "эти", "того", "тот", "так", "такой", "при", "для", "без", "над", "под",
    "про", "его", "её", "они", "она", "оно", "был", "была", "было", "быть", "есть", "ли", "же", "уже", "ещё", "еще",
    "чем", "там", "тут", "все", "всё", "весь", "можно", "нужно", "надо", "который", "которая", "которые", "чтобы",
    "the", "and", "for", "what", "how", "why", "when", "where", "which", "does", "with", "from", "that", "this",
    "are", "was", "were", "has", "have", "into", "about",
];

#[derive(Debug, Clone, PartialEq)]
struct Term {
    /// Строчными; у длинных слов — усечённая основа (без окончаний).
    stem: String,
    weight: f32,
}

fn is_identifier(token: &str) -> bool {
    token.contains('_') || token.chars().any(|c| c.is_ascii_digit()) || token.chars().skip(1).any(char::is_uppercase)
}

/// Термины запросов: слова и идентификаторы без стоп-слов, по одному разу.
fn query_terms(queries: &[String]) -> Vec<Term> {
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    for query in queries {
        for token in query.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|t| !t.is_empty()) {
            let lower = token.to_lowercase();
            let ident = is_identifier(token);
            if !ident && (lower.chars().count() < 3 || STOPWORDS.contains(&lower.as_str())) {
                continue;
            }
            let stem = if ident { lower } else { stem(&lower) };
            if seen.insert(stem.clone()) {
                terms.push(Term { stem, weight: if ident { IDENT_WEIGHT } else { 1.0 } });
            }
        }
    }
    terms
}

/// Грубая основа слова: у длинных слов отбрасываются последние буквы, чтобы
/// «запуском» и «запуск», «интервал» и «интервалы» совпали.
fn stem(word: &str) -> String {
    let n = word.chars().count();
    if n < 6 {
        return word.to_string();
    }
    word.chars().take((n - 3).max(5)).collect()
}

/// Оценка реранкера `heuristic` для каждого кандидата: близость плюс
/// [`TERM_BOOST`] × доля найденных в тексте терминов запроса. Вес термина
/// умножается на его редкость среди кандидатов (IDF): термин, который есть
/// у всех, ничего не различает.
pub fn heuristic_scores(queries: &[String], cosines: &[f32], texts: &[&str]) -> Vec<f32> {
    let terms = query_terms(queries);
    let lower: Vec<String> = texts.iter().map(|t| t.to_lowercase()).collect();
    let n = lower.len() as f32;
    let weighted: Vec<(f32, Vec<bool>)> = terms
        .iter()
        .map(|term| {
            let hits: Vec<bool> = lower.iter().map(|t| t.contains(&term.stem)).collect();
            let df = hits.iter().filter(|&&h| h).count() as f32;
            let idf = if df > 0.0 { (1.0 + n / df).ln() } else { 0.0 };
            (term.weight * idf, hits)
        })
        .collect();
    let total: f32 = weighted.iter().map(|(w, _)| w).sum();
    cosines
        .iter()
        .enumerate()
        .map(|(i, &cos)| {
            if total <= 0.0 {
                return cos;
            }
            let matched: f32 = weighted.iter().filter(|(_, hits)| hits[i]).map(|(w, _)| w).sum();
            cos + TERM_BOOST * matched / total
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Оценка моделью

/// Сколько символов кандидата видит модель-оценщик: начала фрагмента обычно
/// хватает, чтобы понять, о нём ли вопрос, а N полных текстов — дорого.
pub const LLM_PREVIEW_CHARS: usize = 700;

fn llm_prompt(question: &str, texts: &[&str]) -> String {
    let mut s = String::from(
        "Оцени, насколько каждый фрагмент помогает ответить на вопрос. Шкала: 10 — фрагмент содержит ответ; \
         6–8 — содержит часть ответа; 3–5 — по теме, но ответа нет; 0–2 — не относится к вопросу.\n\
         Ответь только JSON без пояснений, по элементу на каждый фрагмент: \
         {\"scores\":[{\"n\":1,\"score\":7},{\"n\":2,\"score\":0}]}\n\n",
    );
    s.push_str(&format!("Вопрос: {question}\n\n"));
    for (i, text) in texts.iter().enumerate() {
        let preview: String = text.chars().take(LLM_PREVIEW_CHARS).collect();
        let cut = if preview.len() < text.len() { " …" } else { "" };
        s.push_str(&format!("[{}]\n{}{cut}\n\n", i + 1, preview.trim()));
    }
    s
}

#[derive(Deserialize)]
struct ScoreItem {
    n: usize,
    score: f32,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ScoresWire {
    Wrapped { scores: Vec<ScoreItem> },
    List(Vec<ScoreItem>),
    Plain(Vec<f32>),
}

/// Оценки 0–1 по номерам фрагментов из ответа модели. Кандидат, которого
/// модель пропустила, получает 0; ни одной оценки — ошибка.
fn parse_llm_scores(reply: &str, count: usize) -> Result<Vec<f32>> {
    let start = reply.find(['{', '[']).context("в ответе модели нет JSON")?;
    let end = reply.rfind(['}', ']']).filter(|&e| e >= start).context("в ответе модели нет JSON")?;
    let wire: ScoresWire = serde_json::from_str(&reply[start..=end])
        .with_context(|| format!("не удалось разобрать оценки модели: {}", crate::rag::retrieve::short(reply, 200)))?;
    let items: Vec<(usize, f32)> = match wire {
        ScoresWire::Wrapped { scores } | ScoresWire::List(scores) => scores.into_iter().map(|s| (s.n, s.score)).collect(),
        ScoresWire::Plain(list) => list.into_iter().enumerate().map(|(i, s)| (i + 1, s)).collect(),
    };
    let mut scores = vec![0.0; count];
    let mut any = false;
    for (n, score) in items {
        if (1..=count).contains(&n) && score.is_finite() {
            scores[n - 1] = (score / 10.0).clamp(0.0, 1.0);
            any = true;
        }
    }
    if !any {
        bail!("модель не оценила ни одного фрагмента");
    }
    Ok(scores)
}

/// Оценка реранкера `llm` (0–1) для каждого кандидата и расход токенов.
pub async fn llm_scores(llm: &LlmStep<'_>, question: &str, texts: &[&str]) -> Result<(Vec<f32>, u64)> {
    let (reply, tokens) = llm.ask(llm_prompt(question, texts)).await.context("оценка фрагментов моделью")?;
    Ok((parse_llm_scores(&reply, texts.len())?, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modes() {
        for mode in Rerank::ALL {
            assert_eq!(Rerank::parse(mode.name()), Some(mode));
        }
        assert_eq!(Rerank::parse("off"), Some(Rerank::None));
        assert_eq!(Rerank::parse("борщ"), None);
    }

    #[test]
    fn terms_skip_stopwords_and_keep_identifiers() {
        let terms = query_terms(&["Какой минимальный интервал у MIN_INTERVAL и порт 8092?".to_string()]);
        let stems: Vec<&str> = terms.iter().map(|t| t.stem.as_str()).collect();
        assert_eq!(stems, ["минималь", "интер", "min_interval", "порт", "8092"]);
        assert_eq!(terms.iter().find(|t| t.stem == "8092").unwrap().weight, IDENT_WEIGHT);
        assert_eq!(terms.iter().find(|t| t.stem == "порт").unwrap().weight, 1.0);
    }

    #[test]
    fn heuristic_lifts_chunk_with_query_terms() {
        let queries = ["минимальный интервал MIN_INTERVAL".to_string()];
        let texts = ["Плановые запуски агентов работают сами.", "const MIN_INTERVAL: Duration = Duration::from_secs(30);"];
        let scores = heuristic_scores(&queries, &[0.70, 0.66], &texts);
        assert!(scores[1] > scores[0], "{scores:?}");
        assert_eq!(scores[0], 0.70);
        // Нет терминов — порядок по близости.
        assert_eq!(heuristic_scores(&["что это".to_string()], &[0.5, 0.4], &texts), vec![0.5, 0.4]);
    }

    #[test]
    fn parses_llm_scores_in_any_shape() {
        let wrapped = "<think>хм</think>```json\n{\"scores\":[{\"n\":2,\"score\":9},{\"n\":1,\"score\":3}]}\n```";
        assert_eq!(parse_llm_scores(strip_thinking(wrapped), 3).unwrap(), vec![0.3, 0.9, 0.0]);
        assert_eq!(parse_llm_scores("[{\"n\":1,\"score\":12}]", 1).unwrap(), vec![1.0]);
        assert_eq!(parse_llm_scores("[5, 10]", 2).unwrap(), vec![0.5, 1.0]);
        assert!(parse_llm_scores("не знаю", 2).is_err());
        assert!(parse_llm_scores("{\"scores\":[{\"n\":7,\"score\":5}]}", 2).is_err());
    }

    #[test]
    fn rewrite_is_one_clean_line() {
        assert_eq!(clean_rewrite("\n Запрос: «MIN_INTERVAL scheduled run interval»\nпояснение"), "MIN_INTERVAL scheduled run interval");
        assert_eq!(clean_rewrite("   "), "");
    }

    #[test]
    fn prompt_numbers_and_trims_candidates() {
        let long = "а".repeat(LLM_PREVIEW_CHARS + 50);
        let p = llm_prompt("вопрос?", &["первый", &long]);
        assert!(p.contains("[1]\nпервый"));
        assert!(p.contains(" …"));
        assert!(p.contains("Вопрос: вопрос?"));
    }
}
