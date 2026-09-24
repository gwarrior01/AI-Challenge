//! Краткое содержание Markdown-документа.
//!
//! Основной способ — LLM через тот же OpenAI-совместимый API, что у клиента
//! (`LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`; клиент из `llm-core`). Длинный
//! документ режется на куски по границам разделов и абзацев: каждый кусок
//! сжимается отдельно — до `DOCUMENTS_LLM_PARALLEL` (4) запросов одновременно,
//! — затем частичные конспекты сводятся в один (и так несколько раундов, если
//! и они не влезают). Рассуждения модели для этих запросов выключаются
//! (`enable_thinking: false`): пересказ их не требует, а на длинном документе
//! они умножают время на число кусков.
//!
//! Без `LLM_API_*` сервер не отказывается работать, а строит извлекающее
//! краткое содержание: заголовки и первое предложение каждого абзаца. Каким
//! способом построен результат, видно в поле `method` — выдавать одно за
//! другое нельзя.

use anyhow::{bail, Result};
use futures_util::{stream, StreamExt, TryStreamExt};
use llm_core::{ChatMessage, ChatOptions, LlmClient};

/// Размер куска для одного запроса к модели, в символах.
const CHUNK_CHARS: usize = 12_000;
/// Предел раундов сведения — страховка от бесконечного цикла, если модель
/// отвечает длиннее, чем получила.
const MAX_ROUNDS: usize = 4;
pub const DEFAULT_MAX_WORDS: usize = 250;
const DEFAULT_PARALLEL: usize = 4;

pub enum Summarizer {
    Llm(LlmClient),
    Extractive,
}

pub struct Options {
    pub max_words: usize,
    /// На что обратить внимание («риски», «цифры и сроки» …).
    pub focus: Option<String>,
}

impl Summarizer {
    pub fn from_env() -> Self {
        match LlmClient::from_env() {
            Ok(client) => Self::Llm(client),
            Err(_) => Self::Extractive,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Llm(client) => format!("llm ({})", client.model()),
            Self::Extractive => "extractive".into(),
        }
    }

    pub async fn summarize(&self, markdown: &str, options: &Options) -> Result<String> {
        match self {
            Self::Llm(client) => llm_summary(client, markdown, options).await,
            Self::Extractive => Ok(extractive_summary(markdown, options.max_words)),
        }
    }
}

async fn llm_summary(client: &LlmClient, markdown: &str, options: &Options) -> Result<String> {
    let mut parts = chunks(markdown, CHUNK_CHARS);
    if parts.len() == 1 {
        return ask(client, &parts[0], options, Stage::Final).await;
    }
    for _ in 0..MAX_ROUNDS {
        let total = parts.len();
        // buffered, а не buffer_unordered: конспекты частей нужны по порядку.
        let partials: Vec<String> = stream::iter(std::mem::take(&mut parts).into_iter().enumerate())
            .map(|(i, part)| async move { ask(client, &part, options, Stage::Partial { index: i + 1, total }).await })
            .buffered(parallel())
            .try_collect()
            .await?;
        let joined = partials.join("\n\n");
        if joined.chars().count() <= CHUNK_CHARS {
            return ask(client, &joined, options, Stage::Combine).await;
        }
        parts = chunks(&joined, CHUNK_CHARS);
    }
    bail!("документ не удалось сжать за {MAX_ROUNDS} раунда — частичные конспекты не становятся короче")
}

enum Stage {
    /// Весь документ за один запрос.
    Final,
    /// Кусок длинного документа.
    Partial { index: usize, total: usize },
    /// Сведение частичных конспектов.
    Combine,
}

async fn ask(client: &LlmClient, text: &str, options: &Options, stage: Stage) -> Result<String> {
    let focus = options
        .focus
        .as_deref()
        .filter(|f| !f.trim().is_empty())
        .map(|f| format!(" Особое внимание: {f}."))
        .unwrap_or_default();
    let words = options.max_words;
    let (task, input_name) = match stage {
        Stage::Final => (
            format!(
                "Составь краткое содержание документа не длиннее {words} слов. Формат Markdown: первая \
                 строка — суть документа одним предложением, затем ключевые пункты списком: факты, цифры, \
                 решения, сроки."
            ),
            "Документ",
        ),
        Stage::Partial { index, total } => (
            format!(
                "Это часть {index} из {total} длинного документа. Выпиши её ключевые факты, цифры, решения и \
                 сроки списком Markdown, не длиннее {} слов.",
                (words * 2).max(150)
            ),
            "Часть документа",
        ),
        Stage::Combine => (
            format!(
                "Ниже — конспекты частей одного документа по порядку. Сведи их в одно краткое содержание не \
                 длиннее {words} слов. Формат Markdown: первая строка — суть документа одним предложением, \
                 затем ключевые пункты списком. Убери повторы."
            ),
            "Конспекты частей",
        ),
    };
    let system = format!(
        "Ты делаешь краткие содержания документов. Пиши на языке документа. Используй только то, что есть \
         в тексте: ничего не додумывай и не добавляй от себя. Отвечай только самим кратким содержанием, без \
         вступлений.{focus}"
    );
    let messages = [ChatMessage::system(system), ChatMessage::user(format!("{task}\n\n{input_name}:\n\n{text}"))];
    let options = ChatOptions { temperature: Some(0.2), reasoning: Some(false), ..Default::default() };
    let reply = client.chat_with_options(&messages, &options).await?;
    let content = reply.content.trim().to_string();
    if content.is_empty() {
        bail!("модель вернула пустой ответ");
    }
    Ok(content)
}

fn parallel() -> usize {
    std::env::var("DOCUMENTS_LLM_PARALLEL")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_PARALLEL)
}

/// Режет текст на куски не длиннее `limit` символов, предпочитая границы
/// разделов (`\n#`), затем абзацев, затем строк; слово пополам — только если
/// иначе нельзя.
pub fn chunks(text: &str, limit: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut rest = text.trim();
    while rest.chars().count() > limit {
        let window_end = rest.char_indices().nth(limit).map(|(i, _)| i).unwrap_or(rest.len());
        let window = &rest[..window_end];
        let min_cut = window.len() / 3;
        let cut = [window.rfind("\n#"), window.rfind("\n\n"), window.rfind('\n'), window.rfind(' ')]
            .into_iter()
            .flatten()
            .find(|&i| i > min_cut)
            .unwrap_or(window_end);
        result.push(rest[..cut].trim().to_string());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() || result.is_empty() {
        result.push(rest.to_string());
    }
    result
}

/// Краткое содержание без модели: заголовок документа, затем по разделам —
/// первое предложение каждого абзаца, пока не наберётся `max_words` слов.
pub fn extractive_summary(markdown: &str, max_words: usize) -> String {
    let mut lines = Vec::new();
    let mut words = 0;
    let mut pending_heading: Option<String> = None;
    for block in markdown.split("\n\n").map(str::trim).filter(|b| !b.is_empty()) {
        if let Some(title) = block.strip_prefix("# ") {
            lines.push(format!("**{}**", title.trim()));
            continue;
        }
        if block.starts_with('#') {
            pending_heading = Some(block.trim_start_matches('#').trim().to_string());
            continue;
        }
        let first_item_or_sentence = block
            .lines()
            .next()
            .map(|l| l.trim_start_matches("- ").to_string())
            .map(|l| first_sentence(&l).to_string())
            .unwrap_or_default();
        let n = first_item_or_sentence.split_whitespace().count();
        if words + n > max_words {
            break;
        }
        words += n;
        let line = match pending_heading.take() {
            Some(heading) => format!("- **{heading}:** {first_item_or_sentence}"),
            None => format!("- {first_item_or_sentence}"),
        };
        lines.push(line);
    }
    lines.join("\n")
}

fn first_sentence(text: &str) -> &str {
    // Конец предложения — «. », «! », «? » перед заглавной буквой или цифрой.
    let bytes: Vec<(usize, char)> = text.char_indices().collect();
    for w in bytes.windows(3) {
        let [(i, c), (_, space), (_, next)] = [w[0], w[1], w[2]];
        if matches!(c, '.' | '!' | '?') && space == ' ' && (next.is_uppercase() || next.is_ascii_digit()) {
            return &text[..i + c.len_utf8()];
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_respect_limit_and_boundaries() {
        let text = format!("# A\n\n{}\n\n## B\n\n{}", "слово ".repeat(300), "другое ".repeat(300));
        let parts = chunks(&text, 2500);
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.chars().count() <= 2500));
        assert!(parts[1].starts_with("## B") || parts.iter().any(|p| p.starts_with("## B")));
        // Ничего не потерялось, кроме пробелов на стыках.
        let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(squash(&parts.join(" ")), squash(&text));
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(chunks("коротко", 100), vec!["коротко".to_string()]);
    }

    #[test]
    fn extractive_takes_first_sentences_under_headings() {
        let md = "# Отчёт\n\n## Итоги\n\nВремя сократилось с 40 до 12 минут. Подробности ниже.\n\n## Риски\n\nУдалённые JVM не поддерживаются. И ещё.";
        let summary = extractive_summary(md, 100);
        assert_eq!(
            summary,
            "**Отчёт**\n- **Итоги:** Время сократилось с 40 до 12 минут.\n- **Риски:** Удалённые JVM не поддерживаются."
        );
        assert_eq!(extractive_summary(md, 8), "**Отчёт**\n- **Итоги:** Время сократилось с 40 до 12 минут.");
    }
}
