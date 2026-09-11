//! Общая логика управления контекстом диалога — используется и именованными
//! агентами ([`crate::agent::Agent`]), и обычным чатом веб-интерфейса, чтобы оба
//! места вели себя одинаково и не дублировали промпты.
//!
//! Реализовано **4 стратегии** управления контекстом, выбираемые через
//! [`ContextStrategy`] в конфигурации агента (`AgentConfig::context_strategy`):
//!
//! - [`ContextStrategy::Full`] — управления нет: в каждый запрос уходит вся
//!   история целиком (поведение по умолчанию, как было раньше).
//! - [`ContextStrategy::Summary`] — сжатие в сводку: каждые
//!   [`context_summary_chunk`] сообщений пользователя вся накопленная с
//!   прошлого пересчёта история сжимается той же моделью отдельным запросом в
//!   текстовую сводку; между пересчётами непросуммированный "хвост" диалога
//!   отправляется как есть.
//! - [`ContextStrategy::SlidingWindow`] — скользящее окно: в модель уходят
//!   только последние [`sliding_window_size`] сообщений пользователя (обменов),
//!   всё более раннее просто отбрасывается — без каких-либо доп. запросов к LLM.
//! - [`ContextStrategy::Facts`] — Sticky Facts / Key-Value Memory: отдельно от
//!   истории ведётся набор "ключ: значение" (цель, ограничения, предпочтения,
//!   принятые решения, договорённости и т.п.), который обновляется отдельным
//!   запросом к LLM после каждого сообщения пользователя ([`update_facts`]); в
//!   основной запрос уходит блок фактов + последние [`sliding_window_size`]
//!   сообщений (то же окно, что и у [`ContextStrategy::SlidingWindow`]).
//! - [`ContextStrategy::Branching`] — ветвление: история хранится как набор
//!   именованных веток, между которыми можно переключаться и от общих
//!   checkpoint'ов отращивать новые (см. `Agent::checkpoint`/`branch_from`/
//!   `switch_branch`); в каждый запрос уходит полная история текущей ветки —
//!   управление контекстом здесь не в обрезании истории, а в том, что ветки не
//!   засоряют контекст друг друга.
//!
//! Единица счёта почти везде здесь — сообщение ПОЛЬЗОВАТЕЛЯ (один его запрос =
//! один ответ ассистента, вместе взятые = один "обмен"), а не отдельная запись
//! в истории: пользователь ожидает, что счётчик в интерфейсах уменьшается на 1
//! за каждое написанное им сообщение, а не на 2 (сообщение + ответ) — см.
//! [`RAW_MESSAGES_PER_EXCHANGE`].

use crate::{ChatMessage, ChatOptions, LlmClient};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Имя переменной окружения, которой можно переопределить [`context_summary_chunk`].
const CONTEXT_SUMMARY_CHUNK_ENV: &str = "LLM_CONTEXT_SUMMARY_CHUNK";

/// Значение [`context_summary_chunk`] по умолчанию, если переменная окружения
/// не задана, пуста или не парсится в положительное целое.
const DEFAULT_CONTEXT_SUMMARY_CHUNK: usize = 10;

/// Сколько сообщений пользователя (обменов "запрос — ответ") нужно накопить с
/// последнего пересчёта сводки, прежде чем она будет пересчитана снова —
/// заново, целиком, охватывая весь накопленный с прошлого раза "хвост".
/// Сводка обновляется не после каждого сообщения, а пачками — иначе каждый
/// запрос удваивался бы лишним обращением к LLM.
///
/// Берётся из переменной окружения `LLM_CONTEXT_SUMMARY_CHUNK` (читается один
/// раз за время жизни процесса и кэшируется — как и остальные `LLM_*`
/// переменные, менять её на лету без перезапуска нельзя); если она не задана
/// или содержит не положительное целое число — используется
/// [`DEFAULT_CONTEXT_SUMMARY_CHUNK`].
pub fn context_summary_chunk() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(CONTEXT_SUMMARY_CHUNK_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_CONTEXT_SUMMARY_CHUNK)
    })
}

/// Имя переменной окружения, которой можно переопределить [`sliding_window_size`].
const SLIDING_WINDOW_ENV: &str = "LLM_SLIDING_WINDOW_SIZE";

/// Значение [`sliding_window_size`] по умолчанию.
const DEFAULT_SLIDING_WINDOW_SIZE: usize = 6;

/// Сколько последних сообщений пользователя (обменов) отправлять в модель при
/// стратегиях [`ContextStrategy::SlidingWindow`] и [`ContextStrategy::Facts`] —
/// всё, что старше, просто отбрасывается (в отличие от [`ContextStrategy::Summary`],
/// здесь ничего не сжимается и не запоминается взамен, кроме фактов у стратегии Facts).
///
/// Берётся из переменной окружения `LLM_SLIDING_WINDOW_SIZE` (читается один раз
/// за время жизни процесса и кэшируется); если она не задана или содержит не
/// положительное целое число — используется [`DEFAULT_SLIDING_WINDOW_SIZE`].
/// Конкретный агент может переопределить это значение через
/// `AgentConfig::window_size`.
pub fn sliding_window_size() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(SLIDING_WINDOW_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_SLIDING_WINDOW_SIZE)
    })
}

/// В истории диалога каждый обмен хранится как два отдельных сообщения
/// (запрос пользователя + ответ ассистента, см. `Agent::history`), но и
/// [`context_summary_chunk`]/[`sliding_window_size`], и счётчики в
/// [`CompressionInfo`](crate::agent::CompressionInfo)/[`crate::agent::SlidingWindowInfo`]
/// считаются в обменах, а не в сырых записях истории — переводит одно в другое.
pub const RAW_MESSAGES_PER_EXCHANGE: usize = 2;

/// Стратегия управления контекстом диалога агента — см. документацию модуля.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContextStrategy {
    /// Управления нет: в каждый запрос уходит вся история целиком.
    #[default]
    Full,
    /// Сжатие устаревшей части истории в текстовую сводку той же моделью.
    Summary,
    /// Только последние [`sliding_window_size`] сообщений, остальное отбрасывается.
    SlidingWindow,
    /// Sticky Facts / Key-Value Memory: факты + последние [`sliding_window_size`] сообщений.
    Facts,
    /// Ветвление диалога: checkpoint'ы и независимые ветки истории.
    Branching,
}

impl ContextStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Summary => "summary",
            Self::SlidingWindow => "sliding-window",
            Self::Facts => "facts",
            Self::Branching => "branching",
        }
    }

    /// Человекочитаемое имя по-русски — для интерфейсов.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Full => "без управления (вся история)",
            Self::Summary => "сжатие в сводку",
            Self::SlidingWindow => "скользящее окно",
            Self::Facts => "sticky facts (ключ-значение)",
            Self::Branching => "ветвление диалога",
        }
    }

    pub const ALL: [ContextStrategy; 5] =
        [Self::Full, Self::Summary, Self::SlidingWindow, Self::Facts, Self::Branching];
}

impl std::str::FromStr for ContextStrategy {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "full" => Ok(Self::Full),
            "summary" => Ok(Self::Summary),
            "sliding-window" | "sliding_window" => Ok(Self::SlidingWindow),
            "facts" => Ok(Self::Facts),
            "branching" => Ok(Self::Branching),
            other => bail!(
                "неизвестная стратегия контекста: «{other}» \
                 (допустимые: full, summary, sliding-window, facts, branching)"
            ),
        }
    }
}

impl std::fmt::Display for ContextStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Системный промпт для обращения к LLM, которое сжимает фрагмент истории
/// диалога в сводку (стратегия [`ContextStrategy::Summary`]).
const CONTEXT_SUMMARY_SYSTEM_PROMPT: &str = "Ты — модуль сжатия контекста диалога между пользователем и \
ассистентом. Тебе дают предыдущую сводку (может быть пустой) и новый фрагмент переписки. Составь новую, \
обновлённую сводку всего рассмотренного диалога: сохрани факты, договорённости, имена, принятые решения, \
незавершённые задачи, предпочтения и стиль общения пользователя — всё, что важно для продолжения разговора. \
Пиши по-русски, кратко и по существу, связным текстом или маркированным списком, без вступлений и оценок \
своей работы. Не придумывай ничего, чего не было в переписке, и не включай технические детали формата \
сообщений (роли, токены и т.п.) — только содержание разговора.";

/// Системный промпт для обращения к LLM, которое обновляет набор фактов
/// (стратегия [`ContextStrategy::Facts`]).
const FACTS_SYSTEM_PROMPT: &str = "Ты — модуль извлечения ключевых фактов из диалога пользователя с \
ассистентом (стратегия управления контекстом Sticky Facts / Key-Value Memory). Тебе дают текущий набор \
фактов в виде JSON-объекта (может быть пустым) и новый фрагмент диалога — последний обмен: сообщение \
пользователя и ответ ассистента. Обнови набор фактов: добавь новые важные данные (цель, ограничения, \
предпочтения, принятые решения, договорённости, имена, цифры, сроки и т.п.), обнови значения, если \
пользователь их изменил, и НЕ удаляй факты, которые всё ещё актуальны и не были явно отменены. Ключи — \
короткие, по-русски, в snake_case (например: цель, бюджет, срок, платформа). Ответь СТРОГО одним JSON-\
объектом вида {\"ключ\": \"значение\", ...} без пояснений, без markdown-разметки (без ```), без текста до \
или после JSON. Если новых или изменившихся фактов нет — верни набор фактов без изменений.";

fn role_label(role: &str) -> &str {
    match role {
        "user" => "Пользователь",
        "assistant" => "Ассистент",
        other => other,
    }
}

/// Сжимает фрагмент диалога `chunk` в обновлённую сводку с учётом `previous_summary`
/// (пустая строка — сводки ещё не было) отдельным обращением к LLM той же моделью,
/// что обслуживает основной диалог.
pub async fn summarize_chunk(
    client: &LlmClient,
    model: &str,
    previous_summary: &str,
    chunk: &[ChatMessage],
) -> Result<String> {
    let chunk_text = chunk
        .iter()
        .map(|m| format!("{}: {}", role_label(&m.role), m.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    let prev_summary_text = if previous_summary.trim().is_empty() {
        "(сводки пока нет — это первый фрагмент)".to_string()
    } else {
        previous_summary.to_string()
    };
    let request = format!(
        "Предыдущая сводка:\n{prev_summary_text}\n\n\
         Новый фрагмент диалога для добавления в сводку:\n{chunk_text}\n\n\
         Составь обновлённую сводку всего диалога с учётом этого фрагмента."
    );
    let messages =
        [ChatMessage::system(CONTEXT_SUMMARY_SYSTEM_PROMPT.to_string()), ChatMessage::user(request)];
    let options = ChatOptions { max_tokens: Some(800), ..ChatOptions::default() };

    let completion = client.chat_with_model(model, &messages, &options).await?;
    Ok(completion.content)
}

/// Обновляет набор фактов (стратегия [`ContextStrategy::Facts`]) с учётом
/// последнего обмена `exchange` (сообщение пользователя + ответ ассистента),
/// отдельным обращением к LLM той же моделью, что обслуживает основной диалог.
pub async fn update_facts(
    client: &LlmClient,
    model: &str,
    previous_facts: &BTreeMap<String, String>,
    exchange: &[ChatMessage],
) -> Result<BTreeMap<String, String>> {
    let prev_json = serde_json::to_string_pretty(previous_facts).unwrap_or_else(|_| "{}".to_string());
    let exchange_text = exchange
        .iter()
        .map(|m| format!("{}: {}", role_label(&m.role), m.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    let request = format!(
        "Текущие факты (JSON):\n{prev_json}\n\n\
         Новый обмен в диалоге:\n{exchange_text}\n\n\
         Верни обновлённый JSON-объект фактов."
    );
    let messages = [ChatMessage::system(FACTS_SYSTEM_PROMPT.to_string()), ChatMessage::user(request)];
    let options = ChatOptions { max_tokens: Some(500), ..ChatOptions::default() };

    let completion = client.chat_with_model(model, &messages, &options).await?;
    parse_facts_json(&completion.content)
        .with_context(|| format!("не удалось разобрать JSON фактов в ответе модели: {}", completion.content))
}

/// Модели (особенно локальные) часто оборачивают JSON в ```markdown``` или
/// добавляют пояснения до/после — вместо строгого парсинга всего ответа
/// вырезаем первую `{...}` подстроку.
fn parse_facts_json(raw: &str) -> Result<BTreeMap<String, String>> {
    let start = raw.find('{').context("в ответе модели нет JSON-объекта")?;
    let end = raw.rfind('}').context("в ответе модели нет JSON-объекта")?;
    if end < start {
        bail!("в ответе модели не найден корректный JSON-объект");
    }
    let slice = &raw[start..=end];
    let value: serde_json::Value = serde_json::from_str(slice)?;
    let obj = value.as_object().context("верхний уровень JSON фактов — не объект")?;
    let mut facts = BTreeMap::new();
    for (k, v) in obj {
        let value_str = match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        facts.insert(k.clone(), value_str);
    }
    Ok(facts)
}

/// Форматирует набор фактов в системное сообщение, подставляемое в запрос при
/// стратегии [`ContextStrategy::Facts`].
pub fn format_facts_block(facts: &BTreeMap<String, String>) -> String {
    let lines: Vec<String> = facts.iter().map(|(k, v)| format!("- {k}: {v}")).collect();
    format!(
        "Известные факты о диалоге (ключ: значение; обновляются после каждого сообщения пользователя — \
         используй их как память о более ранней части разговора, даже если её самой уже нет в контексте):\n\n{}",
        lines.join("\n")
    )
}
