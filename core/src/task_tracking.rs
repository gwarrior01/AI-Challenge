//! Агент сам ведёт задачу диалога ([`crate::AgentConfig::track_task`]): после
//! каждой реплики человека модель обновляет цель задачи, к которой он
//! присоединён, и её заметки — что пользователь уже уточнил, какие
//! ограничения и термины зафиксированы.
//!
//! Это та же общая задача, что `agent task start/join` (см. [`crate::memory`]):
//! цель — [`TaskState::goal`], заметки — [`TaskState::notes`], рядом с явно
//! сохранёнными данными (`task set`). Отличие только в том, кто пишет: данные —
//! человек явно, цель и заметки при `track_task` — модель по ходу диалога.
//! Этапы (planning → … → done) при этом не включаются — это отдельный режим
//! ([`crate::AgentConfig::task_mode`]).
//!
//! ```text
//! вопрос → поиск RAG по запросу с учётом задачи и прошлой реплики
//!        → модель (задача — системным сообщением) → ответ
//!        → отдельный запрос к модели: обновить цель и заметки по этому обмену → shared_tasks
//! ```
//!
//! Модель отвечает JSON, разбирает и сохраняет его код: ответ не по формату не
//! портит задачу — она остаётся как была.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::memory::TaskState;
use crate::{ChatMessage, ChatOptions, LlmClient};

/// Заметки задачи, которые ведёт модель при `track_task`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskNotes {
    /// Что пользователь уже уточнил о себе и своей ситуации.
    #[serde(default)]
    pub clarified: Vec<String>,
    /// Ограничения и требования, которые ответы должны соблюдать.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// Зафиксированные термины: «термин — как его понимать».
    #[serde(default)]
    pub terms: Vec<String>,
}

impl TaskNotes {
    pub fn is_empty(&self) -> bool {
        self.clarified.is_empty() && self.constraints.is_empty() && self.terms.is_empty()
    }

    /// Непустые разделы для показа: (заголовок, пункты).
    pub fn sections(&self) -> Vec<(&'static str, &[String])> {
        [("Уточнено", &self.clarified), ("Ограничения", &self.constraints), ("Термины", &self.terms)]
            .into_iter()
            .filter(|(_, items)| !items.is_empty())
            .map(|(title, items)| (title, items.as_slice()))
            .collect()
    }

    /// Разделы текстом: `Уточнено:\n- …`.
    pub fn render(&self) -> String {
        self.sections()
            .into_iter()
            .map(|(title, items)| {
                let lines: String = items.iter().map(|i| format!("\n- {i}")).collect();
                format!("{title}:{lines}")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Цель и заметки задачи — то, что модель получает и возвращает при
/// обновлении (JSON).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tracked {
    #[serde(default)]
    pub goal: String,
    #[serde(flatten)]
    pub notes: TaskNotes,
}

impl Tracked {
    pub fn of(task: &TaskState) -> Self {
        Self { goal: task.goal.clone().unwrap_or_default(), notes: task.notes.clone() }
    }

    /// Цель и заметки текстом.
    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if !self.goal.is_empty() {
            parts.push(format!("Цель: {}", self.goal));
        }
        if !self.notes.is_empty() {
            parts.push(self.notes.render());
        }
        parts.join("\n")
    }
}

const UPDATE_PROMPT: &str = "Ты ведёшь задачу диалога пользователя с ассистентом. \
Тебе дают текущее состояние задачи (JSON) и последний обмен репликами. Верни обновлённое состояние — СТРОГО один \
JSON-объект с полями:
- \"goal\": цель всего диалога одной фразой — чего пользователь в итоге хочет добиться. Меняй её, только если \
пользователь сам сменил или явно расширил цель; отдельный вопрос по ходу — не новая цель;
- \"clarified\": что пользователь уточнил о себе и своей ситуации (размер команды, отрасль, стек, сроки, что уже \
сделано, выбранный вариант…) — список коротких пунктов;
- \"constraints\": ограничения ситуации пользователя, которые любые ответы должны соблюдать: запреты, обязательные \
условия, регуляторные требования, бюджет, сроки, формат или объём итогового результата — список. Вопрос \
пользователя и то, о чём он просит рассказать в этой реплике, — НЕ ограничение: не записывай пункты вида \
«ответ должен объяснить …»;
- \"terms\": термины, о значении которых договорились, в виде «термин — значение» — список.
Правила: бери факты только из реплик ПОЛЬЗОВАТЕЛЯ (ответ ассистента — лишь контекст; его советы — не уточнения \
пользователя); сохраняй прежние пункты, пока пользователь их не отменил; изменённое — замени, не дублируй; \
пункты короткие, по-русски. Без пояснений, без markdown, без текста до или после JSON.";

/// Обновляет цель и заметки задачи по одному обмену отдельным запросом к модели.
pub async fn update(
    client: &LlmClient,
    model: &str,
    reasoning: Option<bool>,
    previous: &Tracked,
    user: &str,
    assistant: &str,
) -> Result<Tracked> {
    let previous_json = serde_json::to_string_pretty(previous)?;
    let request = format!(
        "Текущее состояние задачи (JSON):\n{previous_json}\n\nПоследний обмен:\nПользователь: {user}\n\nАссистент: {}\n\n\
         Верни обновлённое состояние одним JSON-объектом.",
        crate::rag::retrieve::short(assistant, 1500)
    );
    let messages = [ChatMessage::system(UPDATE_PROMPT.to_string()), ChatMessage::user(request)];
    let options = ChatOptions { temperature: Some(0.0), reasoning, ..ChatOptions::default() };
    let completion = client.chat_with_model(model, &messages, &options).await?;
    parse(&completion.content)
        .with_context(|| format!("задача не обновлена — ответ модели не JSON: {}", completion.content))
}

/// JSON из ответа модели: без блока рассуждений и обёрток вокруг объекта.
fn parse(raw: &str) -> Result<Tracked> {
    let raw = raw.rsplit_once("</think>").map(|(_, rest)| rest).unwrap_or(raw);
    let start = raw.find('{').context("в ответе нет JSON-объекта")?;
    let end = raw.rfind('}').filter(|&end| end > start).context("в ответе нет JSON-объекта")?;
    let mut tracked: Tracked = serde_json::from_str(&raw[start..=end])?;
    tracked.goal = tracked.goal.trim().to_string();
    let notes = &mut tracked.notes;
    for list in [&mut notes.clarified, &mut notes.constraints, &mut notes.terms] {
        list.iter_mut().for_each(|item| *item = item.trim().to_string());
        list.retain(|item| !item.is_empty());
        let mut seen = std::collections::HashSet::new();
        list.retain(|item| seen.insert(item.to_lowercase()));
    }
    Ok(tracked)
}

/// Контекст диалога для поиска: цель и заметки задачи и прошлый обмен — чтобы
/// уточняющий вопрос («а для второго варианта?») искался вместе с тем, к чему
/// он относится. `None` — контекста ещё нет (первый вопрос).
pub fn search_context(tracked: &Tracked, history: &[ChatMessage]) -> Option<String> {
    let mut parts = Vec::new();
    let rendered = tracked.render();
    if !rendered.is_empty() {
        parts.push(rendered);
    }
    let last_user = history.iter().rev().find(|m| m.role == "user");
    let last_assistant = history.iter().rev().find(|m| m.role == "assistant");
    if let Some(m) = last_user {
        parts.push(format!("Прошлый вопрос: {}", crate::rag::retrieve::short(&m.content, 400)));
    }
    if let Some(m) = last_assistant {
        parts.push(format!("Прошлый ответ (начало): {}", crate::rag::retrieve::short(&m.content, 400)));
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_json_with_noise() {
        let raw = "<think>думаю</think>Вот:\n```json\n{\"goal\": \" внедрить AI в SDLC \", \"clarified\": [\"банк\", \"Банк\", \"\"], \
                   \"constraints\": [\"без PII\"]}\n```";
        let t = parse(raw).unwrap();
        assert_eq!(t.goal, "внедрить AI в SDLC");
        assert_eq!(t.notes.clarified, vec!["банк"]);
        assert_eq!(t.notes.constraints, vec!["без PII"]);
        assert!(t.notes.terms.is_empty());
        assert!(parse("нет json").is_err());
    }

    #[test]
    fn renders_only_filled_sections() {
        let t = Tracked {
            goal: "цель".into(),
            notes: TaskNotes { terms: vec!["PRD — документ требований".into()], ..Default::default() },
        };
        assert_eq!(t.render(), "Цель: цель\nТермины:\n- PRD — документ требований");
        assert_eq!(Tracked::default().render(), "");
    }

    #[test]
    fn search_context_needs_task_or_history() {
        assert!(search_context(&Tracked::default(), &[]).is_none());
        let history = [ChatMessage::user("Что такое intent.md?"), ChatMessage::assistant("Это документ замысла.")];
        let ctx = search_context(&Tracked::default(), &history).unwrap();
        assert!(ctx.contains("Прошлый вопрос: Что такое intent.md?"), "{ctx}");
        assert!(ctx.contains("Прошлый ответ (начало): Это документ замысла."), "{ctx}");
    }
}
