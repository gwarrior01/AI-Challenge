//! Ответ на вопрос пользователя с учётом его тикета.
//!
//! ```text
//! вопрос (+ ticket_id, + session_id)
//!   → номер тикета: упомянут в вопросе › передан в запросе (ticket_id) › тикет сессии
//!   → MCP support · get_ticket (кодом, всегда, если тикет известен)  → тикет + профиль клиента
//!     (тикета нет — USR-… в вопросе или клиент сессии → get_user)
//!   → RAG: поиск в базе знаний по вопросу + теме тикета и коду ошибки → фрагменты [n]
//!   → модель: CRM-контекст, фрагменты, история сессии; при нужде сама вызывает
//!       search_knowledge (ещё поиск) и MCP get_ticket / get_user / list_user_tickets
//!   → ответ + «Источники» (процитированные фрагменты), вызовы MCP и RAG отдельно
//! ```
//!
//! Тикет запрашивает код, а не модель: ответ с учётом тикета не зависит от
//! того, догадается ли модель вызвать инструмент. Сессия помнит тикет и
//! последние обмены — «а если код всё равно не придёт?» понимается без
//! повторения номера.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use llm_core::agent::sum_usage;
use llm_core::rag::{Embedder, RagSource};
use llm_core::{
    ChatMessage, ChatOptions, LlmClient, RequestMessage, ToolCallRecord, ToolDefinition, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::crm::{find_id, Crm, Lookup};
use crate::knowledge::{Found, Knowledge};

/// Инструмент поиска — «сервер» `rag` в записях вызовов (не MCP: индекс
/// сервис держит сам).
pub const RAG_SERVER: &str = "rag";
pub const SEARCH_KNOWLEDGE: &str = "search_knowledge";

const MAX_TOOL_ROUNDS: usize = 4;
/// Сколько прошлых обменов сессии уходит модели.
const HISTORY_TURNS: usize = 6;
/// Сколько сессий держать в памяти; лишние — самые давние — забываются.
const MAX_SESSIONS: usize = 200;
const DEFAULT_LLM_TIMEOUT_SEC: u64 = 120;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AskRequest {
    pub question: String,
    /// Продолжение разговора; нет — новая сессия.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Тикет, о котором вопрос, если номер не назван в самом вопросе
    /// (иначе — тикет сессии).
    #[serde(default)]
    pub ticket_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AskReply {
    pub session_id: String,
    pub question: String,
    /// Ответ с блоком «Источники».
    pub answer: String,
    /// Тикет, с учётом которого дан ответ.
    pub ticket_id: Option<String>,
    pub ticket: Option<Value>,
    pub customer: Option<Value>,
    /// Почему данных CRM нет или они неполные (тикет не найден, CRM недоступна).
    pub crm_note: Option<String>,
    /// Почему поиска по базе знаний не было или он не удался.
    pub rag_note: Option<String>,
    pub index_note: Option<String>,
    /// Фрагменты базы знаний, что получила модель (нумерация [n] общая).
    pub sources: Vec<RagSource>,
    /// Номера фрагментов, на которые ответ сослался, — они и есть «Источники».
    pub cited: Vec<usize>,
    /// Поиски по базе знаний: первый — кодом, остальные — моделью.
    pub searches: Vec<String>,
    /// Вызовы MCP и поиска в порядке выполнения.
    pub tool_calls: Vec<ToolCallRecord>,
    pub usage: Option<Usage>,
}

impl AskReply {
    /// Строки контекста над ответом: тикет, CRM, база знаний.
    pub fn context_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        match (&self.ticket, &self.customer) {
            (Some(t), Some(c)) => lines.push(format!(
                "тикет {} «{}» · {} · клиент {} ({}, тариф {}, статус {})",
                s(t, "id"),
                s(t, "subject"),
                s(t, "status"),
                s(c, "name"),
                s(c, "id"),
                s(c, "plan"),
                s(c, "status")
            )),
            (None, Some(c)) => lines.push(format!(
                "клиент {} ({}, тариф {}, статус {})",
                s(c, "name"),
                s(c, "id"),
                s(c, "plan"),
                s(c, "status")
            )),
            _ => {}
        }
        if let Some(note) = &self.crm_note {
            lines.push(format!("CRM: {note}"));
        }
        if let Some(note) = &self.index_note {
            lines.push(format!("индекс: {note}"));
        }
        match &self.rag_note {
            Some(note) => lines.push(format!("RAG: {note}")),
            None => {
                lines.push(format!(
                    "RAG: {} · фрагментов {}",
                    self.searches.join(" · "),
                    self.sources.len()
                ));
                lines.extend(
                    self.sources
                        .iter()
                        .map(|s| format!("[{}] {} · {}", s.n, s.location(), s.scores())),
                );
            }
        }
        lines
    }
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("?")
}

#[derive(Default)]
struct Session {
    ticket_id: Option<String>,
    user_id: Option<String>,
    /// (вопрос, ответ без «Источников»)
    turns: Vec<(String, String)>,
    used: Option<Instant>,
}

/// Модель, база знаний и CRM — одни на процесс, общие для всех запросов.
pub struct Support {
    pub client: LlmClient,
    pub crm: Crm,
    /// `Err` — модель эмбеддингов не настроена: сервис отвечает без базы знаний.
    pub knowledge: Result<Knowledge, String>,
    sessions: Mutex<HashMap<String, Session>>,
}

impl Support {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client: LlmClient::from_env()?,
            crm: Crm::from_env(),
            knowledge: Embedder::from_env()
                .map(Knowledge::new)
                .map_err(|e| format!("{e:#}")),
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// Забыть сессию (кнопка «Новый разговор»).
    pub fn reset(&self, session_id: &str) {
        self.sessions.lock().unwrap().remove(session_id);
    }

    /// Ответ на вопрос — см. документацию модуля. Ошибка — только если не
    /// ответила сама модель; недоступные CRM и база знаний отражаются в ответе.
    pub async fn ask(&self, req: AskRequest) -> Result<AskReply> {
        let question = req.question.trim().to_string();
        anyhow::ensure!(!question.is_empty(), "пустой вопрос");
        let session_id = req
            .session_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(new_session_id);
        let (session_ticket, session_user, history) = {
            let sessions = self.sessions.lock().unwrap();
            match sessions.get(&session_id) {
                Some(s) => (s.ticket_id.clone(), s.user_id.clone(), s.turns.clone()),
                None => (None, None, Vec::new()),
            }
        };

        // 1. CRM: тикет (или клиент) — кодом, до модели.
        let mut tool_calls = Vec::new();
        let ticket_id = find_id(&question, "TCK")
            .or_else(|| {
                req.ticket_id
                    .map(|t| t.trim().to_uppercase())
                    .filter(|t| !t.is_empty())
            })
            .or(session_ticket);
        let (mut ticket, mut customer, mut crm_note) = (None, None, None);
        if let Some(id) = &ticket_id {
            match self
                .crm
                .lookup("get_ticket", json!({ "id": id }), &mut tool_calls)
                .await
            {
                Lookup::Found(v) => {
                    ticket = v.get("ticket").cloned();
                    customer = v.get("user").cloned();
                }
                Lookup::Missing(err) => {
                    crm_note = Some(format!(
                        "тикет {id} не найден в CRM ({})",
                        crate::short(&err, 120)
                    ))
                }
                Lookup::Unavailable(note) => crm_note = Some(note),
            }
        } else if let Some(id) = find_id(&question, "USR").or(session_user) {
            match self
                .crm
                .lookup("get_user", json!({ "id": id }), &mut tool_calls)
                .await
            {
                Lookup::Found(v) => customer = Some(v),
                Lookup::Missing(_) => crm_note = Some(format!("клиент {id} не найден в CRM")),
                Lookup::Unavailable(note) => crm_note = Some(note),
            }
        }
        let ticket_id = ticket.as_ref().map(|t| s(t, "id").to_string());

        // 2. RAG: два поиска — по самому вопросу (вопрос о продукте находит
        // свои статьи, даже если в разговоре есть тикет) и, если тикет есть, по
        // вопросу с темой тикета и кодом ошибки (инструкция под его проблему).
        let mut found = Found::default();
        let (mut rag_note, mut index_note) = (None, None);
        let mut knowledge_text: Vec<String> = Vec::new();
        match &self.knowledge {
            Err(err) => rag_note = Some(format!("база знаний недоступна: {err}")),
            Ok(kb) => {
                match kb.sync().await {
                    Ok(note) => index_note = note,
                    Err(err) => index_note = Some(format!("индекс не обновлён: {err:#}")),
                }
                let previous = history.last().map(|(q, _)| q.as_str());
                for query in search_queries(&question, ticket.as_ref(), previous) {
                    match kb.search(&query).await {
                        Ok(ctx) => {
                            let text = found.merge(&crate::short(&query, 80), ctx);
                            tool_calls.push(crate::record(
                                RAG_SERVER,
                                SEARCH_KNOWLEDGE,
                                &json!({ "query": query }).to_string(),
                                &text,
                                false,
                            ));
                            knowledge_text.push(text);
                        }
                        Err(err) => {
                            rag_note = Some(format!("поиск не удался: {err:#}"));
                            break;
                        }
                    }
                }
            }
        }
        // Фрагменты без повторов: повторно найденный фрагмент в тексте второго
        // поиска уже был в первом.
        let first_search =
            (!found.rag.sources.is_empty() || !knowledge_text.is_empty()).then(|| {
                if found.rag.sources.is_empty() {
                    knowledge_text.join("\n\n")
                } else {
                    found
                        .rag
                        .sources
                        .iter()
                        .map(|s| format!("[{}] {}\n{}", s.n, s.location(), s.text))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
            });

        // 3. Модель.
        let with_search = first_search.is_some();
        let mut tools: Vec<ToolDefinition> = if with_search {
            vec![search_tool()]
        } else {
            Vec::new()
        };
        if self.crm.ensure_connected().await.is_ok() {
            tools.extend(self.crm.model_tools());
        }
        let mut messages = vec![ChatMessage::system(system_prompt())];
        for (q, a) in history.iter().rev().take(HISTORY_TURNS).rev() {
            messages.push(ChatMessage::user(q.clone()));
            messages.push(ChatMessage::assistant(a.clone()));
        }
        messages.push(ChatMessage::user(user_prompt(
            &question,
            ticket.as_ref(),
            customer.as_ref(),
            crm_note.as_deref(),
            first_search.as_deref(),
            rag_note.as_deref(),
        )));
        let mut wire: Vec<RequestMessage> = messages.iter().map(RequestMessage::from).collect();
        let options = ChatOptions {
            temperature: Some(0.2),
            ..ChatOptions::default()
        };
        let mut usage = None;
        // Текст рядом с вызовами инструментов — черновик: модель часто пишет
        // ответ и тут же идёт дочитать данные, а потом отвечает заново. Ответ —
        // текст последнего круга; черновик — только если последний пуст.
        let (mut answer_text, mut draft) = (String::new(), String::new());
        for round in 0..=MAX_TOOL_ROUNDS {
            // Последний круг — без инструментов: ответ по тому, что уже есть.
            let offered: &[ToolDefinition] = if round == MAX_TOOL_ROUNDS {
                &[]
            } else {
                &tools
            };
            let completion = self.chat(&wire, offered, &options).await?;
            usage = sum_usage(usage, completion.usage);
            if completion.tool_calls.is_empty() {
                answer_text = completion.content.clone();
                break;
            }
            if !completion.content.trim().is_empty() {
                draft = completion.content.clone();
            }
            wire.push(RequestMessage::assistant_tool_calls(
                completion.content.clone(),
                completion.tool_calls.clone(),
            ));
            for call in &completion.tool_calls {
                let (name, args) = (
                    call.function.name.as_str(),
                    call.function.arguments.as_str(),
                );
                let (text, is_error, server, tool) = match (name, &self.knowledge) {
                    (SEARCH_KNOWLEDGE, Ok(kb)) => {
                        let (text, is_error) = self.search(kb, args, &mut found).await;
                        (text, is_error, RAG_SERVER.to_string(), name.to_string())
                    }
                    _ => match self.crm.mcp.call(name, args).await {
                        Some(r) => (r.text, r.is_error, r.server, r.tool),
                        None => (
                            format!("инструмента «{name}» нет — используй инструменты из списка"),
                            true,
                            String::new(),
                            name.to_string(),
                        ),
                    },
                };
                wire.push(RequestMessage::tool_result(call.id.clone(), text.clone()));
                tool_calls.push(crate::record(&server, &tool, args, &text, is_error));
            }
        }

        let raw = if answer_text.trim().is_empty() {
            draft
        } else {
            answer_text
        };
        let answer = found.finalize(&raw);
        {
            let mut sessions = self.sessions.lock().unwrap();
            if sessions.len() >= MAX_SESSIONS && !sessions.contains_key(&session_id) {
                if let Some(oldest) = sessions
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(id, _)| id.clone())
                {
                    sessions.remove(&oldest);
                }
            }
            let session = sessions.entry(session_id.clone()).or_default();
            if ticket_id.is_some() {
                session.ticket_id = ticket_id.clone();
            }
            if let Some(c) = &customer {
                session.user_id = Some(s(c, "id").to_string());
            }
            session
                .turns
                .push((question.clone(), raw.trim().to_string()));
            session.used = Some(Instant::now());
        }
        Ok(AskReply {
            session_id,
            question,
            answer,
            ticket_id,
            ticket,
            customer,
            crm_note,
            rag_note,
            index_note,
            cited: found.cited(),
            sources: found.rag.sources,
            searches: found.queries,
            tool_calls,
            usage,
        })
    }

    /// Поиск, который попросила модель.
    async fn search(&self, kb: &Knowledge, arguments: &str, found: &mut Found) -> (String, bool) {
        let query = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| {
                v.get("query")
                    .and_then(Value::as_str)
                    .map(|q| q.trim().to_string())
            })
            .filter(|q| !q.is_empty());
        let Some(query) = query else {
            return ("нужен довод query — что искать".into(), true);
        };
        match kb.search(&query).await {
            Ok(ctx) => (found.merge(&crate::short(&query, 80), ctx), false),
            Err(err) => (format!("поиск по базе знаний не удался: {err:#}"), true),
        }
    }

    /// Запрос к модели с пределом времени (`SUPPORT_LLM_TIMEOUT`, секунды):
    /// у HTTP-клиента ядра таймаута нет.
    async fn chat(
        &self,
        wire: &[RequestMessage],
        tools: &[ToolDefinition],
        options: &ChatOptions,
    ) -> Result<llm_core::ChatCompletion> {
        let secs: u64 = crate::env_or("SUPPORT_LLM_TIMEOUT", "")
            .parse()
            .unwrap_or(DEFAULT_LLM_TIMEOUT_SEC)
            .max(5);
        let request = self
            .client
            .chat_with_tools(self.client.model(), wire, tools, false, options);
        match tokio::time::timeout(Duration::from_secs(secs), request).await {
            Ok(result) => result,
            Err(_) => {
                anyhow::bail!("модель не ответила за {secs} с (предел — SUPPORT_LLM_TIMEOUT)")
            }
        }
    }
}

fn new_session_id() -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("s{:x}{n:x}", nanos)
}

/// Поисковые запросы: вопрос (короткое уточнение — вместе с прошлым
/// вопросом) и, если есть тикет, он же с темой, категорией и кодом ошибки тикета.
fn search_queries(question: &str, ticket: Option<&Value>, previous: Option<&str>) -> Vec<String> {
    let mut base = vec![question.to_string()];
    if question.chars().count() < 60 {
        base.extend(previous.map(str::to_string));
    }
    let mut queries = vec![base.join(" · ")];
    if let Some(t) = ticket {
        let mut parts = vec![question.to_string()];
        for key in ["subject", "category"] {
            parts.extend(t.get(key).and_then(Value::as_str).map(str::to_string));
        }
        parts.extend(
            t.pointer("/environment/last_error")
                .and_then(Value::as_str)
                .map(str::to_string),
        );
        queries.push(parts.join(" · "));
    }
    queries
}

fn search_tool() -> ToolDefinition {
    ToolDefinition {
        name: SEARCH_KNOWLEDGE.into(),
        description: "Поиск по базе знаний продукта CloudDesk (документация и FAQ): возможности, тарифы, \
            команды и роли, проекты и задачи, отчёты и экспорт, интеграции и API, безопасность, вход и 2FA, оплата. Возвращает пронумерованные фрагменты [n], нумерация продолжает уже \
            полученные. Нужен, только если присланных фрагментов не хватает."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": { "query": { "type": "string", "description": "Что искать: суть проблемы и коды ошибок (`AUTH-LOCKED 2FA код не приходит`)." } },
            "required": ["query"]
        }),
    }
}

fn system_prompt() -> String {
    "Ты — ассистент поддержки продукта CloudDesk (облачный сервис для командной работы над проектами). \
     Ты одновременно консультант по продукту и сотрудник поддержки, который разбирает обращения клиентов.\n\n\
     В сообщении пользователя есть:\n\
     - «База знаний» — фрагменты [n] документации и FAQ продукта: возможности, тарифы, команды и роли, \
       проекты, отчёты, интеграции, безопасность, вход и оплата;\n\
     - «Данные CRM» (если тикет или клиент известны) — тикет и профиль клиента из CRM через MCP: тариф, статус \
       аккаунта, окружение, события тикета. Это факты о конкретном клиенте.\n\n\
     Два вида вопросов:\n\
     1. Вопрос о продукте — что умеет CloudDesk, как что-то сделать, чем отличаются тарифы, есть ли \
        интеграция. Отвечай как консультант: прямо и по делу, с шагами, если речь о действии. Номер тикета для \
        этого не нужен — не проси его. Данные CRM используй, только если они меняют ответ (например, что \
        доступно на тарифе этого клиента).\n\
     2. Проблема клиента или вопрос о его обращении — что-то не работает, ошибка, вопрос по тикету. Объясняй по \
        данным тикета: свяжи код ошибки, события и статус аккаунта с инструкцией из базы знаний, а не давай общий \
        совет. Структура: вероятная причина → что сделать сейчас (по шагам) → когда обращаться снова или что \
        передадут специалисту. Если тикета нет, а без данных аккаунта не разобраться — дай общую инструкцию \
        и попроси номер тикета.\n\n\
     Правила:\n\
     - о продукте говори только то, что есть во фрагментах базы знаний, и ставь после этих фактов номер [n]; \
       факты о клиенте — только из данных CRM;\n\
     - не выдумывай функции, цены и настройки: если в базе знаний ответа нет — так и скажи и предложи, что \
       уточнить; если тикет не найден — скажи об этом и не придумывай его содержимое;\n\
     - не проси пароль, полный номер карты, коды из SMS и резервные коды 2FA;\n\
     - если присланного не хватает, можешь вызвать search_knowledge (ещё поиск по базе знаний другими \
       словами) и инструменты CRM: get_ticket — тикет, list_user_tickets — история обращений клиента, \
       get_user — профиль;\n\
     - кратко, без лишних вступлений; отвечай на языке пользователя; к клиенту обращайся по имени, если оно \
       есть в данных CRM;\n\
     - блок «Источники» не пиши — его добавят автоматически."
        .into()
}

fn user_prompt(
    question: &str,
    ticket: Option<&Value>,
    customer: Option<&Value>,
    crm_note: Option<&str>,
    knowledge: Option<&str>,
    rag_note: Option<&str>,
) -> String {
    let mut p = String::new();
    match (ticket, customer) {
        (Some(t), c) => p.push_str(&format!(
            "Данные CRM (MCP support · get_ticket):\n{}\n\n",
            serde_json::to_string_pretty(&json!({ "ticket": t, "user": c })).unwrap_or_default()
        )),
        (None, Some(c)) => p.push_str(&format!(
            "Данные CRM (MCP support · get_user), тикет не указан:\n{}\n\n",
            serde_json::to_string_pretty(c).unwrap_or_default()
        )),
        (None, None) => {}
    }
    match crm_note {
        Some(note) => p.push_str(&format!(
            "Данные CRM: {note}. Не придумывай содержимое тикета.\n\n"
        )),
        None if ticket.is_none() && customer.is_none() => {
            p.push_str("Данные CRM: тикет не указан — отвечай как консультант по продукту.\n\n")
        }
        None => {}
    }
    match (knowledge, rag_note) {
        (Some(k), _) => p.push_str(&format!("База знаний (RAG):\n{k}\n\n")),
        (None, Some(note)) => p.push_str(&format!(
            "База знаний недоступна ({note}) — ссылки [n] не ставь.\n\n"
        )),
        (None, None) => {}
    }
    p.push_str(&format!("Вопрос пользователя: {question}"));
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket() -> Value {
        json!({
            "id": "TCK-1001", "subject": "Не могу войти в аккаунт", "category": "authentication", "status": "open",
            "environment": { "last_error": "AUTH-LOCKED" }
        })
    }

    #[test]
    fn search_queries_cover_question_and_ticket_topic() {
        let q = search_queries("Почему не работает авторизация?", Some(&ticket()), None);
        assert_eq!(
            q,
            vec![
                "Почему не работает авторизация?",
                "Почему не работает авторизация? · Не могу войти в аккаунт · authentication · AUTH-LOCKED",
            ]
        );
        // Вопрос о продукте без тикета — один поиск по самому вопросу.
        let q = search_queries(
            "Какие интеграции поддерживает CloudDesk и как их подключить?",
            None,
            Some("Тикет TCK-1001"),
        );
        assert_eq!(
            q,
            vec!["Какие интеграции поддерживает CloudDesk и как их подключить?"]
        );
        // Короткое уточнение ищется вместе с прошлым вопросом.
        let q = search_queries(
            "А если код не придёт?",
            None,
            Some("Почему не работает вход?"),
        );
        assert_eq!(q, vec!["А если код не придёт? · Почему не работает вход?"]);
    }

    #[test]
    fn prompt_has_crm_facts_knowledge_and_question() {
        let user = json!({ "id": "USR-001", "name": "Анна", "plan": "Pro", "status": "active" });
        let p = user_prompt(
            "Почему?",
            Some(&ticket()),
            Some(&user),
            None,
            Some("[1] authentication.md\n15 минут"),
            None,
        );
        assert!(
            p.contains("Данные CRM (MCP support · get_ticket)")
                && p.contains("AUTH-LOCKED")
                && p.contains("USR-001")
        );
        assert!(p.contains("База знаний (RAG):\n[1] authentication.md"));
        assert!(p.ends_with("Вопрос пользователя: Почему?"));

        let p = user_prompt(
            "Посмотри TCK-9999",
            None,
            None,
            Some("тикет TCK-9999 не найден в CRM"),
            None,
            Some("нет модели"),
        );
        assert!(p.contains("тикет TCK-9999 не найден в CRM. Не придумывай"));
        assert!(p.contains("База знаний недоступна (нет модели)"));

        let p = user_prompt("Как сменить тариф?", None, None, None, None, None);
        assert!(p.contains("тикет не указан"));
    }

    #[test]
    fn context_lines_summarize_ticket_and_sources() {
        let reply = AskReply {
            session_id: "s1".into(),
            question: "q".into(),
            answer: "a".into(),
            ticket_id: Some("TCK-1001".into()),
            ticket: Some(ticket()),
            customer: Some(
                json!({ "id": "USR-001", "name": "Анна", "plan": "Pro", "status": "active" }),
            ),
            crm_note: None,
            rag_note: None,
            index_note: None,
            sources: vec![],
            cited: vec![],
            searches: vec!["Почему?".into()],
            tool_calls: vec![],
            usage: None,
        };
        assert_eq!(
            reply.context_lines(),
            vec![
                "тикет TCK-1001 «Не могу войти в аккаунт» · open · клиент Анна (USR-001, тариф Pro, статус active)",
                "RAG: Почему? · фрагментов 0",
            ]
        );
    }

    #[test]
    fn session_ids_are_unique() {
        assert_ne!(new_session_id(), new_session_id());
    }
}
