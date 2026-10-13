//! `/help <вопрос>` — ответ на вопрос о самом проекте: его устройстве,
//! структуре, командах, API, схемах данных и текущем состоянии репозитория.
//!
//! ```text
//! вопрос → MCP git: git_branch (всегда, кодом)          → контекст репозитория
//!        → модель решает сама, что ей нужно:
//!            search_docs  — RAG по README + docs/        → фрагменты [n]
//!            search_code  — RAG по коду крейтов          → фрагменты [n] (нумерация общая)
//!            MCP git: git_status, git_diff, git_log, list_files, read_file
//!        → ответ + «Источники»: процитированные фрагменты и вызванные инструменты git
//! ```
//!
//! Текущую ветку и последний коммит код запрашивает сам — это локально и
//! дёшево, а нужно почти любому ответу. Всё остальное — инструменты модели:
//! на «на какой я ветке?» или «что не закоммичено?» она отвечает по git, не
//! трогая индексы, а на «как устроен RAG?» — ищет в документации и коде.
//! Индекс сверяется с файлами (SHA-256) при первом поиске по нему за вопрос.
//!
//! Без сервера git или без модели эмбеддингов ассистент всё равно отвечает —
//! тем, что есть, а строки контекста над ответом говорят, чего не хватило.

use anyhow::Result;
use llm_core::agent::sum_usage;
use llm_core::rag::{self, Embedder, RagConfig, RagContext, RagSettings, Rerank};
use llm_core::{ChatMessage, ChatOptions, RequestMessage, ToolCallRecord, ToolDefinition, Usage};
use serde::Serialize;
use serde_json::json;

use crate::context::{code_config, docs_config, record, short, sync_index, Assistant, Progress, STRATEGY};

/// Вопрос для `/help` без текста.
pub const DEFAULT_QUESTION: &str =
    "Кратко расскажи о проекте: что это, из каких частей состоит, где что лежит и с чего начать разработчику.";

/// Инструменты поиска — «сервер» `rag` в записях вызовов (не MCP: индексы
/// ассистент держит сам).
pub const RAG_SERVER: &str = "rag";
pub const SEARCH_DOCS: &str = "search_docs";
pub const SEARCH_CODE: &str = "search_code";

const MAX_TOOL_ROUNDS: usize = 6;

/// Ответ ассистента и то, на чём он построен.
#[derive(Debug, Clone, Serialize)]
pub struct HelpReply {
    pub question: String,
    /// Ответ с блоком «Источники».
    pub text: String,
    /// Контекст репозитория одной строкой (`ветка day-30 · 8eb1422 «…» · изменено файлов: 3`);
    /// `None` — сервер git недоступен, причина в `git_note`.
    pub git: Option<String>,
    pub git_note: Option<String>,
    /// Что изменилось в индексах при поиске, или почему их не удалось обновить.
    pub index_note: Option<String>,
    /// Почему поиска не было совсем (нет модели эмбеддингов); `None` — поиск
    /// был доступен модели.
    pub rag_note: Option<String>,
    /// Все фрагменты, найденные за ответ (нумерация общая для документации и кода).
    pub rag: RagContext,
    /// Поиски, которые сделала модель: `search_docs «запрос»`.
    pub searches: Vec<String>,
    /// Вызовы инструментов: первым — `git_branch`, сделанный кодом, затем
    /// вызовы модели (git и поиск, в порядке вызова).
    pub tool_calls: Vec<ToolCallRecord>,
    pub usage: Option<Usage>,
}

impl HelpReply {
    /// Строки контекста для показа над ответом: git, индекс, поиск.
    pub fn context_lines(&self) -> Vec<String> {
        let mut lines = vec![match (&self.git, &self.git_note) {
            (Some(git), _) => format!("git: {git}"),
            (None, Some(note)) => format!("git: {note}"),
            (None, None) => "git: нет данных".to_string(),
        }];
        if let Some(note) = &self.index_note {
            lines.push(format!("индекс: {note}"));
        }
        match (&self.rag_note, self.searches.is_empty()) {
            (Some(note), _) => lines.push(format!("RAG недоступен: {note}")),
            // Без поиска, но с вызовами MCP — ответ построен на них, их видно отдельно.
            (None, true) if self.tool_calls.iter().any(|c| !c.is_error && c.server.as_deref() != Some(RAG_SERVER)) => {}
            (None, true) => lines.push("RAG: не понадобился — модель ответила без поиска".into()),
            (None, false) => {
                lines.push(format!(
                    "RAG: поисков {}, фрагментов {} ({})",
                    self.searches.len(),
                    self.rag.sources.len(),
                    self.searches.join(" · ")
                ));
                lines.extend(self.rag.sources.iter().map(|s| format!("[{}] {} · {}", s.n, s.location(), s.scores())));
                lines.extend(self.rag.warnings.iter().map(|w| format!("⚠ {w}")));
                if let Some(check) = &self.rag.check {
                    lines.push(check.summary());
                }
            }
        }
        lines
    }
}

/// Настройки поиска: `structure` (разделы Markdown, элементы кода),
/// эвристический реранкинг по идентификаторам. Переписывать запрос не нужно —
/// его формулирует сама модель.
fn search_settings(k: usize) -> RagSettings {
    RagSettings {
        strategy: Some(STRATEGY.name().to_string()),
        k,
        rewrite: false,
        rerank: Rerank::Heuristic,
        ..RagSettings::default()
    }
}

fn search_tools() -> Vec<ToolDefinition> {
    let params = |what: &str| {
        json!({
            "type": "object",
            "properties": { "query": { "type": "string", "description": what } },
            "required": ["query"]
        })
    };
    vec![
        ToolDefinition {
            name: SEARCH_DOCS.into(),
            description: "Поиск по документации проекта (RAG): README.md (на английском) и docs/ (на русском) — \
                архитектура, крейты, API веб-сервера и MCP-серверов, схемы данных (таблицы SQLite, JSON-файлы), \
                сборка и тесты, настройки и переменные окружения. Возвращает пронумерованные фрагменты [n] — \
                ссылайся на них в ответе."
                .into(),
            parameters: params(
                "Что искать: суть вопроса ключевыми словами по-русски и по-английски, с именами модулей, \
                 команд, переменных (`agents table columns / таблица agents поля`).",
            ),
        },
        ToolDefinition {
            name: SEARCH_CODE.into(),
            description: "Семантический поиск по исходному коду проекта (RAG): крейты core, cli, tui, web, \
                assistant, gateway, mcp/* — функции, типы, константы с путём файла и строками. Для вопросов \
                «где реализовано», «как работает функция», «какое значение по умолчанию». Фрагменты [n] \
                продолжают общую нумерацию; найденный файл можно дочитать read_file."
                .into(),
            parameters: params("Что искать: имена функций/типов/констант и суть по-английски (`MCP call timeout`)."),
        },
    ]
}

/// Поиск по индексам как инструменты модели: индекс сверяется с файлами при
/// первом обращении к нему за вопрос, найденные фрагменты копятся с общей
/// нумерацией (повторно найденный фрагмент сохраняет свой номер).
struct Search {
    embedder: Embedder,
    synced_docs: bool,
    synced_code: bool,
    rag: RagContext,
    queries: Vec<String>,
    notes: Vec<String>,
}

impl Search {
    fn new(embedder: Embedder) -> Self {
        Self {
            embedder,
            synced_docs: false,
            synced_code: false,
            rag: RagContext { strategy: STRATEGY.name().into(), ..RagContext::default() },
            queries: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Результат для модели и признак ошибки.
    async fn run(&mut self, tool: &str, arguments: &str) -> (String, bool) {
        let query = serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|v| v.get("query").and_then(|q| q.as_str()).map(|q| q.trim().to_string()))
            .filter(|q| !q.is_empty());
        let Some(query) = query else {
            return ("нужен довод query — что искать".into(), true);
        };
        let docs = tool == SEARCH_DOCS;
        let (cfg, k, what): (RagConfig, usize, &str) =
            if docs { (docs_config(), 6, "документации") } else { (code_config(), 5, "кода") };
        let synced = if docs { &mut self.synced_docs } else { &mut self.synced_code };
        if !*synced {
            match sync_index(&cfg, &self.embedder).await {
                Ok(note) => {
                    self.notes.extend(note);
                    *synced = true;
                }
                Err(err) => {
                    self.notes.push(format!("индекс {what} не обновлён"));
                    return (format!("индекс {what} недоступен: {err:#}"), true);
                }
            }
        }
        self.queries.push(format!("{tool} «{}»", short(&query, 60)));
        let ctx = match rag::retrieve::retrieve_in(&cfg, &self.embedder, &query, &search_settings(k), None, None).await {
            Ok(ctx) => ctx,
            Err(err) => return (format!("поиск по индексу {what} не удался: {err:#}"), true),
        };
        (self.merge(ctx, what), false)
    }

    /// Добавить найденное к общему списку; текст фрагментов — для модели.
    fn merge(&mut self, ctx: RagContext, what: &str) -> String {
        let r = &mut self.rag;
        r.candidates += ctx.candidates;
        r.below_threshold += ctx.below_threshold;
        r.below_rerank += ctx.below_rerank;
        r.min_score = ctx.min_score;
        r.rerank = ctx.rerank;
        r.best_score = match (r.best_score, ctx.best_score) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        r.warnings.extend(ctx.warnings);
        if ctx.sources.is_empty() {
            let best = ctx
                .best_score
                .map(|b| format!(": лучшая близость {b:.2} при пороге {:.2}", ctx.min_score))
                .unwrap_or_default();
            return format!(
                "В индексе {what} близких фрагментов нет{best}. Переформулируй query (другие термины, по-английски, \
                 имена модулей) или используй инструменты git."
            );
        }
        let mut out = String::new();
        for mut src in ctx.sources {
            let n = match r.sources.iter().find(|s| s.chunk_id == src.chunk_id) {
                Some(seen) => seen.n,
                None => {
                    src.n = r.sources.len() + 1;
                    r.sources.push(src.clone());
                    src.n
                }
            };
            out.push_str(&format!("[{n}] {}\n{}\n\n", src.location(), src.text));
        }
        out.trim_end().to_string()
    }
}

/// Ответ на вопрос о проекте — см. документацию модуля. Ошибка — только если
/// не ответила сама модель; недоступные git и индекс отражаются в ответе.
/// `progress` получает этапы и каждый вызов инструмента в момент вызова.
pub async fn ask(a: &Assistant, question: &str, progress: Progress<'_>) -> Result<HelpReply> {
    let (client, mcp, git_server) = (&a.client, &a.mcp, a.git_server.as_str());
    let question = match question.trim() {
        "" => DEFAULT_QUESTION.to_string(),
        q => q.to_string(),
    };

    let mut tool_calls = Vec::new();
    progress(format!("MCP {git_server} · git_branch() — текущая ветка…"));
    let (git, git_json, git_note) = match a.ensure_git().await {
        Ok(()) => match mcp.call_tool(git_server, "git_branch", "{}").await {
            Some(result) => {
                tool_calls.push(record(git_server, "git_branch", "{}", &result.text, result.is_error));
                if result.is_error {
                    (None, None, Some(format!("git_branch не удался: {}", short(&result.text, 200))))
                } else {
                    (Some(crate::context::git_line(&result.text)), Some(result.text), None)
                }
            }
            None => (None, None, Some(format!("у сервера «{git_server}» нет инструмента git_branch"))),
        },
        Err(note) => (None, None, Some(note)),
    };

    let (mut search, rag_note) = match Embedder::from_env() {
        Ok(embedder) => (Some(Search::new(embedder)), None),
        Err(err) => (None, Some(format!("{err:#}"))),
    };
    let mut tools: Vec<ToolDefinition> = if search.is_some() { search_tools() } else { Vec::new() };
    if git.is_some() {
        let prefix = llm_core::mcp::qualified_tool_name(git_server, "");
        let branch = llm_core::mcp::qualified_tool_name(git_server, "git_branch");
        tools.extend(mcp.tool_definitions().into_iter().filter(|t| t.name.starts_with(&prefix) && t.name != branch));
    }
    let messages = vec![
        ChatMessage::system(system_prompt(search.is_some(), git.is_some())),
        ChatMessage::user(user_prompt(&question, git_json.as_deref(), git_note.as_deref(), rag_note.as_deref())),
    ];
    let mut wire: Vec<RequestMessage> = messages.iter().map(RequestMessage::from).collect();
    let options = ChatOptions { temperature: Some(0.2), ..ChatOptions::default() };
    let mut usage: Option<Usage> = None;
    let mut parts: Vec<String> = Vec::new();
    for round in 0..=MAX_TOOL_ROUNDS {
        // Последний круг — без инструментов: ответ текстом по тому, что уже есть.
        let offered: &[ToolDefinition] = if round == MAX_TOOL_ROUNDS { &[] } else { &tools };
        progress("модель думает…".into());
        let completion = client.chat_with_tools(client.model(), &wire, offered, false, &options).await?;
        usage = sum_usage(usage, completion.usage);
        if !completion.content.trim().is_empty() {
            parts.push(completion.content.clone());
        }
        if completion.tool_calls.is_empty() {
            break;
        }
        wire.push(RequestMessage::assistant_tool_calls(completion.content.clone(), completion.tool_calls.clone()));
        for call in &completion.tool_calls {
            let (name, args) = (call.function.name.as_str(), call.function.arguments.as_str());
            let (text, is_error, server, tool) = match (name, search.as_mut()) {
                (SEARCH_DOCS | SEARCH_CODE, Some(search)) => {
                    progress(format!("RAG · {name}({}) — вызвала модель…", short(args.trim(), 80)));
                    let (text, is_error) = search.run(name, args).await;
                    (text, is_error, RAG_SERVER.to_string(), name.to_string())
                }
                _ => {
                    let (server, tool) =
                        mcp.tool_origin(name).unwrap_or_else(|| (git_server.to_string(), name.to_string()));
                    progress(format!("MCP {server} · {tool}({}) — вызвала модель…", short(args.trim(), 80)));
                    match mcp.call(name, args).await {
                        Some(r) => (r.text, r.is_error, r.server, r.tool),
                        None => (format!("инструмента «{name}» нет — используй инструменты из списка"), true, server, tool),
                    }
                }
            };
            wire.push(RequestMessage::tool_result(call.id.clone(), text.clone()));
            tool_calls.push(record(&server, &tool, args, &text, is_error));
        }
    }

    let (mut rag, searches, notes) = match search {
        Some(s) => (s.rag, s.queries, s.notes),
        None => (RagContext::default(), Vec::new(), Vec::new()),
    };
    let answer = parts.join("\n\n");
    let text = finalize(&answer, &mut rag);
    Ok(HelpReply {
        question,
        text,
        git,
        git_note,
        index_note: (!notes.is_empty()).then(|| notes.join("; ")),
        rag_note,
        rag,
        searches,
        tool_calls,
        usage,
    })
}

fn system_prompt(with_search: bool, with_git: bool) -> String {
    let mut s = String::from(
        "Ты — ассистент разработчика этого проекта. Отвечаешь на вопросы о нём: как он устроен, из каких \
         частей состоит, где что лежит, как запустить и проверить, какие у него команды, API и схемы данных, \
         в каком состоянии репозиторий.\n\n\
         Контекст репозитория от MCP-сервера git (текущая ветка, последний коммит, число изменённых файлов) уже \
         есть в сообщении — за веткой инструмент не нужен.\n\nИнструменты:\n",
    );
    if with_search {
        s.push_str(
            "- search_docs — поиск по документации (README и docs/), search_code — поиск по коду. Возвращают \
             фрагменты [n]; после фактов из них ставь номер фрагмента.\n",
        );
    }
    if with_git {
        s.push_str(
            "- git (MCP, только чтение): git_status — что изменено и не закоммичено, git_diff — сами изменения \
             (base=\"main\" — всё сделанное в ветке; для большого диффа сначала stat_only=true), git_log — \
             история, list_files — какие файлы есть (depth=1 или 2 — обзор структуры, pattern — поиск по имени), \
             read_file — текст файла.\n",
        );
    }
    s.push_str("\nКак выбирать:\n");
    if with_git {
        s.push_str(
            "- вопрос о текущем состоянии репозитория — ветка, коммиты, что изменено или не закоммичено, какие \
             файлы есть — отвечай по контексту git и инструментам git; искать в документации для этого не нужно;\n",
        );
    }
    if with_search {
        s.push_str(
            "- вопрос об устройстве проекта — архитектура, крейты, команды, API, схемы данных, настройки, «как», \
             «где», «почему» — сначала search_docs; если речь о конкретном коде (где реализовано, как работает, \
             значения по умолчанию) — ещё search_code, а при необходимости дочитай файл read_file. На такие \
             вопросы не отвечай по памяти без поиска;\n\
             - ничего не нашлось — переформулируй запрос один-два раза (другие термины, по-английски) или \
             посмотри файлы через git;\n",
        );
    } else {
        s.push_str("- поиск по документации и коду недоступен — используй list_files и read_file;\n");
    }
    s.push_str(
        "\nПравила:\n\
         - отвечай на языке вопроса, конкретно и по делу: пути файлов, имена крейтов, модулей и функций, команды;\n\
         - не выдумывай файлы, команды, эндпоинты и поля — только то, что есть в контексте и результатах \
         инструментов; документация и код могут расходиться — тогда скажи об этом;\n\
         - ссылки [n] ставь только на фрагменты, которые вернул поиск;\n\
         - если данных не хватает — так и скажи и подскажи, где посмотреть;\n\
         - блок «Источники» не пиши — его добавят автоматически.",
    );
    s
}

fn user_prompt(question: &str, git_json: Option<&str>, git_note: Option<&str>, rag_note: Option<&str>) -> String {
    let mut s = String::new();
    match (git_json, git_note) {
        (Some(json), _) => s.push_str(&format!("Контекст репозитория (MCP git · git_branch):\n{json}\n\n")),
        (None, Some(note)) => s.push_str(&format!("Контекст репозитория недоступен: {note}\n\n")),
        (None, None) => {}
    }
    if let Some(note) = rag_note {
        s.push_str(&format!("Поиск по документации и коду недоступен ({note}) — ссылки [n] не ставь.\n\n"));
    }
    s.push_str(&format!("Вопрос: {question}"));
    s
}

/// Ответ с блоком «Источники»: фрагменты, на которые сослалась модель (не
/// сослалась ни на один — все полученные; из данных поиска, не с её слов).
/// Вызовы MCP сюда не входят — интерфейсы показывают их отдельно. Проверка
/// ссылок — в `rag.check`.
fn finalize(answer: &str, rag: &mut RagContext) -> String {
    let mut text = answer.trim().to_string();
    let mut sources = String::new();
    if !rag.sources.is_empty() {
        let mut check = rag::answer::check(answer, rag);
        // Модель не поставила ни одной ссылки [n] — источниками считаются все
        // фрагменты, что ушли ей: ответ построен на них.
        if check.cited.is_empty() {
            check.cited = rag.sources.iter().map(|s| s.n).collect();
        }
        sources.push_str(&rag::answer::sources_block(rag, &check.cited));
        rag.check = Some(check);
    }
    if !sources.is_empty() {
        text.push_str(&format!("\n\nИсточники:{sources}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::rag::RagSource;

    fn source(n: usize, text: &str) -> RagSource {
        RagSource {
            n,
            chunk_id: format!("docs/architecture.md@v1/structure/{n:04}"),
            source: "docs/architecture.md".into(),
            name: "docs/architecture.md".into(),
            version: 1,
            section: Some("Крейты".into()),
            start_line: 10,
            end_line: 20,
            score: 0.7,
            rerank_score: None,
            rank_before: n,
            text: text.into(),
        }
    }

    #[test]
    fn prompt_carries_git_context_and_question() {
        let p = user_prompt("на какой я ветке?", Some(r#"{"branch":"main"}"#), None, None);
        assert!(p.contains("MCP git · git_branch"));
        assert!(p.ends_with("Вопрос: на какой я ветке?"));
        assert!(!p.contains("Фрагменты"));
        let p = user_prompt("как устроен RAG?", None, Some("MCP-сервер «git» выключен"), Some("не задана LLM_EMBEDDING_MODEL"));
        assert!(p.contains("Контекст репозитория недоступен: MCP-сервер «git» выключен"));
        assert!(p.contains("Поиск по документации и коду недоступен"));
    }

    #[test]
    fn system_prompt_routes_git_questions_away_from_search() {
        let p = system_prompt(true, true);
        assert!(p.contains("search_docs") && p.contains("git_status"));
        assert!(p.contains("искать в документации для этого не нужно"));
        let p = system_prompt(false, true);
        assert!(!p.contains("search_docs —"));
        assert!(p.contains("поиск по документации и коду недоступен"));
        let names: Vec<String> = search_tools().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec![SEARCH_DOCS, SEARCH_CODE]);
    }

    #[test]
    fn found_fragments_share_numbering_across_searches() {
        let mut search = Search::new(Embedder::new("http://127.0.0.1:9", None, "m", 8));
        let docs = RagContext { sources: vec![source(1, "core — ядро"), source(2, "tui")], ..RagContext::default() };
        let out = search.merge(docs, "документации");
        assert!(out.starts_with("[1] docs/architecture.md › Крейты · строки 10–20\ncore — ядро"));
        // Код: новый фрагмент получает №3, повторно найденный №2 — свой прежний номер.
        let mut code_src = source(9, "fn run()");
        code_src.chunk_id = "core/src/agent.rs@v1/structure/0001".into();
        let code = RagContext { sources: vec![code_src, source(2, "tui")], ..RagContext::default() };
        let out = search.merge(code, "кода");
        assert!(out.starts_with("[3] "));
        assert!(out.contains("[2] docs/architecture.md"));
        let ns: Vec<usize> = search.rag.sources.iter().map(|s| s.n).collect();
        assert_eq!(ns, vec![1, 2, 3]);
        let empty = RagContext { best_score: Some(0.31), min_score: 0.5, ..RagContext::default() };
        assert!(search.merge(empty, "кода").contains("близких фрагментов нет: лучшая близость 0.31 при пороге 0.50"));
    }

    #[test]
    fn context_lines_say_when_search_was_not_needed() {
        let reply = HelpReply {
            question: "q".into(),
            text: "t".into(),
            git: Some("ветка main".into()),
            git_note: None,
            index_note: None,
            rag_note: None,
            rag: RagContext::default(),
            searches: vec![],
            tool_calls: vec![],
            usage: None,
        };
        assert_eq!(reply.context_lines(), vec!["git: ветка main", "RAG: не понадобился — модель ответила без поиска"]);

        // Ответ построен на вызовах MCP — строки про RAG нет.
        let with_mcp = HelpReply { tool_calls: vec![record("git", "git_status", "{}", "{}", false)], ..reply };
        assert_eq!(with_mcp.context_lines(), vec!["git: ветка main"]);
    }

    #[test]
    fn sources_are_cited_fragments_only() {
        let mut rag = RagContext {
            sources: vec![source(1, "core — ядро"), source(2, "tui — интерфейс")],
            ..RagContext::default()
        };
        let text = finalize("Интерфейс — в tui [2].", &mut rag);
        assert!(text.starts_with("Интерфейс — в tui [2].\n\nИсточники:\n- [2] `docs/architecture.md`"));
        assert!(!text.contains("- [1]"));
        assert!(!text.contains("MCP"));
        assert_eq!(rag.check.as_ref().unwrap().cited, vec![2]);

        // Ни одной ссылки [n] — в источниках все фрагменты, что получила модель.
        let mut uncited = RagContext {
            sources: vec![source(1, "core — ядро"), source(2, "tui — интерфейс")],
            ..RagContext::default()
        };
        let text = finalize("Интерфейс — в tui.", &mut uncited);
        assert!(text.contains("- [1] `docs/architecture.md`"));
        assert!(text.contains("- [2] `docs/architecture.md`"));
        assert_eq!(uncited.check.as_ref().unwrap().cited, vec![1, 2]);

        // Без фрагментов (ответ по MCP или без данных) — ответ как есть, без «Источников».
        let mut empty = RagContext::default();
        assert_eq!(finalize(" Не знаю. ", &mut empty), "Не знаю.");
    }
}
