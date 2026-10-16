//! Задача на уровне цели: «найди все места, где используется X», «обнови
//! документацию по коду», «сгенерируй CHANGELOG». Какие файлы искать,
//! читать и править, решает модель сама — через инструменты MCP.
//!
//! ```text
//! цель → git_branch (кодом, если git подключён) и ожидающие изменения (кодом)
//!      → модель, до FILES_MAX_ROUNDS кругов:
//!          fs: list_files, search, read_file          — разведка и чтение
//!          fs: edit_file, write_file, discard_changes — правки (в оверлей, не на диск)
//!          fs: pending_changes                        — проверить свой дифф
//!          git: git_log, git_diff, git_status         — история и изменения
//!      → отчёт модели (Markdown) + дифф ожидающих изменений (кодом)
//!      → запуск сохраняется в files/runs/<id>/ (см. runs.rs)
//! ```
//!
//! На диск ничего не пишется: изменения ждут `/apply` (или `--apply`).

use anyhow::Result;
use llm_core::agent::sum_usage;
use llm_core::{invariants, ChatMessage, ChatOptions, RequestMessage, ToolCallRecord, Usage};
use serde::Serialize;

use crate::session::{chat, env_or, record, short, Pending, Session};

/// Сколько кругов «модель → инструменты» за задачу по умолчанию (`FILES_MAX_ROUNDS`).
pub const DEFAULT_MAX_ROUNDS: usize = 30;
/// Температура по умолчанию (`FILES_TEMPERATURE`): 0 — повторный запуск даёт тот же результат.
pub const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Длиннее результат инструмента модели не уходит (`FILES_TOOL_RESULT_CHARS`).
const DEFAULT_RESULT_CHARS: usize = 24_000;
/// За сколько шагов до предела модель получает предупреждение — пора к правкам и отчёту.
const WRAP_UP_ROUNDS: usize = 5;
/// Инструменты, после которых файлы выглядят иначе — повторное чтение снова имеет смысл.
const CHANGING_TOOLS: &[&str] = &["edit_file", "write_file", "discard_changes"];

/// Шаг выполнения — для интерфейса, по мере работы.
pub enum Step {
    Status(String),
    /// Вызов инструмента моделью — уже с результатом.
    Tool(ToolCallRecord),
}

pub type Progress<'a> = &'a (dyn Fn(Step) + Send + Sync);

/// Один запуск задачи и всё, на чём он построен.
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub goal: String,
    /// Сценарий из `files/scenarios`, если задача взята оттуда.
    pub scenario: Option<String>,
    /// Отчёт модели (Markdown).
    pub report: String,
    /// Ожидающие изменения после задачи (включая оставшиеся от прошлых задач).
    pub pending: Pending,
    pub tool_calls: Vec<ToolCallRecord>,
    pub rounds: usize,
    /// Задача упёрлась в предел кругов.
    pub hit_limit: bool,
    pub model: String,
    pub temperature: f32,
    pub usage: Option<Usage>,
    /// Чего не хватило (git не подключён и т. п.).
    pub notes: Vec<String>,
}

impl Outcome {
    /// Файлы, которые модель читала, искала или правила, — по вызовам.
    pub fn touched_files(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in self.tool_calls.iter().filter(|c| !c.is_error) {
            if !matches!(c.tool.as_str(), "read_file" | "edit_file" | "write_file") {
                continue;
            }
            let path = serde_json::from_str::<serde_json::Value>(&c.arguments)
                .ok()
                .and_then(|v| v["path"].as_str().map(str::to_string));
            if let Some(p) = path.filter(|p| !out.contains(p)) {
                out.push(p);
            }
        }
        out
    }
}

/// Выполнить задачу — см. документацию модуля. Ошибка — если сервер файлов
/// недоступен или не ответила модель.
pub async fn run(s: &Session, goal: &str, scenario: Option<&str>, progress: Progress<'_>) -> Result<Outcome> {
    let goal = goal.trim().to_string();
    if goal.is_empty() {
        anyhow::bail!("пустая цель — опишите, что нужно сделать с файлами проекта");
    }
    progress(Step::Status(format!("подключаю MCP {}…", s.fs_server)));
    s.ensure_fs().await.map_err(anyhow::Error::msg)?;

    let mut notes = Vec::new();
    let mut tool_calls = Vec::new();
    let git = match s.git_line().await {
        Ok(line) => {
            tool_calls.push(record(&s.git_server, "git_branch", "{}", &line, false));
            Some(line)
        }
        Err(err) => {
            notes.push(format!("git недоступен — без истории коммитов: {}", short(&err, 160)));
            None
        }
    };
    let before = s.pending(&mut tool_calls).await.unwrap_or_default();

    let tools = s.model_tools(git.is_some());
    let max_rounds: usize = env_or("FILES_MAX_ROUNDS", "").parse().unwrap_or(DEFAULT_MAX_ROUNDS).clamp(1, 200);
    let result_chars: usize = env_or("FILES_TOOL_RESULT_CHARS", "").parse().unwrap_or(DEFAULT_RESULT_CHARS).max(2_000);
    let temperature: f32 = env_or("FILES_TEMPERATURE", "").parse().unwrap_or(DEFAULT_TEMPERATURE);
    let options = ChatOptions { temperature: Some(temperature), ..ChatOptions::default() };

    let messages = [
        ChatMessage::system(system_prompt(git.is_some())),
        ChatMessage::user(user_prompt(&goal, git.as_deref(), &before, max_rounds)),
    ];
    let mut wire: Vec<RequestMessage> = messages.iter().map(RequestMessage::from).collect();
    let mut usage: Option<Usage> = None;
    let mut rounds = 0;
    let mut hit_limit = false;
    let mut seen = Repeats::default();
    let mut report = loop {
        rounds += 1;
        let last = rounds > max_rounds;
        if rounds + WRAP_UP_ROUNDS == max_rounds + 1 && max_rounds > WRAP_UP_ROUNDS {
            wire.push(RequestMessage::from(&ChatMessage::user(format!(
                "Осталось {WRAP_UP_ROUNDS} шагов из {max_rounds}. Новых файлов больше не изучай: если цель требует \
                 правок — делай их сейчас тем, что уже знаешь, потом отчёт."
            ))));
        }
        if last {
            hit_limit = true;
            wire.push(RequestMessage::from(&ChatMessage::user(
                "Лимит шагов исчерпан. Больше никаких инструментов — напиши итоговый отчёт по тому, что уже \
                 сделано и найдено, и что осталось несделанным.",
            )));
        }
        progress(Step::Status(format!("модель думает… (шаг {rounds})")));
        let offered = if last { &[][..] } else { &tools[..] };
        let completion = match chat_retry(s, &wire, offered, &options, progress).await {
            Ok(c) => c,
            // Модель отвалилась посреди задачи — не терять сделанное: правки в оверлее, вызовы, частичный отчёт.
            Err(err) if rounds > 1 => {
                notes.push(format!("модель не ответила на шаге {rounds}, задача прервана: {}", short(&format!("{err:#}"), 200)));
                break String::new();
            }
            Err(err) => return Err(err),
        };
        usage = sum_usage(usage, completion.usage);
        if completion.tool_calls.is_empty() || last {
            break completion.content.trim().to_string();
        }
        wire.push(RequestMessage::assistant_tool_calls(completion.content.clone(), completion.tool_calls.clone()));
        for call in &completion.tool_calls {
            let (name, args) = (call.function.name.as_str(), call.function.arguments.as_str());
            let (server, tool) = s.mcp.tool_origin(name).unwrap_or_else(|| ("?".into(), name.to_string()));
            progress(Step::Status(format!("MCP {server} · {tool}({})…", short(args, 80))));
            // Только инструменты из выданного списка: apply_changes модели не положен, даже если она его назовёт.
            let allowed = tools.iter().any(|t| t.name == name);
            let (text, is_error, to_model) = if !allowed {
                let text = format!("инструмента «{name}» нет — используй инструменты из списка");
                (text.clone(), true, text)
            } else if let Some(first) = seen.repeat_of(&tool, args, rounds) {
                // Тот же вызов с теми же доводами, а файлы с тех пор не менялись: результат у модели уже есть.
                let text = format!(
                    "Этот вызов с теми же доводами уже был на шаге {first}, и файлы с тех пор не менялись — \
                     результат тот же, он выше в диалоге. Не повторяй его: используй то, что уже получено, \
                     другой диапазон строк или переходи к следующему шагу цели."
                );
                (text.clone(), true, text)
            } else {
                match s.mcp.call(name, args).await {
                    Some(r) => {
                        if !r.is_error && CHANGING_TOOLS.contains(&tool.as_str()) {
                            seen.files_changed();
                        }
                        let to_model = clip(&r.text, result_chars);
                        (r.text, r.is_error, to_model)
                    }
                    None => {
                        let text = format!("инструмента «{name}» нет — используй инструменты из списка");
                        (text.clone(), true, text)
                    }
                }
            };
            let rec = record(&server, &tool, args, &text, is_error);
            progress(Step::Tool(rec.clone()));
            tool_calls.push(rec);
            wire.push(RequestMessage::tool_result(call.id.clone(), to_model));
        }
    };
    if report.is_empty() {
        report = "_Модель не написала отчёт — смотрите вызовы инструментов и дифф._".into();
    }
    if hit_limit {
        notes.push(format!("задача упёрлась в предел шагов ({max_rounds}, FILES_MAX_ROUNDS)"));
    }

    progress(Step::Status("собираю дифф…".into()));
    let pending = s.pending(&mut Vec::new()).await.unwrap_or_default();
    Ok(Outcome {
        goal,
        scenario: scenario.map(str::to_string),
        report,
        pending,
        tool_calls,
        rounds,
        hit_limit,
        model: s.client.model().to_string(),
        temperature,
        usage,
        notes,
    })
}

/// Запрос к модели; сбой (таймаут, сеть, ошибка провайдера) — ещё одна попытка.
async fn chat_retry(
    s: &Session,
    wire: &[RequestMessage],
    tools: &[llm_core::ToolDefinition],
    options: &ChatOptions,
    progress: Progress<'_>,
) -> Result<llm_core::ChatCompletion> {
    match chat(&s.client, wire, tools, options).await {
        Ok(c) => Ok(c),
        Err(err) => {
            progress(Step::Status(format!("модель не ответила ({}) — повторяю…", short(&format!("{err:#}"), 80))));
            chat(&s.client, wire, tools, options).await
        }
    }
}

/// Вызовы, уже сделанные за задачу: инструмент + доводы (без учёта порядка
/// ключей) → шаг. Правка файлов начинает отсчёт заново — после неё то же
/// чтение даёт другой результат.
#[derive(Default)]
struct Repeats {
    calls: std::collections::HashMap<String, usize>,
}

impl Repeats {
    /// Шаг, на котором такой вызов уже был; иначе запоминает его и `None`.
    fn repeat_of(&mut self, tool: &str, args: &str, round: usize) -> Option<usize> {
        let args = serde_json::from_str::<serde_json::Value>(args).map(|v| canonical(&v)).unwrap_or_else(|_| args.trim().to_string());
        let key = format!("{tool} {args}");
        match self.calls.get(&key) {
            Some(&first) => Some(first),
            None => {
                self.calls.insert(key, round);
                None
            }
        }
    }

    fn files_changed(&mut self) {
        self.calls.clear();
    }
}

/// JSON с ключами по алфавиту — один и тот же вызов при любом порядке доводов.
fn canonical(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys.iter().map(|k| format!("{:?}:{}", k, canonical(&map[*k]))).collect();
            format!("{{{}}}", parts.join(","))
        }
        serde_json::Value::Array(items) => format!("[{}]", items.iter().map(canonical).collect::<Vec<_>>().join(",")),
        other => other.to_string(),
    }
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}\n…[обрезано: результат длиннее {max} символов — сузь запрос: path, pattern, start_line/max_lines]")
}

fn system_prompt(with_git: bool) -> String {
    let git = if with_git {
        "\n- История и изменения проекта — инструменты git: git_log (коммиты, в том числе одного пути), git_diff \
         (что изменено; base=\"main\" — всё сделанное в ветке), git_status."
    } else {
        ""
    };
    let mut prompt = format!(
        "Ты — ассистент, который работает с файлами проекта сам: тебе ставят цель, а не говорят, какой файл \
         открыть. Ты сам находишь нужные файлы, читаешь их, анализируешь и, если цель этого требует, создаёшь или \
         правишь файлы.\n\n\
         Инструменты:\n\
         - Структура и поиск файлов по имени — list_files (depth=1/2 — обзор, pattern=\"*.rs\" — по имени).\n\
         - Где встречается имя, API, строка — search (pattern сужает по типу файлов, regex=true — регулярное \
         выражение, context — строки вокруг). Результат даёт число совпадений в каждом файле — полная картина.\n\
         - Текст файла — read_file (start_line/max_lines для больших файлов).\n\
         - Правки — edit_file (точечная замена: old_text символ в символ как в файле, с отступами, без номеров \
         строк, уникальный) и write_file (новый файл или полная перезапись). Правки копятся как ожидающие \
         изменения и НЕ попадают на диск: их увидит пользователь диффом и сам решит, записать ли.\n\
         - pending_changes — твой текущий дифф; discard_changes — откатить неудачную правку.{git}\n\n\
         Как работать:\n\
         1. Разведка: сначала найди, где лежит нужное (search, list_files), не гадай по названиям.\n\
         2. Прочитай найденные места, прежде чем делать выводы или править. Опирайся только на то, что прочитал: \
         не выдумывай содержимое файлов, имена, номера строк.\n\
         3. Меняй файлы, только если этого требует цель. Цель «найди», «проверь», «перечисли» — только отчёт, без \
         правок. Правки минимальные, в стиле и на языке соседнего текста; не трогай то, что к цели не относится.\n\
         4. После правок проверь себя: pending_changes — дифф именно такой, как задумано.\n\
         5. Работай экономно: ищи прицельно, большие файлы читай частями.\n\n\
         Итог — отчёт в Markdown на русском:\n\
         - `## Итог` — одно-два предложения: что найдено или сделано.\n\
         - Находки — со ссылками `путь:строка` (по результатам search/read_file), сгруппированные по смыслу.\n\
         - Если были правки — `## Изменения`: какие файлы и зачем (сам дифф не вставляй — его покажут отдельно).\n\
         - Если что-то не удалось или осталось под вопросом — честно скажи."
    );
    let block = invariants::format_invariants_block(&invariants::load_all());
    if !block.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&block);
    }
    prompt
}

fn user_prompt(goal: &str, git: Option<&str>, before: &Pending, max_rounds: usize) -> String {
    let mut out = String::new();
    if let Some(git) = git {
        out.push_str(&format!("Репозиторий: {git}\n"));
    }
    if !before.is_empty() {
        out.push_str(&format!(
            "Уже есть ожидающие изменения от прошлых задач — {} (read_file и search видят их, pending_changes \
             покажет дифф):\n{}\n",
            before.summary(),
            before.listing()
        ));
    }
    out.push_str(&format!(
        "Шагов на задачу: {max_rounds} (шаг — один твой ответ с вызовами инструментов; несколько независимых \
         вызовов в одном ответе — один шаг).\n\nЦель: {goal}"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::PendingFile;

    #[test]
    fn prompt_keeps_disk_writes_with_user() {
        let p = system_prompt(true);
        assert!(p.contains("НЕ попадают на диск"));
        assert!(p.contains("git_log"));
        assert!(!system_prompt(false).contains("git_log"));
    }

    #[test]
    fn user_prompt_mentions_previous_changes() {
        let before = Pending {
            files: vec![PendingFile { path: "docs/api.md".into(), status: "modified".into(), added: 2, deleted: 1 }],
            added: 2,
            deleted: 1,
            patch: String::new(),
        };
        let p = user_prompt("обнови README", Some("ветка day-34"), &before, 30);
        assert!(p.starts_with("Репозиторий: ветка day-34\n"));
        assert!(p.contains("~ docs/api.md  +2 −1"));
        assert!(p.contains("Шагов на задачу: 30"));
        assert!(p.ends_with("Цель: обнови README"));
        assert!(user_prompt("x", None, &Pending::default(), 5).starts_with("Шагов на задачу: 5"));
    }

    #[test]
    fn repeated_calls_are_caught_until_files_change() {
        let mut seen = Repeats::default();
        assert_eq!(seen.repeat_of("read_file", r#"{"path":"README.md","start_line":1}"#, 3), None);
        // Те же доводы в другом порядке и с пробелами — тот же вызов.
        assert_eq!(seen.repeat_of("read_file", r#"{ "start_line": 1, "path": "README.md" }"#, 5), Some(3));
        assert_eq!(seen.repeat_of("read_file", r#"{"path":"README.md","start_line":400}"#, 5), None);
        assert_eq!(seen.repeat_of("search", r#"{"path":"README.md","start_line":1}"#, 5), None);
        seen.files_changed();
        assert_eq!(seen.repeat_of("read_file", r#"{"path":"README.md","start_line":1}"#, 7), None);
    }

    #[test]
    fn long_results_are_clipped() {
        let text = "а".repeat(50);
        assert_eq!(clip(&text, 100), text);
        let clipped = clip(&text, 10);
        assert!(clipped.starts_with("аааааааааа\n…[обрезано"));
    }

    #[test]
    fn touched_files_come_from_calls() {
        let call = |tool: &str, args: &str, is_error| record("fs", tool, args, "{}", is_error);
        let o = Outcome {
            goal: "g".into(),
            scenario: None,
            report: String::new(),
            pending: Pending::default(),
            tool_calls: vec![
                call("search", r#"{"query":"x"}"#, false),
                call("read_file", r#"{"path":"a.rs"}"#, false),
                call("read_file", r#"{"path":"a.rs","start_line":40}"#, false),
                call("edit_file", r#"{"path":"b.md"}"#, false),
                call("read_file", r#"{"path":"missing.rs"}"#, true),
            ],
            rounds: 1,
            hit_limit: false,
            model: "m".into(),
            temperature: 0.0,
            usage: None,
            notes: vec![],
        };
        assert_eq!(o.touched_files(), vec!["a.rs", "b.md"]);
    }
}
