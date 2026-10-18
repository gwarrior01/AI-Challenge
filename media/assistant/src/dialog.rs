//! Диалог об одной записи. В отличие от file-assistant, задача здесь — не
//! одна реплика: ассистент спрашивает («какой WebM?», «делать саммари?»), и
//! ответ пользователя продолжает тот же разговор с той же историей.
//!
//! ```text
//! реплика пользователя
//!   → модель, до MEDIA_MAX_ROUNDS кругов на реплику:
//!       inspect                         — чекпоинты и следующий шаг (код сервера)
//!       convert_to_mp4 / extract_mp3 /
//!       split_audio / transcribe /
//!       summarize                       — задание; его ждёт КОД (wait_job), модель
//!                                         получает уже итог: succeeded / failed
//!       list_llm_models, write_summary_prompt, probe, list_recordings
//!   → ответ модели: вопрос пользователю или отчёт
//! ```
//!
//! Код, а не модель, ещё и подставляет каталог записи в вызовы, где модель
//! его не указала, — чтобы шаг не ушёл в корень.

use anyhow::Result;
use llm_core::agent::sum_usage;
use llm_core::{ChatMessage, ChatOptions, RequestMessage, ToolCallRecord, Usage};
use serde::Serialize;
use serde_json::{json, Value};

use crate::session::{chat, env_or, record, short, Session};

/// Сколько кругов «модель → инструменты» на одну реплику по умолчанию (`MEDIA_MAX_ROUNDS`).
pub const DEFAULT_MAX_ROUNDS: usize = 25;
/// Сколько секунд один вызов `wait_job` ждёт задание — потом ход работы обновляется в интерфейсе.
const WAIT_SLICE_SEC: u64 = 5;
/// Сколько подряд неудачных `wait_job` терпеть, прежде чем сдаться.
const WAIT_FAILURES: usize = 3;
/// Инструменты сервера, которые работают с каталогом записи (`dir`).
const DIR_TOOLS: &[&str] =
    &["inspect", "probe", "convert_to_mp4", "extract_mp3", "split_audio", "transcribe", "summarize", "write_summary_prompt"];

/// Шаг выполнения — для интерфейса, по мере работы.
pub enum Step {
    Status(String),
    /// Вызов инструмента — уже с результатом (у заданий — с итогом задания).
    Tool(ToolCallRecord),
    /// Ход идущего задания: `convert_to_mp4 j1 · 12:01 из 58:10 (20%)`.
    Job(String),
}

pub type Progress<'a> = &'a (dyn Fn(Step) + Send + Sync);

/// Одна реплика и ответ на неё.
#[derive(Debug, Clone, Serialize)]
pub struct Turn {
    pub user: String,
    pub reply: String,
    pub rounds: usize,
    pub usage: Option<Usage>,
    /// Что пошло не так: упёрлись в предел шагов, модель отвалилась.
    pub notes: Vec<String>,
}

pub struct Dialog {
    /// Каталог записи относительно корня сервера (`MEDIA_ROOT`).
    pub dir: String,
    wire: Vec<RequestMessage>,
    pub turns: Vec<Turn>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub usage: Option<Usage>,
    pub temperature: f32,
}

impl Dialog {
    pub fn new(dir: &str) -> Self {
        Self {
            dir: dir.to_string(),
            wire: vec![RequestMessage::from(&ChatMessage::system(system_prompt()))],
            turns: Vec::new(),
            tool_calls: Vec::new(),
            usage: None,
            temperature: env_or("MEDIA_TEMPERATURE", "").parse().unwrap_or(0.0),
        }
    }

    /// Реплика пользователя → ответ ассистента. Ошибка — если сервер `media`
    /// недоступен или модель не ответила на первом же шаге (реплика тогда в
    /// историю не попадает).
    pub async fn send(&mut self, s: &Session, input: &str, progress: Progress<'_>) -> Result<Turn> {
        let input = input.trim();
        if input.is_empty() {
            anyhow::bail!("пустая реплика");
        }
        progress(Step::Status(format!("подключаю MCP {}…", s.server)));
        s.ensure().await.map_err(anyhow::Error::msg)?;

        let tools = s.model_tools();
        let max_rounds: usize = env_or("MEDIA_MAX_ROUNDS", "").parse().unwrap_or(DEFAULT_MAX_ROUNDS).clamp(1, 200);
        let options = ChatOptions { temperature: Some(self.temperature), ..ChatOptions::default() };
        let mark = self.wire.len();
        self.wire.push(RequestMessage::from(&ChatMessage::user(user_message(input, &self.dir, self.turns.is_empty()))));

        let mut usage: Option<Usage> = None;
        let mut notes = Vec::new();
        let mut rounds = 0;
        let reply = loop {
            rounds += 1;
            let last = rounds > max_rounds;
            if last {
                notes.push(format!("реплика упёрлась в предел шагов ({max_rounds}, MEDIA_MAX_ROUNDS)"));
                self.wire.push(RequestMessage::from(&ChatMessage::user(
                    "Лимит шагов на эту реплику исчерпан. Больше никаких инструментов — скажи пользователю, что сделано, \
                     на чём остановились и что осталось.",
                )));
            }
            progress(Step::Status(format!("модель думает… (шаг {rounds})")));
            let offered = if last { &[][..] } else { &tools[..] };
            let completion = match chat_retry(s, &self.wire, offered, &options, progress).await {
                Ok(c) => c,
                // Модель отвалилась посреди реплики — сделанное (задания, файлы) остаётся, история тоже.
                Err(err) if rounds > 1 => {
                    notes.push(format!("модель не ответила на шаге {rounds}: {}", short(&format!("{err:#}"), 200)));
                    break String::new();
                }
                Err(err) => {
                    self.wire.truncate(mark);
                    return Err(err);
                }
            };
            usage = sum_usage(usage, completion.usage);
            if completion.tool_calls.is_empty() || last {
                break completion.content.trim().to_string();
            }
            self.wire.push(RequestMessage::assistant_tool_calls(completion.content.clone(), completion.tool_calls.clone()));
            for call in &completion.tool_calls {
                let (name, args) = (call.function.name.as_str(), call.function.arguments.as_str());
                let (server, tool) = s.mcp.tool_origin(name).unwrap_or_else(|| ("?".into(), name.to_string()));
                // Только выданные инструменты: wait_job модели не положен — ожидание делает код.
                let rec = if !tools.iter().any(|t| t.name == name) {
                    record(&server, &tool, args, &format!("инструмента «{name}» нет — используй инструменты из списка"), true)
                } else {
                    let args = with_dir(&tool, args, &self.dir);
                    progress(Step::Status(format!("MCP {server} · {tool}({})…", short(&args, 80))));
                    let (text, is_error) = match s.mcp.call(name, &args).await {
                        Some(r) if !r.is_error && started_job(&r.text).is_some() => follow_job(s, &r.text, progress).await,
                        Some(r) => (r.text, r.is_error),
                        None => (format!("инструмента «{name}» нет — используй инструменты из списка"), true),
                    };
                    record(&server, &tool, &args, &text, is_error)
                };
                progress(Step::Tool(rec.clone()));
                self.wire.push(RequestMessage::tool_result(call.id.clone(), rec.result.clone()));
                self.tool_calls.push(rec);
            }
        };
        let reply = if reply.is_empty() { "_Модель ничего не ответила — смотрите вызовы инструментов выше._".to_string() } else { reply };
        self.wire.push(RequestMessage::from(&ChatMessage::assistant(reply.clone())));
        self.usage = sum_usage(self.usage, usage);
        let turn = Turn { user: input.to_string(), reply, rounds, usage, notes };
        self.turns.push(turn.clone());
        Ok(turn)
    }
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

/// Доводы с каталогом записи: если модель не указала `dir`, подставляется текущий.
pub fn with_dir(tool: &str, args: &str, dir: &str) -> String {
    if !DIR_TOOLS.contains(&tool) {
        return args.to_string();
    }
    let mut v: Value = match args.trim() {
        "" => json!({}),
        a => match serde_json::from_str(a) {
            Ok(v) => v,
            Err(_) => return args.to_string(),
        },
    };
    let Some(obj) = v.as_object_mut() else { return args.to_string() };
    let missing = obj.get("dir").and_then(Value::as_str).is_none_or(|d| d.trim().is_empty());
    if missing {
        obj.insert("dir".into(), json!(dir));
    }
    v.to_string()
}

/// Номер задания, если инструмент запустил задание и оно ещё идёт.
pub fn started_job(text: &str) -> Option<String> {
    let v: Value = serde_json::from_str(text).ok()?;
    (v["job"]["state"] == "running").then(|| v["job"]["id"].as_str().map(str::to_string)).flatten()
}

/// Дождаться задания: `wait_job` короткими отрезками, ход — в интерфейс.
/// Возвращает итог задания (как увидит его модель) и признак ошибки.
async fn follow_job(s: &Session, started: &str, progress: Progress<'_>) -> (String, bool) {
    let Some(id) = started_job(started) else { return (started.to_string(), false) };
    let mut failures = 0;
    loop {
        match s.call("wait_job", json!({ "id": id, "seconds": WAIT_SLICE_SEC })).await {
            Ok(text) => {
                failures = 0;
                let job: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                if job["state"] == "running" {
                    progress(Step::Job(job_line(&job)));
                    continue;
                }
                let failed = job["state"] != "succeeded";
                return (json!({ "job": job }).to_string(), failed);
            }
            Err(err) => {
                failures += 1;
                if failures >= WAIT_FAILURES {
                    return (
                        format!(
                            "Не удалось дождаться задания {id}: {}. Задание, возможно, ещё идёт на сервере — не запускай \
                             этот шаг заново; скажи пользователю и предложи проверить позже через inspect.",
                            short(&err, 200)
                        ),
                        true,
                    );
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }
}

/// `convert_to_mp4 j1 · 12:01 из 58:10 (20%) · 3:05`.
pub fn job_line(job: &Value) -> String {
    let secs = job["elapsed_sec"].as_u64().unwrap_or(0);
    format!(
        "{} {} · {} · идёт {}:{:02}",
        job["kind"].as_str().unwrap_or("задание"),
        job["id"].as_str().unwrap_or("?"),
        job["progress"].as_str().unwrap_or(""),
        secs / 60,
        secs % 60
    )
}

fn user_message(input: &str, dir: &str, first: bool) -> String {
    if first {
        format!("Каталог записи: `{dir}` (от корня записей сервера).\n\n{input}")
    } else {
        input.to_string()
    }
}

fn system_prompt() -> String {
    "Ты — ассистент, который обрабатывает записи встреч книжного клуба: видео (WebM или MP4) → output.mp4 → \
     output.mp3 → MP3-чанки → транскрипт output-transcript.md → саммари. Работу делают инструменты MCP-сервера \
     media; ты ведёшь процесс и разговариваешь с пользователем.\n\n\
     Как работать:\n\
     1. Начинай с inspect для каталога записи. Он сам находит последний годный чекпоинт (готовый транскрипт, чанки, \
     MP3, output.mp4 со сверкой длительности по WebM, исходники) и в поле next называет следующий шаг. Не выбирай \
     шаг сам и не перескакивай: вызывай next.tool с next.args.\n\
     2. Шаги convert_to_mp4, extract_mp3, split_audio, transcribe, summarize — долгие (до часа и больше). Вызов \
     вернётся, только когда работа закончится: результат — job с state succeeded или failed. Долгое выполнение — не \
     ошибка. После каждого шага снова inspect и следующий шаг — пока не дойдёшь до ask_summary, done или ошибки.\n\
     3. next.step = choose_source — спроси пользователя, какой файл взять (варианты — next.choices), и остановись. \
     Ответ передай как source в inspect.\n\
     4. next.step = ask_summary — спроси дословно «Сделать саммари? (да/нет)» и остановись. «да», «yes», «y» — \
     list_llm_models: если obvious не null — summarize без model (сервер возьмёт её сам); если null — \
     спроси, какую из chat_models взять (загруженные — loaded_chat_models), и остановись. «нет», «no», «n» — \
     write_summary_prompt; LM Studio не вызывай.\n\
     5. next.step = done — саммари уже есть и переиспользуется. summarize с force=true — только если пользователь \
     прямо попросил сделать саммари заново.\n\
     6. Ошибка инструмента или задание failed — не обходи её: не меняй порядок шагов, не запускай шаг повторно сам, \
     не выдумывай файлы. Скажи пользователю, что случилось (с цифрами из ответа: длительности, какие чанки) и что \
     сохранено; повторить шаг — только если он попросит.\n\
     7. next.step = nothing — обрабатывать нечего: скажи это; list_recordings покажет, где есть записи.\n\
     Опирайся только на ответы инструментов.\n\n\
     Итоговый отчёт (когда обработка закончена или остановилась) — Markdown на русском:\n\
     - Переиспользовано — какие чекпоинты уже были (output.mp3, transcript-chunks/chunk-*.mp3, …).\n\
     - Сделано — что выполнено в этот раз (например: «Транскрибированы все MP3-чанки, TXT перезаписаны», «Создан \
     output-transcript.md»).\n\
     - Если конвертировали WebM или сверяли output.mp4 с WebM — строки «WebM duration: N seconds», «MP4 duration: N \
     seconds», «Duration check: OK» (или несовпадение).\n\
     - Итоговые файлы — только те, что действительно есть по последнему inspect: output.mp4, output.mp3, \
     transcript-chunks/, output-transcript.md, output-summary.md, output-summary-prompt.md.\n\
     - Модель LM Studio, если саммари делалось; если создан только промпт — «LM Studio не вызывался».\n\
     Вопрос пользователю — коротко, без отчёта."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_is_filled_only_where_missing() {
        assert_eq!(with_dir("inspect", "{}", "2026-10-15"), r#"{"dir":"2026-10-15"}"#);
        assert_eq!(with_dir("transcribe", "", "rec"), r#"{"dir":"rec"}"#);
        assert_eq!(with_dir("extract_mp3", r#"{"dir":"","source":"output.mp4"}"#, "rec"), r#"{"dir":"rec","source":"output.mp4"}"#);
        assert_eq!(with_dir("inspect", r#"{"dir":"other"}"#, "rec"), r#"{"dir":"other"}"#);
        assert_eq!(with_dir("list_llm_models", "{}", "rec"), "{}");
        assert_eq!(with_dir("list_recordings", "{}", "rec"), "{}");
        assert_eq!(with_dir("inspect", "не json", "rec"), "не json");
    }

    #[test]
    fn running_jobs_are_recognised() {
        let started = r#"{"job":{"id":"j3","kind":"transcribe","state":"running","progress":"чанки: 2 из 14 готово","elapsed_sec":125},"note":"…"}"#;
        assert_eq!(started_job(started).as_deref(), Some("j3"));
        assert_eq!(started_job(r#"{"job":{"id":"j3","state":"succeeded"}}"#), None);
        assert_eq!(started_job(r#"{"next":{"step":"transcribe"}}"#), None);
        assert_eq!(started_job("ошибка"), None);
        let job: Value = serde_json::from_str(started).unwrap();
        assert_eq!(job_line(&job["job"]), "transcribe j3 · чанки: 2 из 14 готово · идёт 2:05");
    }

    #[test]
    fn prompt_keeps_skill_rules() {
        let p = system_prompt();
        assert!(p.contains("Начинай с inspect"));
        assert!(p.contains("«Сделать саммари? (да/нет)»"));
        assert!(p.contains("LM Studio не вызывай"));
        assert!(p.contains("Duration check: OK"));
        assert!(user_message("обработай", "rec", true).starts_with("Каталог записи: `rec`"));
        assert_eq!(user_message("да", "rec", false), "да");
    }
}
