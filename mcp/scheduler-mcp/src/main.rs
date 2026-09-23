//! MCP-сервер планировщика: отложенные и периодические задания, доступный по
//! Streamable HTTP.
//!
//! Запуск: `cargo run -p scheduler-mcp` — сервер слушает
//! `http://127.0.0.1:8092/mcp` (адрес — `SCHEDULER_MCP_ADDR`). Задания и их
//! запуски хранятся в SQLite (`scheduler.db`, путь — `SCHEDULER_DB`), поэтому
//! переживают перезапуск. Планировщик универсален: задание — это действие
//! (напоминание, HTTP-запрос, вызов инструмента другого MCP-сервера) и
//! расписание, а сводка по накопленным запускам считается одинаково для
//! любого результата (см. [`metrics`]).

mod actions;
mod metrics;
mod scheduler;
mod store;
mod time;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use rmcp::{
    handler::server::wrapper::{Json, Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ServerHandler,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use actions::Executor;
use metrics::{Aggregate, Extract};
use scheduler::Scheduler;
use store::{Event, Job, NewJob, Outcome, Run, Schedule, Store};
use time::now_ms;

const DEFAULT_ADDR: &str = "127.0.0.1:8092";
const DEFAULT_DB: &str = "scheduler.db";
/// Чаще нельзя: задания создаёт модель, и «каждую секунду» быстро забило бы БД.
const MIN_INTERVAL_MS: i64 = 5_000;
const DEFAULT_MAX_FAILURES: i64 = 5;
const DEFAULT_RETENTION_DAYS: u64 = 7;
const DEFAULT_RUNS_LIMIT: usize = 20;
const MAX_RUNS_LIMIT: usize = 200;
const DEFAULT_EVENTS_LIMIT: usize = 50;
const MAX_NAME_CHARS: usize = 60;

// ---------- параметры инструментов ----------

#[derive(Debug, Deserialize, JsonSchema)]
struct ScheduleJobParams {
    /// Что делать: `http` — HTTP-запрос (params: url, method GET|HEAD|POST, headers,
    /// body, timeout_sec); `mcp_tool` — вызвать инструмент другого MCP-сервера
    /// (params: server — имя из конфигурации MCP, или url; tool; arguments);
    /// `reminder` — напоминание (params: text).
    action: String,
    /// Параметры действия (см. action).
    params: Map<String, Value>,
    /// Интервал периодического запуска: `30s`, `5m`, `1h`, `1d`, не чаще 5s.
    /// Без него задание однократное.
    #[serde(default)]
    every: Option<String>,
    /// Через сколько выполнить (однократное) или начать (периодическое): `10m`, `2h`.
    #[serde(default, rename = "in")]
    delay: Option<String>,
    /// Когда выполнить или начать: RFC 3339, «2026-09-23 18:00» (местное время)
    /// или «18:00» — ближайшее такое время. Текущее время сервера — поле `now` в ответах.
    #[serde(default)]
    at: Option<String>,
    /// Короткое имя задания для списков и сводок.
    #[serde(default)]
    name: Option<String>,
    /// Какие метрики брать из результата для сводки: имя → {path, regex}. path — путь
    /// к полю через точку (`body.status`, `heap`), regex — для текстового поля, значение —
    /// первая группа (`used (\d+)K`). Без extract в метрики идут все скалярные поля результата.
    #[serde(default)]
    extract: Option<Extract>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ScheduleReminderParams {
    /// Текст напоминания.
    text: String,
    /// Через сколько напомнить: `20m`, `1h30m`.
    #[serde(default, rename = "in")]
    delay: Option<String>,
    /// Когда напомнить: RFC 3339, «2026-09-23 18:00» или «18:00».
    #[serde(default)]
    at: Option<String>,
    /// Повторять с этим интервалом (`1d`, `2h`); первый раз — по in/at или сразу.
    #[serde(default)]
    every: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListJobsParams {
    /// Показать и выполненные/отменённые задания. По умолчанию нет.
    #[serde(default)]
    include_finished: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct JobIdParams {
    /// Номер задания (из list_jobs).
    job_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetRunsParams {
    /// Номер задания (из list_jobs).
    job_id: i64,
    /// Начало окна: длительность назад (`30m`, `24h`) или момент времени.
    #[serde(default)]
    since: Option<String>,
    /// Сколько последних запусков вернуть, по умолчанию 20, максимум 200.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetSummaryParams {
    /// Номер задания; без него — сводка по всем заданиям.
    #[serde(default)]
    job_id: Option<i64>,
    /// Начало окна: длительность назад (`30m`, `24h`) или момент времени. Без него — все
    /// хранящиеся запуски (старые удаляются через несколько дней).
    #[serde(default)]
    since: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetEventsParams {
    /// Только непрочитанные (не подтверждённые ack_events). По умолчанию да.
    #[serde(default)]
    unread_only: Option<bool>,
    /// Начало окна: длительность назад (`1h`) или момент времени.
    #[serde(default)]
    since: Option<String>,
    /// Сколько событий вернуть, по умолчанию 50.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AckEventsParams {
    /// Номера обработанных событий (из get_events).
    ids: Vec<i64>,
}

// ---------- ответы ----------

#[derive(Debug, Serialize, JsonSchema)]
struct JobView {
    id: i64,
    name: String,
    action: String,
    params: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    extract: Option<Extract>,
    /// «once 2026-09-23T18:00:00+03:00» или «every 30s».
    schedule: String,
    /// active, paused, completed, cancelled.
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_run_at: Option<String>,
    consecutive_failures: i64,
    /// Почему планировщик сам поставил задание на паузу.
    #[serde(skip_serializing_if = "Option::is_none")]
    pause_reason: Option<String>,
    created_at: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct JobReply {
    /// Текущее время сервера.
    now: String,
    job: JobView,
}

#[derive(Debug, Serialize, JsonSchema)]
struct JobList {
    now: String,
    jobs: Vec<JobView>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RunView {
    id: i64,
    scheduled_at: String,
    started_at: String,
    duration_ms: i64,
    /// ok или error.
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RunList {
    now: String,
    job: JobView,
    runs: Vec<RunView>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct JobSummary {
    id: i64,
    name: String,
    action: String,
    schedule: String,
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pause_reason: Option<String>,
    /// Запусков в окне: всего, успешных, с ошибкой.
    runs: usize,
    ok: usize,
    errors: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_duration_ms: Option<f64>,
    /// Агрегаты метрик успешных запусков: числа — count/min/max/avg/first/last/
    /// change_percent, строки — last/changes/частые значения.
    metrics: Aggregate,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SummaryReply {
    now: String,
    /// Начало окна, если задано.
    #[serde(skip_serializing_if = "Option::is_none")]
    since: Option<String>,
    jobs: Vec<JobSummary>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct EventView {
    id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_id: Option<i64>,
    created_at: String,
    /// reminder — напоминание; job_paused — задание остановлено после ошибок подряд;
    /// job_failed — однократное задание не выполнилось.
    kind: String,
    text: String,
    acked: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
struct EventList {
    now: String,
    events: Vec<EventView>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct AckReply {
    now: String,
    /// Сколько событий отмечено прочитанными.
    acked: usize,
}

// ---------- сервер ----------

#[derive(Clone)]
struct SchedulerServer {
    store: Arc<Store>,
    executor: Executor,
    scheduler: Scheduler,
}

#[tool_router]
impl SchedulerServer {
    #[tool(
        name = "schedule_job",
        description = "Создаёт отложенное или периодическое задание. Сервер выполняет его сам по \
            расписанию, круглосуточно, и сохраняет каждый запуск; накопленное смотрят get_summary и \
            get_runs. Действия: http — HTTP-запрос (health-check, API, метрики); mcp_tool — вызов \
            инструмента другого MCP-сервера по имени из конфигурации (например, периодически \
            снимать данные чужим инструментом); reminder — напоминание (удобнее schedule_reminder). \
            Расписание: every — периодически, in/at — однократно (или начало периодического)."
    )]
    async fn schedule_job(&self, Parameters(p): Parameters<ScheduleJobParams>) -> Result<Json<JobReply>, String> {
        let params = Value::Object(p.params);
        self.executor.validate(&p.action, &params).map_err(error_text)?;
        if let Some(extract) = &p.extract {
            metrics::validate_extract(extract).map_err(error_text)?;
        }
        let name = p.name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| default_name(&p.action, &params));
        self.create(name, p.action, params, p.extract, p.every, p.delay, p.at)
    }

    #[tool(
        name = "schedule_reminder",
        description = "Напоминание ЧЕЛОВЕКУ: в срок (in или at) сервер создаёт событие с коротким текстом, \
            и оно показывается человеку как есть. Напоминание ничего не выполняет — не клади в него \
            инструкции для агента (что проверить, какие задания создать): регулярную работу агента задаёт \
            его плановый запуск с промптом. С every напоминание повторяется — только если человек сам \
            просил повторять."
    )]
    async fn schedule_reminder(
        &self,
        Parameters(p): Parameters<ScheduleReminderParams>,
    ) -> Result<Json<JobReply>, String> {
        if p.delay.is_none() && p.at.is_none() && p.every.is_none() {
            return Err("задайте, когда напомнить: in, at или every".into());
        }
        let params = serde_json::json!({ "text": p.text });
        self.executor.validate("reminder", &params).map_err(error_text)?;
        let name = default_name("reminder", &params);
        self.create(name, "reminder".into(), params, None, p.every, p.delay, p.at)
    }

    #[tool(
        name = "list_jobs",
        description = "Список заданий планировщика: расписание, состояние, следующий и последний \
            запуск, ошибки подряд.",
        annotations(read_only_hint = true)
    )]
    async fn list_jobs(&self, Parameters(p): Parameters<ListJobsParams>) -> Result<Json<JobList>, String> {
        let jobs = self.store.jobs(p.include_finished).map_err(error_text)?;
        Ok(Json(JobList { now: time::format(now_ms()), jobs: jobs.iter().map(job_view).collect() }))
    }

    #[tool(name = "cancel_job", description = "Отменяет задание: больше оно не запускается, запуски остаются.")]
    async fn cancel_job(&self, Parameters(p): Parameters<JobIdParams>) -> Result<Json<JobReply>, String> {
        let job = self.store.cancel(p.job_id).map_err(error_text)?;
        Ok(self.reply(&job))
    }

    #[tool(name = "pause_job", description = "Ставит задание на паузу (resume_job — продолжить).")]
    async fn pause_job(&self, Parameters(p): Parameters<JobIdParams>) -> Result<Json<JobReply>, String> {
        let job = self.store.pause(p.job_id, None).map_err(error_text)?;
        Ok(self.reply(&job))
    }

    #[tool(
        name = "resume_job",
        description = "Снимает задание с паузы, в том числе поставленное планировщиком после ошибок \
            подряд; периодическое запускается сразу."
    )]
    async fn resume_job(&self, Parameters(p): Parameters<JobIdParams>) -> Result<Json<JobReply>, String> {
        let job = self.store.resume(p.job_id, now_ms()).map_err(error_text)?;
        self.scheduler.wake();
        Ok(self.reply(&job))
    }

    #[tool(
        name = "get_runs",
        description = "Сырые запуски задания: время, длительность, статус, результат, извлечённые \
            метрики или ошибка. Для разбора отдельных запусков; обзор — get_summary.",
        annotations(read_only_hint = true)
    )]
    async fn get_runs(&self, Parameters(p): Parameters<GetRunsParams>) -> Result<Json<RunList>, String> {
        let now = now_ms();
        let job = self.job(p.job_id)?;
        let since = p.since.as_deref().map(|s| time::parse_since(s, now)).transpose().map_err(error_text)?;
        let limit = p.limit.unwrap_or(DEFAULT_RUNS_LIMIT).clamp(1, MAX_RUNS_LIMIT);
        let runs = self.store.runs(job.id, since, Some(limit)).map_err(error_text)?;
        Ok(Json(RunList { now: time::format(now), job: job_view(&job), runs: runs.into_iter().map(run_view).collect() }))
    }

    #[tool(
        name = "get_summary",
        description = "Агрегированная сводка по накопленным запускам — по одному заданию или по всем: \
            сколько запусков, успешных и с ошибкой, последняя ошибка, среднее время, а по метрикам \
            результата — min/max/avg/первое/последнее значение и изменение в % для чисел, последнее \
            значение и число смен для строк (например, UP → DOWN). Считается детерминированно, без LLM.",
        annotations(read_only_hint = true)
    )]
    async fn get_summary(&self, Parameters(p): Parameters<GetSummaryParams>) -> Result<Json<SummaryReply>, String> {
        let now = now_ms();
        let since = p.since.as_deref().map(|s| time::parse_since(s, now)).transpose().map_err(error_text)?;
        let jobs = match p.job_id {
            Some(id) => vec![self.job(id)?],
            None => self.store.jobs(true).map_err(error_text)?,
        };
        let mut summaries = Vec::with_capacity(jobs.len());
        for job in jobs {
            let runs = self.store.runs(job.id, since, None).map_err(error_text)?;
            // Завершённые задания без запусков в окне — шум в общей сводке.
            let finished = !matches!(job.state, store::JobState::Active | store::JobState::Paused);
            if p.job_id.is_none() && finished && runs.is_empty() {
                continue;
            }
            summaries.push(summarize(&job, &runs));
        }
        Ok(Json(SummaryReply { now: time::format(now), since: since.map(time::format), jobs: summaries }))
    }

    #[tool(
        name = "get_events",
        description = "События для агента: сработавшие напоминания, задания, остановленные после ошибок \
            подряд, и несработавшие однократные задания. По умолчанию — непрочитанные; после \
            обработки подтвердите их ack_events, иначе они вернутся снова.",
        annotations(read_only_hint = true)
    )]
    async fn get_events(&self, Parameters(p): Parameters<GetEventsParams>) -> Result<Json<EventList>, String> {
        let now = now_ms();
        let since = p.since.as_deref().map(|s| time::parse_since(s, now)).transpose().map_err(error_text)?;
        let limit = p.limit.unwrap_or(DEFAULT_EVENTS_LIMIT).clamp(1, MAX_RUNS_LIMIT);
        let events = self.store.events(since, p.unread_only.unwrap_or(true), limit).map_err(error_text)?;
        Ok(Json(EventList { now: time::format(now), events: events.into_iter().map(event_view).collect() }))
    }

    #[tool(name = "ack_events", description = "Отмечает события прочитанными — get_events их больше не вернёт.")]
    async fn ack_events(&self, Parameters(p): Parameters<AckEventsParams>) -> Result<Json<AckReply>, String> {
        let now = now_ms();
        let acked = self.store.ack_events(&p.ids, now).map_err(error_text)?;
        Ok(Json(AckReply { now: time::format(now), acked }))
    }
}

impl SchedulerServer {
    #[allow(clippy::too_many_arguments)]
    fn create(
        &self,
        name: String,
        action: String,
        params: Value,
        extract: Option<Extract>,
        every: Option<String>,
        delay: Option<String>,
        at: Option<String>,
    ) -> Result<Json<JobReply>, String> {
        let now = now_ms();
        let start = match (delay, at) {
            (Some(_), Some(_)) => return Err("задайте либо in, либо at".into()),
            (Some(delay), None) => Some(now + time::parse_duration(&delay).map_err(error_text)?),
            (None, Some(at)) => Some(time::parse_at(&at, now).map_err(error_text)?),
            (None, None) => None,
        };
        let (schedule, first_run_at) = match every {
            Some(every) => {
                let interval_ms = time::parse_duration(&every).map_err(error_text)?;
                if interval_ms < MIN_INTERVAL_MS {
                    return Err(format!("слишком часто: интервал не меньше {} с", MIN_INTERVAL_MS / 1000));
                }
                (Schedule::Every { interval_ms }, start.unwrap_or(now))
            }
            None => {
                let at = start.ok_or("задайте расписание: every — периодически, in или at — однократно")?;
                (Schedule::Once { at }, at)
            }
        };
        let job = NewJob { name, action, params, extract, schedule, first_run_at };
        let job = self.store.insert_job(job, now).map_err(error_text)?;
        self.scheduler.wake();
        Ok(self.reply(&job))
    }

    fn job(&self, id: i64) -> Result<Job, String> {
        self.store.job(id).map_err(error_text)?.ok_or_else(|| format!("задания #{id} нет"))
    }

    fn reply(&self, job: &Job) -> Json<JobReply> {
        Json(JobReply { now: time::format(now_ms()), job: job_view(job) })
    }
}

#[tool_handler]
impl ServerHandler for SchedulerServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("scheduler-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Планировщик заданий, работающий круглосуточно. schedule_job ставит периодический сбор \
                 данных (http-запрос или инструмент другого MCP-сервера), schedule_reminder — напоминание. \
                 Сервер сам выполняет задания и хранит результаты; get_summary даёт агрегированную \
                 сводку, get_events — напоминания и сбои (подтверждайте их ack_events). Время в ответах \
                 — время сервера (поле now); относительные сроки задавайте как 30s, 10m, 2h, 1d.",
            )
    }
}

// ---------- представление ----------

fn default_name(action: &str, params: &Value) -> String {
    let s = |key: &str| params.get(key).and_then(Value::as_str).unwrap_or_default().to_string();
    let name = match action {
        "reminder" => s("text"),
        "http" => s("url"),
        "mcp_tool" => {
            let server = params.get("server").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| s("url"));
            format!("{server} · {}", s("tool"))
        }
        other => other.to_string(),
    };
    if name.chars().count() > MAX_NAME_CHARS {
        name.chars().take(MAX_NAME_CHARS).collect::<String>() + "…"
    } else {
        name
    }
}

fn schedule_text(schedule: Schedule) -> String {
    match schedule {
        Schedule::Once { at } => format!("once {}", time::format(at)),
        Schedule::Every { interval_ms } => format!("every {}", format_interval(interval_ms)),
    }
}

/// `90000` → `1m30s`.
fn format_interval(ms: i64) -> String {
    let mut sec = ms / 1000;
    let mut out = String::new();
    for (unit, size) in [("d", 86_400), ("h", 3_600), ("m", 60), ("s", 1)] {
        if sec >= size {
            out.push_str(&format!("{}{unit}", sec / size));
            sec %= size;
        }
    }
    if out.is_empty() {
        format!("{ms}ms")
    } else {
        out
    }
}

fn job_view(job: &Job) -> JobView {
    JobView {
        id: job.id,
        name: job.name.clone(),
        action: job.action.clone(),
        params: job.params.clone(),
        extract: job.extract.clone(),
        schedule: schedule_text(job.schedule),
        state: job.state.as_str().to_string(),
        next_run_at: job.next_run_at.map(time::format),
        last_run_at: job.last_run_at.map(time::format),
        consecutive_failures: job.consecutive_failures,
        pause_reason: job.pause_reason.clone(),
        created_at: time::format(job.created_at),
    }
}

fn run_view(run: Run) -> RunView {
    let (status, result, metrics, error) = match run.outcome {
        Outcome::Ok { result, metrics } => ("ok", Some(result), Some(metrics.into_iter().collect()), None),
        Outcome::Error(e) => ("error", None, None, Some(e)),
    };
    RunView {
        id: run.id,
        scheduled_at: time::format(run.scheduled_at),
        started_at: time::format(run.started_at),
        duration_ms: run.finished_at - run.started_at,
        status: status.to_string(),
        result,
        metrics,
        error,
    }
}

fn summarize(job: &Job, runs: &[Run]) -> JobSummary {
    let ok: Vec<_> = runs
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Ok { metrics, .. } => Some(metrics),
            Outcome::Error(_) => None,
        })
        .collect();
    let last_error = runs.iter().rev().find_map(|r| match &r.outcome {
        Outcome::Error(e) => Some(e.clone()),
        Outcome::Ok { .. } => None,
    });
    let avg_duration_ms = (!runs.is_empty()).then(|| {
        let total: i64 = runs.iter().map(|r| r.finished_at - r.started_at).sum();
        (total as f64 / runs.len() as f64 * 10.0).round() / 10.0
    });
    JobSummary {
        id: job.id,
        name: job.name.clone(),
        action: job.action.clone(),
        schedule: schedule_text(job.schedule),
        state: job.state.as_str().to_string(),
        pause_reason: job.pause_reason.clone(),
        runs: runs.len(),
        ok: ok.len(),
        errors: runs.len() - ok.len(),
        last_error,
        first_run_at: runs.first().map(|r| time::format(r.started_at)),
        last_run_at: runs.last().map(|r| time::format(r.started_at)),
        avg_duration_ms,
        metrics: metrics::aggregate(ok),
    }
}

fn event_view(event: Event) -> EventView {
    EventView {
        id: event.id,
        job_id: event.job_id,
        created_at: time::format(event.created_at),
        kind: event.kind,
        text: event.text,
        acked: event.acked,
    }
}

fn error_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

fn env_number<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("SCHEDULER_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let db = PathBuf::from(std::env::var("SCHEDULER_DB").unwrap_or_else(|_| DEFAULT_DB.into()));
    // Тот же файл конфигурации, что у MCP-клиента: задания mcp_tool находят
    // серверы по имени.
    let mcp_config = PathBuf::from(std::env::var("LLM_MCP_CONFIG").unwrap_or_else(|_| "mcp.json".into()));
    let max_failures = env_number("SCHEDULER_MAX_FAILURES", DEFAULT_MAX_FAILURES);
    let retention_days = env_number("SCHEDULER_RETENTION_DAYS", DEFAULT_RETENTION_DAYS);

    let store = Arc::new(Store::open(&db)?);
    let active = store.jobs(false)?.len();
    eprintln!("scheduler-mcp: БД {} — активных и приостановленных заданий: {active}", db.display());

    let executor = Executor::new(mcp_config, addr.clone());
    let scheduler = Scheduler::new(store.clone(), executor.clone(), max_failures);
    tokio::spawn(scheduler.clone().run());
    tokio::spawn(scheduler::prune_loop(
        store.clone(),
        std::time::Duration::from_secs(retention_days.max(1) * 86_400),
    ));

    let service = StreamableHttpService::new(
        move || Ok(SchedulerServer { store: store.clone(), executor: executor.clone(), scheduler: scheduler.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("не удалось занять адрес {addr}"))?;

    eprintln!("scheduler-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_intervals() {
        assert_eq!(format_interval(30_000), "30s");
        assert_eq!(format_interval(90_000), "1m30s");
        assert_eq!(format_interval(86_400_000 + 3_600_000), "1d1h");
    }

    #[test]
    fn summary_counts_runs_and_aggregates_ok_ones() {
        let store = Store::in_memory();
        let job = store
            .insert_job(
                NewJob {
                    name: "health".into(),
                    action: "http".into(),
                    params: json!({"url": "http://x"}),
                    extract: None,
                    schedule: Schedule::Every { interval_ms: 60_000 },
                    first_run_at: 0,
                },
                0,
            )
            .unwrap();
        let ok = |status: &str, latency: f64| Outcome::Ok {
            result: json!({}),
            metrics: serde_json::from_value(json!({"body.status": status, "latency_ms": latency})).unwrap(),
        };
        store.record_run(job.id, 0, 0, 10, &ok("UP", 10.0)).unwrap();
        store.record_run(job.id, 1, 1, 21, &Outcome::Error("таймаут".into())).unwrap();
        store.record_run(job.id, 2, 2, 32, &ok("DOWN", 30.0)).unwrap();

        let s = summarize(&job, &store.runs(job.id, None, None).unwrap());
        assert_eq!((s.runs, s.ok, s.errors), (3, 2, 1));
        assert_eq!(s.last_error.as_deref(), Some("таймаут"));
        assert_eq!(s.avg_duration_ms, Some(20.0));
        assert_eq!(s.metrics.numeric["latency_ms"].change_percent, Some(200.0));
        assert_eq!(s.metrics.text["body.status"].last, "DOWN");
        assert_eq!(s.schedule, "every 1m");
    }

    #[test]
    fn default_names_describe_the_action() {
        assert_eq!(default_name("mcp_tool", &json!({"server": "prof", "tool": "info"})), "prof · info");
        assert_eq!(default_name("http", &json!({"url": "http://x"})), "http://x");
        assert!(default_name("reminder", &json!({"text": "я".repeat(100)})).ends_with('…'));
    }
}
