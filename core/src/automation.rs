//! Плановые запуски агентов — агент «на дежурстве», который работает сам,
//! без реплик человека.
//!
//! Плановый запуск ([`ScheduledRun`]) — это агент, промпт и интервал: раз в
//! интервал агент получает промпт (например, «собери сводку планировщика за
//! последние 30 минут») и отвечает в свой чат с пометкой
//! [`crate::agent::SCHEDULED_MARK`]. Запуски хранятся в `agents.db` рядом с
//! агентами и переживают перезапуск; выполняет их [`spawn`] в любом
//! долгоживущем процессе — веб-сервере, TUI или `llm-cli agent daemon`.
//!
//! Там же [`spawn`] доставляет **события** MCP-серверов без обращения к
//! модели: у любого подключённого сервера с инструментами `get_events` и
//! `ack_events` (такой — `scheduler-mcp` этого репозитория) непрочитанные
//! события раз в [`EVENTS_POLL`] переносятся в чат дежурных агентов и
//! подтверждаются. Так напоминание «через 20 минут» приходит вовремя, а не со
//! следующей плановой сводкой, и не стоит вызова модели.
//!
//! Если запущено несколько процессов над одним `agents.db` (например, веб и
//! TUI сразу), каждый запуск и каждый опрос событий забирает ровно один из
//! них: срок в БД переносится условным `UPDATE … WHERE next_run_at = <старый>`,
//! и выполняет тот, чей `UPDATE` прошёл.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::agent::{AgentManager, ScheduledOutcome, SCHEDULED_MARK};

/// Как часто цикл проверяет сроки плановых запусков.
const TICK: Duration = Duration::from_secs(5);
/// Как часто опрашивать события MCP-серверов.
pub const EVENTS_POLL: Duration = Duration::from_secs(15);
/// Чаще нельзя: каждый плановый запуск — это обращение к модели.
pub const MIN_INTERVAL: Duration = Duration::from_secs(30);

/// Промпт планового запуска по умолчанию — сводка `scheduler-mcp`.
/// `{every}` заменяется интервалом запуска.
pub const DEFAULT_PROMPT: &str = "Плановая сводка. Вызови get_summary планировщика с since=\"{every}\" — \
    это данные за время с прошлой сводки — и get_events с unread_only=false и since=\"{every}\". Кратко \
    по-русски: что собиралось, как менялись ключевые метрики (рост, падение, аномалии), были ли ошибки, \
    остановленные задания и сработавшие напоминания. Если данных нет, скажи об этом одной строкой.";

/// Плановый запуск агента.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScheduledRun {
    pub id: i64,
    pub agent: String,
    /// Подпись в чате: `⏰ <name>`.
    pub name: String,
    pub prompt: String,
    pub interval_ms: i64,
    pub enabled: bool,
    /// Миллисекунды Unix.
    pub next_run_at: i64,
    pub last_run_at: Option<i64>,
    /// Итог последнего запуска: `ok`, `пропущен: …` или `ошибка: …`.
    pub last_status: Option<String>,
}

impl ScheduledRun {
    /// Интервал: `30m`.
    pub fn every(&self) -> String {
        format_duration(self.interval_ms)
    }
}

/// Что произошло в фоне — для интерфейсов, которым надо обновить открытый чат.
#[derive(Debug, Clone, Serialize)]
pub struct Activity {
    /// Возрастающий номер: интерфейс помнит последний увиденный.
    pub seq: u64,
    pub agent: String,
    pub text: String,
}

/// Запускает фоновый цикл плановых запусков и доставки событий.
pub fn spawn(manager: Arc<AgentManager>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let now = now_ms();
            match manager.claim_due_scheduled(now) {
                Ok(due) => {
                    for run in due {
                        tokio::spawn(execute(manager.clone(), run));
                    }
                }
                Err(e) => eprintln!("плановые запуски: {e:#}"),
            }
            if manager.claim_lease("mcp_events", EVENTS_POLL, now).unwrap_or(false) {
                deliver_events(&manager).await;
            }
            tokio::time::sleep(TICK).await;
        }
    })
}

async fn execute(manager: Arc<AgentManager>, run: ScheduledRun) {
    let status = match manager.get(&run.agent) {
        None => format!("ошибка: агента «{}» нет", run.agent),
        Some(agent) => match agent.run_scheduled(&run.name, &run.prompt).await {
            Ok(ScheduledOutcome::Replied(reply)) => {
                manager.push_activity(&run.agent, &reply.text);
                "ok".to_string()
            }
            Ok(ScheduledOutcome::Skipped(reason)) => format!("пропущен: {reason}"),
            Err(e) => {
                let status = format!("ошибка: {e:#}");
                manager.push_activity(&run.agent, &status);
                status
            }
        },
    };
    if let Err(e) = manager.record_scheduled_result(run.id, now_ms(), &status) {
        eprintln!("плановые запуски: не удалось сохранить итог: {e:#}");
    }
}

#[derive(Deserialize)]
struct EventList {
    #[serde(default)]
    events: Vec<McpEvent>,
}

#[derive(Deserialize)]
struct McpEvent {
    id: i64,
    #[serde(default)]
    kind: String,
    text: String,
}

/// Переносит непрочитанные события MCP-серверов в чаты дежурных агентов —
/// тех, у кого есть включённый плановый запуск, а если таких нет, всех
/// запущенных. Пока доставить некому, события не подтверждаются и ждут.
async fn deliver_events(manager: &AgentManager) {
    let mcp = manager.mcp();
    let sources: Vec<String> = mcp
        .servers()
        .into_iter()
        .filter(|s| s.status == crate::mcp::McpStatus::Connected)
        .filter(|s| ["get_events", "ack_events"].iter().all(|t| s.tools.iter().any(|tool| tool.name == *t)))
        .map(|s| s.name)
        .collect();
    if sources.is_empty() {
        return;
    }
    let targets = manager.duty_agents();
    if targets.is_empty() {
        return;
    }
    for server in sources {
        let Some(result) = mcp.call_tool(&server, "get_events", r#"{"unread_only":true}"#).await else { continue };
        if result.is_error {
            continue;
        }
        let Ok(list) = serde_json::from_str::<EventList>(&result.text) else { continue };
        if list.events.is_empty() {
            continue;
        }
        for event in &list.events {
            let text = format!("{SCHEDULED_MARK} {}: {}", event_label(&event.kind), event.text);
            for agent in &targets {
                if let Some(a) = manager.get(agent) {
                    a.append_notice(&text);
                    manager.push_activity(agent, &text);
                }
            }
        }
        let ids: Vec<i64> = list.events.iter().map(|e| e.id).collect();
        let args = serde_json::json!({ "ids": ids }).to_string();
        let _ = mcp.call_tool(&server, "ack_events", &args).await;
    }
}

fn event_label(kind: &str) -> &str {
    match kind {
        "reminder" => "Напоминание",
        "job_paused" => "Задание остановлено",
        "job_failed" => "Задание не выполнилось",
        other => other,
    }
}

/// Задание MCP-планировщика (`scheduler-mcp`) — как его отдаёт `list_jobs`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchedulerJob {
    pub id: i64,
    pub name: String,
    pub action: String,
    /// «every 30s» или «once 2026-09-23T18:00:00+03:00».
    pub schedule: String,
    /// active, paused, completed, cancelled.
    pub state: String,
    #[serde(default)]
    pub next_run_at: Option<String>,
    #[serde(default)]
    pub last_run_at: Option<String>,
    #[serde(default)]
    pub consecutive_failures: i64,
    #[serde(default)]
    pub pause_reason: Option<String>,
}

impl SchedulerJob {
    /// `#8 «java-monitor» · every 20s · active · последний 20:00:25`.
    pub fn describe(&self) -> String {
        let mut line = format!("#{} «{}» · {} · {} · {}", self.id, self.name, self.action, self.schedule, self.state);
        if let Some(at) = &self.last_run_at {
            line.push_str(&format!(" · последний {}", clock(at)));
        }
        if self.consecutive_failures > 0 {
            line.push_str(&format!(" · ошибок подряд: {}", self.consecutive_failures));
        }
        if let Some(reason) = &self.pause_reason {
            line.push_str(&format!(" · {reason}"));
        }
        line
    }
}

/// Время суток из RFC 3339: `2026-09-23T20:00:25+03:00` → `20:00:25`.
fn clock(moment: &str) -> &str {
    moment.get(11..19).unwrap_or(moment)
}

/// Что можно сделать с заданием планировщика.
pub const JOB_ACTIONS: &[&str] = &["cancel", "pause", "resume"];

/// Подключённый MCP-сервер-планировщик: у него есть `list_jobs` и `cancel_job`.
pub fn scheduler_server(mcp: &crate::mcp::McpManager) -> Option<String> {
    mcp.servers()
        .into_iter()
        .filter(|s| s.status == crate::mcp::McpStatus::Connected)
        .find(|s| ["list_jobs", "cancel_job"].iter().all(|t| s.tools.iter().any(|tool| tool.name == *t)))
        .map(|s| s.name)
}

/// Задания планировщика; `include_finished` — и выполненные/отменённые.
pub async fn scheduler_jobs(mcp: &crate::mcp::McpManager, include_finished: bool) -> Result<Vec<SchedulerJob>> {
    #[derive(Deserialize)]
    struct JobList {
        jobs: Vec<SchedulerJob>,
    }
    let server = scheduler_server(mcp).ok_or_else(|| anyhow!("MCP-планировщик не подключён"))?;
    let args = serde_json::json!({ "include_finished": include_finished }).to_string();
    let result = mcp.call_tool(&server, "list_jobs", &args).await.ok_or_else(|| anyhow!("планировщик отключился"))?;
    if result.is_error {
        bail!("{}", result.text);
    }
    Ok(serde_json::from_str::<JobList>(&result.text)
        .map_err(|e| anyhow!("не удалось разобрать ответ list_jobs: {e}"))?
        .jobs)
}

/// Отменить, приостановить или возобновить задание (`action` — из [`JOB_ACTIONS`]).
pub async fn scheduler_job_action(mcp: &crate::mcp::McpManager, id: i64, action: &str) -> Result<SchedulerJob> {
    #[derive(Deserialize)]
    struct JobReply {
        job: SchedulerJob,
    }
    let tool = match action {
        "cancel" => "cancel_job",
        "pause" => "pause_job",
        "resume" => "resume_job",
        other => bail!("неизвестное действие «{other}»: {}", JOB_ACTIONS.join(", ")),
    };
    let server = scheduler_server(mcp).ok_or_else(|| anyhow!("MCP-планировщик не подключён"))?;
    let args = serde_json::json!({ "job_id": id }).to_string();
    let result = mcp.call_tool(&server, tool, &args).await.ok_or_else(|| anyhow!("планировщик отключился"))?;
    if result.is_error {
        bail!("{}", result.text);
    }
    Ok(serde_json::from_str::<JobReply>(&result.text)
        .map_err(|e| anyhow!("не удалось разобрать ответ {tool}: {e}"))?
        .job)
}

/// Команды заданий планировщика для интерфейсов с командной строкой; `args` —
/// слова после `jobs`: пусто или `list [all]` — список, `cancel|pause|resume <id>`.
pub async fn run_jobs_command(mcp: &crate::mcp::McpManager, args: &[&str]) -> Result<String> {
    match args.first().copied() {
        None | Some("list") => {
            let all = args.get(1) == Some(&"all");
            let jobs = scheduler_jobs(mcp, all).await?;
            if jobs.is_empty() {
                return Ok("Заданий планировщика нет.".into());
            }
            Ok(jobs.iter().map(SchedulerJob::describe).collect::<Vec<_>>().join("\n"))
        }
        Some(action) if JOB_ACTIONS.contains(&action) => {
            let id = args
                .get(1)
                .and_then(|s| s.trim_start_matches('#').parse().ok())
                .ok_or_else(|| anyhow!("укажите номер задания: jobs {action} <id>"))?;
            Ok(scheduler_job_action(mcp, id, action).await?.describe())
        }
        Some(other) => bail!("неизвестное действие «{other}»: jobs [list [all]|cancel|pause|resume <id>]"),
    }
}

/// Команды плановых запусков одного агента — общие для интерфейсов с
/// командной строкой (CLI, экран памяти TUI). `args` — слова после `schedule`:
///
/// - пусто или `list` — список;
/// - `add <интервал> [промпт…]` — добавить (без промпта — сводка планировщика);
/// - `remove|on|off|run <id>` — удалить, включить, выключить, выполнить сейчас.
pub fn run_command(manager: &AgentManager, agent: &str, args: &[&str]) -> Result<String> {
    let id = |args: &[&str]| -> Result<i64> {
        args.get(1)
            .and_then(|s| s.trim_start_matches('#').parse().ok())
            .ok_or_else(|| anyhow!("укажите номер планового запуска (из schedule list)"))
    };
    match args.first().copied() {
        None | Some("list") => {
            let runs = manager.scheduled_runs(Some(agent))?;
            if runs.is_empty() {
                return Ok(format!(
                    "У агента «{agent}» нет плановых запусков. Добавить: schedule add <интервал> [промпт] \
                     (например, schedule add 30m — сводка планировщика раз в 30 минут)."
                ));
            }
            let now = now_ms();
            let lines: Vec<String> = runs.iter().map(|r| r.describe(now)).collect();
            Ok(lines.join("\n"))
        }
        Some("add") => {
            let every = args.get(1).ok_or_else(|| anyhow!("укажите интервал: schedule add <интервал> [промпт]"))?;
            let prompt = args[2..].join(" ");
            let run = manager.add_scheduled_run(agent, every, Some(prompt.as_str()), None)?;
            Ok(format!("Добавлен плановый запуск:\n{}", run.describe(now_ms())))
        }
        Some("remove") => {
            let id = id(args)?;
            manager.remove_scheduled_run(id)?;
            Ok(format!("Плановый запуск #{id} удалён."))
        }
        Some("on") | Some("off") => {
            let run = manager.set_scheduled_run_enabled(id(args)?, args[0] == "on")?;
            Ok(run.describe(now_ms()))
        }
        Some("run") => {
            let run = manager.trigger_scheduled_run(id(args)?)?;
            Ok(format!("Плановый запуск #{} выполнится при ближайшей проверке (до {} с).", run.id, TICK.as_secs()))
        }
        Some(other) => bail!("неизвестное действие «{other}»: schedule [list|add|remove|on|off|run]"),
    }
}

impl ScheduledRun {
    /// Строка для списков: `#1 «Сводка» · каждые 30m · вкл · следующий через 12 мин · последний 3 мин назад: ok`.
    pub fn describe(&self, now: i64) -> String {
        let state = if self.enabled {
            format!("вкл · следующий {}", relative(self.next_run_at, now))
        } else {
            "выкл".to_string()
        };
        let last = match (&self.last_run_at, &self.last_status) {
            (Some(at), Some(status)) => format!(" · последний {}: {status}", relative(*at, now)),
            _ => String::new(),
        };
        format!("#{} «{}» · каждые {} · {state}{last}", self.id, self.name, self.every())
    }
}

/// «через 12 мин», «3 ч назад», «сейчас».
pub fn relative(at: i64, now: i64) -> String {
    let diff = (at - now) / 1000;
    let abs = diff.unsigned_abs();
    let text = match abs {
        0..=4 => return "сейчас".into(),
        5..=59 => format!("{abs} с"),
        60..=3599 => format!("{} мин", abs / 60),
        3600..=86_399 => format!("{} ч {} мин", abs / 3600, abs % 3600 / 60),
        _ => format!("{} д", abs / 86_400),
    };
    if diff > 0 {
        format!("через {text}")
    } else {
        format!("{text} назад")
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Следующий срок по сетке `scheduled + k·interval` после `now` — после
/// простоя запуск выполняется один раз, а не за каждый пропущенный интервал.
pub(crate) fn next_on_grid(scheduled: i64, interval_ms: i64, now: i64) -> i64 {
    let interval = interval_ms.max(1);
    if now < scheduled {
        return scheduled + interval;
    }
    scheduled + interval * ((now - scheduled) / interval + 1)
}

/// Длительность вида `30s`, `5m`, `1h30m`, `1d`; без единицы — секунды.
pub fn parse_duration(text: &str) -> Result<Duration> {
    let text = text.trim();
    if let Ok(sec) = text.parse::<u64>() {
        return Ok(Duration::from_secs(sec));
    }
    let mut total = 0u64;
    let mut number = String::new();
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let unit = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            'd' => 86_400,
            _ => bail!("некорректный интервал «{text}»: ожидается вроде 30s, 5m, 1h30m, 1d"),
        };
        let value: u64 = number.parse().map_err(|_| anyhow!("некорректный интервал «{text}»"))?;
        total += value * unit;
        number.clear();
    }
    if !number.is_empty() || total == 0 {
        bail!("некорректный интервал «{text}»: ожидается вроде 30s, 5m, 1h30m, 1d");
    }
    Ok(Duration::from_secs(total))
}

/// `5400000` → `1h30m`.
pub fn format_duration(ms: i64) -> String {
    let mut sec = ms / 1000;
    let mut out = String::new();
    for (unit, size) in [("d", 86_400), ("h", 3_600), ("m", 60), ("s", 1)] {
        if sec >= size {
            out.push_str(&format!("{}{unit}", sec / size));
            sec %= size;
        }
    }
    if out.is_empty() {
        "0s".into()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
        PaginatedRequestParams, Tool,
    };
    use rmcp::service::RequestContext;
    use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt};

    /// Поддельный планировщик: одно задание #8, ответы в формате scheduler-mcp.
    struct FakeScheduler;

    impl ServerHandler for FakeScheduler {
        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let serde_json::Value::Object(schema) = serde_json::json!({ "type": "object" }) else { unreachable!() };
            let tools = ["list_jobs", "cancel_job", "pause_job", "resume_job"]
                .into_iter()
                .map(|name| Tool::new(name, name, schema.clone()))
                .collect();
            Ok(ListToolsResult::with_all_items(tools))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            let job = |state: &str| {
                serde_json::json!({
                    "id": 8, "name": "java-monitor", "action": "mcp_tool", "params": {}, "schedule": "every 20s",
                    "state": state, "last_run_at": "2026-09-23T20:00:25+03:00", "consecutive_failures": 0,
                    "created_at": "2026-09-23T19:55:00+03:00"
                })
            };
            let id = request.arguments.as_ref().and_then(|a| a.get("job_id")).and_then(|v| v.as_i64());
            let body = match (request.name.as_ref(), id) {
                ("list_jobs", _) => serde_json::json!({ "now": "x", "jobs": [job("active")] }),
                (_, Some(id)) if id != 8 => {
                    let error = CallToolResult::error(vec![ContentBlock::text(format!("задания #{id} нет"))]);
                    return Ok(CallToolResponse::Complete(error));
                }
                ("cancel_job", _) => serde_json::json!({ "now": "x", "job": job("cancelled") }),
                ("pause_job", _) => serde_json::json!({ "now": "x", "job": job("paused") }),
                _ => serde_json::json!({ "now": "x", "job": job("active") }),
            };
            Ok(CallToolResponse::Complete(CallToolResult::success(vec![ContentBlock::text(body.to_string())])))
        }
    }

    #[tokio::test]
    async fn lists_and_manages_scheduler_jobs() {
        let mcp = crate::mcp::McpManager::empty();
        assert!(scheduler_jobs(&mcp, false).await.is_err(), "планировщик не подключён");

        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            if let Ok(service) = FakeScheduler.serve(server_io).await {
                let _ = service.waiting().await;
            }
        });
        mcp.attach_for_tests("sched", client_io).await;
        mcp.attach_for_tests("calc", crate::mcp::test_support::spawn_calculator()).await;
        assert_eq!(scheduler_server(&mcp).as_deref(), Some("sched"), "калькулятор — не планировщик");

        let jobs = scheduler_jobs(&mcp, false).await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].describe(), "#8 «java-monitor» · mcp_tool · every 20s · active · последний 20:00:25");

        assert_eq!(scheduler_job_action(&mcp, 8, "pause").await.unwrap().state, "paused");
        assert!(scheduler_job_action(&mcp, 8, "drop").await.is_err());
        let missing = scheduler_job_action(&mcp, 99, "cancel").await.unwrap_err().to_string();
        assert!(missing.contains("задания #99 нет"), "{missing}");

        let text = run_jobs_command(&mcp, &["cancel", "#8"]).await.unwrap();
        assert!(text.contains("cancelled"), "{text}");
        assert!(run_jobs_command(&mcp, &[]).await.unwrap().starts_with("#8"));
        assert!(run_jobs_command(&mcp, &["cancel"]).await.is_err(), "без номера");
    }

    #[test]
    fn durations_round_trip() {
        assert_eq!(parse_duration("30m").unwrap(), Duration::from_secs(1800));
        assert_eq!(parse_duration("1h30m").unwrap(), Duration::from_secs(5400));
        assert_eq!(parse_duration("90").unwrap(), Duration::from_secs(90));
        assert!(parse_duration("0m").is_err());
        assert!(parse_duration("полчаса").is_err());
        assert_eq!(format_duration(5_400_000), "1h30m");
        assert_eq!(format_duration(86_400_000), "1d");
    }

    #[test]
    fn grid_skips_missed_runs() {
        assert_eq!(next_on_grid(0, 10, 35), 40);
        assert_eq!(next_on_grid(0, 10, 0), 10);
        assert_eq!(next_on_grid(50, 10, 20), 60);
    }
}
