//! Хранилище планировщика в SQLite: задания (`jobs`), их запуски (`runs`) и
//! события для агента (`events`). Всё расписание живёт здесь, а не в памяти,
//! поэтому после перезапуска сервер продолжает с того же места.
//!
//! Время — миллисекунды Unix (UTC), см. [`crate::time`].

use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde_json::Value;

use crate::metrics::{Extract, Metrics};

/// Когда выполнять задание.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Один раз в момент `at`.
    Once { at: i64 },
    /// Каждые `interval_ms`, начиная с первого запуска; сетка запусков
    /// сохраняется и после простоя (см. [`next_on_grid`]).
    Every { interval_ms: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Active,
    Paused,
    /// Однократное задание выполнено.
    Completed,
    Cancelled,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Active => "active",
            JobState::Paused => "paused",
            JobState::Completed => "completed",
            JobState::Cancelled => "cancelled",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "paused" => JobState::Paused,
            "completed" => JobState::Completed,
            "cancelled" => JobState::Cancelled,
            _ => JobState::Active,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: i64,
    pub name: String,
    /// Что делает задание: `reminder`, `http`, `mcp_tool` (см. [`crate::actions`]).
    pub action: String,
    pub params: Value,
    pub extract: Option<Extract>,
    pub schedule: Schedule,
    pub state: JobState,
    /// Следующий запуск; `None` — запусков больше не будет (пауза, выполнено, отменено).
    pub next_run_at: Option<i64>,
    pub last_run_at: Option<i64>,
    pub consecutive_failures: i64,
    /// Почему задание на паузе, если его остановил сам планировщик.
    pub pause_reason: Option<String>,
    pub created_at: i64,
}

pub struct NewJob {
    pub name: String,
    pub action: String,
    pub params: Value,
    pub extract: Option<Extract>,
    pub schedule: Schedule,
    /// Первый запуск.
    pub first_run_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Ok { result: Value, metrics: Metrics },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Run {
    pub id: i64,
    /// Когда запуск должен был начаться по расписанию.
    pub scheduled_at: i64,
    pub started_at: i64,
    pub finished_at: i64,
    pub outcome: Outcome,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub id: i64,
    pub job_id: Option<i64>,
    pub created_at: i64,
    /// `reminder`, `job_paused` и т. п.
    pub kind: String,
    pub text: String,
    pub acked: bool,
}

pub struct Store {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS jobs (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        name                 TEXT NOT NULL,
        action               TEXT NOT NULL,
        params_json          TEXT NOT NULL,
        extract_json         TEXT,
        schedule_kind        TEXT NOT NULL,
        schedule_value       INTEGER NOT NULL,
        state                TEXT NOT NULL,
        next_run_at          INTEGER,
        last_run_at          INTEGER,
        consecutive_failures INTEGER NOT NULL DEFAULT 0,
        pause_reason         TEXT,
        created_at           INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state, next_run_at);
    CREATE TABLE IF NOT EXISTS runs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id       INTEGER NOT NULL REFERENCES jobs(id),
        scheduled_at INTEGER NOT NULL,
        started_at   INTEGER NOT NULL,
        finished_at  INTEGER NOT NULL,
        status       TEXT NOT NULL,
        result_json  TEXT,
        metrics_json TEXT,
        error        TEXT
    );
    CREATE INDEX IF NOT EXISTS runs_by_job ON runs(job_id, started_at);
    CREATE TABLE IF NOT EXISTS events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id     INTEGER,
        created_at INTEGER NOT NULL,
        kind       TEXT NOT NULL,
        text       TEXT NOT NULL,
        acked_at   INTEGER
    );
    CREATE INDEX IF NOT EXISTS events_unread ON events(acked_at, id);
";

const JOB_COLUMNS: &str = "id, name, action, params_json, extract_json, schedule_kind, schedule_value, state, \
     next_run_at, last_run_at, consecutive_failures, pause_reason, created_at";
const RUN_COLUMNS: &str = "id, job_id, scheduled_at, started_at, finished_at, status, result_json, metrics_json, error";
const EVENT_COLUMNS: &str = "id, job_id, created_at, kind, text, acked_at";

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("не удалось открыть {}", path.display()))?;
        Self::init(conn)
    }

    #[cfg(test)]
    pub fn in_memory() -> Self {
        Self::init(Connection::open_in_memory().unwrap()).unwrap()
    }

    fn init(conn: Connection) -> Result<Self> {
        // WAL — чтобы чтение сводки не ждало записи очередного запуска.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("соединение с БД планировщика отравлено паникой")
    }

    pub fn insert_job(&self, job: NewJob, now: i64) -> Result<Job> {
        let (kind, value) = match job.schedule {
            Schedule::Once { at } => ("once", at),
            Schedule::Every { interval_ms } => ("every", interval_ms),
        };
        let extract = job.extract.as_ref().map(serde_json::to_string).transpose()?;
        let id = {
            let conn = self.conn();
            conn.execute(
                "INSERT INTO jobs (name, action, params_json, extract_json, schedule_kind, schedule_value, state, \
                     next_run_at, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?8)",
                params![job.name, job.action, job.params.to_string(), extract, kind, value, job.first_run_at, now],
            )?;
            conn.last_insert_rowid()
        };
        self.job(id)?.context("задание пропало сразу после создания")
    }

    pub fn job(&self, id: i64) -> Result<Option<Job>> {
        let conn = self.conn();
        Ok(conn
            .query_row(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE id = ?1"), [id], job_from_row)
            .optional()?)
    }

    /// Задания по порядку создания; `include_finished` — и выполненные/отменённые.
    pub fn jobs(&self, include_finished: bool) -> Result<Vec<Job>> {
        let conn = self.conn();
        let filter = if include_finished { "" } else { "WHERE state IN ('active', 'paused')" };
        let mut stmt = conn.prepare(&format!("SELECT {JOB_COLUMNS} FROM jobs {filter} ORDER BY id"))?;
        let jobs = stmt.query_map([], job_from_row)?.collect::<rusqlite::Result<_>>()?;
        Ok(jobs)
    }

    /// Ближайший запуск среди активных заданий.
    pub fn next_due_at(&self) -> Result<Option<i64>> {
        let conn = self.conn();
        Ok(conn.query_row("SELECT MIN(next_run_at) FROM jobs WHERE state = 'active'", [], |r| r.get(0))?)
    }

    /// Забирает задания, чей срок наступил, и сразу переносит их следующий
    /// запуск — так цикл планировщика не возьмёт их второй раз, пока они
    /// выполняются. Возвращает задания вместе с моментом, на который был
    /// назначен запуск.
    ///
    /// Пропущенные за время простоя запуски не воспроизводятся: задание
    /// выполняется один раз, а следующий запуск — ближайший по его сетке
    /// после `now` (см. [`next_on_grid`]).
    pub fn claim_due(&self, now: i64) -> Result<Vec<(Job, i64)>> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let due: Vec<Job> = {
            let mut stmt = tx.prepare(&format!(
                "SELECT {JOB_COLUMNS} FROM jobs WHERE state = 'active' AND next_run_at <= ?1 ORDER BY next_run_at"
            ))?;
            let rows = stmt.query_map([now], job_from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let mut claimed = Vec::with_capacity(due.len());
        for job in due {
            let scheduled = job.next_run_at.unwrap_or(now);
            let next = match job.schedule {
                Schedule::Once { .. } => None,
                Schedule::Every { interval_ms } => Some(next_on_grid(scheduled, interval_ms, now)),
            };
            tx.execute("UPDATE jobs SET next_run_at = ?1 WHERE id = ?2", params![next, job.id])?;
            claimed.push((job, scheduled));
        }
        tx.commit()?;
        Ok(claimed)
    }

    /// Сохраняет запуск и обновляет задание: время последнего запуска, счётчик
    /// ошибок подряд, у однократного — состояние «выполнено». Возвращает
    /// число ошибок подряд после этого запуска.
    pub fn record_run(&self, job_id: i64, scheduled_at: i64, started_at: i64, finished_at: i64, outcome: &Outcome) -> Result<i64> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let (status, result, metrics, error) = match outcome {
            Outcome::Ok { result, metrics } => {
                ("ok", Some(result.to_string()), Some(serde_json::to_string(metrics)?), None)
            }
            Outcome::Error(e) => ("error", None, None, Some(e.as_str())),
        };
        tx.execute(
            &format!("INSERT INTO runs ({}) VALUES (NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)", RUN_COLUMNS),
            params![job_id, scheduled_at, started_at, finished_at, status, result, metrics, error],
        )?;
        let failures_sql = if matches!(outcome, Outcome::Ok { .. }) { "0" } else { "consecutive_failures + 1" };
        tx.execute(
            &format!(
                "UPDATE jobs SET last_run_at = ?1, consecutive_failures = {failures_sql}, \
                     state = CASE WHEN schedule_kind = 'once' AND state = 'active' THEN 'completed' ELSE state END \
                 WHERE id = ?2"
            ),
            params![finished_at, job_id],
        )?;
        let failures = tx.query_row("SELECT consecutive_failures FROM jobs WHERE id = ?1", [job_id], |r| r.get(0))?;
        tx.commit()?;
        Ok(failures)
    }

    /// Ставит задание на паузу; `reason` — если это сделал сам планировщик.
    pub fn pause(&self, id: i64, reason: Option<&str>) -> Result<Job> {
        self.change_state(id, |job| {
            if job.state != JobState::Active {
                bail!("задание #{id} не активно (состояние: {})", job.state.as_str());
            }
            Ok(("paused", None, reason.map(str::to_string)))
        })
    }

    /// Снимает задание с паузы. Периодическое запускается сразу и дальше по
    /// сетке от этого момента; однократное — в свой срок или сразу, если он прошёл.
    pub fn resume(&self, id: i64, now: i64) -> Result<Job> {
        self.change_state(id, |job| {
            if job.state != JobState::Paused {
                bail!("задание #{id} не на паузе (состояние: {})", job.state.as_str());
            }
            let next = match job.schedule {
                Schedule::Once { at } => at.max(now),
                Schedule::Every { .. } => now,
            };
            Ok(("active", Some(next), None))
        })
    }

    pub fn cancel(&self, id: i64) -> Result<Job> {
        self.change_state(id, |job| {
            if matches!(job.state, JobState::Completed | JobState::Cancelled) {
                bail!("задание #{id} уже завершено (состояние: {})", job.state.as_str());
            }
            Ok(("cancelled", None, None))
        })
    }

    fn change_state(
        &self,
        id: i64,
        decide: impl FnOnce(&Job) -> Result<(&'static str, Option<i64>, Option<String>)>,
    ) -> Result<Job> {
        {
            let conn = self.conn();
            let job = conn
                .query_row(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE id = ?1"), [id], job_from_row)
                .optional()?
                .with_context(|| format!("задания #{id} нет"))?;
            let (state, next, reason) = decide(&job)?;
            conn.execute(
                "UPDATE jobs SET state = ?1, next_run_at = ?2, pause_reason = ?3, \
                     consecutive_failures = CASE WHEN ?1 = 'active' THEN 0 ELSE consecutive_failures END \
                 WHERE id = ?4",
                params![state, next, reason, id],
            )?;
        }
        self.job(id)?.context("задание пропало")
    }

    /// Запуски задания в окне `[since, until]`, от старых к новым; `limit` —
    /// последние столько.
    pub fn runs(&self, job_id: i64, since: Option<i64>, limit: Option<usize>) -> Result<Vec<Run>> {
        let conn = self.conn();
        let limit = limit.map(|l| l as i64).unwrap_or(-1);
        let mut stmt = conn.prepare(&format!(
            "SELECT * FROM (SELECT {RUN_COLUMNS} FROM runs WHERE job_id = ?1 AND started_at >= ?2 \
                 ORDER BY started_at DESC, id DESC LIMIT ?3) ORDER BY started_at, id"
        ))?;
        let runs = stmt
            .query_map(params![job_id, since.unwrap_or(i64::MIN), limit], run_from_row)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(runs)
    }

    pub fn add_event(&self, job_id: Option<i64>, kind: &str, text: &str, now: i64) -> Result<i64> {
        let conn = self.conn();
        conn.execute(
            "INSERT INTO events (job_id, created_at, kind, text) VALUES (?1, ?2, ?3, ?4)",
            params![job_id, now, kind, text],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// События от старых к новым: только непрочитанные или все после `since`.
    pub fn events(&self, since: Option<i64>, unread_only: bool, limit: usize) -> Result<Vec<Event>> {
        let conn = self.conn();
        let unread = if unread_only { "AND acked_at IS NULL" } else { "" };
        let mut stmt = conn.prepare(&format!(
            "SELECT {EVENT_COLUMNS} FROM events WHERE created_at >= ?1 {unread} ORDER BY id LIMIT ?2"
        ))?;
        let events = stmt
            .query_map(params![since.unwrap_or(i64::MIN), limit as i64], event_from_row)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(events)
    }

    /// Отмечает события прочитанными; возвращает, сколько отмечено впервые.
    pub fn ack_events(&self, ids: &[i64], now: i64) -> Result<usize> {
        let conn = self.conn();
        let mut acked = 0;
        for id in ids {
            acked += conn.execute("UPDATE events SET acked_at = ?1 WHERE id = ?2 AND acked_at IS NULL", params![now, id])?;
        }
        Ok(acked)
    }

    /// Удаляет запуски и прочитанные события старше `before`.
    pub fn prune(&self, before: i64) -> Result<(usize, usize)> {
        let conn = self.conn();
        let runs = conn.execute("DELETE FROM runs WHERE started_at < ?1", [before])?;
        let events = conn.execute("DELETE FROM events WHERE created_at < ?1 AND acked_at IS NOT NULL", [before])?;
        Ok((runs, events))
    }
}

/// Следующий запуск периодического задания после `now` по сетке
/// `scheduled + k·interval`: после простоя задание выполняется один раз, а
/// дальше идёт по своей прежней сетке (12:22 → 12:25 → 12:30 для «каждые 5
/// минут» с запусками в 12:00, 12:05, …), а не «через интервал от сейчас».
pub fn next_on_grid(scheduled: i64, interval_ms: i64, now: i64) -> i64 {
    let interval = interval_ms.max(1);
    if now < scheduled {
        return scheduled + interval;
    }
    scheduled + interval * ((now - scheduled) / interval + 1)
}

fn job_from_row(row: &Row) -> rusqlite::Result<Job> {
    let kind: String = row.get(5)?;
    let value: i64 = row.get(6)?;
    let params: String = row.get(3)?;
    let extract: Option<String> = row.get(4)?;
    let state: String = row.get(7)?;
    Ok(Job {
        id: row.get(0)?,
        name: row.get(1)?,
        action: row.get(2)?,
        params: serde_json::from_str(&params).unwrap_or(Value::Null),
        extract: extract.and_then(|e| serde_json::from_str(&e).ok()),
        schedule: if kind == "once" { Schedule::Once { at: value } } else { Schedule::Every { interval_ms: value } },
        state: JobState::parse(&state),
        next_run_at: row.get(8)?,
        last_run_at: row.get(9)?,
        consecutive_failures: row.get(10)?,
        pause_reason: row.get(11)?,
        created_at: row.get(12)?,
    })
}

fn run_from_row(row: &Row) -> rusqlite::Result<Run> {
    let status: String = row.get(5)?;
    let outcome = if status == "ok" {
        let result: Option<String> = row.get(6)?;
        let metrics: Option<String> = row.get(7)?;
        Outcome::Ok {
            result: result.and_then(|r| serde_json::from_str(&r).ok()).unwrap_or(Value::Null),
            metrics: metrics.and_then(|m| serde_json::from_str(&m).ok()).unwrap_or_default(),
        }
    } else {
        Outcome::Error(row.get::<_, Option<String>>(8)?.unwrap_or_default())
    };
    Ok(Run {
        id: row.get(0)?,
        scheduled_at: row.get(2)?,
        started_at: row.get(3)?,
        finished_at: row.get(4)?,
        outcome,
    })
}

fn event_from_row(row: &Row) -> rusqlite::Result<Event> {
    Ok(Event {
        id: row.get(0)?,
        job_id: row.get(1)?,
        created_at: row.get(2)?,
        kind: row.get(3)?,
        text: row.get(4)?,
        acked: row.get::<_, Option<i64>>(5)?.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MIN: i64 = 60_000;

    fn every(store: &Store, first: i64, interval: i64) -> Job {
        let job = NewJob {
            name: "t".into(),
            action: "http".into(),
            params: json!({}),
            extract: None,
            schedule: Schedule::Every { interval_ms: interval },
            first_run_at: first,
        };
        store.insert_job(job, 0).unwrap()
    }

    #[test]
    fn grid_skips_missed_runs() {
        // Каждые 5 минут, запуск назначен на 12:10, сервер ожил в 12:22.
        assert_eq!(next_on_grid(10 * MIN, 5 * MIN, 22 * MIN), 25 * MIN);
        assert_eq!(next_on_grid(10 * MIN, 5 * MIN, 10 * MIN), 15 * MIN);
        assert_eq!(next_on_grid(10 * MIN, 5 * MIN, 15 * MIN), 20 * MIN);
    }

    #[test]
    fn claim_runs_a_missed_periodic_job_once() {
        let store = Store::in_memory();
        let job = every(&store, 10 * MIN, 5 * MIN);
        assert!(store.claim_due(9 * MIN).unwrap().is_empty());

        let claimed = store.claim_due(22 * MIN).unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].1, 10 * MIN, "назначенный момент запуска");
        assert!(store.claim_due(22 * MIN).unwrap().is_empty(), "второй раз не берётся");
        assert_eq!(store.job(job.id).unwrap().unwrap().next_run_at, Some(25 * MIN));
        assert_eq!(store.next_due_at().unwrap(), Some(25 * MIN));
    }

    #[test]
    fn once_job_completes_and_failures_are_counted() {
        let store = Store::in_memory();
        let once = store
            .insert_job(
                NewJob {
                    name: "r".into(),
                    action: "reminder".into(),
                    params: json!({"text": "x"}),
                    extract: None,
                    schedule: Schedule::Once { at: MIN },
                    first_run_at: MIN,
                },
                0,
            )
            .unwrap();
        let claimed = store.claim_due(MIN).unwrap();
        assert_eq!(claimed.len(), 1);
        let ok = Outcome::Ok { result: json!(1), metrics: Metrics::new() };
        store.record_run(once.id, MIN, MIN, MIN + 5, &ok).unwrap();
        let once = store.job(once.id).unwrap().unwrap();
        assert_eq!((once.state, once.next_run_at), (JobState::Completed, None));

        let periodic = every(&store, 0, MIN);
        let err = Outcome::Error("нет связи".into());
        assert_eq!(store.record_run(periodic.id, 0, 0, 1, &err).unwrap(), 1);
        assert_eq!(store.record_run(periodic.id, MIN, MIN, MIN + 1, &err).unwrap(), 2);
        assert_eq!(store.record_run(periodic.id, 2 * MIN, 2 * MIN, 2 * MIN + 1, &ok).unwrap(), 0);

        let runs = store.runs(periodic.id, None, None).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].outcome, err);
        assert_eq!(store.runs(periodic.id, Some(MIN), None).unwrap().len(), 2);
        let last = store.runs(periodic.id, None, Some(1)).unwrap();
        assert_eq!((last.len(), last[0].started_at), (1, 2 * MIN));
    }

    #[test]
    fn pause_resume_cancel() {
        let store = Store::in_memory();
        let job = every(&store, 0, MIN);
        let paused = store.pause(job.id, Some("5 ошибок подряд")).unwrap();
        assert_eq!((paused.state, paused.pause_reason.as_deref()), (JobState::Paused, Some("5 ошибок подряд")));
        assert!(store.claim_due(10 * MIN).unwrap().is_empty(), "на паузе не запускается");
        assert!(store.pause(job.id, None).is_err());

        let resumed = store.resume(job.id, 7 * MIN).unwrap();
        assert_eq!((resumed.state, resumed.next_run_at, resumed.pause_reason), (JobState::Active, Some(7 * MIN), None));
        let cancelled = store.cancel(job.id).unwrap();
        assert_eq!((cancelled.state, cancelled.next_run_at), (JobState::Cancelled, None));
        assert!(store.cancel(job.id).is_err());
        assert!(store.jobs(false).unwrap().is_empty());
        assert_eq!(store.jobs(true).unwrap().len(), 1);
        assert!(store.pause(999, None).is_err());
    }

    #[test]
    fn events_are_read_until_acked() {
        let store = Store::in_memory();
        let a = store.add_event(None, "reminder", "раз", 10).unwrap();
        let b = store.add_event(Some(1), "job_paused", "два", 20).unwrap();
        assert_eq!(store.events(None, true, 10).unwrap().len(), 2);
        assert_eq!(store.ack_events(&[a], 30).unwrap(), 1);
        assert_eq!(store.ack_events(&[a], 31).unwrap(), 0, "повторно не считается");
        let unread = store.events(None, true, 10).unwrap();
        assert_eq!((unread.len(), unread[0].id), (1, b));
        let all = store.events(Some(0), false, 10).unwrap();
        assert!(all[0].acked && !all[1].acked);

        assert_eq!(store.prune(25).unwrap(), (0, 1), "удаляется только прочитанное старое событие");
    }
}
