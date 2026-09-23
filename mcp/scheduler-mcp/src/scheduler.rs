//! Фоновый цикл планировщика: одна tokio-задача ждёт ближайший срок из БД,
//! забирает наступившие задания, выполняет каждое в своей задаче и пишет
//! результат в `runs`. Новые задания и снятие с паузы будят цикл сразу
//! ([`Scheduler::wake`]), иначе он спит до ближайшего срока.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;

use crate::actions::Executor;
use crate::metrics;
use crate::store::{Job, Outcome, Schedule, Store};
use crate::time::{self, now_ms};

/// Даже без заданий цикл просыпается раз в минуту — на случай, если БД
/// поменяли снаружи.
const MAX_SLEEP: Duration = Duration::from_secs(60);
/// Опоздание, после которого напоминание честно говорит, что оно опоздало
/// (сервер был выключен в свой срок).
const LATE_NOTICE_MS: i64 = 60_000;

#[derive(Clone)]
pub struct Scheduler {
    store: Arc<Store>,
    executor: Executor,
    wake: Arc<Notify>,
    /// Задания, которые выполняются прямо сейчас: медленный запуск не
    /// накладывается на следующий по расписанию — тот пропускается.
    in_flight: Arc<Mutex<HashSet<i64>>>,
    /// Сколько ошибок подряд терпит периодическое задание, прежде чем
    /// планировщик ставит его на паузу.
    max_failures: i64,
}

impl Scheduler {
    pub fn new(store: Arc<Store>, executor: Executor, max_failures: i64) -> Self {
        Self {
            store,
            executor,
            wake: Arc::new(Notify::new()),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            max_failures: max_failures.max(1),
        }
    }

    /// Разбудить цикл: расписание изменилось.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    pub async fn run(self) {
        loop {
            let now = now_ms();
            match self.store.claim_due(now) {
                Ok(due) => {
                    for (job, scheduled_at) in due {
                        if !self.in_flight.lock().expect("in_flight отравлен").insert(job.id) {
                            eprintln!("scheduler-mcp: задание #{} ещё выполняется, запуск пропущен", job.id);
                            continue;
                        }
                        let this = self.clone();
                        tokio::spawn(async move {
                            let id = job.id;
                            this.run_job(job, scheduled_at).await;
                            this.in_flight.lock().expect("in_flight отравлен").remove(&id);
                        });
                    }
                }
                Err(e) => eprintln!("scheduler-mcp: не удалось выбрать задания: {e:#}"),
            }

            let sleep = match self.store.next_due_at() {
                Ok(Some(next)) => Duration::from_millis((next - now_ms()).max(0) as u64).min(MAX_SLEEP),
                _ => MAX_SLEEP,
            };
            tokio::select! {
                _ = tokio::time::sleep(sleep) => {}
                _ = self.wake.notified() => {}
            }
        }
    }

    /// Выполняет задание один раз и сохраняет запуск.
    pub async fn run_job(&self, job: Job, scheduled_at: i64) {
        let started_at = now_ms();
        let executed = self.executor.execute(&job.action, &job.params).await;
        let finished_at = now_ms();

        let outcome = match executed {
            Ok(output) => {
                if let Some(text) = output.event {
                    let late = started_at - scheduled_at;
                    let text = if late > LATE_NOTICE_MS {
                        format!("{text} (с опозданием: срок был {}, сервер тогда не работал)", time::format(scheduled_at))
                    } else {
                        text
                    };
                    self.event(Some(job.id), &job.action, &text, finished_at);
                }
                let metrics = metrics::collect(&output.result, job.extract.as_ref());
                Outcome::Ok { result: output.result, metrics }
            }
            Err(e) => Outcome::Error(format!("{e:#}")),
        };

        let failures = match self.store.record_run(job.id, scheduled_at, started_at, finished_at, &outcome) {
            Ok(failures) => failures,
            Err(e) => {
                eprintln!("scheduler-mcp: не удалось сохранить запуск задания #{}: {e:#}", job.id);
                return;
            }
        };
        let Outcome::Error(error) = &outcome else { return };
        match job.schedule {
            Schedule::Once { .. } => {
                let text = format!("Задание #{} «{}» не выполнилось: {error}", job.id, job.name);
                self.event(Some(job.id), "job_failed", &text, finished_at);
            }
            Schedule::Every { .. } if failures >= self.max_failures => {
                let reason = format!("{failures} ошибок подряд, последняя: {error}");
                if self.store.pause(job.id, Some(&reason)).is_ok() {
                    let text = format!("Задание #{} «{}» поставлено на паузу: {reason}", job.id, job.name);
                    self.event(Some(job.id), "job_paused", &text, finished_at);
                }
            }
            Schedule::Every { .. } => {}
        }
    }

    fn event(&self, job_id: Option<i64>, kind: &str, text: &str, now: i64) {
        if let Err(e) = self.store.add_event(job_id, kind, text, now) {
            eprintln!("scheduler-mcp: не удалось сохранить событие: {e:#}");
        }
    }
}

/// Раз в час удаляет запуски и прочитанные события старше `retention`.
pub async fn prune_loop(store: Arc<Store>, retention: Duration) {
    loop {
        let before = now_ms() - retention.as_millis() as i64;
        match store.prune(before) {
            Ok((0, 0)) => {}
            Ok((runs, events)) => eprintln!("scheduler-mcp: удалено старых запусков: {runs}, событий: {events}"),
            Err(e) => eprintln!("scheduler-mcp: не удалось удалить старые данные: {e:#}"),
        }
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{JobState, NewJob};
    use serde_json::json;

    fn scheduler(store: Arc<Store>, max_failures: i64) -> Scheduler {
        let config = std::env::temp_dir().join("scheduler-mcp-test-missing.json");
        Scheduler::new(store, Executor::new(config, "127.0.0.1:8092".into()), max_failures)
    }

    #[tokio::test]
    async fn late_reminder_says_so_and_completes() {
        let store = Arc::new(Store::in_memory());
        let s = scheduler(store.clone(), 3);
        store
            .insert_job(
                NewJob {
                    name: "r".into(),
                    action: "reminder".into(),
                    params: json!({"text": "созвон"}),
                    extract: None,
                    schedule: Schedule::Once { at: 0 },
                    first_run_at: 0,
                },
                0,
            )
            .unwrap();
        let (job, scheduled) = store.claim_due(now_ms()).unwrap().remove(0);
        s.run_job(job, scheduled).await;

        let events = store.events(None, true, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "reminder");
        assert!(events[0].text.starts_with("созвон (с опозданием"), "{}", events[0].text);
        assert_eq!(store.job(1).unwrap().unwrap().state, JobState::Completed);
        assert_eq!(store.runs(1, None, None).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn failing_periodic_job_is_paused_with_an_event() {
        let store = Arc::new(Store::in_memory());
        let s = scheduler(store.clone(), 2);
        // Сервера «nope» в конфигурации нет — каждый запуск падает.
        let job = store
            .insert_job(
                NewJob {
                    name: "prof".into(),
                    action: "mcp_tool".into(),
                    params: json!({"server": "nope", "tool": "x"}),
                    extract: None,
                    schedule: Schedule::Every { interval_ms: 1000 },
                    first_run_at: 0,
                },
                0,
            )
            .unwrap();
        s.run_job(job.clone(), 0).await;
        assert_eq!(store.job(job.id).unwrap().unwrap().state, JobState::Active);
        s.run_job(job.clone(), 1000).await;
        let paused = store.job(job.id).unwrap().unwrap();
        assert_eq!(paused.state, JobState::Paused);
        assert!(paused.pause_reason.unwrap().starts_with("2 ошибок подряд"));
        let events = store.events(None, true, 10).unwrap();
        assert_eq!((events.len(), events[0].kind.as_str()), (1, "job_paused"));
    }
}
