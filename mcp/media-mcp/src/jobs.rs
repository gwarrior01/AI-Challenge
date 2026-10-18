//! Задания: долгие операции (конвертация, извлечение, нарезка, транскрипция,
//! саммари) идут в фоне сервера, а вызов инструмента сразу возвращает номер
//! задания. Процессом владеет сервер: он ждёт его выхода и решает по коду
//! возврата, а не по тому, что файл появился или растёт.
//!
//! В одном каталоге записи одновременно идёт не больше одного задания: пока
//! пишется `output.mp4`, ни извлечь из него MP3, ни запустить вторую
//! конвертацию нельзя — [`Jobs::start`] откажет.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::watch;

/// Сколько завершённых заданий помнить.
const KEEP_FINISHED: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Succeeded,
    Failed,
}

/// Задание глазами клиента.
#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    /// Инструмент, который его запустил: `convert_to_mp4`, `transcribe`, …
    pub kind: String,
    /// Каталог записи относительно корня.
    pub dir: String,
    pub state: JobState,
    pub elapsed_sec: u64,
    /// Ход работы словами: `12:01 из 58:10 (21%)`, `чанки: 7 из 14 готово`.
    pub progress: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Неудача задания: сообщение и, если есть, подробности (длительности, чанки).
#[derive(Debug)]
pub struct Failure {
    pub message: String,
    pub details: Option<Value>,
}

impl Failure {
    pub fn with(message: impl Into<String>, details: Value) -> Self {
        Self { message: message.into(), details: Some(details) }
    }
}

impl From<anyhow::Error> for Failure {
    fn from(e: anyhow::Error) -> Self {
        Self { message: format!("{e:#}"), details: None }
    }
}

struct Job {
    kind: String,
    dir: PathBuf,
    label: String,
    started: Instant,
    finished: Option<Instant>,
    state: JobState,
    progress: String,
    result: Option<Value>,
    error: Option<String>,
}

impl Job {
    fn view(&self, id: &str) -> JobView {
        let end = self.finished.unwrap_or_else(Instant::now);
        JobView {
            id: id.to_string(),
            kind: self.kind.clone(),
            dir: self.label.clone(),
            state: self.state,
            elapsed_sec: end.duration_since(self.started).as_secs(),
            progress: self.progress.clone(),
            result: self.result.clone(),
            error: self.error.clone(),
        }
    }
}

#[derive(Default)]
struct Inner {
    next: u64,
    /// По номеру — `j1`, `j2`, …; ключ — число, чтобы порядок был по времени.
    jobs: BTreeMap<u64, Job>,
}

pub struct Jobs {
    inner: Mutex<Inner>,
    /// Счётчик завершений: ожидающие просыпаются, когда он меняется.
    finished: watch::Sender<u64>,
}

/// Ручка задания внутри его работы: сообщать ход.
#[derive(Clone)]
pub struct JobCtx {
    jobs: Arc<Jobs>,
    id: u64,
}

impl JobCtx {
    pub fn progress(&self, text: impl Into<String>) {
        if let Some(job) = self.jobs.lock().jobs.get_mut(&self.id) {
            job.progress = text.into();
        }
    }
}

impl Jobs {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(Inner::default()), finished: watch::channel(0).0 })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Идущее задание в каталоге записи.
    pub fn running_in(&self, dir: &Path) -> Option<JobView> {
        let inner = self.lock();
        inner.jobs.iter().find(|(_, j)| j.state == JobState::Running && j.dir == dir).map(|(id, j)| j.view(&job_id(*id)))
    }

    /// Запустить задание в каталоге `dir`. Отказ — если там уже идёт другое.
    pub fn start<F, Fut>(self: &Arc<Self>, kind: &str, dir: &Path, label: &str, work: F) -> Result<JobView, String>
    where
        F: FnOnce(JobCtx) -> Fut,
        Fut: Future<Output = Result<Value, Failure>> + Send + 'static,
    {
        let id = {
            let mut inner = self.lock();
            if let Some((id, j)) = inner.jobs.iter().find(|(_, j)| j.state == JobState::Running && j.dir == dir) {
                return Err(format!(
                    "в каталоге «{label}» уже идёт задание {} ({}) — дождись его (wait_job), второе не запускается",
                    job_id(*id),
                    j.kind
                ));
            }
            inner.next += 1;
            let id = inner.next;
            inner.jobs.insert(
                id,
                Job {
                    kind: kind.to_string(),
                    dir: dir.to_path_buf(),
                    label: label.to_string(),
                    started: Instant::now(),
                    finished: None,
                    state: JobState::Running,
                    progress: "запущено".into(),
                    result: None,
                    error: None,
                },
            );
            prune(&mut inner);
            id
        };
        let fut = work(JobCtx { jobs: self.clone(), id });
        let jobs = self.clone();
        // Работа — отдельной задачей: паника в ней не оставит задание «идущим» навсегда.
        let handle = tokio::spawn(fut);
        tokio::spawn(async move {
            let outcome = match handle.await {
                Ok(r) => r,
                Err(e) => Err(Failure { message: format!("задание аварийно завершилось: {e}"), details: None }),
            };
            jobs.finish(id, outcome);
        });
        Ok(self.get(&job_id(id)).expect("задание только что добавлено"))
    }

    fn finish(&self, id: u64, outcome: Result<Value, Failure>) {
        {
            let mut inner = self.lock();
            if let Some(job) = inner.jobs.get_mut(&id) {
                job.finished = Some(Instant::now());
                match outcome {
                    Ok(result) => {
                        job.state = JobState::Succeeded;
                        job.progress = "готово".into();
                        job.result = Some(result);
                    }
                    Err(f) => {
                        job.state = JobState::Failed;
                        job.progress = "ошибка".into();
                        job.error = Some(f.message);
                        job.result = f.details;
                    }
                }
            }
        }
        self.finished.send_modify(|n| *n += 1);
    }

    pub fn get(&self, id: &str) -> Option<JobView> {
        let n = parse_id(id)?;
        self.lock().jobs.get(&n).map(|j| j.view(&job_id(n)))
    }

    /// Все задания, новые в конце.
    pub fn list(&self) -> Vec<JobView> {
        self.lock().jobs.iter().map(|(id, j)| j.view(&job_id(*id))).collect()
    }

    /// Ждать завершения задания не дольше `timeout`. `None` — такого нет.
    pub async fn wait(&self, id: &str, timeout: Duration) -> Option<JobView> {
        let mut rx = self.finished.subscribe();
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let view = self.get(id)?;
            if view.state != JobState::Running {
                return Some(view);
            }
            // Подписка взята до проверки: завершение между ними не потеряется.
            match tokio::time::timeout_at(deadline, rx.changed()).await {
                Ok(Ok(())) => continue,
                _ => return self.get(id),
            }
        }
    }
}

fn prune(inner: &mut Inner) {
    let finished: Vec<u64> = inner.jobs.iter().filter(|(_, j)| j.state != JobState::Running).map(|(id, _)| *id).collect();
    for id in finished.iter().take(finished.len().saturating_sub(KEEP_FINISHED)) {
        inner.jobs.remove(id);
    }
}

fn job_id(n: u64) -> String {
    format!("j{n}")
}

fn parse_id(id: &str) -> Option<u64> {
    id.trim().trim_start_matches('j').parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn one_job_per_directory_and_waiting() {
        let jobs = Jobs::new();
        let dir = Path::new("/rec/a");
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let job = jobs
            .start("convert_to_mp4", dir, "a", |ctx| async move {
                ctx.progress("половина");
                let _ = rx.await;
                Ok(json!({ "output": "output.mp4" }))
            })
            .unwrap();
        assert_eq!(job.id, "j1");
        assert_eq!(job.state, JobState::Running);

        let err = jobs.start("extract_mp3", dir, "a", |_| async { Ok(json!({})) }).unwrap_err();
        assert!(err.contains("уже идёт задание j1 (convert_to_mp4)"), "{err}");
        // В другом каталоге — можно.
        let other = jobs.start("split_audio", Path::new("/rec/b"), "b", |_| async { Ok(json!({})) }).unwrap();
        assert_eq!(other.id, "j2");

        // Пока не отпустили — ожидание возвращает «идёт» по таймауту.
        let still = jobs.wait("j1", Duration::from_millis(50)).await.unwrap();
        assert_eq!((still.state, still.progress.as_str()), (JobState::Running, "половина"));
        assert!(jobs.running_in(dir).is_some());

        tx.send(()).unwrap();
        let done = jobs.wait("j1", Duration::from_secs(5)).await.unwrap();
        assert_eq!(done.state, JobState::Succeeded);
        assert_eq!(done.result.unwrap()["output"], "output.mp4");
        assert!(jobs.running_in(dir).is_none());
        assert!(jobs.wait("j99", Duration::from_millis(1)).await.is_none());
    }

    #[tokio::test]
    async fn failures_and_panics_finish_the_job() {
        let jobs = Jobs::new();
        jobs.start("transcribe", Path::new("/r"), "r", |_| async {
            Err(Failure::with("2 чанка не расшифрованы", json!({ "failed": ["chunk-001.mp3"] })))
        })
        .unwrap();
        let v = jobs.wait("j1", Duration::from_secs(5)).await.unwrap();
        assert_eq!(v.state, JobState::Failed);
        assert_eq!(v.error.as_deref(), Some("2 чанка не расшифрованы"));
        assert_eq!(v.result.unwrap()["failed"][0], "chunk-001.mp3");

        jobs.start("split_audio", Path::new("/r"), "r", |_| async { panic!("сбой") }).unwrap();
        let v = jobs.wait("j2", Duration::from_secs(5)).await.unwrap();
        assert_eq!(v.state, JobState::Failed);
        assert!(v.error.unwrap().contains("аварийно"));
        assert!(jobs.running_in(Path::new("/r")).is_none());
    }
}
