//! Ограничения шлюза: лимит запросов в минуту на пользователя и очередь к слотам модели.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

/// Скользящее окно в одну минуту: помнит моменты последних запросов каждого пользователя.
pub struct RateLimiter {
    window: Duration,
    hits: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl RateLimiter {
    pub fn per_minute() -> Self {
        Self { window: Duration::from_secs(60), hits: Mutex::new(HashMap::new()) }
    }

    /// Засчитывает запрос, если лимит не исчерпан. Иначе — через сколько освободится место.
    pub fn check(&self, user: &str, limit: u32, now: Instant) -> Result<(), Duration> {
        let mut hits = self.hits.lock().unwrap();
        let q = hits.entry(user.to_string()).or_default();
        while q.front().is_some_and(|t| now.duration_since(*t) >= self.window) {
            q.pop_front();
        }
        if q.len() >= limit as usize {
            // При limit = 0 очередь пуста — ждать придётся всё окно.
            return Err(q.front().map_or(self.window, |t| self.window - now.duration_since(*t)));
        }
        q.push_back(now);
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum AdmissionError {
    /// В очереди уже max_queue запросов.
    QueueFull,
    /// Слот не освободился за queue_timeout.
    Timeout,
}

/// Допуск к модели: не больше `slots` запросов одновременно и не больше `max_queue` ждущих.
pub struct Admission {
    slots: Arc<Semaphore>,
    total: usize,
    waiting: AtomicUsize,
    max_queue: usize,
}

impl Admission {
    pub fn new(slots: usize, max_queue: usize) -> Self {
        Self { slots: Arc::new(Semaphore::new(slots)), total: slots, waiting: AtomicUsize::new(0), max_queue }
    }

    pub async fn enter(&self, timeout: Duration) -> Result<OwnedSemaphorePermit, AdmissionError> {
        if let Ok(p) = self.slots.clone().try_acquire_owned() {
            return Ok(p);
        }
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.max_queue {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(AdmissionError::QueueFull);
        }
        let res = tokio::time::timeout(timeout, self.slots.clone().acquire_owned()).await;
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        match res {
            Ok(p) => Ok(p.expect("семафор не закрывается")),
            Err(_) => Err(AdmissionError::Timeout),
        }
    }

    pub fn active(&self) -> usize {
        self.total - self.slots.available_permits()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_blocks_until_window_passes() {
        let rl = RateLimiter::per_minute();
        let t0 = Instant::now();
        assert!(rl.check("a", 2, t0).is_ok());
        assert!(rl.check("a", 2, t0 + Duration::from_secs(10)).is_ok());
        let retry = rl.check("a", 2, t0 + Duration::from_secs(20)).unwrap_err();
        assert_eq!(retry, Duration::from_secs(40));
        // Другой пользователь считается отдельно.
        assert!(rl.check("b", 2, t0 + Duration::from_secs(20)).is_ok());
        // Первый запрос выпал из окна — место освободилось.
        assert!(rl.check("a", 2, t0 + Duration::from_secs(60)).is_ok());
    }

    #[tokio::test]
    async fn admission_queues_then_rejects() {
        let adm = Arc::new(Admission::new(1, 1));
        let held = adm.enter(Duration::from_secs(1)).await.unwrap();
        assert_eq!(adm.active(), 1);

        let a = adm.clone();
        let waiter = tokio::spawn(async move { a.enter(Duration::from_secs(5)).await.map(|_| ()) });
        while adm.waiting() == 0 {
            tokio::task::yield_now().await;
        }
        // Очередь на одно место уже занята.
        assert_eq!(adm.enter(Duration::from_secs(1)).await.unwrap_err(), AdmissionError::QueueFull);

        drop(held);
        assert!(waiter.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn admission_times_out() {
        let adm = Admission::new(1, 5);
        let _held = adm.enter(Duration::from_secs(1)).await.unwrap();
        assert_eq!(adm.enter(Duration::from_millis(20)).await.unwrap_err(), AdmissionError::Timeout);
        assert_eq!(adm.waiting(), 0);
    }
}
