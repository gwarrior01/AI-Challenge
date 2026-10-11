//! Конфигурация шлюза (TOML): адрес бэкенда, лимиты и пользователи с ключами.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Где слушает шлюз, например "0.0.0.0:8080".
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Адрес llama-server без /v1 на конце.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Имя модели, которое шлюз отдаёт в /v1/models и подставляет в запросы к бэкенду:
    /// клиент может прислать любое `model`, обслуживается всё равно единственная модель.
    pub model: String,
    /// Контекст одного слота llama-server (CTX_SIZE / PARALLEL). Промпт вместе с
    /// max_tokens ответа обязан в него влезать — иначе 400 context_length_exceeded.
    #[serde(default = "default_max_context")]
    pub max_context_tokens: u64,
    /// Потолок длины ответа: больший max_tokens клиента урезается до него.
    #[serde(default = "default_max_output")]
    pub max_output_tokens: u64,
    /// max_tokens, если клиент его не указал.
    #[serde(default = "default_max_tokens")]
    pub default_max_tokens: u64,
    /// Сколько запросов одновременно уходит в модель (= PARALLEL у llama-server).
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    /// Сколько запросов может ждать свободного слота; сверх этого — 503.
    #[serde(default = "default_max_queue")]
    pub max_queue: usize,
    /// Сколько запрос ждёт в очереди, прежде чем получить 503.
    #[serde(default = "default_queue_timeout")]
    pub queue_timeout_secs: u64,
    /// Таймаут одного запроса к модели целиком (вместе с генерацией).
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    /// Лимит запросов в минуту для пользователей, у которых не задан свой `rpm`.
    #[serde(default = "default_rpm")]
    pub default_rpm: u32,
    /// Сколько запросов одного пользователя выполняется одновременно: остальные его
    /// запросы ждут, а не занимают все слоты модели, — так 2–3 пользователя делят её честно.
    #[serde(default = "default_per_user_parallel")]
    pub per_user_parallel: usize,
    pub users: Vec<User>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub name: String,
    pub key: String,
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Администратор видит в /stats всех пользователей, остальные — только себя.
    #[serde(default)]
    pub admin: bool,
}

fn default_listen() -> String {
    "0.0.0.0:8080".into()
}
fn default_backend() -> String {
    "http://127.0.0.1:8081".into()
}
fn default_max_context() -> u64 {
    2048
}
fn default_max_output() -> u64 {
    512
}
fn default_max_tokens() -> u64 {
    256
}
fn default_max_concurrent() -> usize {
    3
}
fn default_max_queue() -> usize {
    6
}
fn default_queue_timeout() -> u64 {
    120
}
fn default_request_timeout() -> u64 {
    300
}
fn default_rpm() -> u32 {
    20
}
fn default_per_user_parallel() -> usize {
    1
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("не удалось прочитать {}", path.display()))?;
        let cfg: Config = toml::from_str(&text).with_context(|| format!("ошибка в {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.users.is_empty() {
            bail!("в конфиге нет ни одного пользователя ([[users]])");
        }
        let mut keys = std::collections::HashSet::new();
        for u in &self.users {
            if u.key.len() < 16 {
                bail!("ключ пользователя {} короче 16 символов", u.name);
            }
            if !keys.insert(&u.key) {
                bail!("ключ пользователя {} совпадает с чужим", u.name);
            }
        }
        if self.max_output_tokens >= self.max_context_tokens {
            bail!("max_output_tokens должен быть меньше max_context_tokens");
        }
        if self.max_concurrent == 0 || self.per_user_parallel == 0 {
            bail!("max_concurrent и per_user_parallel должны быть больше нуля");
        }
        Ok(())
    }

    pub fn rpm(&self, user: &User) -> u32 {
        user.rpm.unwrap_or(self.default_rpm)
    }
}
