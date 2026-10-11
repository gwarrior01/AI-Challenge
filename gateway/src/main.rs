//! llm-gateway — приватный AI-сервис поверх локальной LLM (llama-server).
//!
//! `llm-gateway serve --config gateway.toml` — OpenAI-совместимый шлюз с ключами,
//! rate limit, лимитом контекста и очередью к слотам модели.
//! `llm-gateway loadtest --url http://host:8080 --keys k1,k2,k3` — проверка сервиса по сети:
//! несколько пользователей одновременно и срабатывание ограничений.

mod config;
mod limits;
mod loadtest;
mod proxy;

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

const USAGE: &str = "\
Использование:
  llm-gateway serve [--config gateway.toml]
  llm-gateway loadtest --url http://host:8080 --keys key1,key2,key3 [--requests 3] [--max-tokens 64]";

/// Значение флага `--name value` из списка аргументов.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let path = PathBuf::from(flag(&args, "--config").unwrap_or_else(|| "gateway.toml".into()));
            let cfg = config::Config::load(&path)?;
            let listen = cfg.listen.clone();
            println!(
                "llm-gateway: {} пользователей, модель {} через {}, контекст {} токенов, слотов {}",
                cfg.users.len(),
                cfg.model,
                cfg.backend,
                cfg.max_context_tokens,
                cfg.max_concurrent
            );
            let app = proxy::App::new(cfg)?;
            let listener = tokio::net::TcpListener::bind(&listen)
                .await
                .with_context(|| format!("не удалось занять {listen}"))?;
            println!("Шлюз запущен: http://{listen}");
            axum::serve(listener, proxy::router(app)).await?;
            Ok(())
        }
        Some("loadtest") => {
            let url = flag(&args, "--url").context("нужен --url")?;
            let keys: Vec<String> = flag(&args, "--keys")
                .context("нужен --keys")?
                .split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect();
            let requests = flag(&args, "--requests").map_or(Ok(3), |v| v.parse()).context("--requests")?;
            let max_tokens = flag(&args, "--max-tokens").map_or(Ok(64), |v| v.parse()).context("--max-tokens")?;
            loadtest::run(url.trim_end_matches('/'), &keys, requests, max_tokens).await
        }
        _ => bail!("{USAGE}"),
    }
}
