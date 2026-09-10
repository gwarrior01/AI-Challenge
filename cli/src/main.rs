//! Терминальный агент: отправляет запрос в LLM через API и печатает ответ в консоль.
//!
//! Два режима работы:
//!   1) Прямой разовый/интерактивный запрос к LLM (без именованного агента) — как раньше.
//!   2) Управление именованными агентами: у каждого своя конфигурация (системный промпт,
//!      модель, лимиты генерации, показывать ли токены) и жизненный цикл — агента можно
//!      добавить, запустить и остановить. Запрос, пока агент остановлен, отклоняется.
//!
//! Использование:
//!   llm-cli "запрос"                  -- разовый запрос напрямую к LLM
//!   llm-cli                           -- интерактивный REPL напрямую к LLM
//!   llm-cli agent list                -- список агентов и их статус
//!   llm-cli agent add <имя> [флаги]   -- добавить агента (по умолчанию остановлен)
//!   llm-cli agent remove <имя>        -- удалить агента
//!   llm-cli agent start <имя>         -- запустить агента и открыть чат с ним
//!   llm-cli agent stop <имя>          -- остановить агента
//!
//! Флаги `agent add`:
//!   --system TEXT            системный промпт
//!   --model NAME              модель (иначе используется LLM_MODEL)
//!   --show-tokens             выводить расход токенов в ответах
//!   --max-tokens N
//!   --temperature N
//!   --top-p N
//!   --reasoning on|off
//!   --compress                включить управление контекстом: последние сообщения
//!                              отправляются как есть, остальное периодически сжимается
//!                              той же моделью в сводку (см. AgentConfig::context_compression)
//!
//! Обязательные переменные окружения: LLM_API_URL, LLM_API_KEY (см. .env.example).
//! Реестр агентов и история их диалогов хранятся в SQLite-файле AGENTS_STORE_PATH
//! (по умолчанию agents.db) — после перезапуска агент помнит прошлые сообщения.

use anyhow::{anyhow, bail, Result};
use llm_core::{Agent, AgentConfig, AgentManager, LlmClient};
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(String::as_str) == Some("agent") {
        return run_agent_cli(&args[1..]).await;
    }

    let client = LlmClient::from_env()?;

    if !args.is_empty() {
        let prompt = args.join(" ");
        let completion = client.ask(&prompt).await?;
        println!("{}", completion.content);
        if let Some(usage) = completion.usage {
            println!("{}", format_usage(&usage));
        }
        return Ok(());
    }

    println!("Challenger (CLI). Введите запрос и нажмите Enter. Ctrl+D для выхода.");
    println!("Подсказка: `llm-cli agent list` — управление именованными агентами.\n");
    let stdin = io::stdin();
    let mut session_tokens: u64 = 0;
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF (Ctrl+D)
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        match client.ask(prompt).await {
            Ok(completion) => {
                println!("{}", completion.content);
                if let Some(usage) = completion.usage {
                    session_tokens += usage.total_tokens as u64;
                    println!("{}  ·  всего за сессию: {session_tokens}", format_usage(&usage));
                }
                println!();
            }
            Err(err) => eprintln!("Ошибка: {err:#}\n"),
        }
    }

    Ok(())
}

fn format_usage(usage: &llm_core::Usage) -> String {
    format!(
        "[токены: запрос {} + ответ {} = всего {}]",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    )
}

/// Разбирает и выполняет подкоманды `llm-cli agent ...`.
async fn run_agent_cli(args: &[String]) -> Result<()> {
    let client = LlmClient::from_env()?;
    let manager = AgentManager::from_env(client)?;

    match args.first().map(String::as_str) {
        Some("list") => {
            let agents = manager.list();
            if agents.is_empty() {
                println!("Агентов пока нет. Добавьте: llm-cli agent add <имя> [флаги]");
            } else {
                for info in agents {
                    let status = if info.running { "запущен" } else { "остановлен" };
                    let model = info.config.model.as_deref().unwrap_or("(модель по умолчанию)");
                    let tokens = if info.config.show_tokens { "показывать" } else { "скрывать" };
                    let compression = if info.config.context_compression { "вкл" } else { "выкл" };
                    println!(
                        "- {} [{status}] · модель: {model} · токены: {tokens} · сжатие контекста: {compression}",
                        info.config.name
                    );
                }
            }
            Ok(())
        }
        Some("add") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent add <имя> [флаги]"))?;
            let config = parse_agent_add_flags(name, &args[2..])?;
            let info = manager.create(config)?;
            println!(
                "Агент «{}» добавлен (остановлен). Запустите: llm-cli agent start {}",
                info.config.name, info.config.name
            );
            Ok(())
        }
        Some("remove") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent remove <имя>"))?;
            manager.remove(&name)?;
            println!("Агент «{name}» удалён.");
            Ok(())
        }
        Some("start") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent start <имя>"))?;
            manager.start(&name)?;
            let agent = manager.get(&name).expect("агент только что запущен");
            println!("Агент «{name}» запущен. Введите запрос, `stop` или Ctrl+D — остановить и выйти.\n");
            print_restored_history(&agent);
            run_agent_chat(&manager, agent).await
        }
        Some("stop") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent stop <имя>"))?;
            manager.stop(&name)?;
            println!("Агент «{name}» остановлен.");
            Ok(())
        }
        _ => {
            print_agent_usage();
            Ok(())
        }
    }
}

fn print_agent_usage() {
    println!(
        "Использование:\n\
         \x20 llm-cli agent list\n\
         \x20 llm-cli agent add <имя> [--system TEXT] [--model NAME] [--show-tokens]\n\
         \x20                        [--max-tokens N] [--temperature N] [--top-p N] [--reasoning on|off]\n\
         \x20                        [--compress]\n\
         \x20 llm-cli agent remove <имя>\n\
         \x20 llm-cli agent start <имя>\n\
         \x20 llm-cli agent stop <имя>"
    );
}

fn parse_agent_add_flags(name: String, flags: &[String]) -> Result<AgentConfig> {
    let mut config = AgentConfig::new(name);
    let mut i = 0;
    while i < flags.len() {
        match flags[i].as_str() {
            "--system" => {
                i += 1;
                let value = flags.get(i).ok_or_else(|| anyhow!("--system требует значение"))?;
                config.system_prompt = Some(value.clone());
            }
            "--model" => {
                i += 1;
                let value = flags.get(i).ok_or_else(|| anyhow!("--model требует значение"))?;
                config.model = Some(value.clone());
            }
            "--show-tokens" => {
                config.show_tokens = true;
            }
            "--compress" => {
                config.context_compression = true;
            }
            "--max-tokens" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--max-tokens требует значение"))?;
                config.max_tokens =
                    Some(raw.parse().map_err(|_| anyhow!("--max-tokens должно быть целым числом"))?);
            }
            "--temperature" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--temperature требует значение"))?;
                config.temperature =
                    Some(raw.parse().map_err(|_| anyhow!("--temperature должно быть числом"))?);
            }
            "--top-p" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--top-p требует значение"))?;
                config.top_p = Some(raw.parse().map_err(|_| anyhow!("--top-p должно быть числом"))?);
            }
            "--reasoning" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--reasoning требует значение on|off"))?;
                config.reasoning = match raw.as_str() {
                    "on" => Some(true),
                    "off" => Some(false),
                    _ => bail!("--reasoning принимает on или off"),
                };
            }
            other => bail!("неизвестный флаг: {other}"),
        }
        i += 1;
    }
    Ok(config)
}

/// Печатает восстановленную из БД историю диалога (если она не пуста) —
/// например, после перезапуска приложения, когда агент помнит прошлые
/// сообщения и продолжает разговор, как будто его не выключали.
fn print_restored_history(agent: &Agent) {
    let history = agent.history();
    if history.is_empty() {
        return;
    }
    println!("— восстановлена история диалога ({} сообщений) —", history.len());
    for message in &history {
        let label = match message.role.as_str() {
            "user" => "Вы",
            "assistant" => agent.name(),
            other => other,
        };
        println!("[{label}] {}", message.content);
    }
    println!();
}

/// Интерактивный чат с конкретным запущенным агентом. Завершается по Ctrl+D или
/// командам `stop`/`exit`, при этом агент останавливается и состояние сохраняется.
async fn run_agent_chat(manager: &AgentManager, agent: Arc<Agent>) -> Result<()> {
    let stdin = io::stdin();
    let mut session_tokens: u64 = 0;
    let show_tokens = agent.config().show_tokens;

    loop {
        print!("[{}] > ", agent.name());
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF (Ctrl+D)
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("stop") || prompt.eq_ignore_ascii_case("exit") {
            break;
        }

        match agent.handle_request(prompt).await {
            Ok(reply) => {
                println!("{}", reply.text);
                if let Some(usage) = reply.usage {
                    session_tokens += usage.total_tokens as u64;
                    if show_tokens {
                        println!("[всего за сессию: {session_tokens} ток.]");
                    }
                }
                println!();
            }
            Err(err) => eprintln!("Ошибка: {err:#}\n"),
        }
    }

    manager.stop(agent.name())?;
    println!("Агент «{}» остановлен.", agent.name());
    Ok(())
}
