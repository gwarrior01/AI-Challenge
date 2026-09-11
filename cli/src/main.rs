//! Терминальный агент: отправляет запрос в LLM через API и печатает ответ в консоль.
//!
//! Два режима работы:
//!   1) Прямой разовый/интерактивный запрос к LLM (без именованного агента) — как раньше.
//!   2) Управление именованными агентами: у каждого своя конфигурация (системный промпт,
//!      модель, лимиты генерации, показывать ли токены, стратегия управления контекстом)
//!      и жизненный цикл — агента можно добавить, запустить и остановить. Запрос, пока
//!      агент остановлен, отклоняется.
//!
//! Использование:
//!   llm-cli "запрос"                  -- разовый запрос напрямую к LLM
//!   llm-cli                           -- интерактивный REPL напрямую к LLM
//!   llm-cli agent list                -- список агентов и их статус
//!   llm-cli agent add <имя> [флаги]   -- добавить агента (по умолчанию остановлен)
//!   llm-cli agent remove <имя>        -- удалить агента
//!   llm-cli agent start <имя>         -- запустить агента и открыть чат с ним
//!   llm-cli agent stop <имя>          -- остановить агента
//!   llm-cli agent strategy <имя> <стратегия>       -- сменить стратегию контекста на лету
//!   llm-cli agent checkpoint <имя> <метка>         -- отметить точку ветвления (strategy branching)
//!   llm-cli agent branch <имя> <новая-ветка> [--from <метка>]  -- ответвить новую ветку
//!   llm-cli agent switch <имя> <ветка>             -- переключиться на другую ветку
//!   llm-cli agent branches <имя>                   -- список веток и checkpoint'ов
//!
//! ## Стратегии управления контекстом (`--context-strategy`)
//!   full             -- без управления: вся история в каждом запросе (по умолчанию)
//!   summary          -- сжатие устаревшей части истории в сводку той же моделью
//!   sliding-window   -- только последние N сообщений пользователя, остальное отбрасывается
//!   facts            -- sticky facts (ключ-значение) + последние N сообщений
//!   branching        -- ветки диалога с checkpoint'ами (см. agent checkpoint/branch/switch)
//! N берётся из LLM_SLIDING_WINDOW_SIZE (по умолчанию 6) или флага --window-size у агента.
//!
//! Флаги `agent add`:
//!   --system TEXT              системный промпт
//!   --model NAME                модель (иначе используется LLM_MODEL)
//!   --show-tokens               выводить расход токенов в ответах
//!   --max-tokens N
//!   --temperature N
//!   --top-p N
//!   --reasoning on|off
//!   --context-strategy STRAT    full|summary|sliding-window|facts|branching (см. выше)
//!   --window-size N              переопределить размер окна для sliding-window/facts
//!   --compress                   устаревший алиас --context-strategy summary
//!
//! Обязательные переменные окружения: LLM_API_URL, LLM_API_KEY (см. .env.example).
//! Реестр агентов и история их диалогов хранятся в SQLite-файле AGENTS_STORE_PATH
//! (по умолчанию agents.db) — после перезапуска агент помнит прошлые сообщения.

use anyhow::{anyhow, bail, Result};
use llm_core::{Agent, AgentConfig, AgentManager, ContextStrategy, LlmClient};
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
                    println!(
                        "- {} [{status}] · модель: {model} · токены: {tokens} · стратегия контекста: {}",
                        info.config.name, info.config.context_strategy
                    );
                    if let Some(branching) = &info.branching {
                        println!(
                            "    ветка: {} · веток всего: {} · checkpoint'ов: {}",
                            branching.current_branch,
                            branching.branches.len(),
                            branching.checkpoints.len()
                        );
                    }
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
        Some("strategy") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent strategy <имя> <стратегия>"))?;
            let raw = args
                .get(2)
                .cloned()
                .ok_or_else(|| anyhow!("укажите стратегию: full|summary|sliding-window|facts|branching"))?;
            let strategy: ContextStrategy = raw.parse()?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            let mut config = agent.config();
            config.context_strategy = strategy;
            agent.set_config(config)?;
            println!("Стратегия контекста агента «{name}» переключена на «{strategy}».");
            Ok(())
        }
        Some("checkpoint") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent checkpoint <имя> <метка>"))?;
            let label =
                args.get(2).cloned().ok_or_else(|| anyhow!("укажите метку checkpoint'а"))?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            agent.checkpoint(&label)?;
            println!("Checkpoint «{label}» сохранён (ветка «{}»).", agent.current_branch());
            Ok(())
        }
        Some("branch") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent branch <имя> <новая-ветка> [--from <метка>]"))?;
            let new_branch = args.get(2).cloned().ok_or_else(|| anyhow!("укажите имя новой ветки"))?;
            let mut from_checkpoint: Option<String> = None;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--from" {
                    i += 1;
                    from_checkpoint =
                        Some(args.get(i).cloned().ok_or_else(|| anyhow!("--from требует значение"))?);
                } else {
                    bail!("неизвестный флаг: {}", args[i]);
                }
                i += 1;
            }
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            agent.branch_from(from_checkpoint.as_deref(), &new_branch)?;
            match &from_checkpoint {
                Some(cp) => println!("Ветка «{new_branch}» создана от checkpoint'а «{cp}»."),
                None => println!("Ветка «{new_branch}» создана от текущего конца ветки «{}».", agent.current_branch()),
            }
            println!("Переключиться на неё: llm-cli agent switch {name} {new_branch}");
            Ok(())
        }
        Some("switch") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent switch <имя> <ветка>"))?;
            let branch = args.get(2).cloned().ok_or_else(|| anyhow!("укажите имя ветки"))?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            agent.switch_branch(&branch)?;
            println!("Агент «{name}» переключён на ветку «{branch}».");
            Ok(())
        }
        Some("branches") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent branches <имя>"))?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            let current = agent.current_branch();
            println!("Текущая ветка: {current}");
            println!("Ветки:");
            for branch in agent.list_branches() {
                let marker = if branch == current { "*" } else { " " };
                println!("  {marker} {branch}");
            }
            let checkpoints = agent.list_checkpoints();
            if checkpoints.is_empty() {
                println!("Checkpoint'ов пока нет.");
            } else {
                println!("Checkpoint'ы:");
                for cp in checkpoints {
                    println!("  - {cp}");
                }
            }
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
         \x20                        [--context-strategy full|summary|sliding-window|facts|branching]\n\
         \x20                        [--window-size N] [--compress]\n\
         \x20 llm-cli agent remove <имя>\n\
         \x20 llm-cli agent start <имя>\n\
         \x20 llm-cli agent stop <имя>\n\
         \x20 llm-cli agent strategy <имя> <стратегия>\n\
         \x20 llm-cli agent checkpoint <имя> <метка>                    (стратегия branching)\n\
         \x20 llm-cli agent branch <имя> <новая-ветка> [--from <метка>] (стратегия branching)\n\
         \x20 llm-cli agent switch <имя> <ветка>                        (стратегия branching)\n\
         \x20 llm-cli agent branches <имя>                              (стратегия branching)"
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
                config.context_strategy = ContextStrategy::Summary;
            }
            "--context-strategy" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--context-strategy требует значение"))?;
                config.context_strategy = raw.parse()?;
            }
            "--window-size" => {
                i += 1;
                let raw = flags.get(i).ok_or_else(|| anyhow!("--window-size требует значение"))?;
                config.window_size =
                    Some(raw.parse().map_err(|_| anyhow!("--window-size должно быть целым числом"))?);
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

    // Стоимость диалога (см. llm_core::pricing) — в отличие от session_tokens
    // выше (который считает только запросы этого запуска CLI), она сразу
    // включает и восстановленную из SQLite историю, поэтому число с первого же
    // сообщения отражает, сколько уже стоил весь диалог с этим агентом, а не
    // только эта сессия. Печатается, только если для ХОТЯ БЫ ОДНОГО сообщения
    // стоимость удалось определить (реальная от провайдера или оценка по
    // ставкам LLM_PRICE_INPUT_PER_1M/LLM_PRICE_OUTPUT_PER_1M) — иначе печатать
    // нечего. dialogue_cost_approx — true, если хоть один вклад в сумму был
    // оценкой, а не реальной стоимостью от провайдера: тогда итог помечается "≈".
    let currency = llm_core::pricing::currency();
    let mut dialogue_cost: f64 = 0.0;
    let mut dialogue_cost_known = false;
    let mut dialogue_cost_approx = false;
    for (_, usage) in agent.history_with_usage() {
        let Some(usage) = usage else { continue };
        if let Some(total) = llm_core::pricing::resolve_total(&usage) {
            dialogue_cost += total;
            dialogue_cost_known = true;
            if llm_core::pricing::source(&usage) == Some(llm_core::pricing::CostSource::Estimated) {
                dialogue_cost_approx = true;
            }
        }
    }

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
                if let Some(cost) = reply.cost {
                    dialogue_cost += cost.total;
                    dialogue_cost_known = true;
                    let request_approx = cost.source == llm_core::pricing::CostSource::Estimated;
                    if request_approx {
                        dialogue_cost_approx = true;
                    }
                    if show_tokens {
                        let request_mark = if request_approx { "≈" } else { "" };
                        let dialogue_mark = if dialogue_cost_approx { "≈" } else { "" };
                        println!(
                            "[💰 запрос: {request_mark}{currency}{:.8} · весь диалог: {dialogue_mark}{currency}{dialogue_cost:.8}]",
                            cost.total
                        );
                    }
                } else if dialogue_cost_known && show_tokens {
                    // Стоимость этого конкретного обмена определить не удалось (см.
                    // AgentReply::cost), но по диалогу в целом что-то уже накоплено —
                    // печатаем хотя бы итог, чтобы счётчик не пропадал молча.
                    let dialogue_mark = if dialogue_cost_approx { "≈" } else { "" };
                    println!("[💰 весь диалог: {dialogue_mark}{currency}{dialogue_cost:.8}]");
                }
                if let (true, Some(n)) = (reply.summarized, reply.summary_covers) {
                    println!("[🗜️ сводка контекста пересчитана — покрывает {n} сообщений]");
                }
                if reply.facts_updated {
                    if let Some(facts) = agent.info().facts {
                        if facts.facts.is_empty() {
                            println!("[📌 facts: (пусто)]");
                        } else {
                            let rendered: Vec<String> =
                                facts.facts.iter().map(|(k, v)| format!("{k}={v}")).collect();
                            println!("[📌 facts обновлены: {}]", rendered.join(", "));
                        }
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
