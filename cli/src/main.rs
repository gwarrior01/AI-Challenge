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
//!   llm-cli agent memory <имя>                     -- показать все три уровня памяти
//!   llm-cli agent remember <имя> <ключ> <значение> [--category CAT]  -- долговременная память (общая для всех)
//!   llm-cli agent forget <имя> <ключ>                                -- удалить запись (общей) долговременной памяти
//!   llm-cli agent task <имя> start <название> [--goal TEXT]  -- создать общую задачу и присоединиться
//!   llm-cli agent task <имя> join <название>                 -- присоединиться к чужой общей задаче
//!   llm-cli agent task <имя> set <ключ> <значение>            -- записать данные (видно всем участникам)
//!   llm-cli agent task <имя> show                             -- показать текущую задачу (в т.ч. состояние автомата)
//!   llm-cli agent task <имя> advance <этап> [--step T] [--expect T]  -- перейти на следующий легальный этап
//!   llm-cli agent task <имя> step <текст>                     -- обновить текущий шаг, не меняя этап
//!   llm-cli agent task <имя> expect <текст>                   -- обновить ожидаемое действие, не меняя этап
//!   llm-cli agent task <имя> pause                            -- поставить задачу на паузу (на любом этапе)
//!   llm-cli agent task <имя> resume                           -- снять паузу (без повторных объяснений агенту)
//!   llm-cli agent task <имя> approve                          -- применить переход, предложенный моделью
//!   llm-cli agent task <имя> reject <причина>                 -- отклонить предложенный моделью переход
//!   llm-cli agent task <имя> finish                           -- завершить задачу ДЛЯ ВСЕХ участников
//!   llm-cli agent task <имя> invariant set <id> <текст>       -- инвариант ТОЛЬКО этой задачи
//!   llm-cli agent task <имя> invariant remove <id>            -- снять инвариант этой задачи
//!   llm-cli agent task <имя> forbid <из> <в>                  -- запретить переход для этой задачи (абсолютно)
//!   llm-cli agent task <имя> allow <из> <в>                   -- снять запрет перехода
//!   llm-cli agent task <имя> require-approval <из> <в>        -- доп. гейт согласия (для перехода МОДЕЛИ)
//!   llm-cli agent task <имя> unrequire-approval <из> <в>      -- снять доп. гейт согласия
//!   llm-cli agent tasks                                       -- список всех общих задач и их участников
//!   llm-cli agent profile <имя>                    -- показать текущий профиль персонализации агента
//!   llm-cli agent profile <имя> <профиль|none>     -- сменить профиль (или отключить персонализацию)
//!   llm-cli agent profiles                         -- список всех профилей из каталога профилей
//!   llm-cli agent profiles create <профиль>        -- создать новый профиль (пустой шаблон) на диске
//!   llm-cli agent invariants                       -- список жёстких инвариантов (общих для всех агентов)
//!   llm-cli agent invariants show <id>             -- показать текст одного инварианта
//!   llm-cli agent invariants create <id>           -- создать новый инвариант (пустой шаблон) на диске
//!   llm-cli agent invariants remove <id>           -- удалить инвариант (единственный способ его снять)
//!
//! ## Инварианты (см. llm_core::invariants)
//!   Три источника, каждый со своим масштабом действия:
//!   - Глобальные-файлы: markdown в каталоге инвариантов (по умолчанию
//!     `invariants/`, переопределяется `LLM_INVARIANTS_DIR`) — архитектура,
//!     технические решения, ограничения по стеку, бизнес-правила.
//!   - Глобальные-память: записи `agent remember <ключ> <текст> --category
//!     invariant` — та же сила действия, без отдельного файла.
//!   - Задачи: `agent task <имя> invariant set/remove` (текстовые) и
//!     `agent task <имя> forbid/allow/require-approval/unrequire-approval`
//!     (структурные — реально проверяются кодом, не только промптом) —
//!     действуют ТОЛЬКО пока агент работает над этой конкретной задачей.
//!
//!   Оба глобальных источника подключаются ПЕРВЫМ системным сообщением к
//!   каждому запросу — модель обязана отказываться от того, что их нарушает,
//!   называя нарушенный инвариант и причину; задачные — часть блока задачи.
//!
//! ## Персонализация (см. llm_core::profile)
//!   Отдельная от памяти ось: markdown-файл в каталоге профилей (по умолчанию
//!   `profiles/`, переопределяется `LLM_PROFILES_DIR`) описывает манеру
//!   общения, язык ответа, формат и ограничения — подключается к КАЖДОМУ
//!   запросу агента независимо от context-strategy. `AgentConfig::profile`
//!   хранит только ИМЯ файла (без `.md`) — не заданное значение разрешается в
//!   профиль `default`, значение `none` явно отключает персонализацию.
//!
//! ## Модель памяти агента (см. llm_core::memory)
//!   краткосрочная -- текущий диалог (history/branches выше, управляется --context-strategy)
//!   рабочая        -- данные ОБЩЕЙ ЗАДАЧИ: один агент создаёт её (task start), другие
//!                      присоединяются по имени (task join) и делят её данные (task set) —
//!                      завершение (task finish) удаляет задачу и её данные для ВСЕХ участников разом
//!   долговременная -- решения/знания, ОДНА НА ВСЁ ПРИЛОЖЕНИЕ (agent remember/forget):
//!                      правка через одного агента сразу видна через любого другого,
//!                      независимо от задачи, живёт, пока не удалят явно
//! Все три хранятся раздельно (разные таблицы SQLite) и заполняются только явно —
//! никакого автоматического попадания данных не по адресу.
//!
//! ## Конечный автомат задачи (Task State Machine, см. llm_core::memory::Stage)
//!   Формализует рабочую память задачи выше: этап (planning -> execution -> validation
//!   -> done, с откатом validation -> execution) + текущий шаг + ожидаемое действие +
//!   независимая от этапа пауза (agent task <имя> pause/resume). Всё это попадает в
//!   каждый запрос к LLM (как и остальная рабочая память) — поэтому resume не требует
//!   заново объяснять агенту контекст: он читает его из БД, а не из истории диалога.
//!
//!   Модель двигает автомат САМА — вызовом инструментов (function calling)
//!   move_stage/update_step, которые ей предлагаются, пока есть активная
//!   задача, не на паузе и не ждущая утверждения. Программа проверяет
//!   предложенный переход по карте (Stage::allowed_next) и применяет его —
//!   КРОМЕ переходов planning->execution и validation->done: они требуют
//!   утверждения человеком (Stage::requires_approval_to) и остаются в
//!   ожидании (agent task <имя> approve/reject), пока их явно не откроют.
//!   Именно это не даёт агенту решить задачу целиком в первом же ответе, минуя
//!   этап планирования: пока модель не подтвердила план и не дождалась
//!   approve, она вообще не может перейти к исполнению.
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
//!   --profile NAME               профиль персонализации (иначе используется "default")
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
        Some("memory") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent memory <имя>"))?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            print_memory_overview(&agent);
            Ok(())
        }
        Some("remember") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent remember <имя> <ключ> <значение> [--category CAT]"))?;
            let key = args.get(2).cloned().ok_or_else(|| anyhow!("укажите ключ"))?;
            let value = args.get(3).cloned().ok_or_else(|| anyhow!("укажите значение"))?;
            let mut category = "knowledge".to_string();
            let mut i = 4;
            while i < args.len() {
                if args[i] == "--category" {
                    i += 1;
                    category = args.get(i).cloned().ok_or_else(|| anyhow!("--category требует значение"))?;
                } else {
                    bail!("неизвестный флаг: {}", args[i]);
                }
                i += 1;
            }
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            agent.remember(&key, &value, &category)?;
            println!("Долговременная память (общая для всех агентов) обновлена: [{category}] {key} = {value}");
            Ok(())
        }
        Some("forget") => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("укажите имя агента: llm-cli agent forget <имя> <ключ>"))?;
            let key = args.get(2).cloned().ok_or_else(|| anyhow!("укажите ключ"))?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            if agent.forget(&key)? {
                println!("Запись «{key}» удалена из долговременной памяти (общей для всех агентов).");
            } else {
                println!("В долговременной памяти нет записи «{key}».");
            }
            Ok(())
        }
        Some("task") => {
            let name = args.get(1).cloned().ok_or_else(|| {
                anyhow!("укажите имя агента: llm-cli agent task <имя> <start|join|set|show|finish> ...")
            })?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            match args.get(2).map(String::as_str) {
                Some("start") => {
                    let task_name = args
                        .get(3)
                        .cloned()
                        .ok_or_else(|| anyhow!("укажите название задачи: llm-cli agent task {name} start <название> [--goal TEXT]"))?;
                    let mut goal: Option<String> = None;
                    let mut i = 4;
                    while i < args.len() {
                        if args[i] == "--goal" {
                            i += 1;
                            goal = Some(args.get(i).cloned().ok_or_else(|| anyhow!("--goal требует значение"))?);
                        } else {
                            bail!("неизвестный флаг: {}", args[i]);
                        }
                        i += 1;
                    }
                    agent.task_start(&task_name, goal.as_deref())?;
                    println!("Задача «{task_name}» создана и агент «{name}» присоединён к ней.");
                    Ok(())
                }
                Some("join") => {
                    let task_name = args
                        .get(3)
                        .cloned()
                        .ok_or_else(|| anyhow!("укажите название задачи: llm-cli agent task {name} join <название>"))?;
                    agent.task_join(&task_name)?;
                    println!(
                        "Агент «{name}» присоединён к задаче «{task_name}» — видит и меняет её рабочую память \
                         наравне с остальными участниками."
                    );
                    Ok(())
                }
                Some("set") => {
                    let key = args
                        .get(3)
                        .cloned()
                        .ok_or_else(|| anyhow!("укажите ключ: llm-cli agent task {name} set <ключ> <значение>"))?;
                    let value = args.get(4).cloned().ok_or_else(|| anyhow!("укажите значение"))?;
                    agent.task_set(&key, &value)?;
                    println!("Рабочая память задачи обновлена (видно всем участникам): {key} = {value}");
                    Ok(())
                }
                Some("show") => {
                    match agent.task_state() {
                        Some(task) => print_task(&task),
                        None => println!("У агента «{name}» сейчас нет активной задачи."),
                    }
                    Ok(())
                }
                Some("advance") => {
                    let stage_raw = args
                        .get(3)
                        .cloned()
                        .ok_or_else(|| anyhow!(
                            "укажите этап: llm-cli agent task {name} advance <planning|execution|validation|done> \
                             [--step TEXT] [--expect TEXT]"
                        ))?;
                    let stage: llm_core::Stage = stage_raw.parse()?;
                    let mut step: Option<String> = None;
                    let mut expect: Option<String> = None;
                    let mut i = 4;
                    while i < args.len() {
                        match args[i].as_str() {
                            "--step" => {
                                i += 1;
                                step = Some(args.get(i).cloned().ok_or_else(|| anyhow!("--step требует значение"))?);
                            }
                            "--expect" => {
                                i += 1;
                                expect =
                                    Some(args.get(i).cloned().ok_or_else(|| anyhow!("--expect требует значение"))?);
                            }
                            other => bail!("неизвестный флаг: {other}"),
                        }
                        i += 1;
                    }
                    agent.task_advance(stage, step.as_deref(), expect.as_deref())?;
                    println!("Задача переведена на этап «{stage}».");
                    Ok(())
                }
                Some("step") => {
                    let text = args.get(3..).map(|s| s.join(" ")).filter(|s| !s.is_empty()).ok_or_else(|| {
                        anyhow!("укажите текущий шаг: llm-cli agent task {name} step <текст>")
                    })?;
                    agent.task_step(&text)?;
                    println!("Текущий шаг задачи обновлён: {text}");
                    Ok(())
                }
                Some("expect") => {
                    let text = args.get(3..).map(|s| s.join(" ")).filter(|s| !s.is_empty()).ok_or_else(|| {
                        anyhow!("укажите ожидаемое действие: llm-cli agent task {name} expect <текст>")
                    })?;
                    agent.task_expect(&text)?;
                    println!("Ожидаемое действие задачи обновлено: {text}");
                    Ok(())
                }
                Some("pause") => {
                    agent.task_pause()?;
                    println!("Задача поставлена на паузу.");
                    Ok(())
                }
                Some("resume") => {
                    agent.task_resume()?;
                    println!("Задача возобновлена — этап/шаг/ожидаемое действие не менялись.");
                    Ok(())
                }
                Some("approve") => {
                    agent.task_approve()?;
                    println!("Предложенный моделью переход применён.");
                    Ok(())
                }
                Some("reject") => {
                    let note = args.get(3..).map(|s| s.join(" ")).unwrap_or_default();
                    agent.task_reject(&note)?;
                    println!("Предложенный моделью переход отклонён — этап не изменился.");
                    Ok(())
                }
                Some("finish") => match agent.task_finish()? {
                    Some(task) => {
                        println!(
                            "Задача «{}» завершена и удалена ДЛЯ ВСЕХ агентов, которые были к ней присоединены.",
                            task.name
                        );
                        Ok(())
                    }
                    None => {
                        println!("У агента «{name}» не было активной задачи.");
                        Ok(())
                    }
                },
                Some("invariant") => match args.get(3).map(String::as_str) {
                    Some("set") => {
                        let id = args.get(4).cloned().ok_or_else(|| {
                            anyhow!("укажите id: llm-cli agent task {name} invariant set <id> <текст>")
                        })?;
                        let text = args.get(5..).map(|s| s.join(" ")).filter(|s| !s.is_empty()).ok_or_else(
                            || anyhow!("укажите текст: llm-cli agent task {name} invariant set <id> <текст>"),
                        )?;
                        agent.task_invariant_set(&id, &text)?;
                        println!(
                            "Инвариант задачи «{id}» сохранён — обязателен наравне с глобальными, пока идёт \
                             работа над этой задачей: {text}"
                        );
                        Ok(())
                    }
                    Some("remove") => {
                        let id = args.get(4).cloned().ok_or_else(|| {
                            anyhow!("укажите id: llm-cli agent task {name} invariant remove <id>")
                        })?;
                        if agent.task_invariant_remove(&id)? {
                            println!("Инвариант задачи «{id}» удалён.");
                        } else {
                            println!("Инварианта задачи «{id}» нет.");
                        }
                        Ok(())
                    }
                    _ => bail!(
                        "укажите действие: llm-cli agent task {name} invariant <set <id> <текст>|remove <id>>"
                    ),
                },
                Some("forbid") => {
                    let (from, to) = parse_transition_pair(&name, "forbid", &args[3..])?;
                    agent.task_forbid_transition(from, to)?;
                    println!(
                        "Переход «{from}» -> «{to}» запрещён для этой задачи (абсолютно, не обойти даже \
                         вручную) — снять: llm-cli agent task {name} allow {from} {to}"
                    );
                    Ok(())
                }
                Some("allow") => {
                    let (from, to) = parse_transition_pair(&name, "allow", &args[3..])?;
                    if agent.task_allow_transition(from, to)? {
                        println!("Запрет на переход «{from}» -> «{to}» для этой задачи снят.");
                    } else {
                        println!("Переход «{from}» -> «{to}» для этой задачи и не был запрещён.");
                    }
                    Ok(())
                }
                Some("require-approval") => {
                    let (from, to) = parse_transition_pair(&name, "require-approval", &args[3..])?;
                    agent.task_require_approval(from, to)?;
                    println!(
                        "Переход «{from}» -> «{to}», предложенный МОДЕЛЬЮ, теперь для этой задачи \
                         дополнительно требует подтверждения человеком — снять: llm-cli agent task {name} \
                         unrequire-approval {from} {to}"
                    );
                    Ok(())
                }
                Some("unrequire-approval") => {
                    let (from, to) = parse_transition_pair(&name, "unrequire-approval", &args[3..])?;
                    if agent.task_unrequire_approval(from, to)? {
                        println!("Дополнительное требование подтверждения на «{from}» -> «{to}» снято.");
                    } else {
                        println!("Для перехода «{from}» -> «{to}» дополнительного требования и не было.");
                    }
                    Ok(())
                }
                _ => {
                    bail!(
                        "укажите действие: llm-cli agent task {name} \
                         <start|join|set|show|advance|step|expect|pause|resume|approve|reject|finish|\
                         invariant|forbid|allow|require-approval|unrequire-approval> ..."
                    );
                }
            }
        }
        Some("profile") => {
            let name = args.get(1).cloned().ok_or_else(|| {
                anyhow!("укажите имя агента: llm-cli agent profile <имя> [<профиль|none>]")
            })?;
            let agent = manager.get(&name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
            match args.get(2).cloned() {
                Some(new_profile) => {
                    let mut config = agent.config();
                    config.profile = Some(new_profile.clone());
                    agent.set_config(config)?;
                    if llm_core::profile::is_disabled(&new_profile) {
                        println!("Персонализация агента «{name}» отключена.");
                    } else {
                        println!("Профиль агента «{name}» переключён на «{new_profile}».");
                    }
                }
                None => print_profile_status(&agent),
            }
            Ok(())
        }
        Some("profiles") => match args.get(1).map(String::as_str) {
            Some("create") => {
                let profile_name = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| anyhow!("укажите имя профиля: llm-cli agent profiles create <имя>"))?;
                let path = llm_core::profile::profiles_dir().join(format!("{profile_name}.md"));
                llm_core::profile::create(&profile_name, &llm_core::profile::template(&profile_name))?;
                println!("Профиль «{profile_name}» создан: {}", path.display());
                println!("Подключить агенту: llm-cli agent profile <имя-агента> {profile_name}");
                Ok(())
            }
            None => {
                let profiles = llm_core::list_profiles();
                if profiles.is_empty() {
                    println!(
                        "Профилей пока нет в каталоге «{}» — создайте: llm-cli agent profiles create <имя>.",
                        llm_core::profile::profiles_dir().display()
                    );
                } else {
                    println!("Каталог профилей: {}", llm_core::profile::profiles_dir().display());
                    for profile in profiles {
                        println!("- {profile}");
                    }
                }
                Ok(())
            }
            Some(other) => bail!("неизвестное действие «{other}» — llm-cli agent profiles [create <имя>]"),
        },
        Some("invariants") => match args.get(1).map(String::as_str) {
            Some("create") => {
                let id = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| anyhow!("укажите идентификатор: llm-cli agent invariants create <id>"))?;
                let path = llm_core::invariants::invariants_dir().join(format!("{id}.md"));
                llm_core::invariants::create(&id, &llm_core::invariants::template(&id))?;
                println!("Инвариант «{id}» создан: {}", path.display());
                println!(
                    "Отредактируйте файл, чтобы описать правило — оно подключится к каждому запросу \
                     каждого агента без перезапуска."
                );
                Ok(())
            }
            Some("remove") => {
                let id = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| anyhow!("укажите идентификатор: llm-cli agent invariants remove <id>"))?;
                if llm_core::invariants::remove(&id)? {
                    println!("Инвариант «{id}» удалён — его действие снято для всех агентов.");
                } else {
                    println!("Инварианта «{id}» нет в каталоге.");
                }
                Ok(())
            }
            Some("show") => {
                let id = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| anyhow!("укажите идентификатор: llm-cli agent invariants show <id>"))?;
                match llm_core::invariants::load(&id) {
                    Some(content) => {
                        println!("=== Инвариант «{id}» ===\n\n{content}");
                        Ok(())
                    }
                    None => bail!("инвариант «{id}» не найден в каталоге инвариантов"),
                }
            }
            None => {
                print_invariants_list(Some(&manager.long_term_memory()));
                Ok(())
            }
            Some(other) => {
                bail!("неизвестное действие «{other}» — llm-cli agent invariants [show|create|remove] <id>")
            }
        },
        Some("tasks") => {
            let tasks = manager.list_tasks();
            if tasks.is_empty() {
                println!("Общих задач пока нет — создайте: llm-cli agent task <имя-агента> start <название>");
            } else {
                for task in tasks {
                    let goal = task.goal.as_deref().map(|g| format!(" (цель: {g})")).unwrap_or_default();
                    let members = if task.members.is_empty() {
                        "без участников".to_string()
                    } else {
                        format!("участники: {}", task.members.join(", "))
                    };
                    println!("- {}{goal} · {members}", task.name);
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

/// Печатает все три уровня памяти агента: краткосрочную (сводка по диалогу),
/// рабочую (текущая задача) и долговременную (решения/знания) — по
/// отдельности, чтобы граница между ними была видна и в интерфейсе, а не
/// только в хранилище.
fn print_memory_overview(agent: &Agent) {
    let config = agent.config();
    let long_term = agent.long_term_memory();
    println!("=== Память агента «{}» ===\n", agent.name());

    println!("-- Инварианты (жёсткие правила, общие для ВСЕХ агентов, приоритетнее всего остального) --");
    print_invariants_list(Some(&long_term));
    println!();

    println!("-- Персонализация (профиль, отдельная ось — не память, а КАК отвечать) --");
    print_profile_status(agent);
    println!();

    println!("-- Краткосрочная (текущий диалог) --");
    println!(
        "Сообщений в активной ветке «{}»: {} · стратегия контекста: {}",
        agent.current_branch(),
        agent.history().len(),
        config.context_strategy
    );
    println!();

    println!("-- Рабочая (данные общей задачи, к которой присоединён этот агент) --");
    match agent.task_state() {
        Some(task) => print_task(&task),
        None => println!(
            "(активной задачи нет — llm-cli agent task {} start <название> или ... join <название>)",
            agent.name()
        ),
    }
    println!();

    println!("-- Долговременная (решения, знания — общая для ВСЕХ агентов) --");
    if long_term.is_empty() {
        println!("(пусто — llm-cli agent remember {} <ключ> <значение>)", agent.name());
    } else {
        for (key, item) in long_term {
            println!("- [{}] {key} = {}", item.category, item.value);
        }
    }
}

/// Печатает статус персонализации агента: используемый профиль, найден ли он
/// на диске (и подключён ли поэтому к запросам), и какие ещё профили есть в
/// каталоге — доступно и как часть `agent memory`, и отдельно как `agent profile <имя>`.
fn print_profile_status(agent: &Agent) {
    let status = agent.profile_status();
    if !status.enabled {
        println!("Профиль: отключён явно (llm-cli agent profile {} <профиль> — включить снова).", agent.name());
    } else if status.found {
        println!("Профиль: «{}» — подключён к каждому запросу.", status.name);
    } else {
        println!(
            "Профиль: «{}» — файл не найден в каталоге профилей, персонализация сейчас не применяется.",
            status.name
        );
    }
    if status.available.is_empty() {
        println!("Доступных профилей в каталоге пока нет.");
    } else {
        println!("Доступные профили: {}", status.available.join(", "));
    }
}

/// Печатает файловые инварианты и, если передана долговременная память,
/// также инварианты без файла (категория `"invariant"`, см.
/// llm_core::invariants::from_long_term) — оба источника глобальные, общие
/// для ВСЕХ агентов, поэтому не принимает имя агента.
fn print_invariants_list(long_term: Option<&llm_core::LongTermMemory>) {
    let ids = llm_core::invariants::list_ids();
    if ids.is_empty() {
        println!(
            "Файловых инвариантов пока нет в каталоге «{}» — создайте: llm-cli agent invariants create <id>.",
            llm_core::invariants::invariants_dir().display()
        );
    } else {
        println!("Каталог инвариантов: {}", llm_core::invariants::invariants_dir().display());
        for id in ids {
            println!("- {id}");
        }
    }
    if let Some(long_term) = long_term {
        let memory_invariants = llm_core::invariants::from_long_term(long_term);
        if !memory_invariants.is_empty() {
            println!(
                "Инварианты из долговременной памяти (категория «invariant», без отдельного файла):"
            );
            for inv in memory_invariants {
                println!("- [{}] {}", inv.id, inv.content);
            }
        }
    }
}

fn print_task(task: &llm_core::TaskState) {
    match &task.goal {
        Some(goal) => println!("Задача «{}» (цель: {goal})", task.name),
        None => println!("Задача «{}»", task.name),
    }
    let pause_suffix = if task.paused { "  ⏸ НА ПАУЗЕ" } else { "" };
    println!("Этап: {}{pause_suffix}", task.stage);
    if let Some(step) = task.current_step.as_deref().filter(|s| !s.is_empty()) {
        println!("Текущий шаг: {step}");
    }
    if let Some(expect) = task.expected_action.as_deref().filter(|s| !s.is_empty()) {
        println!("Ожидаемое действие: {expect}");
    }
    if let Some(pending) = task.pending_stage {
        let outcome = task.pending_outcome.as_deref().unwrap_or("(не указан)");
        println!("⏳ Предложен переход на этап «{pending}» (итог: {outcome}) — ждёт approve/reject");
    }
    if !task.blocked_transitions.is_empty() {
        let items: Vec<String> = task.blocked_transitions.iter().map(|(f, t)| format!("{f}->{t}")).collect();
        println!("🚫 Запрещено инвариантом этой задачи: {}", items.join(", "));
    }
    if !task.extra_approval_transitions.is_empty() {
        let items: Vec<String> =
            task.extra_approval_transitions.iter().map(|(f, t)| format!("{f}->{t}")).collect();
        println!("🔒 Дополнительно требует подтверждения (для переходов модели): {}", items.join(", "));
    }
    if !task.invariants.is_empty() {
        println!("Инварианты этой задачи:");
        for (id, text) in &task.invariants {
            println!("  - [{id}] {text}");
        }
    }
    if task.data.is_empty() {
        println!("(данных пока нет)");
    } else {
        for (key, value) in &task.data {
            println!("- {key} = {value}");
        }
    }
}

/// Разбирает пару этапов `<from> <to>` из хвоста аргументов подкоманд
/// forbid/allow/require-approval/unrequire-approval — общая для всех четырёх,
/// чтобы не повторять один и тот же разбор и текст ошибки четыре раза.
fn parse_transition_pair(name: &str, action: &str, args: &[String]) -> Result<(llm_core::Stage, llm_core::Stage)> {
    let from_raw = args
        .first()
        .ok_or_else(|| anyhow!("укажите этапы: llm-cli agent task {name} {action} <из-этапа> <в-этап>"))?;
    let to_raw = args
        .get(1)
        .ok_or_else(|| anyhow!("укажите этапы: llm-cli agent task {name} {action} <из-этапа> <в-этап>"))?;
    let from: llm_core::Stage = from_raw.parse()?;
    let to: llm_core::Stage = to_raw.parse()?;
    Ok((from, to))
}

fn print_agent_usage() {
    println!(
        "Использование:\n\
         \x20 llm-cli agent list\n\
         \x20 llm-cli agent add <имя> [--system TEXT] [--model NAME] [--show-tokens]\n\
         \x20                        [--max-tokens N] [--temperature N] [--top-p N] [--reasoning on|off]\n\
         \x20                        [--context-strategy full|summary|sliding-window|facts|branching]\n\
         \x20                        [--window-size N] [--compress] [--profile NAME]\n\
         \x20 llm-cli agent remove <имя>\n\
         \x20 llm-cli agent start <имя>\n\
         \x20 llm-cli agent stop <имя>\n\
         \x20 llm-cli agent strategy <имя> <стратегия>\n\
         \x20 llm-cli agent checkpoint <имя> <метка>                    (стратегия branching)\n\
         \x20 llm-cli agent branch <имя> <новая-ветка> [--from <метка>] (стратегия branching)\n\
         \x20 llm-cli agent switch <имя> <ветка>                        (стратегия branching)\n\
         \x20 llm-cli agent branches <имя>                              (стратегия branching)\n\
         \x20 llm-cli agent memory <имя>                                (все 3 уровня памяти)\n\
         \x20 llm-cli agent remember <имя> <ключ> <значение> [--category CAT]  (долговременная память, общая)\n\
         \x20 llm-cli agent forget <имя> <ключ>                              (долговременная память, общая)\n\
         \x20 llm-cli agent task <имя> start <название> [--goal TEXT]  (создать общую задачу и присоединиться)\n\
         \x20 llm-cli agent task <имя> join <название>                (присоединиться к чужой общей задаче)\n\
         \x20 llm-cli agent task <имя> set <ключ> <значение>           (видно всем участникам задачи)\n\
         \x20 llm-cli agent task <имя> show                            (рабочая память задачи)\n\
         \x20 llm-cli agent task <имя> finish                          (завершает задачу ДЛЯ ВСЕХ участников)\n\
         \x20 llm-cli agent task <имя> invariant set <id> <текст>      (инвариант ТОЛЬКО этой задачи)\n\
         \x20 llm-cli agent task <имя> invariant remove <id>           (снять инвариант этой задачи)\n\
         \x20 llm-cli agent task <имя> forbid <из> <в>                 (запретить переход для этой задачи, абсолютно)\n\
         \x20 llm-cli agent task <имя> allow <из> <в>                  (снять запрет перехода)\n\
         \x20 llm-cli agent task <имя> require-approval <из> <в>       (доп. гейт согласия — только для переходов модели)\n\
         \x20 llm-cli agent task <имя> unrequire-approval <из> <в>     (снять доп. гейт согласия)\n\
         \x20 llm-cli agent tasks                                      (список всех общих задач и их участников)\n\
         \x20 llm-cli agent profile <имя>                              (показать текущий профиль персонализации)\n\
         \x20 llm-cli agent profile <имя> <профиль|none>               (сменить профиль / отключить персонализацию)\n\
         \x20 llm-cli agent profiles                                   (список профилей в каталоге профилей)\n\
         \x20 llm-cli agent profiles create <профиль>                  (создать новый профиль — пустой шаблон)\n\
         \x20 llm-cli agent invariants                                 (жёсткие правила, общие для ВСЕХ агентов)\n\
         \x20 llm-cli agent invariants show <id>                       (показать текст одного инварианта)\n\
         \x20 llm-cli agent invariants create <id>                     (создать новый инвариант — пустой шаблон)\n\
         \x20 llm-cli agent invariants remove <id>                     (снять инвариант — удалить его файл)\n\
         \x20 llm-cli agent remember <имя> <ключ> <текст> --category invariant  (глобальный инвариант без файла)"
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
            "--profile" => {
                i += 1;
                let value = flags.get(i).ok_or_else(|| anyhow!("--profile требует значение"))?;
                config.profile = Some(value.clone());
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
