//! file-assistant — ассистент, который сам работает с файлами проекта: ищет,
//! читает, анализирует, создаёт и правит их по цели, поставленной словами.
//!
//! ```text
//! file-assistant                                  — TUI: цель словами, /scenario, /diff, /apply, /discard
//! file-assistant run <цель…> [--apply]            — одна задача без интерфейса
//! file-assistant run --scenario <имя> [--apply]   — готовая цель из files/scenarios
//! file-assistant scenarios                        — список сценариев
//! ```
//!
//! Файлы — через свой MCP-сервер `fs` (`mcp/fs-mcp`), история — через `git`
//! (`mcp/git-mcp`); конфигурация — `files/mcp.json`. Правки модели копятся в
//! оверлее сервера и попадают на диск только по `/apply` или `--apply`;
//! каждый запуск сохраняется в `files/runs/<id>/` вместе с `changes.patch`.
//!
//! Коды возврата: 0 — готово, 1 — при `--apply` были конфликты, 2 — ошибка.

mod runs;
mod scenarios;
mod session;
mod task;
mod tui;

use std::io::Write;
use std::process::ExitCode;

use anyhow::{bail, Result};

use session::Session;
use task::Step;

const USAGE: &str = "file-assistant — ассистент, который сам работает с файлами проекта

  file-assistant                                  TUI (команды — /? внутри)
  file-assistant run <цель…> [флаги]              одна задача без интерфейса
  file-assistant run --scenario <имя> [флаги]     готовая цель из files/scenarios
  file-assistant scenarios                        список сценариев

Флаги run:
  --apply          записать изменения на диск (по умолчанию — только дифф и changes.patch)
  --out <каталог>  куда сохранить запуск (по умолчанию files/runs)
  --quiet          не печатать вызовы инструментов в stderr

Результат: отчёт и дифф — в stdout, запуск — в files/runs/<id>/ (goal.md, report.md,
changes.patch, tools.json, run.json); без --apply изменения применяются так:
  git apply files/runs/<id>/changes.patch
Коды возврата: 0 — готово, 1 — при --apply были конфликты, 2 — ошибка.
Настройки — переменные окружения LLM_* и FILES_* (см. docs/day-34-file-assistant.md).";

#[tokio::main]
async fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()).await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("Ошибка: {err:#}");
            ExitCode::from(2)
        }
    }
}

async fn run(args: Vec<String>) -> Result<ExitCode> {
    match args.first().map(String::as_str) {
        None => {
            tui::run(Session::from_env()?).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Some("scenarios") => {
            println!("Сценарии ({}):\n{}", scenarios::dir().display(), scenarios::listing(&scenarios::list()?));
            Ok(ExitCode::SUCCESS)
        }
        Some("run") => cli_run(RunArgs::parse(&args[1..])?).await,
        Some(other) => bail!("неизвестная команда «{other}»\n\n{USAGE}"),
    }
}

#[derive(Debug, Default, PartialEq)]
struct RunArgs {
    goal: Vec<String>,
    scenario: Option<String>,
    apply: bool,
    out: Option<String>,
    quiet: bool,
}

impl RunArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut r = RunArgs::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--apply" => r.apply = true,
                "--quiet" | "-q" => r.quiet = true,
                "--scenario" | "-s" => match it.next() {
                    Some(name) => r.scenario = Some(name.clone()),
                    None => bail!("--scenario — нужно имя сценария (file-assistant scenarios)"),
                },
                "--out" => match it.next() {
                    Some(dir) => r.out = Some(dir.clone()),
                    None => bail!("--out — нужен каталог"),
                },
                flag if flag.starts_with("--") => bail!("неизвестный флаг {flag}\n\n{USAGE}"),
                word => r.goal.push(word.to_string()),
            }
        }
        match (&r.scenario, r.goal.is_empty()) {
            (Some(_), false) => bail!("либо цель словами, либо --scenario — не оба сразу"),
            (None, true) => bail!("нужна цель: file-assistant run <цель…> или --scenario <имя>"),
            _ => Ok(r),
        }
    }
}

async fn cli_run(args: RunArgs) -> Result<ExitCode> {
    let (goal, scenario) = match &args.scenario {
        Some(name) => (scenarios::find(name)?.goal, Some(name.as_str())),
        None => (args.goal.join(" "), None),
    };
    let s = Session::from_env()?;
    let quiet = args.quiet;
    let progress = move |step: Step| {
        if quiet {
            return;
        }
        match step {
            Step::Status(_) => {}
            Step::Tool(call) => {
                let mark = if call.is_error { "✗" } else { "✓" };
                eprintln!(
                    "{mark} {} › {}({})  ↳ {}",
                    call.server.as_deref().unwrap_or("?"),
                    call.tool,
                    session::short(&call.arguments, 100),
                    call.result_preview(120)
                );
            }
        }
    };
    let outcome = task::run(&s, &goal, scenario, &progress).await?;
    let root = args.out.map(Into::into).unwrap_or_else(runs::runs_dir);
    let dir = runs::save(&root, &outcome)?;

    let mut out = std::io::stdout().lock();
    writeln!(out, "{}", outcome.report)?;
    for note in &outcome.notes {
        writeln!(out, "\n⚠ {note}")?;
    }
    if outcome.pending.is_empty() {
        writeln!(out, "\nИзменений в файлах нет. Запуск сохранён: {}", dir.display())?;
        return Ok(ExitCode::SUCCESS);
    }
    writeln!(out, "\nИзменения — {}:\n{}\n\n{}", outcome.pending.summary(), outcome.pending.listing(), outcome.pending.patch.trim_end())?;
    writeln!(out, "\nЗапуск сохранён: {}", dir.display())?;
    if !args.apply {
        writeln!(out, "На диск не записано. Применить: git apply {}", dir.join("changes.patch").display())?;
        return Ok(ExitCode::SUCCESS);
    }
    let applied = s.apply(&[], &mut Vec::new()).await.map_err(anyhow::Error::msg)?;
    runs::mark_applied(&dir, &applied)?;
    writeln!(out, "Записано: {}", if applied.written.is_empty() { "ничего".into() } else { applied.written.join(", ") })?;
    for c in &applied.conflicts {
        writeln!(out, "Конфликт: {} — {}", c.path, c.reason)?;
    }
    Ok(if applied.conflicts.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<RunArgs> {
        RunArgs::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn run_arguments() {
        let r = parse(&["найди", "McpManager", "--apply", "-q"]).unwrap();
        assert_eq!(r.goal, vec!["найди", "McpManager"]);
        assert!(r.apply && r.quiet);
        let r = parse(&["--scenario", "changelog", "--out", "/tmp/runs"]).unwrap();
        assert_eq!((r.scenario.as_deref(), r.out.as_deref()), (Some("changelog"), Some("/tmp/runs")));
        assert!(parse(&[]).is_err());
        assert!(parse(&["--scenario", "x", "цель"]).is_err());
        assert!(parse(&["--scenario"]).is_err());
        assert!(parse(&["цель", "--force"]).is_err());
    }
}
