//! media-assistant — ассистент, который обрабатывает записи встреч: видео →
//! MP3 → транскрипт → саммари. Работу делает свой MCP-сервер `media`
//! (`mcp/media-mcp`): чекпоинты, ожидание `ffmpeg`, транскрипция и LM Studio —
//! в его коде; модель ведёт процесс, спрашивает пользователя, где нужен выбор,
//! и пишет отчёт.
//!
//! ```text
//! media-assistant [каталог]                       — TUI: реплики, /dir, /status, /jobs
//! media-assistant run [каталог] [--summary yes|no] — одна обработка без интерфейса
//! ```
//!
//! Каталог записи — от корня `MEDIA_ROOT` сервера (по умолчанию корень,
//! `MEDIA_DIR`). Коды возврата: 0 — готово, 2 — ошибка.

mod dialog;
mod runs;
mod session;
mod tui;

use std::io::Write;
use std::process::ExitCode;

use anyhow::{bail, Result};

use dialog::{Dialog, Step};
use session::{env_or, Session};

const USAGE: &str = "media-assistant — обработка записей встреч: видео → MP3 → транскрипт → саммари

  media-assistant [каталог]             TUI (команды — /? внутри)
  media-assistant run [каталог] [флаги] одна обработка без интерфейса

Каталог записи — от корня MEDIA_ROOT (по умолчанию — сам корень).

Флаги run:
  --summary yes|no   ответ на вопрос о саммари заранее (иначе run остановится на вопросе)
  --quiet            не печатать вызовы инструментов и ход заданий в stderr

Результат: ответ ассистента — в stdout, диалог — в media/runs/<id>/.
Настройки — переменные окружения LLM_* и MEDIA_* (см. docs/day-35-media-assistant.md).";

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
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Some("run") => cli_run(RunArgs::parse(&args[1..])?).await,
        Some(flag) if flag.starts_with('-') => bail!("неизвестный флаг {flag}\n\n{USAGE}"),
        Some(dir) if args.len() == 1 => {
            tui::run(Session::from_env()?, dir.to_string()).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some(_) => bail!("лишние аргументы\n\n{USAGE}"),
        None => {
            tui::run(Session::from_env()?, env_or("MEDIA_DIR", ".")).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct RunArgs {
    dir: Option<String>,
    summary: Option<bool>,
    quiet: bool,
}

impl RunArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut r = RunArgs::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--quiet" | "-q" => r.quiet = true,
                "--summary" => {
                    // Основная форма — yes/no; y/n и да/нет — синонимы.
                    r.summary = match it.next().map(|s| s.to_lowercase()).as_deref() {
                        Some("yes" | "y" | "да") => Some(true),
                        Some("no" | "n" | "нет") => Some(false),
                        Some(other) => bail!("--summary yes|no, а не «{other}»"),
                        None => bail!("--summary yes|no — нужно значение"),
                    }
                }
                flag if flag.starts_with("--") => bail!("неизвестный флаг {flag}\n\n{USAGE}"),
                dir if r.dir.is_none() => r.dir = Some(dir.to_string()),
                _ => bail!("каталог записи — один\n\n{USAGE}"),
            }
        }
        Ok(r)
    }

    /// Реплика для ассистента: обработать запись, ответ на вопрос о саммари — заранее.
    fn request(&self) -> String {
        let summary = match self.summary {
            Some(true) => "На вопрос о саммари ответ уже есть: да — делай саммари, не спрашивая.",
            Some(false) => "На вопрос о саммари ответ уже есть: нет — только промпт саммари, LM Studio не вызывай.",
            None => "Если дойдёшь до вопроса о саммари — задай его и остановись.",
        };
        format!(
            "Обработай запись. {summary} Это запуск без интерфейса: если понадобится выбор файла — перечисли варианты \
             и остановись."
        )
    }
}

async fn cli_run(args: RunArgs) -> Result<ExitCode> {
    let s = Session::from_env()?;
    let dir = args.dir.clone().unwrap_or_else(|| env_or("MEDIA_DIR", "."));
    let quiet = args.quiet;
    let progress = move |step: Step| {
        if quiet {
            return;
        }
        match step {
            Step::Status(_) => {}
            Step::Job(line) => eprintln!("… {line}"),
            Step::Tool(call) => {
                let mark = if call.is_error { "✗" } else { "✓" };
                eprintln!(
                    "{mark} {} › {}({})  ↳ {}",
                    call.server.as_deref().unwrap_or("?"),
                    call.tool,
                    session::short(&call.arguments, 100),
                    call.result_preview(160)
                );
            }
        }
    };
    let mut d = Dialog::new(&dir);
    let turn = d.send(&s, &args.request(), &progress).await?;
    let saved = runs::save(&runs::runs_dir(), None, &d, s.client.model())?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{}", turn.reply)?;
    for note in &turn.notes {
        writeln!(out, "\n⚠ {note}")?;
    }
    writeln!(out, "\nДиалог сохранён: {}", saved.display())?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<RunArgs> {
        RunArgs::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn run_arguments() {
        let r = parse(&["2026-10-15", "--summary", "yes", "-q"]).unwrap();
        assert_eq!(r, RunArgs { dir: Some("2026-10-15".into()), summary: Some(true), quiet: true });
        assert!(r.request().contains("да — делай саммари"));
        assert_eq!(parse(&["--summary", "no"]).unwrap().summary, Some(false));
        assert_eq!(parse(&["--summary", "YES"]).unwrap().summary, Some(true));
        assert_eq!(parse(&["--summary", "нет"]).unwrap().summary, Some(false));
        assert!(parse(&["--summary"]).unwrap_err().to_string().contains("нужно значение"));
        assert!(parse(&[]).unwrap().request().contains("задай его и остановись"));
        assert!(parse(&["--summary", "может"]).is_err());
        assert!(parse(&["a", "b"]).is_err());
        assert!(parse(&["--force"]).is_err());
    }
}
