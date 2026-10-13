//! code-assistant — ассистент разработчика этого проекта, отдельный от агентов.
//!
//! ```text
//! code-assistant                         — TUI: /help, /review, /diff, /status, /files, /log, /save
//! code-assistant help [вопрос]           — ответ о проекте (README + docs/ через RAG, git через MCP)
//! code-assistant review [цель] [флаги]   — ревью без интерфейса (годится для CI)
//!     цель: <base> (по умолчанию main) | <base>..<head> | --wip
//!     --base <ветка> --head <ветка>      — то же, что <base>..<head>
//!     --format md|json                   — текст ревью или JSON (по умолчанию md)
//!     --out <файл>                       — записать в файл (иначе stdout)
//!     --fail-on low|medium|high|never    — код возврата 1, если есть замечание этого уровня или выше
//! ```
//!
//! Коды возврата: 0 — готово, 1 — сработал `--fail-on`, 2 — ошибка.

mod context;
mod help;
mod input;
mod markdown;
mod review;
mod tui;

use std::process::ExitCode;

use anyhow::{bail, Context, Result};

use context::Assistant;
use review::{Severity, Target};

const USAGE: &str = "code-assistant — ассистент разработчика проекта

  code-assistant                         TUI (команды — /? внутри)
  code-assistant help [вопрос]           вопрос о проекте: README + docs/ (RAG) и git (MCP)
  code-assistant review [цель] [флаги]   ревью изменений без интерфейса

Цель ревью:
  <base>             текущая ветка с незакоммиченным против точки ответвления от base (по умолчанию main)
  <base>..<head>     закоммиченное в head против base (или --base <base> --head <head>)
  --wip              только незакоммиченное

Флаги ревью:
  --format md|json                   текст ревью (Markdown) или JSON, по умолчанию md
  --out <файл>                       записать результат в файл, иначе — в stdout
  --fail-on low|medium|high|never    код 1, если есть замечание этого уровня или выше (по умолчанию never)

Коды возврата: 0 — готово, 1 — сработал --fail-on, 2 — ошибка.
Настройки — переменные окружения LLM_* и ASSISTANT_* (см. docs/developer-assistant.md).";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("Ошибка: {err:#}");
            ExitCode::from(2)
        }
    }
}

async fn run() -> Result<ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => {
            tui::run(Assistant::from_env()?).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Some("help") => {
            cli_help(&args[1..].join(" ")).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some("review") => cli_review(&args[1..]).await,
        Some(other) => bail!("неизвестная команда «{other}»\n\n{USAGE}"),
    }
}

async fn cli_help(question: &str) -> Result<()> {
    let a = Assistant::from_env()?;
    let progress = |msg: String| eprintln!("… {msg}");
    let reply = help::ask(&a, question, &progress).await?;
    for line in reply.context_lines() {
        println!("🧭 {line}");
    }
    for call in &reply.tool_calls {
        let mark = if call.is_error { "⚠" } else { "🔧" };
        println!("[{mark} {} → {}]", call.headline(120), call.result_preview(200));
    }
    println!("\n{}", reply.text);
    if let Some(u) = reply.usage {
        println!("\n[токены: {} (запрос {}, ответ {})]", u.total_tokens, u.prompt_tokens, u.completion_tokens);
    }
    Ok(())
}

/// Разобранные флаги `review`.
#[derive(Debug, PartialEq)]
struct ReviewArgs {
    target: Target,
    json: bool,
    out: Option<String>,
    fail_on: Option<Severity>,
}

fn parse_review_args(args: &[String], default_base: &str) -> Result<ReviewArgs> {
    let mut positional: Option<String> = None;
    let (mut base, mut head, mut wip) = (None, None, false);
    let (mut json, mut out, mut fail_on) = (false, None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().with_context(|| format!("{name}: не указано значение"));
        match arg.as_str() {
            "--wip" => wip = true,
            "--base" => base = Some(value("--base")?),
            "--head" => head = Some(value("--head")?),
            "--out" => out = Some(value("--out")?),
            "--format" => {
                json = match value("--format")?.as_str() {
                    "json" => true,
                    "md" | "markdown" => false,
                    other => bail!("--format: md или json, а не «{other}»"),
                }
            }
            "--fail-on" => {
                let v = value("--fail-on")?;
                fail_on = match v.as_str() {
                    "never" | "none" => None,
                    _ => Some(Severity::parse(&v).with_context(|| format!("--fail-on: low, medium, high или never, а не «{v}»"))?),
                }
            }
            flag if flag.starts_with("--") => bail!("неизвестный флаг «{flag}»\n\n{USAGE}"),
            _ if positional.is_none() => positional = Some(arg.clone()),
            _ => bail!("лишний аргумент «{arg}»"),
        }
    }
    let target = match (wip, positional, base, head) {
        (true, None, None, None) => Target::Wip,
        (true, ..) => bail!("--wip не сочетается с другими целями"),
        (false, Some(_), Some(_), _) | (false, Some(_), _, Some(_)) => bail!("цель — либо позиционно, либо --base/--head"),
        (false, Some(p), None, None) => Target::parse(&p, default_base)?,
        (false, None, base, Some(head)) => Target::Range { base: base.unwrap_or_else(|| default_base.into()), head },
        (false, None, Some(base), None) => Target::Branch { base },
        (false, None, None, None) => Target::Branch { base: default_base.into() },
    };
    Ok(ReviewArgs { target, json, out, fail_on })
}

async fn cli_review(args: &[String]) -> Result<ExitCode> {
    let default_base = context::env_or("ASSISTANT_REVIEW_BASE", review::DEFAULT_BASE);
    let args = parse_review_args(args, &default_base)?;
    let a = Assistant::from_env()?;
    eprintln!("ревью: {}", args.target.label());
    let progress = |msg: String| eprintln!("… {msg}");
    let review = review::run(&a, &args.target, &progress).await?;
    let text = if args.json { review.to_json() } else { review.to_markdown() };
    match &args.out {
        Some(path) => {
            std::fs::write(path, &text).with_context(|| format!("не удалось записать {path}"))?;
            eprintln!("ревью записано в {path}");
        }
        None => println!("{text}"),
    }
    let worst = review.max_severity();
    eprintln!(
        "замечаний: баги {}, архитектура {}, рекомендации {}",
        review.bugs.len(),
        review.architecture.len(),
        review.recommendations.len()
    );
    match (args.fail_on, worst) {
        (Some(limit), Some(worst)) if worst >= limit => {
            eprintln!("--fail-on {limit:?}: есть замечание уровня {worst:?} — код 1");
            Ok(ExitCode::from(1))
        }
        _ => Ok(ExitCode::SUCCESS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn review_args() {
        let r = parse_review_args(&args(""), "main").unwrap();
        assert_eq!(r, ReviewArgs { target: Target::Branch { base: "main".into() }, json: false, out: None, fail_on: None });
        let r = parse_review_args(&args("main..feature --format json --out r.json --fail-on high"), "main").unwrap();
        assert_eq!(r.target, Target::Range { base: "main".into(), head: "feature".into() });
        assert!(r.json);
        assert_eq!(r.out.as_deref(), Some("r.json"));
        assert_eq!(r.fail_on, Some(Severity::High));
        let r = parse_review_args(&args("--head feature"), "dev").unwrap();
        assert_eq!(r.target, Target::Range { base: "dev".into(), head: "feature".into() });
        assert_eq!(parse_review_args(&args("--wip"), "main").unwrap().target, Target::Wip);
        assert!(parse_review_args(&args("--wip main"), "main").is_err());
        assert!(parse_review_args(&args("--format xml"), "main").is_err());
        assert!(parse_review_args(&args("--fail-on"), "main").is_err());
        assert!(parse_review_args(&args("--oops"), "main").is_err());
    }
}
