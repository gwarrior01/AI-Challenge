//! code-assistant — ассистент разработчика этого проекта, отдельный от агентов.
//!
//! ```text
//! code-assistant                         — TUI: /help, /review, /diff, /status, /files, /log, /save
//! code-assistant help [вопрос]           — ответ о проекте (README + docs/ через RAG, git через MCP)
//! code-assistant review [цель] [путь…] [флаги]   — ревью без интерфейса (годится для CI)
//!     цель: <base> (по умолчанию main) | <base>..<head> | --wip
//!     путь: файлы и каталоги, которыми ограничено ревью (или --path <путь>, или после --)
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
  code-assistant review [цель] [путь…] [флаги]   ревью изменений без интерфейса

Цель ревью:
  <base>             текущая ветка с незакоммиченным против точки ответвления от base (по умолчанию main)
  <base>..<head>     закоммиченное в head против base (или --base <base> --head <head>)
  --wip              только незакоммиченное

Пути (необязательно): файлы и каталоги, которыми ограничено ревью —
  code-assistant review --wip assistant/src/tui.rs
  code-assistant review main core/src/rag docs
  code-assistant review main..feature -- assistant/      (-- — всё дальше пути)
  --path <путь>      то же флагом, можно несколько раз

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
    paths: Vec<String>,
    json: bool,
    out: Option<String>,
    fail_on: Option<Severity>,
}

/// `is_path` — существует ли такой файл или каталог (см. [`review::parse_args`]).
fn parse_review_args(args: &[String], default_base: &str, is_path: impl Fn(&str) -> bool) -> Result<ReviewArgs> {
    let mut words: Vec<String> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    let (mut base, mut head, mut wip) = (None, None, false);
    let (mut json, mut out, mut fail_on) = (false, None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().with_context(|| format!("{name}: не указано значение"));
        match arg.as_str() {
            "--" => {
                paths.extend(it.by_ref().cloned());
            }
            "--wip" => wip = true,
            "--base" => base = Some(value("--base")?),
            "--head" => head = Some(value("--head")?),
            "--path" => paths.push(value("--path")?),
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
            _ => words.push(arg.clone()),
        }
    }
    let flags = wip || base.is_some() || head.is_some();
    let target = if flags {
        // Цель задана флагами — все слова пути.
        paths.splice(0..0, words.drain(..));
        match (wip, base, head) {
            (true, None, None) => Target::Wip,
            (true, ..) => bail!("--wip не сочетается с --base/--head"),
            (false, base, Some(head)) => Target::Range { base: base.unwrap_or_else(|| default_base.into()), head },
            (false, Some(base), None) => Target::Branch { base },
            (false, None, None) => unreachable!("flags"),
        }
    } else {
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let (target, scoped) = review::parse_args(&refs, default_base, &is_path)?;
        paths.splice(0..0, scoped);
        target
    };
    let paths = paths
        .iter()
        .map(|p| p.trim().trim_start_matches("./").trim_end_matches('/').to_string())
        .filter(|p| !p.is_empty() && p != ".")
        .collect();
    Ok(ReviewArgs { target, paths, json, out, fail_on })
}

async fn cli_review(args: &[String]) -> Result<ExitCode> {
    let default_base = context::env_or("ASSISTANT_REVIEW_BASE", review::DEFAULT_BASE);
    let args = parse_review_args(args, &default_base, |p| std::path::Path::new(p).exists())?;
    let a = Assistant::from_env()?;
    match args.paths.is_empty() {
        true => eprintln!("ревью: {}", args.target.label()),
        false => eprintln!("ревью: {} · {}", args.target.label(), args.paths.join(", ")),
    }
    let progress = |msg: String| eprintln!("… {msg}");
    let review = review::run(&a, &args.target, &args.paths, &progress).await?;
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

    fn parse(s: &str) -> Result<ReviewArgs> {
        parse_review_args(&args(s), "main", |p| matches!(p, "assistant/src/tui.rs" | "docs"))
    }

    #[test]
    fn review_args() {
        let r = parse("").unwrap();
        assert_eq!(
            r,
            ReviewArgs { target: Target::Branch { base: "main".into() }, paths: vec![], json: false, out: None, fail_on: None }
        );
        let r = parse("main..feature --format json --out r.json --fail-on high").unwrap();
        assert_eq!(r.target, Target::Range { base: "main".into(), head: "feature".into() });
        assert!(r.json);
        assert_eq!(r.out.as_deref(), Some("r.json"));
        assert_eq!(r.fail_on, Some(Severity::High));
        let r = parse_review_args(&args("--head feature"), "dev", |_| false).unwrap();
        assert_eq!(r.target, Target::Range { base: "dev".into(), head: "feature".into() });
        assert_eq!(parse("--wip").unwrap().target, Target::Wip);
        assert!(parse("--wip --base main").is_err());
        assert!(parse("--format xml").is_err());
        assert!(parse("--fail-on").is_err());
        assert!(parse("--oops").is_err());
    }

    #[test]
    fn review_args_with_paths() {
        let r = parse("--wip assistant/src/tui.rs").unwrap();
        assert_eq!((r.target, r.paths), (Target::Wip, vec!["assistant/src/tui.rs".to_string()]));
        let r = parse("assistant/src/tui.rs --fail-on high").unwrap();
        assert_eq!((r.target, r.paths), (Target::Branch { base: "main".into() }, vec!["assistant/src/tui.rs".to_string()]));
        let r = parse("dev ./docs/ --path core/src").unwrap();
        assert_eq!((r.target, r.paths), (Target::Branch { base: "dev".into() }, vec!["docs".to_string(), "core/src".to_string()]));
        let r = parse("--format json -- main").unwrap();
        assert!(r.json);
        assert_eq!((r.target, r.paths), (Target::Branch { base: "main".into() }, vec!["main".to_string()]));
    }
}
