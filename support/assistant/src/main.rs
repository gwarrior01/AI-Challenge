//! support-assistant — мини-сервис поддержки пользователей продукта CloudDesk.
//!
//! Отвечает на вопросы о продукте по базе знаний (RAG: FAQ и документация из
//! `support/knowledge`) с учётом тикета и клиента из CRM (MCP: свой сервер
//! `support-mcp`, данные `support/data/*.json`).
//!
//! ```text
//! support-assistant [serve]                       — HTTP: чат и JSON API на 127.0.0.1:8096
//! support-assistant ask [--ticket TCK-1001] вопрос — один ответ в терминал (без сервера)
//! support-assistant tickets                       — тикеты из CRM
//! ```
//!
//! Коды возврата: 0 — готово, 2 — ошибка.

mod crm;
mod http;
mod knowledge;
mod service;

use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use llm_core::ToolCallRecord;

use crate::crm::Lookup;
use crate::service::{AskRequest, Support};

const DEFAULT_ADDR: &str = "127.0.0.1:8096";

const USAGE: &str = "support-assistant — ассистент поддержки пользователей (RAG по базе знаний + CRM через MCP)

  support-assistant [serve]                         HTTP-сервис: страница чата и API (SUPPORT_ADDR, по умолчанию 127.0.0.1:8096)
  support-assistant ask [--ticket <id>] <вопрос>    один ответ в терминал
  support-assistant tickets                         тикеты из CRM

Переменные: LLM_* и LLM_EMBEDDING_MODEL — модель и эмбеддинги (см. README),
SUPPORT_KNOWLEDGE (support/knowledge), SUPPORT_RAG_DB (rag/support.db),
SUPPORT_MCP_CONFIG (support/mcp.json), SUPPORT_CRM_SERVER (support), SUPPORT_LLM_TIMEOUT (120).";

#[tokio::main]
async fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ошибка: {err:#}");
            ExitCode::from(2)
        }
    }
}

async fn run(args: Vec<String>) -> Result<()> {
    match args.first().map(String::as_str) {
        None | Some("serve") => serve().await,
        Some("ask") => ask(&args[1..]).await,
        Some("tickets") => tickets().await,
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("неизвестная команда «{other}»\n\n{USAGE}"),
    }
}

async fn serve() -> Result<()> {
    let support = Arc::new(Support::from_env()?);
    let addr = env_or("SUPPORT_ADDR", DEFAULT_ADDR);
    eprintln!("support-assistant: модель {}", support.client.model());
    match &support.knowledge {
        Ok(kb) => match kb.sync().await {
            Ok(note) => eprintln!(
                "support-assistant: база знаний {} → {}{}",
                kb.cfg.corpus.join(", "),
                kb.cfg.db_path.display(),
                note.map(|n| format!(" ({n})")).unwrap_or_default()
            ),
            Err(err) => eprintln!("support-assistant: база знаний не проиндексирована: {err:#}"),
        },
        Err(err) => eprintln!("support-assistant: без базы знаний — {err}"),
    }
    match support.crm.ensure_connected().await {
        Ok(()) => eprintln!(
            "support-assistant: CRM подключена (MCP «{}»)",
            support.crm.server
        ),
        Err(note) => eprintln!("support-assistant: CRM недоступна — {note}"),
    }
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("не удалось занять адрес {addr}"))?;
    eprintln!("support-assistant: http://{addr}");
    axum::serve(listener, http::router(support))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn ask(args: &[String]) -> Result<()> {
    let mut req = AskRequest::default();
    let mut words = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--ticket" | "-t" => {
                req.ticket_id = Some(it.next().context("--ticket требует номер тикета")?.clone())
            }
            _ => words.push(arg.as_str()),
        }
    }
    req.question = words.join(" ");
    if req.question.trim().is_empty() {
        bail!("укажите вопрос: support-assistant ask [--ticket TCK-1001] <вопрос>");
    }
    let support = Support::from_env()?;
    let reply = support.ask(req).await?;
    for call in &reply.tool_calls {
        eprintln!("{}", call_line(call));
    }
    for line in reply.context_lines() {
        eprintln!("{line}");
    }
    println!("\n{}", reply.answer);
    Ok(())
}

async fn tickets() -> Result<()> {
    let support = Support::from_env()?;
    match support
        .crm
        .lookup("list_tickets", serde_json::json!({}), &mut Vec::new())
        .await
    {
        Lookup::Found(v) => {
            for t in v
                .get("tickets")
                .and_then(|t| t.as_array())
                .into_iter()
                .flatten()
            {
                let f = |k: &str| t.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
                println!(
                    "{}  {:<9} {:<7} {} — {}",
                    f("id"),
                    f("status"),
                    f("priority"),
                    f("user_name"),
                    f("subject")
                );
            }
            Ok(())
        }
        Lookup::Missing(err) | Lookup::Unavailable(err) => bail!(err),
    }
}

/// `MCP support · get_ticket({"id":"TCK-1001"}) → …` — строка вызова для терминала.
fn call_line(call: &ToolCallRecord) -> String {
    let kind = if call.server.as_deref() == Some(service::RAG_SERVER) {
        "RAG"
    } else {
        "MCP"
    };
    let mark = if call.is_error { "✗" } else { "→" };
    format!(
        "{kind} {} · {}({}) {mark} {}",
        call.server.as_deref().unwrap_or("?"),
        call.tool,
        short(&call.arguments, 80),
        short(&call.result.replace('\n', " "), 100)
    )
}

pub fn record(
    server: &str,
    tool: &str,
    arguments: &str,
    result: &str,
    is_error: bool,
) -> ToolCallRecord {
    ToolCallRecord {
        server: Some(server.to_string()),
        tool: tool.to_string(),
        arguments: arguments.to_string(),
        result: result.to_string(),
        is_error,
    }
}

pub fn short(s: &str, max: usize) -> String {
    llm_core::rag::retrieve::short(s, max)
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.into())
}
