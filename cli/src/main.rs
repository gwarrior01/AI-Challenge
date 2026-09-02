//! Минимальный CLI-агент: отправляет запрос в LLM через API и печатает ответ в консоль.
//!
//! Использование:
//!   llm-cli "твой вопрос"      -- разовый запрос
//!   llm-cli                    -- интерактивный режим (REPL)
//!
//! Обязательные переменные окружения: LLM_API_URL, LLM_API_KEY (см. .env.example).

use anyhow::Result;
use llm_core::LlmClient;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<()> {
    let client = LlmClient::from_env()?;

    let args: Vec<String> = std::env::args().skip(1).collect();
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
