//! Проверка развёрнутого сервиса по сети: параллельная нагрузка от нескольких
//! пользователей и срабатывание ограничений. Печатает отчёт в Markdown.

use anyhow::Result;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const PROMPTS: &[&str] = &[
    "Объясни в трёх предложениях, что такое HTTP.",
    "Назови три преимущества локальных языковых моделей.",
    "Кратко: чем отличается процесс от потока?",
    "Придумай название для кофейни и объясни его.",
];

struct Sample {
    user: usize,
    status: u16,
    latency: Duration,
    completion_tokens: u64,
}

async fn chat(http: &reqwest::Client, url: &str, key: &str, body: &Value) -> (u16, Value, Duration) {
    let started = Instant::now();
    let resp = http.post(format!("{url}/v1/chat/completions")).bearer_auth(key).json(body).send().await;
    let (status, json) = match resp {
        Ok(r) => (r.status().as_u16(), r.json().await.unwrap_or(Value::Null)),
        Err(e) => (0, json!({ "error": { "message": e.to_string() } })),
    };
    (status, json, started.elapsed())
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn error_code(v: &Value) -> String {
    v["error"]["code"].as_str().unwrap_or("-").to_string()
}

pub async fn run(url: &str, keys: &[String], requests: usize, max_tokens: u64) -> Result<()> {
    let http = reqwest::Client::builder().timeout(Duration::from_secs(600)).build()?;

    let health: Value = http.get(format!("{url}/health")).send().await?.json().await?;
    println!("# Нагрузочная проверка {url}\n");
    println!("Состояние: `{health}`\n");

    // 1. Все пользователи одновременно, каждый — серия запросов подряд.
    println!("## {} пользователя × {requests} запросов одновременно (max_tokens = {max_tokens})\n", keys.len());
    let wall = Instant::now();
    let mut tasks = Vec::new();
    for (user, key) in keys.iter().cloned().enumerate() {
        let http = http.clone();
        let url = url.to_string();
        tasks.push(tokio::spawn(async move {
            let mut out = Vec::new();
            for i in 0..requests {
                let body = json!({
                    "messages": [{ "role": "user", "content": PROMPTS[(user + i) % PROMPTS.len()] }],
                    "max_tokens": max_tokens,
                });
                let (status, v, latency) = chat(&http, &url, &key, &body).await;
                let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                if status != 200 {
                    eprintln!("пользователь {}: {status} {}", user + 1, v["error"]["message"]);
                }
                out.push(Sample { user, status, latency, completion_tokens });
            }
            out
        }));
    }
    let mut samples = Vec::new();
    for t in tasks {
        samples.extend(t.await?);
    }
    let wall = wall.elapsed();

    println!("| Пользователь | Запросов | Успешно | Средняя задержка | Токенов ответа |");
    println!("|---|---|---|---|---|");
    for user in 0..keys.len() {
        let mine: Vec<_> = samples.iter().filter(|s| s.user == user).collect();
        let ok = mine.iter().filter(|s| s.status == 200).count();
        let avg = mine.iter().map(|s| s.latency).sum::<Duration>() / mine.len().max(1) as u32;
        let tokens: u64 = mine.iter().map(|s| s.completion_tokens).sum();
        println!("| {} | {} | {ok} | {:.1} с | {tokens} |", user + 1, mine.len(), avg.as_secs_f64());
    }
    let mut lat: Vec<_> = samples.iter().filter(|s| s.status == 200).map(|s| s.latency).collect();
    lat.sort();
    let ok = lat.len();
    let tokens: u64 = samples.iter().map(|s| s.completion_tokens).sum();
    println!();
    println!("- Всего: {} запросов, успешно {ok}, ошибок {}", samples.len(), samples.len() - ok);
    println!(
        "- Задержка: p50 {:.1} с, p95 {:.1} с, max {:.1} с",
        percentile(&lat, 0.5).as_secs_f64(),
        percentile(&lat, 0.95).as_secs_f64(),
        lat.last().copied().unwrap_or_default().as_secs_f64()
    );
    println!(
        "- Общая скорость генерации: {:.1} ток/с ({tokens} токенов за {:.1} с)\n",
        tokens as f64 / wall.as_secs_f64(),
        wall.as_secs_f64()
    );

    // 2. Ограничения.
    println!("## Ограничения\n");
    println!("| Проверка | Ожидание | Результат |");
    println!("|---|---|---|");
    let key = &keys[0];

    let (status, v, _) = chat(&http, url, "wrong-key-for-loadtest", &json!({ "messages": [{ "role": "user", "content": "hi" }] })).await;
    println!("| Неверный ключ | 401 | {status} `{}` |", error_code(&v));

    let long = "слово ".repeat(5000);
    let (status, v, _) = chat(&http, url, key, &json!({ "messages": [{ "role": "user", "content": long }] })).await;
    println!("| Промпт на ~5000 слов | 400 context_length_exceeded | {status} `{}` |", error_code(&v));

    let stats: Value = http.get(format!("{url}/stats")).bearer_auth(key).send().await?.json().await?;
    let rpm = stats["limits"]["rpm"].as_u64().unwrap_or(20);
    let mut hit = None;
    for i in 1..=rpm + 2 {
        let body = json!({ "messages": [{ "role": "user", "content": "Ответь одним словом: да" }], "max_tokens": 1 });
        let (status, v, _) = chat(&http, url, key, &body).await;
        if status == 429 {
            hit = Some((i, error_code(&v)));
            break;
        }
    }
    match hit {
        Some((i, code)) => println!("| Частые запросы (лимит {rpm}/мин, с учётом прошлых) | 429 | 429 `{code}` на {i}-м запросе |"),
        None => println!("| Частые запросы (лимит {rpm}/мин) | 429 | лимит не сработал за {} запросов |", rpm + 2),
    }

    let stats: Value = http.get(format!("{url}/stats")).bearer_auth(key).send().await?.json().await?;
    println!("\n## /stats после проверки\n\n```json\n{}\n```", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
