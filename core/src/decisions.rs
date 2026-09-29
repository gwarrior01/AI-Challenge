//! Решения моделью Jev (TypeSafe) через OpenRouter Decisions API
//! (`POST /api/alpha/decisions`).
//!
//! Jev не генерирует текст: на каждый вопрос она возвращает типизированный
//! ответ с вероятностями — «да/нет» (`noul`: вероятность «да»), выбор одного
//! из заранее заданных вариантов (`choice`) или балл на шкале уровней
//! (`score`). Ситуация, о которой задаются вопросы, передаётся полем `state`.
//!
//! Настройки — из переменных окружения:
//! - `JEV_API_KEY` (или `OPENROUTER_API_KEY`) — ключ OpenRouter; без него
//!   используется `LLM_API_KEY`, если `LLM_API_URL` указывает на OpenRouter;
//! - `JEV_API_URL` — адрес эндпоинта (по умолчанию [`DEFAULT_URL`]);
//! - `JEV_MODEL` — модель (по умолчанию [`DEFAULT_MODEL`]).
//!
//! Вопросы собираются командами ([`DecisionSession::apply`]) — общими для
//! всех интерфейсов — или приходят готовым списком [`Question`] (веб-форма).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_URL: &str = "https://openrouter.ai/api/alpha/decisions";
pub const DEFAULT_MODEL: &str = "typesafe/jev-1.13";

/// Адрес, ключ и модель Decisions API.
#[derive(Debug, Clone)]
pub struct DecisionsConfig {
    pub url: String,
    /// `None` — ключ не задан; ошибка появится только при запросе, чтобы
    /// экран вопросов открывался и без ключа.
    pub api_key: Option<String>,
    pub model: String,
}

impl DecisionsConfig {
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let api_key = var("JEV_API_KEY").or_else(|| var("OPENROUTER_API_KEY")).or_else(|| {
            var("LLM_API_URL").filter(|url| url.contains("openrouter.ai")).and_then(|_| var("LLM_API_KEY"))
        });
        Self {
            url: var("JEV_API_URL").unwrap_or_else(|| DEFAULT_URL.to_string()),
            api_key,
            model: var("JEV_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        }
    }
}

/// Вопрос к модели. `key` — имя, под которым вернётся ответ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    pub key: String,
    pub instructions: String,
    #[serde(flatten)]
    pub kind: QuestionKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum QuestionKind {
    /// «Да или нет»: когда ответ «да» и когда «нет».
    Noul { yes: String, no: String },
    /// Выбор одного варианта из списка.
    Choice { options: Vec<ChoiceOption> },
    /// Балл на шкале: уровни от младшего (0) к старшему.
    Score { levels: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub key: String,
    pub text: String,
}

impl QuestionKind {
    pub fn name(&self) -> &'static str {
        match self {
            QuestionKind::Noul { .. } => "noul",
            QuestionKind::Choice { .. } => "choice",
            QuestionKind::Score { .. } => "score",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            QuestionKind::Noul { .. } => "да/нет",
            QuestionKind::Choice { .. } => "выбор",
            QuestionKind::Score { .. } => "шкала",
        }
    }
}

impl Question {
    /// Проверяет вопрос до отправки — понятнее, чем 400 от API.
    pub fn validate(&self) -> Result<()> {
        if !valid_key(&self.key) {
            bail!("ключ вопроса «{}»: нужны буквы, цифры, «_» или «-», без пробелов", self.key);
        }
        if self.instructions.trim().is_empty() {
            bail!("вопрос {}: пустая формулировка", self.key);
        }
        match &self.kind {
            QuestionKind::Noul { yes, no } => {
                if yes.trim().is_empty() || no.trim().is_empty() {
                    bail!("вопрос {}: нужно описать, когда ответ «да» и когда «нет»", self.key);
                }
            }
            QuestionKind::Choice { options } => {
                if options.len() < 2 {
                    bail!("вопрос {}: у выбора нужно хотя бы два варианта", self.key);
                }
                for (i, option) in options.iter().enumerate() {
                    if !valid_key(&option.key) {
                        bail!("вопрос {}: вариант «{}» — ключ без пробелов", self.key, option.key);
                    }
                    if options[..i].iter().any(|o| o.key == option.key) {
                        bail!("вопрос {}: вариант {} повторяется", self.key, option.key);
                    }
                }
            }
            QuestionKind::Score { levels } => {
                if levels.is_empty() || levels.iter().any(|l| l.trim().is_empty()) {
                    bail!("вопрос {}: у шкалы нужен хотя бы один непустой уровень", self.key);
                }
            }
        }
        Ok(())
    }

    fn to_wire(&self) -> serde_json::Value {
        let criteria = match &self.kind {
            QuestionKind::Noul { yes, no } => serde_json::json!({ "true": yes, "false": no }),
            QuestionKind::Choice { options } => {
                serde_json::Value::Object(options.iter().map(|o| (o.key.clone(), o.text.clone().into())).collect())
            }
            QuestionKind::Score { levels } => serde_json::json!(levels),
        };
        serde_json::json!({
            "type": self.kind.name(),
            "instructions": self.instructions,
            "criteria": criteria,
        })
    }

    /// Вопрос одной строкой — в синтаксисе команды, которой он задаётся.
    pub fn describe(&self) -> String {
        let parts: Vec<String> = match &self.kind {
            QuestionKind::Noul { yes, no } => vec![yes.clone(), no.clone()],
            QuestionKind::Choice { options } => options.iter().map(|o| format!("{}: {}", o.key, o.text)).collect(),
            QuestionKind::Score { levels } => levels.clone(),
        };
        format!("{} {}: {} ; {}", self.kind.name(), self.key, self.instructions, parts.join(" ; "))
    }
}

fn valid_key(key: &str) -> bool {
    !key.is_empty() && key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Ответ на один вопрос — в том виде, в каком его присылает API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Вероятность ответа «да» (0–1).
    Noul { noul: f64 },
    Choice {
        choice: String,
        #[serde(default)]
        confidence: Option<f64>,
        #[serde(default)]
        probabilities: serde_json::Map<String, serde_json::Value>,
    },
    Score {
        /// Ожидаемый уровень (дробный: 1.99 — почти наверняка уровень 2).
        score: f64,
        #[serde(default)]
        confidence: Option<f64>,
        #[serde(default)]
        probabilities: serde_json::Map<String, serde_json::Value>,
        #[serde(default)]
        legend: serde_json::Map<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub model: String,
    #[serde(default)]
    pub provider: Option<String>,
    /// Ответы по ключам вопросов — в порядке ответа API.
    pub answers: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub usage: DecisionUsage,
}

/// Итог запроса: разобранный ответ, текст для показа и сырые JSON.
#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub response: DecisionResponse,
    /// Ответы в порядке вопросов; `None` — ответ не разобрался (виден в JSON).
    pub answers: Vec<(Question, Option<Answer>)>,
    pub text: String,
    pub request_json: String,
    pub response_json: String,
}

/// `state` — текст ситуации; JSON-объект или массив отправляется как есть
/// (API принимает и структурированное состояние).
fn state_value(state: &str) -> serde_json::Value {
    let trimmed = state.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return value;
        }
    }
    serde_json::Value::String(trimmed.to_string())
}

pub fn request_body(model: &str, questions: &[Question], state: &str) -> serde_json::Value {
    let questions: serde_json::Map<String, serde_json::Value> =
        questions.iter().map(|q| (q.key.clone(), q.to_wire())).collect();
    serde_json::json!({ "model": model, "questions": questions, "state": state_value(state) })
}

/// Отправляет вопросы о ситуации `state` и возвращает ответы модели.
pub async fn decide(cfg: &DecisionsConfig, questions: &[Question], state: &str) -> Result<Decision> {
    if questions.is_empty() {
        bail!("нет ни одного вопроса — добавьте хотя бы один");
    }
    if state.trim().is_empty() {
        bail!("пустое описание ситуации — модели не о чем решать");
    }
    for (i, q) in questions.iter().enumerate() {
        q.validate()?;
        if questions[..i].iter().any(|p| p.key == q.key) {
            bail!("ключ вопроса {} повторяется", q.key);
        }
    }
    let api_key = cfg.api_key.as_deref().context(
        "не задан ключ OpenRouter для Jev: переменная окружения JEV_API_KEY (или OPENROUTER_API_KEY)",
    )?;

    let body = request_body(&cfg.model, questions, state);
    let request_json = serde_json::to_string_pretty(&body).context("не удалось сериализовать запрос")?;
    let response = reqwest::Client::new()
        .post(&cfg.url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("ошибка запроса к {}", cfg.url))?;
    let status = response.status();
    let raw = response.text().await.context("не удалось прочитать тело ответа")?;
    if !status.is_success() {
        bail!("Decisions API вернул {status}: {}", api_error_message(&raw));
    }
    let parsed: DecisionResponse =
        serde_json::from_str(&raw).with_context(|| format!("не удалось разобрать ответ Decisions API: {raw}"))?;
    let response_json = serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or(raw);

    let answers: Vec<(Question, Option<Answer>)> = questions
        .iter()
        .map(|q| {
            let answer = parsed.answers.get(&q.key).and_then(|v| serde_json::from_value(v.clone()).ok());
            (q.clone(), answer)
        })
        .collect();
    let text = format_decision(&parsed, &answers);
    Ok(Decision { response: parsed, answers, text, request_json, response_json })
}

/// Текст ошибки из тела ответа OpenRouter (`{"error":{"message":…}}`) или
/// само тело, если оно другое.
fn api_error_message(raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_string))
        .unwrap_or_else(|| raw.trim().to_string())
}

fn prob(map: &serde_json::Map<String, serde_json::Value>, key: &str) -> f64 {
    map.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn percent(p: f64) -> String {
    format!("{:>3.0}%", p * 100.0)
}

fn bar(p: f64) -> String {
    let filled = (p.clamp(0.0, 1.0) * 20.0).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(20 - filled))
}

/// Ответы текстом — для TUI и CLI (веб рисует их сам по JSON).
pub fn format_decision(response: &DecisionResponse, answers: &[(Question, Option<Answer>)]) -> String {
    let mut out = Vec::new();
    for (question, answer) in answers {
        out.push(format!("{} ({}) — {}", question.key, question.kind.label(), question.instructions));
        match (answer, &question.kind) {
            (None, _) => out.push("  нет ответа (см. JSON ответа)".to_string()),
            (Some(Answer::Noul { noul }), _) => {
                let verdict = if *noul >= 0.5 { "ДА" } else { "НЕТ" };
                out.push(format!("  → {verdict} · вероятность «да» {}  {}", percent(*noul).trim(), bar(*noul)));
            }
            (Some(Answer::Choice { choice, confidence, probabilities }), kind) => {
                let conf = confidence.map(|c| format!(" · уверенность {}", percent(c).trim())).unwrap_or_default();
                out.push(format!("  → {choice}{conf}"));
                let keys: Vec<String> = match kind {
                    QuestionKind::Choice { options } => options.iter().map(|o| o.key.clone()).collect(),
                    _ => probabilities.keys().cloned().collect(),
                };
                let width = keys.iter().map(|k| k.chars().count()).max().unwrap_or(0);
                for key in keys {
                    let p = prob(probabilities, &key);
                    let mark = if key == *choice { "▸" } else { " " };
                    out.push(format!("  {mark} {key:<width$} {} {}", percent(p), bar(p)));
                }
            }
            (Some(Answer::Score { score, confidence, probabilities, legend }), kind) => {
                let levels: Vec<String> = match kind {
                    QuestionKind::Score { levels } => levels.clone(),
                    _ => (0..legend.len())
                        .map(|i| legend.get(&i.to_string()).and_then(|v| v.as_str()).unwrap_or_default().to_string())
                        .collect(),
                };
                let top = levels.len().saturating_sub(1);
                let nearest = (score.round().max(0.0) as usize).min(top);
                let conf = confidence.map(|c| format!(" · уверенность {}", percent(c).trim())).unwrap_or_default();
                let name = levels.get(nearest).cloned().unwrap_or_default();
                out.push(format!("  → {score:.2} из 0–{top} («{name}»){conf}"));
                for (i, level) in levels.iter().enumerate() {
                    let p = prob(probabilities, &i.to_string());
                    let mark = if i == nearest { "▸" } else { " " };
                    out.push(format!("  {mark} {i} {} {}  {level}", percent(p), bar(p)));
                }
            }
        }
        out.push(String::new());
    }
    let mut footer = format!("модель {}", response.model);
    if let Some(provider) = &response.provider {
        footer.push_str(&format!(" · {provider}"));
    }
    footer.push_str(&format!(" · токены {} → {}", response.usage.input_tokens, response.usage.output_tokens));
    if let Some(cost) = response.usage.cost {
        footer.push_str(&format!(" · ${cost:.6}"));
    }
    out.push(footer);
    out.join("\n")
}

pub const HELP: &str = "Команды (части вопроса разделяются «;»):
  noul <ключ>: <вопрос> ; <когда «да»> ; <когда «нет»>   — вопрос «да/нет»
  choice <ключ>: <вопрос> ; <вариант>: <описание> ; …      — выбор варианта (от двух)
  score <ключ>: <вопрос> ; <уровень 0> ; <уровень 1> ; …   — шкала, уровни по возрастанию
  ask <ситуация>   — отправить все вопросы о ситуации (текст или JSON)
  list             — список вопросов
  del <ключ>       — удалить вопрос
  clear            — удалить все вопросы
  example          — пример: разбор обращения в поддержку";

/// Итог команды: текст для показа или запрос, который нужно отправить.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Output(String),
    Ask(String),
}

/// Набор вопросов, собираемый командами — одинаково в любом интерфейсе.
#[derive(Debug, Clone, Default)]
pub struct DecisionSession {
    pub questions: Vec<Question>,
}

impl DecisionSession {
    pub fn apply(&mut self, line: &str) -> Result<Step> {
        let line = line.trim();
        let (command, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        match command {
            "" | "list" => Ok(Step::Output(self.list())),
            "help" => Ok(Step::Output(HELP.to_string())),
            "ask" => {
                if self.questions.is_empty() {
                    bail!("сначала добавьте вопросы (noul/choice/score) или загрузите пример: example");
                }
                if rest.is_empty() {
                    bail!("после ask опишите ситуацию: ask <текст>");
                }
                Ok(Step::Ask(rest.to_string()))
            }
            "noul" | "choice" | "score" => {
                let question = parse_question(command, rest)?;
                let key = question.key.clone();
                let replaced = match self.questions.iter_mut().find(|q| q.key == key) {
                    Some(existing) => {
                        *existing = question;
                        true
                    }
                    None => {
                        self.questions.push(question);
                        false
                    }
                };
                let verb = if replaced { "заменён" } else { "добавлен" };
                Ok(Step::Output(format!("Вопрос {key} {verb}.\n\n{}", self.list())))
            }
            "del" => {
                let before = self.questions.len();
                self.questions.retain(|q| q.key != rest);
                if self.questions.len() == before {
                    bail!("нет вопроса с ключом «{rest}»");
                }
                Ok(Step::Output(format!("Вопрос {rest} удалён.\n\n{}", self.list())))
            }
            "clear" => {
                self.questions.clear();
                Ok(Step::Output("Все вопросы удалены.".to_string()))
            }
            "example" => {
                self.questions = example_questions();
                Ok(Step::Output(format!(
                    "{}\n\nТеперь, например:\nask {EXAMPLE_STATE}",
                    self.list()
                )))
            }
            other => bail!("неизвестная команда «{other}»\n\n{HELP}"),
        }
    }

    pub fn list(&self) -> String {
        if self.questions.is_empty() {
            return "Вопросов пока нет.".to_string();
        }
        let mut out = vec![format!("Вопросы ({}):", self.questions.len())];
        out.extend(self.questions.iter().map(|q| format!("  {}", q.describe())));
        out.join("\n")
    }
}

/// Разбирает `<ключ>: <вопрос> ; <часть> ; …` для команды `kind`.
pub fn parse_question(kind: &str, text: &str) -> Result<Question> {
    let mut parts = text.split(';').map(str::trim);
    let head = parts.next().unwrap_or_default();
    let (key, instructions) = head
        .split_once(':')
        .with_context(|| format!("формат: {kind} <ключ>: <вопрос> ; …"))?;
    let rest: Vec<&str> = parts.filter(|p| !p.is_empty()).collect();
    let kind = match kind {
        "noul" => match rest.as_slice() {
            [yes, no] => QuestionKind::Noul { yes: yes.to_string(), no: no.to_string() },
            _ => bail!("у noul две части после вопроса: ; <когда «да»> ; <когда «нет»>"),
        },
        "choice" => QuestionKind::Choice {
            options: rest
                .iter()
                .map(|part| match part.split_once(':') {
                    Some((key, text)) => ChoiceOption { key: key.trim().to_string(), text: text.trim().to_string() },
                    None => ChoiceOption { key: part.to_string(), text: part.to_string() },
                })
                .collect(),
        },
        "score" => QuestionKind::Score { levels: rest.iter().map(|s| s.to_string()).collect() },
        other => bail!("неизвестный тип вопроса «{other}»: noul, choice или score"),
    };
    let question = Question { key: key.trim().to_string(), instructions: instructions.trim().to_string(), kind };
    question.validate()?;
    Ok(question)
}

pub const EXAMPLE_STATE: &str =
    "После нажатия «Оплатить» страница оформления заказа становится пустой. Пробовал в двух браузерах.";

/// Пример из документации OpenRouter: разбор обращения в поддержку.
pub fn example_questions() -> Vec<Question> {
    let option = |key: &str, text: &str| ChoiceOption { key: key.to_string(), text: text.to_string() };
    vec![
        Question {
            key: "is_bug".into(),
            instructions: "Сообщает ли клиент о дефекте в программе?".into(),
            kind: QuestionKind::Noul {
                yes: "Клиент описывает сломанное или неожиданное поведение продукта".into(),
                no: "Клиент задаёт вопрос или просит новую функцию".into(),
            },
        },
        Question {
            key: "team".into(),
            instructions: "Какая команда должна взять обращение?".into(),
            kind: QuestionKind::Choice {
                options: vec![
                    option("account", "Вход, права доступа, профиль"),
                    option("frontend", "Отрисовка, вёрстка, совместимость браузеров"),
                    option("payments", "Оформление заказа, счета, платежи"),
                ],
            },
        },
        Question {
            key: "urgency".into(),
            instructions: "Насколько срочное обращение?".into(),
            kind: QuestionKind::Score {
                levels: vec![
                    "Подождёт до следующего релиза".into(),
                    "Исправить на этой неделе".into(),
                    "Прямо сейчас блокирует выручку".into(),
                ],
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_question_kinds() {
        let q = parse_question("noul", "is_bug: Это баг? ; сломано ; вопрос").unwrap();
        assert_eq!(q.kind, QuestionKind::Noul { yes: "сломано".into(), no: "вопрос".into() });

        let q = parse_question("choice", "team: Кому? ; account: Вход ; frontend").unwrap();
        let QuestionKind::Choice { options } = &q.kind else { panic!() };
        assert_eq!(options[0], ChoiceOption { key: "account".into(), text: "Вход".into() });
        assert_eq!(options[1], ChoiceOption { key: "frontend".into(), text: "frontend".into() });

        let q = parse_question("score", "urgency: Срочно? ; нет ; да").unwrap();
        assert_eq!(q.kind, QuestionKind::Score { levels: vec!["нет".into(), "да".into()] });
    }

    #[test]
    fn rejects_malformed_questions() {
        assert!(parse_question("noul", "is_bug Это баг? ; a ; b").is_err());
        assert!(parse_question("noul", "is_bug: Это баг? ; a").is_err());
        assert!(parse_question("choice", "team: Кому? ; одна").is_err());
        assert!(parse_question("choice", "team: Кому? ; a ; a").is_err());
        assert!(parse_question("score", "u: Срочно?").is_err());
        assert!(parse_question("noul", "два слова: Баг? ; a ; b").is_err());
    }

    #[test]
    fn request_body_matches_api_shape() {
        let body = request_body("typesafe/jev-1.13", &example_questions(), "текст");
        assert_eq!(body["questions"]["is_bug"]["type"], "noul");
        assert!(body["questions"]["is_bug"]["criteria"]["true"].is_string());
        assert!(body["questions"]["team"]["criteria"]["payments"].is_string());
        assert_eq!(body["questions"]["urgency"]["criteria"].as_array().unwrap().len(), 3);
        assert_eq!(body["state"], "текст");
        let body = request_body("m", &example_questions(), r#"{"tier":"enterprise"}"#);
        assert_eq!(body["state"]["tier"], "enterprise");
    }

    #[test]
    fn parses_documented_response() {
        let raw = r#"{"id":"gen-dec-1","model":"typesafe/jev-1.13-20260917","provider":"TypeSafe",
            "answers":{"is_bug":{"type":"noul","noul":0.96},
            "team":{"type":"choice","choice":"payments","confidence":0.75,"probabilities":{"account":0,"frontend":0.16,"payments":0.84}},
            "urgency":{"type":"score","score":1.99,"confidence":0.99,"probabilities":{"0":0,"1":0.01,"2":0.99},
              "legend":{"0":"a","1":"b","2":"c"}}},
            "usage":{"input_tokens":476,"output_tokens":70,"cost":0.000019992}}"#;
        let parsed: DecisionResponse = serde_json::from_str(raw).unwrap();
        let answers: Vec<_> = example_questions()
            .into_iter()
            .map(|q| {
                let a = parsed.answers.get(&q.key).and_then(|v| serde_json::from_value(v.clone()).ok());
                (q, a)
            })
            .collect();
        assert!(answers.iter().all(|(_, a)| a.is_some()));
        let text = format_decision(&parsed, &answers);
        assert!(text.contains("→ ДА"));
        assert!(text.contains("→ payments · уверенность 75%"));
        assert!(text.contains("→ 1.99 из 0–2 («Прямо сейчас блокирует выручку»)"));
        assert!(text.contains("токены 476 → 70"));
    }

    #[test]
    fn session_commands() {
        let mut s = DecisionSession::default();
        assert!(s.apply("ask что-то").is_err());
        s.apply("example").unwrap();
        assert_eq!(s.questions.len(), 3);
        s.apply("noul team: Платёж? ; да ; нет").unwrap();
        assert_eq!(s.questions.len(), 3);
        assert_eq!(s.questions[1].kind.name(), "noul");
        s.apply("del team").unwrap();
        assert_eq!(s.questions.len(), 2);
        assert_eq!(s.apply("ask  пусто на экране ").unwrap(), Step::Ask("пусто на экране".into()));
        assert!(s.apply("foo").is_err());
    }
}
