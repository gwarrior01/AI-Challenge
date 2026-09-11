//! Денежная стоимость запросов к LLM — используется в интерфейсах именованных
//! агентов (CLI, TUI, веб), чтобы показывать "сколько стоил этот запрос" и
//! накопительную стоимость диалога рядом с расходом токенов.
//!
//! Два источника стоимости, в порядке предпочтения (см. [`resolve_total`] и
//! [`source`]):
//!
//! 1. **Реальная стоимость от провайдера** ([`crate::Usage::cost`]) — некоторые
//!    OpenAI-совместимые API (например, OpenRouter) присылают в `usage.cost`
//!    фактически выставленную сумму в USD. Она персистится вместе с сообщением
//!    в SQLite (см. `Agent`/`Db` в [`crate::agent`]), потому что её, в отличие
//!    от оценки ниже, нельзя пересчитать задним числом — это то, что было
//!    заплачено по факту.
//! 2. **Оценка по ставкам** ([`Pricing`]/[`from_env`]) — когда провайдер
//!    `cost` не присылает (большинство: OpenAI, Ollama, LM Studio...), но
//!    заданы ОБЕ переменные окружения `LLM_PRICE_INPUT_PER_1M` /
//!    `LLM_PRICE_OUTPUT_PER_1M`. В отличие от реальной стоимости, оценка НЕ
//!    персистится — она всегда пересчитывается на лету из уже сохранённых
//!    метрик токенов по ставкам, действующим на момент отображения: если
//!    ставки поменять, стоимость старых сообщений в истории пересчитается по
//!    новым — признанный компромисс ради простоты.
//!
//! Если ни то ни другое не доступно — стоимость не показывается вовсе
//! (`None`), а не молчаливый "$0.00" для локальных бесплатных моделей.

use crate::Usage;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

const PRICE_INPUT_ENV: &str = "LLM_PRICE_INPUT_PER_1M";
const PRICE_OUTPUT_ENV: &str = "LLM_PRICE_OUTPUT_PER_1M";
const PRICE_CURRENCY_ENV: &str = "LLM_PRICE_CURRENCY";
const DEFAULT_CURRENCY: &str = "$";

/// Ставки цены модели: сколько стоит миллион входных (`prompt`) и миллион
/// выходных (`completion`) токенов. Валюта в ставках не хранится — это просто
/// число, единица измерения даётся отдельно через [`currency`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
}

impl Pricing {
    /// Стоимость `tokens` входных токенов по этим ставкам.
    pub fn input_cost(&self, tokens: u32) -> f64 {
        tokens as f64 / 1_000_000.0 * self.input_per_million
    }

    /// Стоимость `tokens` выходных токенов по этим ставкам.
    pub fn output_cost(&self, tokens: u32) -> f64 {
        tokens as f64 / 1_000_000.0 * self.output_per_million
    }

    /// Суммарная стоимость одного обмена (запрос + ответ) по метрикам `usage`.
    pub fn total_cost(&self, usage: &Usage) -> f64 {
        self.input_cost(usage.prompt_tokens) + self.output_cost(usage.completion_tokens)
    }
}

/// Ставки цены из переменных окружения `LLM_PRICE_INPUT_PER_1M` /
/// `LLM_PRICE_OUTPUT_PER_1M` (цена за миллион входных/выходных токенов) —
/// `None`, если хотя бы одна из них не задана или не парсится в
/// неотрицательное число: расчёт стоимости — это опциональная фича,
/// включаемая только явным указанием ОБЕИХ ставок сразу, а не наполовину.
///
/// Читается один раз за время жизни процесса и кэшируется, как и остальные
/// `LLM_*` переменные — менять ставки на лету без перезапуска нельзя.
pub fn from_env() -> Option<Pricing> {
    static VALUE: OnceLock<Option<Pricing>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let input_per_million = parse_non_negative(PRICE_INPUT_ENV)?;
        let output_per_million = parse_non_negative(PRICE_OUTPUT_ENV)?;
        Some(Pricing { input_per_million, output_per_million })
    })
}

fn parse_non_negative(var: &str) -> Option<f64> {
    std::env::var(var).ok()?.trim().parse::<f64>().ok().filter(|v| *v >= 0.0)
}

/// Символ/код валюты для отображения рядом с суммами — из `LLM_PRICE_CURRENCY`,
/// по умолчанию `"$"`. Реальная стоимость от провайдера ([`Usage::cost`]) тоже
/// показывается под этой же меткой: у известных провайдеров, присылающих это
/// поле (OpenRouter), она в USD, что совпадает со значением по умолчанию.
pub fn currency() -> String {
    static VALUE: OnceLock<String> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            std::env::var(PRICE_CURRENCY_ENV)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_CURRENCY.to_string())
        })
        .clone()
}

/// Откуда взята стоимость, которую вернул один из `resolve_*` — интерфейсы
/// показывают это как "≈" перед оценкой, чтобы не выдавать прикидку за
/// точную сумму.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    /// Реальная стоимость, которую вернул сам провайдер (`usage.cost`).
    Api,
    /// Оценка по ставкам [`from_env`] — провайдер `usage.cost` не прислал.
    Estimated,
}

/// Источник стоимости для `usage` без вычисления самой суммы — `None`, если
/// стоимость взять неоткуда (провайдер её не прислал и ставки не заданы).
pub fn source(usage: &Usage) -> Option<CostSource> {
    if usage.cost.is_some() {
        Some(CostSource::Api)
    } else if from_env().is_some() {
        Some(CostSource::Estimated)
    } else {
        None
    }
}

/// Стоимость входной части `usage`: `usage.cost_input`, если провайдер дал
/// разбивку; `0.0`, если провайдер дал только `cost` целиком без разбивки
/// (тогда вся сумма приписывается [`resolve_output`] — обмен целиком не
/// теряется, просто не делится на составляющие); иначе — оценка по ставкам.
/// `None`, если стоимость взять неоткуда.
pub fn resolve_input(usage: &Usage) -> Option<f64> {
    if let Some(input) = usage.cost_input {
        return Some(input);
    }
    if usage.cost.is_some() {
        return Some(0.0);
    }
    from_env().map(|p| p.input_cost(usage.prompt_tokens))
}

/// Стоимость выходной части `usage` — см. [`resolve_input`] (симметрично, с
/// той же оговоркой про обмен без разбивки: без неё сюда уходит `cost` целиком).
pub fn resolve_output(usage: &Usage) -> Option<f64> {
    if let Some(output) = usage.cost_output {
        return Some(output);
    }
    if let Some(cost) = usage.cost {
        return Some(cost);
    }
    from_env().map(|p| p.output_cost(usage.completion_tokens))
}

/// Суммарная стоимость `usage` — `usage.cost`, если провайдер его прислал,
/// иначе оценка по ставкам [`from_env`]. `None`, если ни того ни другого нет.
/// Всегда равна `resolve_input(usage) + resolve_output(usage)` (с точностью
/// округления), когда обе определены.
pub fn resolve_total(usage: &Usage) -> Option<f64> {
    if let Some(cost) = usage.cost {
        return Some(cost);
    }
    from_env().map(|p| p.total_cost(usage))
}
