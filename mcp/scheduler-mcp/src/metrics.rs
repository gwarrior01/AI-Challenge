//! Метрики запуска и их агрегация. Планировщик не знает, что вернуло задание:
//! из результата (JSON) берутся скалярные поля — все подряд по путям вида
//! `body.data.rate` или только перечисленные в `extract`, — и `get_summary`
//! считает по ним одно и то же для любого источника: min/avg/max/последнее и
//! изменение для чисел, последнее значение и число смен для строк.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use regex::Regex;
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Сколько полей результата брать в метрики без `extract`: большой ответ
/// (список процессов, страница JSON) иначе раздул бы каждую строку `runs`.
const MAX_AUTO_METRICS: usize = 40;
/// Длинные строки — это текст, а не значение-состояние; в метрики не идут.
const MAX_TEXT_METRIC_CHARS: usize = 120;
/// Сколько различных значений строковой метрики показывать в сводке.
const MAX_TEXT_VALUES: usize = 5;

/// Метрики одного запуска: путь или имя → число, строка или bool.
pub type Metrics = BTreeMap<String, Value>;

/// Как достать метрику из результата: путь к полю (через точку, индексы
/// массивов — числами; пустой путь — весь результат) и, если значение —
/// текст, регулярное выражение, первая группа которого (или всё совпадение)
/// и есть значение.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExtractRule {
    /// Путь к полю результата, например `heap` или `body.data.0.value`. Пусто — весь результат.
    #[serde(default)]
    pub path: String,
    /// Регулярное выражение для текстового поля; значение — первая группа,
    /// например `used (\d+)K`. Число в группе становится числовой метрикой.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
}

pub type Extract = BTreeMap<String, ExtractRule>;

/// Проверяет правила заранее, чтобы ошибка в регулярке пришла при создании
/// задания, а не молча в каждом запуске.
pub fn validate_extract(extract: &Extract) -> Result<()> {
    for (name, rule) in extract {
        if name.trim().is_empty() {
            bail!("пустое имя метрики в extract");
        }
        if let Some(pattern) = &rule.regex {
            Regex::new(pattern).with_context(|| format!("некорректное регулярное выражение метрики «{name}»"))?;
        }
    }
    Ok(())
}

/// Метрики результата: по правилам `extract`, если они заданы, иначе все
/// скалярные поля результата.
pub fn collect(result: &Value, extract: Option<&Extract>) -> Metrics {
    match extract {
        Some(rules) if !rules.is_empty() => rules
            .iter()
            .filter_map(|(name, rule)| Some((name.clone(), apply_rule(result, rule)?)))
            .collect(),
        _ => {
            let mut metrics = Metrics::new();
            flatten(result, String::new(), &mut metrics);
            metrics
        }
    }
}

fn apply_rule(result: &Value, rule: &ExtractRule) -> Option<Value> {
    let value = lookup(result, &rule.path)?;
    let Some(pattern) = &rule.regex else {
        return scalar(value);
    };
    let text = match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let captures = Regex::new(pattern).ok()?.captures(&text)?;
    let matched = captures.get(1).or_else(|| captures.get(0))?.as_str().trim();
    Some(match matched.parse::<f64>() {
        Ok(number) => number_value(number),
        Err(_) => Value::String(matched.to_string()),
    })
}

fn lookup<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').filter(|s| !s.is_empty()).try_fold(value, |current, segment| match current {
        Value::Object(map) => map.get(segment),
        Value::Array(items) => items.get(segment.parse::<usize>().ok()?),
        _ => None,
    })
}

fn scalar(value: &Value) -> Option<Value> {
    match value {
        Value::Number(_) | Value::Bool(_) => Some(value.clone()),
        Value::String(s) if s.chars().count() <= MAX_TEXT_METRIC_CHARS => Some(value.clone()),
        _ => None,
    }
}

fn flatten(value: &Value, path: String, out: &mut Metrics) {
    if out.len() >= MAX_AUTO_METRICS {
        return;
    }
    let child = |key: &str| if path.is_empty() { key.to_string() } else { format!("{path}.{key}") };
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                flatten(v, child(key), out);
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                flatten(v, child(&i.to_string()), out);
            }
        }
        other => {
            if let Some(v) = scalar(other) {
                out.insert(if path.is_empty() { "value".to_string() } else { path }, v);
            }
        }
    }
}

fn number_value(n: f64) -> Value {
    serde_json::Number::from_f64(n).map(Value::Number).unwrap_or(Value::Null)
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub struct NumericStats {
    /// Сколько запусков дали это значение.
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    /// Значение в первом и последнем запуске окна.
    pub first: f64,
    pub last: f64,
    /// Изменение от первого к последнему, в процентах; нет, если первое — 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub struct TextValue {
    pub value: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub struct TextStats {
    pub count: usize,
    pub last: String,
    /// Сколько раз значение менялось между соседними запусками (например,
    /// UP → DOWN → UP — две смены).
    pub changes: usize,
    /// Самые частые значения.
    pub values: Vec<TextValue>,
}

#[derive(Debug, Default, Clone, Serialize, JsonSchema, PartialEq)]
pub struct Aggregate {
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub numeric: BTreeMap<String, NumericStats>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub text: BTreeMap<String, TextStats>,
}

/// Сводка по метрикам запусков, упорядоченных от старых к новым.
pub fn aggregate<'a>(runs: impl IntoIterator<Item = &'a Metrics>) -> Aggregate {
    let mut numbers: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for metrics in runs {
        for (key, value) in metrics {
            match value {
                Value::Number(n) => {
                    if let Some(n) = n.as_f64() {
                        numbers.entry(key.clone()).or_default().push(n);
                    }
                }
                Value::Bool(b) => texts.entry(key.clone()).or_default().push(b.to_string()),
                Value::String(s) => texts.entry(key.clone()).or_default().push(s.clone()),
                _ => {}
            }
        }
    }

    let numeric = numbers
        .into_iter()
        .map(|(key, values)| {
            let first = values[0];
            let last = values[values.len() - 1];
            let stats = NumericStats {
                count: values.len(),
                min: round(values.iter().copied().fold(f64::INFINITY, f64::min)),
                max: round(values.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                avg: round(values.iter().sum::<f64>() / values.len() as f64),
                first: round(first),
                last: round(last),
                change_percent: (first != 0.0).then(|| round((last - first) / first.abs() * 100.0)),
            };
            (key, stats)
        })
        .collect();

    let text = texts
        .into_iter()
        .map(|(key, values)| {
            let changes = values.windows(2).filter(|w| w[0] != w[1]).count();
            let mut counts: Vec<TextValue> = Vec::new();
            for v in &values {
                match counts.iter_mut().find(|c| &c.value == v) {
                    Some(c) => c.count += 1,
                    None => counts.push(TextValue { value: v.clone(), count: 1 }),
                }
            }
            counts.sort_by_key(|c| std::cmp::Reverse(c.count));
            counts.truncate(MAX_TEXT_VALUES);
            let stats = TextStats { count: values.len(), last: values[values.len() - 1].clone(), changes, values: counts };
            (key, stats)
        })
        .collect();

    Aggregate { numeric, text }
}

fn round(n: f64) -> f64 {
    (n * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flattens_scalar_fields_by_path() {
        let result = json!({
            "status": 200, "ok": true, "latency_ms": 12.5,
            "body": { "status": "UP", "items": [ {"v": 1}, {"v": 2} ], "none": null }
        });
        let m = collect(&result, None);
        assert_eq!(m["status"], json!(200));
        assert_eq!(m["ok"], json!(true));
        assert_eq!(m["body.status"], json!("UP"));
        assert_eq!(m["body.items.1.v"], json!(2));
        assert!(!m.contains_key("body.none"));
        assert_eq!(collect(&json!(42), None)["value"], json!(42));
    }

    #[test]
    fn long_text_and_huge_results_are_bounded() {
        let long = "x".repeat(MAX_TEXT_METRIC_CHARS + 1);
        assert!(collect(&json!({ "log": long }), None).is_empty());
        let big: Vec<i32> = (0..500).collect();
        assert_eq!(collect(&json!(big), None).len(), MAX_AUTO_METRICS);
    }

    #[test]
    fn extracts_numbers_from_text_with_regex() {
        let result = json!({ "pid": 7, "heap": " garbage-first heap   total 262144K, used 41234K [0x...]" });
        let extract: Extract = serde_json::from_value(json!({
            "heap_used_kb": { "path": "heap", "regex": "used (\\d+)K" },
            "heap_total_kb": { "path": "heap", "regex": "total (\\d+)K" },
            "pid": { "path": "pid" },
            "missing": { "path": "nope" }
        }))
        .unwrap();
        validate_extract(&extract).unwrap();
        let m = collect(&result, Some(&extract));
        assert_eq!(m["heap_used_kb"], json!(41234.0));
        assert_eq!(m["heap_total_kb"], json!(262144.0));
        assert_eq!(m["pid"], json!(7));
        assert!(!m.contains_key("missing"));

        let bad: Extract = serde_json::from_value(json!({ "x": { "regex": "(" } })).unwrap();
        assert!(validate_extract(&bad).is_err());
    }

    #[test]
    fn aggregates_numbers_and_text_in_run_order() {
        let runs: Vec<Metrics> = [
            json!({ "heap": 400, "state": "UP" }),
            json!({ "heap": 500, "state": "DOWN" }),
            json!({ "heap": 600, "state": "UP", "ok": true }),
        ]
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
        let a = aggregate(&runs);
        let heap = &a.numeric["heap"];
        assert_eq!((heap.count, heap.min, heap.max, heap.avg), (3, 400.0, 600.0, 500.0));
        assert_eq!((heap.first, heap.last, heap.change_percent), (400.0, 600.0, Some(50.0)));
        let state = &a.text["state"];
        assert_eq!((state.count, state.last.as_str(), state.changes), (3, "UP", 2));
        assert_eq!(state.values[0], TextValue { value: "UP".into(), count: 2 });
        assert_eq!(a.text["ok"].count, 1);
        assert_eq!(aggregate(&[]), Aggregate::default());
    }

    #[test]
    fn change_from_zero_is_undefined() {
        let runs: Vec<Metrics> = vec![
            serde_json::from_value(json!({ "n": 0 })).unwrap(),
            serde_json::from_value(json!({ "n": 5 })).unwrap(),
        ];
        assert_eq!(aggregate(&runs).numeric["n"].change_percent, None);
    }
}
