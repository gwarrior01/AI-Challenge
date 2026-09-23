//! Время в планировщике: хранится как миллисекунды Unix (UTC), наружу
//! отдаётся в RFC 3339 с часовым поясом машины. Модель не знает текущего
//! времени, поэтому интервалы она задаёт относительно («30s», «1h30m»), а
//! абсолютные моменты сверяет с полем `now` в каждом ответе сервера.

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Момент времени для ответа: `2026-09-23T14:05:00+03:00`.
pub fn format(ms: i64) -> String {
    match Local.timestamp_millis_opt(ms).single() {
        Some(t) => t.to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
        None => ms.to_string(),
    }
}

/// Длительность вида `30s`, `5m`, `1h30m`, `2d`; без единицы — секунды.
pub fn parse_duration(text: &str) -> Result<i64> {
    let text = text.trim();
    if text.is_empty() {
        bail!("пустая длительность");
    }
    if let Ok(sec) = text.parse::<u64>() {
        return Ok(sec as i64 * 1000);
    }
    let mut total: i64 = 0;
    let mut number = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        if c.is_whitespace() {
            continue;
        }
        let unit_ms = match c {
            's' => 1_000,
            'm' => 60_000,
            'h' => 3_600_000,
            'd' => 86_400_000,
            _ => bail!("некорректная длительность «{text}»: ожидается вроде 30s, 5m, 1h30m, 2d"),
        };
        let value: i64 = number
            .parse()
            .map_err(|_| anyhow!("некорректная длительность «{text}»: перед «{c}» нет числа"))?;
        total = total.saturating_add(value.saturating_mul(unit_ms));
        number.clear();
    }
    if !number.is_empty() {
        bail!("некорректная длительность «{text}»: у числа {number} нет единицы (s, m, h, d)");
    }
    Ok(total)
}

/// Абсолютный момент: RFC 3339 (`2026-09-23T18:00:00+03:00`), местное время
/// (`2026-09-23 18:00`, `2026-09-23T18:00:00`) или только время суток
/// (`18:00`) — ближайшее такое время после `now`.
pub fn parse_at(text: &str, now: i64) -> Result<i64> {
    let text = text.trim();
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Ok(t.timestamp_millis());
    }
    for format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return local_ms(naive);
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return local_ms(date.and_time(NaiveTime::MIN));
    }
    for format in ["%H:%M:%S", "%H:%M"] {
        if let Ok(time) = NaiveTime::parse_from_str(text, format) {
            let today = Local.timestamp_millis_opt(now).single().ok_or_else(|| anyhow!("некорректное текущее время"))?;
            let mut candidate = local_ms(today.date_naive().and_time(time))?;
            if candidate <= now {
                candidate = local_ms(today.date_naive().succ_opt().unwrap_or(today.date_naive()).and_time(time))?;
            }
            return Ok(candidate);
        }
    }
    bail!("некорректный момент «{text}»: ожидается RFC 3339 (2026-09-23T18:00:00+03:00), «2026-09-23 18:00» или «18:00»")
}

/// Начало окна выборки: длительность назад от `now` (`30m`, `24h`) или
/// абсолютный момент, как в [`parse_at`] (кроме «только времени»: «18:00»
/// здесь — сегодня, а не завтра).
pub fn parse_since(text: &str, now: i64) -> Result<i64> {
    if let Ok(duration) = parse_duration(text) {
        return Ok(now - duration);
    }
    let at = parse_at(text, now)?;
    Ok(if at > now { at - 86_400_000 } else { at })
}

fn local_ms(naive: NaiveDateTime) -> Result<i64> {
    Local
        .from_local_datetime(&naive)
        .earliest()
        .map(|t| t.timestamp_millis())
        .ok_or_else(|| anyhow!("момент {naive} не существует в местном часовом поясе"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_durations() {
        assert_eq!(parse_duration("30s").unwrap(), 30_000);
        assert_eq!(parse_duration("5m").unwrap(), 300_000);
        assert_eq!(parse_duration("1h30m").unwrap(), 5_400_000);
        assert_eq!(parse_duration("2d").unwrap(), 172_800_000);
        assert_eq!(parse_duration("45").unwrap(), 45_000);
        assert_eq!(parse_duration("1h 5m").unwrap(), 3_900_000);
        assert!(parse_duration("5").is_ok());
        assert!(parse_duration("5x").is_err());
        assert!(parse_duration("m").is_err());
        assert!(parse_duration("10m5").is_err());
    }

    #[test]
    fn parses_absolute_moments() {
        let t = parse_at("2026-09-23T18:00:00+03:00", 0).unwrap();
        assert_eq!(format(t), format(DateTime::parse_from_rfc3339("2026-09-23T15:00:00Z").unwrap().timestamp_millis()));
        let local = parse_at("2026-09-23 18:00", 0).unwrap();
        assert_eq!(parse_at("2026-09-23T18:00:00", 0).unwrap(), local);
        assert!(parse_at("завтра", 0).is_err());
    }

    #[test]
    fn time_of_day_is_the_next_occurrence() {
        let now = parse_at("2026-09-23 18:00", 0).unwrap();
        assert_eq!(parse_at("19:30", now).unwrap(), parse_at("2026-09-23 19:30", 0).unwrap());
        assert_eq!(parse_at("09:00", now).unwrap(), parse_at("2026-09-24 09:00", 0).unwrap());
        assert_eq!(parse_since("09:00", now).unwrap(), parse_at("2026-09-23 09:00", 0).unwrap());
        assert_eq!(parse_since("1h", now).unwrap(), now - 3_600_000);
    }
}
