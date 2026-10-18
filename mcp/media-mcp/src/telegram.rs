//! Саммари для Telegram из `output-summary.md` (Markdown) — два файла, сам
//! сервер ничего никуда не отправляет:
//!
//! - `output-summary-telegram.md` — для вставки в приложение Telegram руками:
//!   разметка, которую приложение разбирает при отправке (`**жирный**`,
//!   `__курсив__`, `~~зачёркнутый~~`, `` `код` ``, `||спойлер||`), без
//!   экранирования;
//! - `output-summary-telegram-bot.md` — MarkdownV2 для Bot API
//!   (`parse_mode="MarkdownV2"`). Вставленный руками, он показывается как
//!   есть — с `\.`, `\(` и `__*…*__`.
//!
//! Разбирает Markdown Python-библиотека `telegramify-markdown` (`convert` —
//! текст и сущности, `markdownify` — MarkdownV2): заголовки — жирным с эмодзи
//! `✏`, пункты — `⦁`. Версия для вставки собирается из её сущностей.
//! Скрипт `telegram.py` встроен в сервер и запускается интерпретатором
//! `MEDIA_PYTHON`; по умолчанию — `media/.venv/bin/python`, если он есть
//! (`python3 -m venv media/.venv && media/.venv/bin/pip install -r
//! media/requirements.txt`), иначе `python3`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::recording::{SUMMARY, TELEGRAM_SUMMARY};

/// MarkdownV2 для Bot API — рядом с версией для вставки.
pub const TELEGRAM_BOT_SUMMARY: &str = "output-summary-telegram-bot.md";
use crate::transcribe::write_atomic;

const SCRIPT: &str = include_str!("telegram.py");
const DEFAULT_VENV_PYTHON: &str = "media/.venv/bin/python";
/// Предел одного сообщения Telegram — в единицах UTF-16.
pub const MESSAGE_LIMIT: u64 = 4096;
/// Первый запуск Python с импортом библиотеки — с запасом.
const TIMEOUT: Duration = Duration::from_secs(120);

pub fn python() -> PathBuf {
    match std::env::var("MEDIA_PYTHON").ok().filter(|p| !p.trim().is_empty()) {
        Some(p) => PathBuf::from(p.trim()),
        None if Path::new(DEFAULT_VENV_PYTHON).is_file() => PathBuf::from(DEFAULT_VENV_PYTHON),
        None => PathBuf::from("python3"),
    }
}

/// Саммари в двух видах.
#[derive(Debug, Clone)]
pub struct Converted {
    /// Для вставки в приложение Telegram.
    pub paste: String,
    /// MarkdownV2 для Bot API.
    pub markdownv2: String,
    /// Длина отправленного сообщения (без разметки) в единицах UTF-16.
    pub utf16_len: u64,
    pub version: String,
}

/// Markdown → версия для вставки и MarkdownV2.
pub async fn convert(python: &Path, markdown: &str) -> Result<Converted> {
    let mut child = Command::new(python)
        .arg("-c")
        .arg(SCRIPT)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("не удалось запустить {} (MEDIA_PYTHON)", python.display()))?;
    let mut stdin = child.stdin.take().expect("stdin piped");
    stdin.write_all(markdown.as_bytes()).await.context("не удалось передать саммари в Python")?;
    drop(stdin);
    let out = tokio::time::timeout(TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| anyhow::anyhow!("telegramify-markdown не ответил за {} с", TIMEOUT.as_secs()))??;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let last = err.trim().lines().last().unwrap_or("").to_string();
        bail!("{} (Python {}): {last}", "преобразование для Telegram не удалось", python.display());
    }
    let v: Value = serde_json::from_slice(&out.stdout).context("telegram.py вернул не JSON")?;
    let field = |k: &str| v[k].as_str().unwrap_or_default().to_string();
    let c = Converted {
        paste: field("paste"),
        markdownv2: field("markdownv2"),
        utf16_len: v["text_utf16"].as_u64().unwrap_or(0),
        version: v["version"].as_str().unwrap_or("?").to_string(),
    };
    // telegramify-markdown 1.4.0 с pyromark 0.10 молча теряет весь текст, если в нём есть `~` или `$`
    // (см. media/requirements.txt) — пустой результат не записывать.
    if !markdown.trim().is_empty() && (c.paste.trim().is_empty() || c.markdownv2.trim().is_empty()) {
        bail!(
            "telegramify-markdown вернул пустой текст — проверь версии в media/requirements.txt \
             (pyromark должен быть 0.9.x)"
        );
    }
    Ok(c)
}

/// `output-summary.md` → `output-summary-telegram.md` и `output-summary-telegram-bot.md`.
pub async fn write(python: &Path, dir: &Path) -> Result<Value> {
    let markdown = std::fs::read_to_string(dir.join(SUMMARY)).with_context(|| format!("{SUMMARY} не читается"))?;
    let c = convert(python, &markdown).await?;
    // Сначала версия для бота: версия для вставки — чекпоинт шага, она пишется последней.
    write_atomic(&dir.join(TELEGRAM_BOT_SUMMARY), c.markdownv2.as_bytes())?;
    write_atomic(&dir.join(TELEGRAM_SUMMARY), c.paste.as_bytes())?;
    let fits = c.utf16_len <= MESSAGE_LIMIT;
    Ok(json!({
        "created": [TELEGRAM_SUMMARY, TELEGRAM_BOT_SUMMARY],
        "source": SUMMARY,
        "paste": {
            "file": TELEGRAM_SUMMARY,
            "use": "вставить в приложение Telegram и отправить — **жирный**, __курсив__ приложение оформит само",
        },
        "bot": {
            "file": TELEGRAM_BOT_SUMMARY,
            "use": "Bot API: sendMessage с parse_mode=\"MarkdownV2\"; руками не вставлять — будут видны обратные слэши",
        },
        "utf16_len": c.utf16_len,
        "fits_one_message": fits,
        "note": if fits { "помещается в одно сообщение Telegram" } else { "длиннее 4096 (UTF-16) — при отправке придётся делить на сообщения" },
        "library": format!("telegramify-markdown {}", c.version),
        "sent": false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python с библиотекой: `MEDIA_PYTHON` или `media/.venv` проекта. Нет — тест пропускается.
    fn python_with_library() -> Option<PathBuf> {
        let candidates = [
            std::env::var("MEDIA_PYTHON").ok().map(PathBuf::from),
            Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(DEFAULT_VENV_PYTHON)),
        ];
        candidates.into_iter().flatten().find(|p| {
            std::process::Command::new(p).args(["-c", "import telegramify_markdown"]).output().is_ok_and(|o| o.status.success())
        })
    }

    #[tokio::test]
    async fn summary_for_paste_and_for_bot() {
        let Some(py) = python_with_library() else {
            eprintln!("нет Python с telegramify-markdown — тест пропущен");
            return;
        };
        let dir = std::env::temp_dir().join(format!("media-mcp-telegram-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(SUMMARY),
            "## Краткое резюме\n\nГлавное (1%) — **система**, около ~20 человек, $10.\n\n## Ключевые мысли\n\n\
             - *Пункт* 📚 **два**!\n- ~~старое~~ и [ссылка](https://example.com)\n",
        )
        .unwrap();
        let v = write(&py, &dir).await.unwrap();
        assert_eq!((v["fits_one_message"].as_bool(), v["sent"].as_bool()), (Some(true), Some(false)));

        // Для вставки: разметка приложения, без экранирования; `~` и `$` текст не теряют.
        let paste = std::fs::read_to_string(dir.join(TELEGRAM_SUMMARY)).unwrap();
        assert!(paste.starts_with("✏ **Краткое резюме**\n"), "{paste}");
        assert!(paste.contains("Главное (1%) — **система**, около ~20 человек, $10."), "{paste}");
        assert!(paste.contains("⦁ __Пункт__ 📚 **два**!"), "эмодзи не сбивает позиции: {paste}");
        assert!(paste.contains("⦁ ~~старое~~ и ссылка (https://example.com)"), "{paste}");
        assert!(!paste.contains('\\'), "без обратных слэшей: {paste}");

        // Для бота: MarkdownV2 с экранированием.
        let bot = std::fs::read_to_string(dir.join(TELEGRAM_BOT_SUMMARY)).unwrap();
        assert!(bot.contains("__*Краткое резюме*__"), "{bot}");
        assert!(bot.contains("\\(1%\\)") && bot.contains("*система*"), "{bot}");
    }

    #[tokio::test]
    async fn missing_python_is_explained() {
        let err = convert(Path::new("/nonexistent/python"), "x").await.unwrap_err();
        assert!(format!("{err:#}").contains("MEDIA_PYTHON"), "{err:#}");
    }
}
