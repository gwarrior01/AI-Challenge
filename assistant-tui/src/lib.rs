//! Общие куски TUI ассистентов проекта (`code-assistant`, `file-assistant`):
//!
//! - [`feed`] — лента: записи разных видов, Markdown ответов, плашки вызовов
//!   MCP/RAG, цвет диффа, прокрутка;
//! - [`prompt`] — строка ввода с историей, подсветкой `/команды` и подсказкой
//!   к ней в строке состояния;
//! - [`markdown`], [`input`] — разбор Markdown и редактирование строки;
//! - [`terminal`] — вход в полноэкранный режим и выход из него.
//!
//! Что делают команды и как выглядит шапка, решает каждый ассистент сам.

pub mod feed;
pub mod input;
pub mod markdown;
pub mod prompt;
pub mod terminal;

pub use feed::{Entry, Feed, Kind};
pub use prompt::{Action, Hints, Prompt};

/// Анимация «идёт работа» в строке состояния.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
