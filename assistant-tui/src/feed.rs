//! Лента ассистента: что спросил пользователь, контекст, вызовы инструментов
//! (плашки « MCP » и « RAG »), ответы в Markdown, вывод команд с цветом
//! диффа. Прокрутка — от низа ленты.

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};

use llm_core::{ToolCallRecord, Usage};

use crate::markdown;
use crate::prompt::{command_at, command_style, highlight, Hints};

/// «Сервер» поиска по индексам в записях вызовов: плашка « RAG », а не « MCP ».
pub const RAG_SERVER: &str = "rag";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Info,
    User,
    Context,
    /// Вызов MCP-инструмента: первая строка — плашка, дальше — результат.
    Mcp,
    McpError,
    /// Вызов поиска по индексам — та же плашка, но « RAG ».
    Rag,
    RagError,
    /// Ответ модели — Markdown.
    Answer,
    /// Вывод команд как есть; у строк диффа — цвет.
    Output,
    Error,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub kind: Kind,
    pub text: String,
}

impl Entry {
    pub fn new(kind: Kind, text: impl Into<String>) -> Self {
        Self { kind, text: text.into() }
    }
}

#[derive(Default)]
pub struct Feed {
    pub entries: Vec<Entry>,
    /// Прокрутка от низа ленты, в строках.
    pub scroll_back: usize,
}

impl Feed {
    pub fn push(&mut self, kind: Kind, text: impl Into<String>) {
        self.entries.push(Entry::new(kind, text));
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.scroll_back = 0;
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect, hints: Hints) {
        let width = area.width.saturating_sub(2).max(10) as usize;
        let lines = render_entries(&self.entries, width, hints);
        let height = area.height.saturating_sub(2) as usize;
        let max_scroll = lines.len().saturating_sub(height);
        let top = max_scroll.saturating_sub(self.scroll_back.min(max_scroll));
        let title = if self.scroll_back > 0 && max_scroll > 0 {
            format!(" Лента — {}/{} (Ctrl+End — в конец) ", top + height.min(lines.len()), lines.len())
        } else {
            " Лента — PgUp/PgDn прокрутка ".to_string()
        };
        let feed = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(title))
            .scroll((top.min(u16::MAX as usize) as u16, 0));
        frame.render_widget(feed, area);
    }
}

/// Плашка вызова инструмента: `git › read_file({"path":…})  ✓ · вызвала модель`
/// и ниже — начало результата. `who` пустой — вызов сделал сам ассистент.
pub fn tool_entry(call: &ToolCallRecord, who: &str) -> Entry {
    let server = call.server.as_deref().unwrap_or("?");
    let args = match call.arguments.trim() {
        "" | "{}" => String::new(),
        a => short(a, 100),
    };
    let rag = server == RAG_SERVER;
    let (kind, mark) = match (rag, call.is_error) {
        (false, false) => (Kind::Mcp, "✓"),
        (false, true) => (Kind::McpError, "✗ ошибка"),
        (true, false) => (Kind::Rag, "✓"),
        (true, true) => (Kind::RagError, "✗ ошибка"),
    };
    Entry::new(kind, format!("{server} › {}({args})  {mark}{}\n  ↳ {}", call.tool, caller(who), call.result_preview(180)))
}

/// Вызовы одной плашкой — когда их десятки: `git_diff ×25, read_file ×24`.
pub fn tool_summary_entry(calls: &[ToolCallRecord], who: &str) -> Entry {
    let mut counts: Vec<(String, String, usize, usize)> = Vec::new();
    for c in calls {
        let server = c.server.clone().unwrap_or_default();
        match counts.iter_mut().find(|(s, t, ..)| *s == server && *t == c.tool) {
            Some(entry) => {
                entry.2 += 1;
                entry.3 += c.is_error as usize;
            }
            None => counts.push((server, c.tool.clone(), 1, c.is_error as usize)),
        }
    }
    let errors: usize = counts.iter().map(|c| c.3).sum();
    let mut servers: Vec<&str> = counts.iter().map(|c| c.0.as_str()).collect();
    servers.dedup();
    let server = if servers.is_empty() { "?".to_string() } else { servers.join("+") };
    let list: Vec<String> = counts
        .iter()
        .map(|(_, tool, n, err)| if *err > 0 { format!("{tool} ×{n} (ошибок {err})") } else { format!("{tool} ×{n}") })
        .collect();
    let mark = if errors > 0 { format!("✗ ошибок {errors}") } else { "✓".into() };
    Entry::new(
        if errors > 0 { Kind::McpError } else { Kind::Mcp },
        format!("{server} › вызовов {}  {mark}{}\n  ↳ {}", calls.len(), caller(who), list.join(", ")),
    )
}

/// ` · вызвала модель` — кто вызвал; пусто — без подписи.
fn caller(who: &str) -> String {
    if who.is_empty() {
        String::new()
    } else {
        format!(" · {who}")
    }
}

/// Ответ и под ним — токены запроса.
pub fn with_usage(text: String, usage: Option<Usage>) -> String {
    match usage {
        Some(u) => format!("{text}\n\n[токены: {} (запрос {}, ответ {})]", u.total_tokens, u.prompt_tokens, u.completion_tokens),
        None => text,
    }
}

fn short(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > max {
        flat.chars().take(max).collect::<String>() + "…"
    } else {
        flat
    }
}

/// Строки плашки: первая — ` MCP ` на цветном фоне и `сервер › инструмент(…)`,
/// остальные (результат) — приглушённые, с отступом под плашку.
fn badge_lines(kind: Kind, first: bool, raw: &str, width: usize) -> Vec<Line<'static>> {
    let (badge, bg) = match kind {
        Kind::Rag => (" RAG ", Color::Blue),
        Kind::RagError => (" RAG ", Color::Red),
        Kind::McpError => (" MCP ", Color::Red),
        _ => (" MCP ", Color::Magenta),
    };
    let indent = " ".repeat(badge.chars().count() + 1);
    let style = if first {
        Style::default().fg(bg).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let text_width = width.saturating_sub(indent.chars().count()).max(10);
    textwrap::wrap(raw.trim_start(), text_width)
        .into_iter()
        .enumerate()
        .map(|(j, part)| {
            let lead = if first && j == 0 {
                // Белый жирный на цветном фоне — читается и в светлой, и в тёмной теме.
                Span::styled(badge, Style::default().fg(Color::White).bg(bg).add_modifier(Modifier::BOLD))
            } else {
                Span::raw(" ".repeat(badge.chars().count()))
            };
            Line::from(vec![lead, Span::raw(" "), Span::styled(part.into_owned(), style)])
        })
        .collect()
}

pub fn render_entries(entries: &[Entry], width: usize, hints: Hints) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in entries {
        match entry.kind {
            Kind::Answer => lines.extend(markdown::render(&entry.text, width)),
            Kind::Mcp | Kind::McpError | Kind::Rag | Kind::RagError => {
                for (i, raw) in entry.text.lines().enumerate() {
                    lines.extend(badge_lines(entry.kind, i == 0, raw, width));
                }
            }
            kind => {
                let base = match kind {
                    Kind::Info => Style::default().fg(Color::DarkGray),
                    Kind::User => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                    Kind::Context => Style::default().fg(Color::Blue),
                    Kind::Error => Style::default().fg(Color::Red),
                    _ => Style::default(),
                };
                for (i, raw) in entry.text.lines().enumerate() {
                    let text = if kind == Kind::User && i == 0 { format!("› {raw}") } else { raw.to_string() };
                    let style = if kind == Kind::Output { diff_style(raw, base) } else { base };
                    if text.is_empty() {
                        lines.push(Line::default());
                        continue;
                    }
                    for (j, part) in textwrap::wrap(&text, width).into_iter().enumerate() {
                        // Команда в строке ввода пользователя — тем же цветом, что при наборе.
                        let command = (kind == Kind::User && i == 0 && j == 0).then(|| command_at(raw, hints)).flatten();
                        match command {
                            Some((len, hint)) => {
                                lines.push(Line::from(highlight(&part, 2..2 + len, command_style(hint.is_some()), style)))
                            }
                            None => lines.push(Line::from(Span::styled(part.into_owned(), style))),
                        }
                    }
                }
            }
        }
        lines.push(Line::default());
    }
    lines
}

/// Цвет строки диффа.
pub fn diff_style(line: &str, base: Style) -> Style {
    if line.starts_with("+++") || line.starts_with("---") || line.starts_with("diff --git") {
        Style::default().add_modifier(Modifier::BOLD)
    } else if line.starts_with('+') {
        Style::default().fg(Color::Green)
    } else if line.starts_with('-') {
        Style::default().fg(Color::Red)
    } else if line.starts_with("@@") {
        Style::default().fg(Color::Cyan)
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(tool: &str, args: &str, is_error: bool) -> ToolCallRecord {
        ToolCallRecord {
            server: Some("git".into()),
            tool: tool.into(),
            arguments: args.into(),
            result: "{\"branch\": \"day-30\"}".into(),
            is_error,
        }
    }

    #[test]
    fn tool_calls_are_marked() {
        let e = tool_entry(&call("git_branch", "{}", false), "");
        assert!(e.kind == Kind::Mcp);
        assert_eq!(e.text, "git › git_branch()  ✓\n  ↳ {\"branch\": \"day-30\"}");
        let e = tool_entry(&call("read_file", r#"{"path":"x"}"#, true), "вызвала модель");
        assert!(e.kind == Kind::McpError);
        assert!(e.text.starts_with(r#"git › read_file({"path":"x"})  ✗ ошибка · вызвала модель"#));

        let calls = [call("git_diff", "{}", false), call("read_file", "{}", false), call("read_file", "{}", true)];
        let e = tool_summary_entry(&calls, "");
        assert_eq!(e.text, "git › вызовов 3  ✗ ошибок 1\n  ↳ git_diff ×1, read_file ×2 (ошибок 1)");

        let mut rag = call("search_docs", r#"{"query":"RAG"}"#, false);
        rag.server = Some(RAG_SERVER.into());
        let e = tool_entry(&rag, "вызвала модель");
        assert!(e.kind == Kind::Rag);
        assert_eq!(badge_lines(Kind::Rag, true, "rag › search_docs()", 80)[0].spans[0].content, " RAG ");

        let lines = badge_lines(Kind::Mcp, true, "git › git_branch()", 80);
        assert_eq!(lines[0].spans[0].content, " MCP ");
        assert_eq!(lines[0].spans[2].content, "git › git_branch()");
    }

    #[test]
    fn diff_lines_are_colored() {
        assert_eq!(diff_style("+new", Style::default()).fg, Some(Color::Green));
        assert_eq!(diff_style("-old", Style::default()).fg, Some(Color::Red));
        assert_eq!(diff_style("@@ -1 +1 @@", Style::default()).fg, Some(Color::Cyan));
        assert_eq!(diff_style("--- a/x", Style::default()).fg, None);
    }
}
