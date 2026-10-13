//! Markdown ответов модели → строки ratatui с переносом по ширине ленты.
//!
//! Поддерживается то, что модели реально пишут: заголовки, **жирный**,
//! *курсив*, ~~зачёркнутый~~, `код`, блоки кода (у ```diff — цвет строк),
//! списки (вложенные, нумерованные, задачи `[x]`), цитаты, таблицы, ссылки,
//! разделители. Метки серьёзности ревью (🔴/🟠/🟢) красят свой фрагмент.

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Отрисовать `text` в строки шириной не больше `width` столбцов.
pub fn render(text: &str, width: usize) -> Vec<Line<'static>> {
    let mut r = Renderer::new(width.max(20));
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    for event in Parser::new_ext(text, options) {
        r.event(event);
    }
    r.flush();
    while r.lines.last().is_some_and(|l| l.width() == 0) {
        r.lines.pop();
    }
    r.lines
}

/// Ширина строки в столбцах терминала (эмодзи — два).
fn width_of(s: &str) -> usize {
    Span::raw(s.to_string()).width()
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    head_rows: usize,
    cell: String,
}

struct Renderer {
    width: usize,
    lines: Vec<Line<'static>>,
    /// Текст текущего блока (абзаца, пункта, заголовка) — кусками со стилями.
    runs: Vec<(String, Style)>,
    /// Вложенные строчные стили: жирный, курсив, ссылка…
    inline: Vec<Style>,
    /// Открытые списки: `Some(n)` — нумерованный, следующий номер n.
    lists: Vec<Option<u64>>,
    /// Ширина маркера каждого открытого пункта — отступ его продолжения.
    items: Vec<usize>,
    /// Маркер пункта, ещё не выведенный (первая строка пункта).
    marker: Option<String>,
    quote: usize,
    heading: Option<HeadingLevel>,
    /// Блок кода: язык и собранный текст.
    code: Option<(String, String)>,
    links: Vec<String>,
    table: Option<Table>,
}

impl Renderer {
    fn new(width: usize) -> Self {
        Self {
            width,
            lines: Vec::new(),
            runs: Vec::new(),
            inline: Vec::new(),
            lists: Vec::new(),
            items: Vec::new(),
            marker: None,
            quote: 0,
            heading: None,
            code: None,
            links: Vec::new(),
            table: None,
        }
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => {
                if let Some((_, code)) = &mut self.code {
                    code.push_str(&text);
                } else if let Some(table) = &mut self.table {
                    table.cell.push_str(&text);
                } else {
                    let style = severity_style(&text, self.style());
                    self.runs.push((text.into_string(), style));
                }
            }
            Event::Code(code) => {
                if let Some(table) = &mut self.table {
                    table.cell.push_str(&code);
                } else {
                    self.runs.push((code.into_string(), self.style().fg(Color::Yellow)));
                }
            }
            Event::SoftBreak => self.runs.push((" ".into(), Style::default())),
            Event::HardBreak => self.runs.push(("\n".into(), Style::default())),
            Event::Rule => {
                self.flush();
                let rule = "─".repeat(self.width.saturating_sub(self.indent_width()).min(60));
                let mut spans = self.prefix(false);
                spans.push(Span::styled(rule, Style::default().fg(Color::DarkGray)));
                self.lines.push(Line::from(spans));
                self.blank();
            }
            Event::TaskListMarker(done) => {
                self.marker = Some(if done { "☑ ".into() } else { "☐ ".into() });
            }
            Event::Html(html) | Event::InlineHtml(html) => self.runs.push((html.into_string(), Style::default())),
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag) {
        match tag {
            Tag::Heading { level, .. } => {
                self.flush();
                self.blank();
                self.heading = Some(level);
            }
            Tag::Paragraph => self.flush(),
            Tag::BlockQuote(_) => {
                self.flush();
                self.quote += 1;
            }
            Tag::CodeBlock(kind) => {
                self.flush();
                let lang = match kind {
                    CodeBlockKind::Fenced(lang) => lang.split_whitespace().next().unwrap_or("").to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                self.code = Some((lang, String::new()));
            }
            Tag::List(start) => {
                self.flush();
                self.lists.push(start);
            }
            Tag::Item => {
                self.flush();
                let depth = self.lists.len().saturating_sub(1);
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => ["• ", "◦ ", "▪ "][depth % 3].to_string(),
                };
                self.items.push(width_of(&marker));
                self.marker = Some(marker);
            }
            Tag::Emphasis => self.push_inline(Style::default().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.push_inline(Style::default().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => self.push_inline(Style::default().add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { dest_url, .. } => {
                self.links.push(dest_url.into_string());
                self.push_inline(Style::default().fg(Color::LightBlue).add_modifier(Modifier::UNDERLINED));
            }
            Tag::Table(_) => {
                self.flush();
                self.table = Some(Table::default());
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(t) = &mut self.table {
                    t.rows.push(Vec::new());
                }
            }
            Tag::TableCell => {
                if let Some(t) = &mut self.table {
                    t.cell.clear();
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Heading(_) => {
                self.flush();
                self.heading = None;
                self.blank();
            }
            TagEnd::Paragraph => {
                self.flush();
                // В пунктах списка абзацы идут подряд, вне списков — через пустую строку.
                if self.lists.is_empty() {
                    self.blank();
                }
            }
            TagEnd::BlockQuote(_) => {
                self.flush();
                self.quote = self.quote.saturating_sub(1);
                self.blank();
            }
            TagEnd::CodeBlock => {
                if let Some((lang, code)) = self.code.take() {
                    self.code_block(&lang, &code);
                }
                self.blank();
            }
            TagEnd::List(_) => {
                self.flush();
                self.lists.pop();
                if self.lists.is_empty() {
                    self.blank();
                }
            }
            TagEnd::Item => {
                self.flush();
                self.items.pop();
                self.marker = None;
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.inline.pop();
            }
            TagEnd::Link => {
                self.inline.pop();
                if let Some(url) = self.links.pop() {
                    let text: String = self.runs.iter().map(|(t, _)| t.as_str()).collect();
                    if !url.is_empty() && !text.ends_with(url.as_str()) && !url.starts_with('#') {
                        self.runs.push((format!(" ({url})"), Style::default().fg(Color::DarkGray)));
                    }
                }
            }
            TagEnd::TableCell => {
                if let Some(t) = &mut self.table {
                    let cell = std::mem::take(&mut t.cell);
                    if let Some(row) = t.rows.last_mut() {
                        row.push(cell.trim().to_string());
                    }
                }
            }
            TagEnd::TableHead => {
                if let Some(t) = &mut self.table {
                    t.head_rows = t.rows.len();
                }
            }
            TagEnd::Table => {
                if let Some(t) = self.table.take() {
                    self.table_lines(t);
                }
                self.blank();
            }
            _ => {}
        }
    }

    fn push_inline(&mut self, style: Style) {
        let combined = self.style().patch(style);
        self.inline.push(combined);
    }

    /// Стиль текста сейчас: заголовок, цитата и вложенные строчные стили.
    fn style(&self) -> Style {
        let mut style = match self.heading {
            Some(HeadingLevel::H1) => {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            }
            Some(HeadingLevel::H2) => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            Some(_) => Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD),
            None if self.quote > 0 => Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
            None => Style::default(),
        };
        if let Some(inline) = self.inline.last() {
            style = style.patch(*inline);
        }
        style
    }

    fn indent_width(&self) -> usize {
        self.quote * 2 + self.items.iter().sum::<usize>()
    }

    /// Отступ строки: полосы цитат, отступы пунктов и (на первой строке
    /// пункта) его маркер.
    fn prefix(&mut self, first: bool) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        if self.quote > 0 {
            spans.push(Span::styled("▌ ".repeat(self.quote), Style::default().fg(Color::DarkGray)));
        }
        let all: usize = self.items.iter().sum();
        match self.marker.take().filter(|_| first) {
            Some(marker) => {
                let own = self.items.last().copied().unwrap_or(0);
                spans.push(Span::raw(" ".repeat(all - own.min(all))));
                spans.push(Span::styled(marker, Style::default().fg(Color::Cyan)));
            }
            None if all > 0 => spans.push(Span::raw(" ".repeat(all))),
            None => {}
        }
        spans
    }

    /// Вывести накопленный текст блока с переносом по словам.
    fn flush(&mut self) {
        if self.runs.iter().all(|(t, _)| t.trim().is_empty()) {
            self.runs.clear();
            return;
        }
        let runs = std::mem::take(&mut self.runs);
        let first = self.prefix(true);
        let cont = self.prefix(false);
        let lines = wrap(&runs, first, cont, self.width);
        self.lines.extend(lines);
    }

    fn blank(&mut self) {
        if self.lines.last().is_some_and(|l| l.width() > 0) {
            self.lines.push(Line::default());
        }
    }

    fn code_block(&mut self, lang: &str, code: &str) {
        let pad = self.prefix(false);
        let pad_width: usize = pad.iter().map(|s| s.width()).sum();
        let room = self.width.saturating_sub(pad_width + 4).max(10);
        let diff = lang == "diff";
        for raw in code.trim_end_matches('\n').lines() {
            let style = match raw.chars().next() {
                Some('+') if diff => Style::default().fg(Color::Green),
                Some('-') if diff => Style::default().fg(Color::Red),
                Some('@') if diff => Style::default().fg(Color::Cyan),
                _ => Style::default().fg(Color::Yellow),
            };
            // Код не переносится по словам — режется по ширине.
            let chars: Vec<char> = raw.chars().collect();
            let chunks: Vec<String> = if chars.is_empty() {
                vec![String::new()]
            } else {
                chars.chunks(room).map(|c| c.iter().collect()).collect()
            };
            for chunk in chunks {
                let mut spans = pad.clone();
                spans.push(Span::styled("  │ ", Style::default().fg(Color::DarkGray)));
                spans.push(Span::styled(chunk, style));
                self.lines.push(Line::from(spans));
            }
        }
    }

    fn table_lines(&mut self, t: Table) {
        let cols = t.rows.iter().map(Vec::len).max().unwrap_or(0);
        if cols == 0 {
            return;
        }
        let pad = self.prefix(false);
        let pad_width: usize = pad.iter().map(|s| s.width()).sum();
        let mut widths: Vec<usize> = (0..cols)
            .map(|c| t.rows.iter().filter_map(|r| r.get(c)).map(|s| width_of(s)).max().unwrap_or(0).max(1))
            .collect();
        // Не влезает — сужаем самые широкие колонки (текст в них переносится).
        let room = self.width.saturating_sub(pad_width + 3 * (cols - 1)).max(cols * 4);
        while widths.iter().sum::<usize>() > room {
            let (i, &w) = widths.iter().enumerate().max_by_key(|(_, &w)| w).expect("есть колонки");
            if w <= 4 {
                break;
            }
            widths[i] = w - 1;
        }
        let sep = Style::default().fg(Color::DarkGray);
        for (ri, row) in t.rows.iter().enumerate() {
            let head = ri < t.head_rows;
            let cells: Vec<Vec<String>> = (0..cols)
                .map(|c| {
                    let text = row.get(c).map(String::as_str).unwrap_or("");
                    let wrapped: Vec<String> = textwrap::wrap(text, widths[c]).into_iter().map(|s| s.into_owned()).collect();
                    if wrapped.is_empty() {
                        vec![String::new()]
                    } else {
                        wrapped
                    }
                })
                .collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1);
            for h in 0..height {
                let mut spans = pad.clone();
                for (c, cell) in cells.iter().enumerate() {
                    if c > 0 {
                        spans.push(Span::styled(" │ ", sep));
                    }
                    let text = cell.get(h).cloned().unwrap_or_default();
                    let fill = widths[c].saturating_sub(width_of(&text));
                    let style = if head { Style::default().add_modifier(Modifier::BOLD) } else { Style::default() };
                    spans.push(Span::styled(format!("{text}{}", " ".repeat(fill)), style));
                }
                self.lines.push(Line::from(spans));
            }
            if head && ri + 1 == t.head_rows {
                let mut spans = pad.clone();
                let line: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
                spans.push(Span::styled(line.join("─┼─"), sep));
                self.lines.push(Line::from(spans));
            }
        }
    }
}

/// 🔴/🟠/🟢 — цвет серьёзности замечания ревью.
fn severity_style(text: &str, base: Style) -> Style {
    if text.contains('🔴') {
        base.fg(Color::Red)
    } else if text.contains('🟠') {
        base.fg(Color::Yellow)
    } else if text.contains('🟢') {
        base.fg(Color::Green)
    } else {
        base
    }
}

/// Перенос кусков текста со стилями по словам: первая строка — с `first`,
/// остальные — с `cont`. `\n` в куске — принудительный перенос.
fn wrap(runs: &[(String, Style)], first: Vec<Span<'static>>, cont: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
    let cont_width: usize = cont.iter().map(|s| s.width()).sum();
    let mut lines = Vec::new();
    let mut line = first;
    let mut start: usize = line.iter().map(|s| s.width()).sum();
    let mut used = start;
    let mut space: Option<Style> = None;

    let mut newline = |line: &mut Vec<Span<'static>>, used: &mut usize, start: &mut usize| {
        lines.push(Line::from(std::mem::replace(line, cont.clone())));
        *used = cont_width;
        *start = cont_width;
    };

    for (text, style) in runs {
        let mut word = String::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '\n' && !c.is_whitespace() {
                word.push(c);
                if chars.peek().is_some_and(|n| !n.is_whitespace()) {
                    continue;
                }
            }
            if !word.is_empty() {
                let w = width_of(&word);
                let gap = usize::from(space.is_some() && used > start);
                if used + gap + w > width && used > start {
                    newline(&mut line, &mut used, &mut start);
                } else if gap == 1 {
                    line.push(Span::styled(" ", space.unwrap_or_default()));
                    used += 1;
                }
                // Слово длиннее строки — режем по символам.
                let mut rest: Vec<char> = word.chars().collect();
                while width_of(&rest.iter().collect::<String>()) > width.saturating_sub(used) && used < width {
                    let room = width - used;
                    let mut take = 0;
                    let mut acc = 0;
                    for ch in &rest {
                        let cw = width_of(&ch.to_string());
                        if acc + cw > room {
                            break;
                        }
                        acc += cw;
                        take += 1;
                    }
                    if take == 0 {
                        newline(&mut line, &mut used, &mut start);
                        continue;
                    }
                    line.push(Span::styled(rest[..take].iter().collect::<String>(), *style));
                    rest.drain(..take);
                    newline(&mut line, &mut used, &mut start);
                }
                if !rest.is_empty() {
                    let piece: String = rest.into_iter().collect();
                    used += width_of(&piece);
                    line.push(Span::styled(piece, *style));
                }
                word.clear();
                space = None;
            }
            if c == '\n' {
                newline(&mut line, &mut used, &mut start);
                space = None;
            } else if c.is_whitespace() {
                space = Some(*style);
            }
        }
    }
    if used > start || lines.is_empty() {
        lines.push(Line::from(line));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>().trim_end().to_string()).collect()
    }

    fn span_with<'a>(lines: &'a [Line<'static>], text: &str) -> &'a Span<'static> {
        lines.iter().flat_map(|l| l.spans.iter()).find(|s| s.content.contains(text)).expect(text)
    }

    #[test]
    fn headings_emphasis_and_code() {
        let lines = render("# Обзор\n\nТекст с **жирным**, *курсивом* и `кодом`.", 80);
        assert_eq!(plain(&lines), vec!["Обзор", "", "Текст с жирным, курсивом и кодом."]);
        assert!(span_with(&lines, "Обзор").style.add_modifier.contains(Modifier::BOLD));
        assert!(span_with(&lines, "жирным").style.add_modifier.contains(Modifier::BOLD));
        assert!(span_with(&lines, "курсивом").style.add_modifier.contains(Modifier::ITALIC));
        assert_eq!(span_with(&lines, "кодом").style.fg, Some(Color::Yellow));
    }

    #[test]
    fn lists_nest_and_wrap_under_marker() {
        let md = "- первый пункт с длинным текстом, который переносится\n  - вложенный\n1. раз\n2. два";
        let lines = render(md, 30);
        assert_eq!(
            plain(&lines),
            vec!["• первый пункт с длинным", "  текстом, который переносится", "  ◦ вложенный", "", "1. раз", "2. два"]
        );
    }

    #[test]
    fn code_blocks_keep_lines_and_color_diff() {
        let lines = render("```diff\n+ добавлено\n- удалено\n  fn main() {}\n```", 40);
        assert_eq!(plain(&lines), vec!["  │ + добавлено", "  │ - удалено", "  │   fn main() {}"]);
        assert_eq!(span_with(&lines, "+ добавлено").style.fg, Some(Color::Green));
        assert_eq!(span_with(&lines, "- удалено").style.fg, Some(Color::Red));
    }

    #[test]
    fn tables_are_aligned() {
        let md = "| Сервер | Порт |\n|---|---|\n| scheduler | 8092 |\n| git | — |";
        let lines = render(md, 60);
        assert_eq!(plain(&lines), vec!["Сервер    │ Порт", "──────────┼─────", "scheduler │ 8092", "git       │ —"]);
    }

    #[test]
    fn quotes_links_rules_and_severity() {
        let lines = render("> важно\n\n[docs](docs/api.md)\n\n---\n\n- **🔴 high** `a.rs:3` — паника", 60);
        let text = plain(&lines);
        assert_eq!(text[0], "▌ важно");
        assert_eq!(text[2], "docs (docs/api.md)");
        assert!(text[4].starts_with("────"));
        assert_eq!(text[6], "• 🔴 high a.rs:3 — паника");
        assert_eq!(span_with(&lines, "🔴").style.fg, Some(Color::Red));
        assert!(lines.iter().all(|l| l.width() <= 60));
    }

    #[test]
    fn long_words_are_cut_and_lines_fit() {
        let lines = render(&format!("слово {}", "x".repeat(50)), 20);
        assert!(lines.iter().all(|l| l.width() <= 20), "{:?}", plain(&lines));
        assert_eq!(plain(&lines).concat().replace(' ', "").len(), "слово".len() + 50);
    }
}
