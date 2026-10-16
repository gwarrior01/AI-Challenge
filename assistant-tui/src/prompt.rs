//! Строка ввода ассистента: редактирование ([`InputLine`]), история по ↑/↓,
//! подсветка `/команды` (известная — голубым, неизвестная — красным) и
//! подсказка к ней в строке состояния. Клавиши ленты (PgUp/PgDn,
//! Ctrl+Home/End, Ctrl+L) обрабатываются здесь же — у всех ассистентов они
//! одинаковые.

use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};

use crate::feed::Feed;
use crate::input::InputLine;
use crate::SPINNER;

/// Команды и подсказки к ним: `("review", "/review [base] — …")`.
pub type Hints = &'static [(&'static str, &'static str)];

/// Что сделать после события клавиатуры.
#[derive(Debug, PartialEq)]
pub enum Action {
    None,
    /// Enter: введённая строка (без пробелов по краям, не пустая).
    Submit(String),
    /// Esc или Ctrl+C.
    Quit,
}

pub struct Prompt {
    pub input: InputLine,
    hints: Hints,
    history: Vec<String>,
    history_pos: Option<usize>,
}

impl Prompt {
    pub fn new(hints: Hints) -> Self {
        Self { input: InputLine::default(), hints, history: Vec::new(), history_pos: None }
    }

    pub fn hints(&self) -> Hints {
        self.hints
    }

    pub fn handle(&mut self, event: Event, feed: &mut Feed) -> Action {
        let key = match event {
            Event::Paste(text) => {
                self.input.insert_str(&text);
                return Action::None;
            }
            Event::Key(key) if key.kind == KeyEventKind::Press => key,
            _ => return Action::None,
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let word = ctrl || alt;
        let input = &mut self.input;
        match key.code {
            KeyCode::Esc => return Action::Quit,
            KeyCode::Char('c') if ctrl => return Action::Quit,
            KeyCode::Char('l') if ctrl => feed.clear(),
            KeyCode::Char('a') if ctrl => input.home(),
            KeyCode::Char('e') if ctrl => input.end(),
            KeyCode::Char('u') if ctrl => input.kill_to_start(),
            KeyCode::Char('k') if ctrl => input.kill_to_end(),
            KeyCode::Char('w') if ctrl => input.delete_word_back(),
            // Alt+←/→ в Terminal.app и iTerm2 приходят как Esc+b / Esc+f.
            KeyCode::Char('b') if alt => input.word_left(),
            KeyCode::Char('f') if alt => input.word_right(),
            KeyCode::Char('d') if alt => input.delete_word_forward(),
            KeyCode::Char(c) if !ctrl => input.insert(c),
            KeyCode::Backspace if word => input.delete_word_back(),
            KeyCode::Backspace => input.backspace(),
            KeyCode::Delete => input.delete(),
            KeyCode::Left if word => input.word_left(),
            KeyCode::Right if word => input.word_right(),
            KeyCode::Left => input.left(),
            KeyCode::Right => input.right(),
            KeyCode::Home if ctrl => feed.scroll_back = usize::MAX / 2,
            KeyCode::End if ctrl => feed.scroll_back = 0,
            KeyCode::Home => input.home(),
            KeyCode::End => input.end(),
            KeyCode::PageUp => feed.scroll_back += 10,
            KeyCode::PageDown => feed.scroll_back = feed.scroll_back.saturating_sub(10),
            KeyCode::Up if !self.history.is_empty() => {
                let pos = self.history_pos.map(|p| p.saturating_sub(1)).unwrap_or(self.history.len() - 1);
                self.history_pos = Some(pos);
                self.input.set(&self.history[pos]);
            }
            KeyCode::Down => match self.history_pos {
                Some(p) if p + 1 < self.history.len() => {
                    self.history_pos = Some(p + 1);
                    self.input.set(&self.history[p + 1]);
                }
                _ => {
                    self.history_pos = None;
                    self.input.clear();
                }
            },
            KeyCode::Enter => return self.submit(feed),
            _ => {}
        }
        Action::None
    }

    fn submit(&mut self, feed: &mut Feed) -> Action {
        let text = self.input.text().trim().to_string();
        if text.is_empty() {
            return Action::None;
        }
        self.input.clear();
        self.history_pos = None;
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        feed.scroll_back = 0;
        Action::Submit(text)
    }

    /// Поле ввода в рамке с заголовком `title`; курсор — на своём месте.
    pub fn draw(&self, frame: &mut Frame, area: Rect, title: &str) {
        // Длинный ввод — видна его последняя часть.
        let room = area.width.saturating_sub(5) as usize;
        let (shown, col) = self.input.view(room);
        let (start, _) = self.input.window(room);
        let text = self.input.text();
        let mut spans = vec![Span::raw("› ")];
        match command_at(&text, self.hints) {
            // Команда (или её видимая часть) — своим цветом.
            Some((len, hint)) => spans.extend(highlight(
                &shown,
                0..len.saturating_sub(start),
                command_style(hint.is_some()),
                Style::default(),
            )),
            None => spans.push(Span::raw(shown)),
        }
        let prompt = Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(format!(" {title} ")));
        frame.render_widget(prompt, area);
        frame.set_cursor_position((area.x + 3 + col as u16, area.y + 1));
    }

    /// Строка состояния: что выполняется (`busy`), подсказка к набираемой
    /// команде или `idle` — подсказка по умолчанию.
    pub fn status_line(&self, busy: Option<&str>, spinner: usize, idle: &str) -> Line<'static> {
        let text = self.input.text();
        match (busy, command_at(&text, self.hints)) {
            (Some(msg), _) => Line::from(Span::styled(
                format!(" {} {msg}", SPINNER[spinner % SPINNER.len()]),
                Style::default().fg(Color::Yellow),
            )),
            (None, Some((_, Some(hint)))) => Line::from(Span::styled(format!(" {hint}"), Style::default().fg(Color::LightCyan))),
            (None, Some((len, None))) if len > 1 => Line::from(Span::styled(
                format!(" нет команды {} — /? список команд", text.chars().take(len).collect::<String>()),
                Style::default().fg(Color::LightRed),
            )),
            _ => Line::from(Span::styled(format!(" {idle}"), Style::default().fg(Color::DarkGray))),
        }
    }
}

/// Команда в начале строки: длина `/команды` в символах и подсказка к ней
/// (`None` — такой команды нет). Не команда — `None`.
pub fn command_at(text: &str, hints: Hints) -> Option<(usize, Option<&'static str>)> {
    let rest = text.strip_prefix('/')?;
    let name: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
    let hint = hints.iter().find(|(n, _)| *n == name).map(|(_, h)| *h);
    Some((1 + name.chars().count(), hint))
}

/// Известная команда — голубым жирным, неизвестная — красным.
pub fn command_style(known: bool) -> Style {
    let color = if known { Color::LightCyan } else { Color::LightRed };
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

/// `text` кусками: символы `cmd` (диапазон символов) — стилем команды, остальное — `rest`.
pub fn highlight(text: &str, cmd: std::ops::Range<usize>, cmd_style: Style, rest: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let (a, b) = (cmd.start.min(chars.len()), cmd.end.min(chars.len()));
    [(0..a, rest), (a..b, cmd_style), (b..chars.len(), rest)]
        .into_iter()
        .filter(|(r, _)| !r.is_empty())
        .map(|(r, style)| Span::styled(chars[r].iter().collect::<String>(), style))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HINTS: Hints = &[("help", "/help"), ("review", "/review"), ("?", "/?")];

    #[test]
    fn commands_are_recognized_and_highlighted() {
        let at = |t: &str| command_at(t, HINTS).map(|(n, h)| (n, h.is_some()));
        assert_eq!(at("/review main..dev"), Some((7, true)));
        assert_eq!(at("/help"), Some((5, true)));
        assert_eq!(at("/?"), Some((2, true)));
        assert_eq!(at("/revew x"), Some((6, false)));
        assert_eq!(at("где лежит код?"), None);

        let spans = highlight("› /help где", 2..7, command_style(true), Style::default());
        let parts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(parts, vec!["› ", "/help", " где"]);
        assert_eq!(spans[1].style.fg, Some(Color::LightCyan));
        // Команда частично прокручена за левый край — подсвечена видимая часть.
        let spans = highlight("iew main", 0..3, command_style(true), Style::default());
        assert_eq!(spans[0].content, "iew");
    }

    #[test]
    fn submit_keeps_history() {
        use crossterm::event::KeyEvent;
        let mut p = Prompt::new(HINTS);
        let mut feed = Feed::default();
        let key = |code| Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        p.handle(Event::Paste("  /help x ".into()), &mut feed);
        assert_eq!(p.handle(key(KeyCode::Enter), &mut feed), Action::Submit("/help x".into()));
        assert_eq!(p.handle(key(KeyCode::Enter), &mut feed), Action::None);
        p.handle(key(KeyCode::Up), &mut feed);
        assert_eq!(p.input.text(), "/help x");
        assert_eq!(p.handle(key(KeyCode::Esc), &mut feed), Action::Quit);
    }
}
