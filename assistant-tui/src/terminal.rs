//! Полноэкранный режим терминала: raw mode, альтернативный экран, вставка из
//! буфера одним событием, события мыши (колесо прокручивает ленту; выделить
//! текст мышью — с зажатым Option в iTerm2 или Fn в Terminal.app).
//! [`leave`] вызывается и после ошибки цикла событий.

use std::io;

use anyhow::Result;
use crossterm::{
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

pub type Term = Terminal<CrosstermBackend<io::Stdout>>;

pub fn enter() -> Result<Term> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste, EnableMouseCapture)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

pub fn leave(terminal: &mut Term) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableMouseCapture, DisableBracketedPaste, LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
