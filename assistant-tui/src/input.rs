//! Строка ввода TUI с курсором: перемещение по символам и словам, удаление
//! слова, начало/конец строки — как в обычной командной строке.
//!
//! | Клавиши | Действие |
//! |---|---|
//! | ←/→ | символ |
//! | Alt+←/→, Ctrl+←/→, Alt+B/F | слово (Alt+стрелки в Terminal.app приходят как Alt+B/F) |
//! | Home/End, Ctrl+A/E | начало/конец строки |
//! | Backspace/Delete | символ до/после курсора |
//! | Alt+Backspace, Ctrl+W | слово до курсора |
//! | Alt+D | слово после курсора |
//! | Ctrl+U/K | всё до/после курсора |

/// Текст по символам (кириллица — один символ, один столбец) и позиция курсора
/// между ними: `0` — перед первым, `len` — после последнего.
#[derive(Debug, Default, Clone)]
pub struct InputLine {
    chars: Vec<char>,
    cursor: usize,
}

impl InputLine {
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Заменить текст (история ввода) — курсор в конец.
    pub fn set(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    pub fn clear(&mut self) {
        self.set("");
    }

    pub fn insert(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }

    /// Вставка (в том числе из буфера обмена) — переводы строк становятся пробелами.
    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            self.insert(if c == '\n' || c == '\r' { ' ' } else { c });
        }
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.chars.len();
    }

    /// К началу текущего или предыдущего слова.
    pub fn word_left(&mut self) {
        self.cursor = self.prev_word_start();
    }

    /// К концу текущего или следующего слова.
    pub fn word_right(&mut self) {
        self.cursor = self.next_word_end();
    }

    pub fn delete_word_back(&mut self) {
        let start = self.prev_word_start();
        self.chars.drain(start..self.cursor);
        self.cursor = start;
    }

    pub fn delete_word_forward(&mut self) {
        let end = self.next_word_end();
        self.chars.drain(self.cursor..end);
    }

    pub fn kill_to_start(&mut self) {
        self.chars.drain(..self.cursor);
        self.cursor = 0;
    }

    pub fn kill_to_end(&mut self) {
        self.chars.truncate(self.cursor);
    }

    /// Видимая часть строки шириной `width` и столбец курсора в ней:
    /// окно сдвигается так, чтобы курсор всегда был виден.
    pub fn view(&self, width: usize) -> (String, usize) {
        let (start, end) = self.window(width);
        (self.chars[start..end].iter().collect(), self.cursor - start)
    }

    /// Видимый диапазон символов `start..end` для ширины `width` (см. [`InputLine::view`]).
    pub fn window(&self, width: usize) -> (usize, usize) {
        let width = width.max(1);
        let start = (self.cursor + 1).saturating_sub(width);
        (start, (start + width).min(self.chars.len()))
    }

    fn prev_word_start(&self) -> usize {
        let mut i = self.cursor;
        while i > 0 && !is_word(self.chars[i - 1]) {
            i -= 1;
        }
        while i > 0 && is_word(self.chars[i - 1]) {
            i -= 1;
        }
        i
    }

    fn next_word_end(&self) -> usize {
        let mut i = self.cursor;
        let n = self.chars.len();
        while i < n && !is_word(self.chars[i]) {
            i += 1;
        }
        while i < n && is_word(self.chars[i]) {
            i += 1;
        }
        i
    }
}

/// Часть слова: буквы любого алфавита, цифры, `_`. Пробелы, `/`, `.`, `-` —
/// границы (так `main..feature` и пути проходятся по частям).
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, cursor: usize) -> InputLine {
        InputLine { chars: text.chars().collect(), cursor }
    }

    #[test]
    fn inserts_and_deletes_at_cursor() {
        let mut l = InputLine::default();
        l.insert_str("/help где RAG");
        l.home();
        l.right();
        l.insert('х');
        assert_eq!(l.text(), "/хhelp где RAG");
        l.backspace();
        l.delete();
        assert_eq!(l.text(), "/elp где RAG");
        l.end();
        l.left();
        l.backspace();
        assert_eq!(l.text(), "/elp где RG");
        l.insert_str("a\nb");
        assert_eq!(l.text(), "/elp где Ra bG");
    }

    #[test]
    fn moves_by_words() {
        let mut l = line("/review main..feature где", 25);
        l.word_left();
        assert_eq!(l.cursor, 22); // «где»
        l.word_left();
        assert_eq!(l.cursor, 14); // «feature»
        l.word_left();
        assert_eq!(l.cursor, 8); // «main»
        l.word_left();
        assert_eq!(l.cursor, 1); // «review»
        l.word_left();
        assert_eq!(l.cursor, 0);
        l.word_left();
        assert_eq!(l.cursor, 0);
        l.word_right();
        assert_eq!(l.cursor, 7); // после «review»
        l.word_right();
        assert_eq!(l.cursor, 12); // после «main»
    }

    #[test]
    fn deletes_words_and_kills() {
        let mut l = line("вопрос о проекте", 16);
        l.delete_word_back();
        assert_eq!(l.text(), "вопрос о ");
        l.home();
        l.delete_word_forward();
        assert_eq!(l.text(), " о ");
        let mut l = line("abc def", 3);
        l.kill_to_end();
        assert_eq!(l.text(), "abc");
        let mut l = line("abc def", 4);
        l.kill_to_start();
        assert_eq!((l.text().as_str(), l.cursor), ("def", 0));
    }

    #[test]
    fn view_keeps_cursor_visible() {
        let l = line("0123456789", 10);
        assert_eq!(l.view(4), ("789".to_string(), 3));
        let l = line("0123456789", 2);
        assert_eq!(l.view(4), ("0123".to_string(), 2));
        let l = line("", 0);
        assert_eq!(l.view(4), (String::new(), 0));
    }
}
