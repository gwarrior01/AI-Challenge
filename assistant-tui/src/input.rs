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
//!
//! «Символ» здесь — то, что человек видит как одну букву (графема), а не
//! `char`. Пути из Finder и имена файлов macOS приходят в разложенной форме
//! (NFD): «й» — это «и» + U+0306, два `char` в одной колонке. Поэтому текст
//! приводится к NFC, курсор ходит по графемам, а его колонка на экране
//! считается по ширине отображения (эмодзи и CJK — две колонки).

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Текст по графемам и позиция курсора между ними: `0` — перед первой,
/// `len` — после последней.
#[derive(Debug, Default, Clone)]
pub struct InputLine {
    cells: Vec<String>,
    cursor: usize,
}

impl InputLine {
    pub fn text(&self) -> String {
        self.cells.concat()
    }

    /// Заменить текст (история ввода) — курсор в конец.
    pub fn set(&mut self, text: &str) {
        self.cells = cells(&clean(text));
        self.cursor = self.cells.len();
    }

    pub fn clear(&mut self) {
        self.set("");
    }

    pub fn insert(&mut self, c: char) {
        self.insert_str(c.encode_utf8(&mut [0; 4]));
    }

    /// Вставка (в том числе из буфера обмена): переводы строк и табуляция
    /// становятся пробелами, прочие управляющие символы отбрасываются, текст
    /// приводится к NFC. Комбинируемый знак, набранный после буквы,
    /// соединяется с ней в одну графему.
    pub fn insert_str(&mut self, s: &str) {
        // Текст до курсора разбирается заново вместе со вставкой: так знак
        // ударения или бреве садится на предыдущую букву.
        let after = self.cells.split_off(self.cursor);
        self.cells = cells(&clean(&(self.cells.concat() + s)));
        self.cursor = self.cells.len();
        self.cells.extend(after);
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.cells.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.cells.len() {
            self.cells.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.cells.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.cells.len();
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
        self.cells.drain(start..self.cursor);
        self.cursor = start;
    }

    pub fn delete_word_forward(&mut self) {
        let end = self.next_word_end();
        self.cells.drain(self.cursor..end);
    }

    pub fn kill_to_start(&mut self) {
        self.cells.drain(..self.cursor);
        self.cursor = 0;
    }

    pub fn kill_to_end(&mut self) {
        self.cells.truncate(self.cursor);
    }

    /// Видимая часть строки шириной `width` колонок и колонка курсора в ней:
    /// окно сдвигается так, чтобы курсор всегда был виден.
    pub fn view(&self, width: usize) -> (String, usize) {
        let (start, end) = self.window(width);
        let col = self.cells[start..self.cursor].iter().map(|c| cell_width(c)).sum();
        (self.cells[start..end].concat(), col)
    }

    /// Видимый диапазон графем `start..end` для ширины `width` колонок (см.
    /// [`InputLine::view`]): от курсора влево, сколько поместится вместе с
    /// местом под сам курсор, и дальше вправо до края.
    pub fn window(&self, width: usize) -> (usize, usize) {
        let width = width.max(1);
        // Место под курсор: графема под ним или одна колонка в конце строки.
        let mut used = self.cells.get(self.cursor).map(|c| cell_width(c).max(1)).unwrap_or(1);
        let mut start = self.cursor;
        while start > 0 && used + cell_width(&self.cells[start - 1]) <= width {
            start -= 1;
            used += cell_width(&self.cells[start]);
        }
        let mut end = self.cursor;
        let mut shown: usize = self.cells[start..self.cursor].iter().map(|c| cell_width(c)).sum();
        while end < self.cells.len() && shown + cell_width(&self.cells[end]) <= width {
            shown += cell_width(&self.cells[end]);
            end += 1;
        }
        (start, end)
    }

    fn prev_word_start(&self) -> usize {
        let mut i = self.cursor;
        while i > 0 && !is_word(&self.cells[i - 1]) {
            i -= 1;
        }
        while i > 0 && is_word(&self.cells[i - 1]) {
            i -= 1;
        }
        i
    }

    fn next_word_end(&self) -> usize {
        let mut i = self.cursor;
        let n = self.cells.len();
        while i < n && !is_word(&self.cells[i]) {
            i += 1;
        }
        while i < n && is_word(&self.cells[i]) {
            i += 1;
        }
        i
    }
}

/// Текст для строки ввода: одна строка, без управляющих символов, NFC.
fn clean(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            '\n' | '\r' | '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .nfc()
        .collect()
}

fn cells(s: &str) -> Vec<String> {
    s.graphemes(true).map(str::to_string).collect()
}

fn cell_width(cell: &str) -> usize {
    cell.width()
}

/// Часть слова: буквы любого алфавита, цифры, `_`. Пробелы, `/`, `.`, `-` —
/// границы (так `main..feature` и пути проходятся по частям).
fn is_word(cell: &str) -> bool {
    cell.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, cursor: usize) -> InputLine {
        InputLine { cells: cells(text), cursor }
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
        l.insert_str("a\nb\tc\u{7}");
        assert_eq!(l.text(), "/elp где Ra b cG");
    }

    /// Путь из Finder в NFD: «й» — две `char`, но одна буква и одна колонка.
    #[test]
    fn decomposed_paste_is_one_letter_per_column() {
        let nfd = "/dir Встречи/Май-1.webm".nfd().collect::<String>();
        assert_eq!(nfd.chars().count(), 24, "в NFD «й» — два char");
        let mut l = InputLine::default();
        l.insert_str(&nfd);
        assert_eq!(l.text(), "/dir Встречи/Май-1.webm", "текст приведён к NFC");
        assert_eq!(l.view(80), ("/dir Встречи/Май-1.webm".to_string(), 23), "курсор — сразу за последней буквой");
        // Курсор за «й»: Backspace удаляет букву целиком, а не только бреве.
        for _ in 0.."-1.webm".len() {
            l.left();
        }
        l.backspace();
        assert_eq!(l.text(), "/dir Встречи/Ма-1.webm");
        assert_eq!(l.view(80).1, "/dir Встречи/Ма".chars().count());
    }

    #[test]
    fn combining_mark_typed_after_letter_joins_it() {
        let mut l = InputLine::default();
        l.insert_str("ии");
        l.left();
        l.insert('\u{306}');
        assert_eq!(l.text(), "йи");
        assert_eq!(l.view(10).1, 1);
        l.backspace();
        assert_eq!(l.text(), "и");
    }

    #[test]
    fn wide_characters_take_two_columns() {
        let mut l = InputLine::default();
        l.insert_str("a📁b");
        assert_eq!(l.view(80).1, 4);
        l.left();
        l.backspace();
        assert_eq!(l.text(), "ab");
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
        // Широкие символы: в 5 колонок перед курсором в конце влезают два по 2.
        let l = line("📁📁📁", 3);
        assert_eq!(l.view(5), ("📁📁".to_string(), 4));
    }
}
