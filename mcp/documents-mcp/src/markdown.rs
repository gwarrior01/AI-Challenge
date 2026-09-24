//! Текст страниц PDF → Markdown.
//!
//! `pdf-extract` отдаёт текст страницы как есть: строки вёрстки через `\n`,
//! блоки (абзацы, заголовки) через пустую строку. Разметки в PDF нет, поэтому
//! структура восстанавливается эвристиками, и они намеренно осторожные: лучше
//! оставить заголовок абзацем, чем превратить абзац в заголовок.
//!
//! - колонтитулы (строка, повторяющаяся на большинстве страниц) и номера
//!   страниц убираются;
//! - строки блока склеиваются в абзац; перенос со слогом (`пере-\nход`)
//!   склеивается без дефиса, но `PDF-\nдокумент` остаётся с дефисом;
//! - внутри блока абзац заканчивается на короткой строке с точкой в конце,
//!   за которой идёт строка с заглавной буквы — так выглядят и пункты
//!   списка, у которых PDF потерял маркеры;
//! - блок из одной короткой строки без точки в конце — заголовок: первый —
//!   `#`, пронумерованный `1.2` — `###` по глубине, остальные — `##`;
//! - абзац, разорванный границей страницы, склеивается обратно;
//! - маркеры списков (`•`, `●`, `▪`, `–` …) становятся `- `.

use std::collections::HashMap;

/// Результат конвертации: Markdown и заголовок документа (первый заголовок,
/// если он нашёлся).
pub struct Converted {
    pub markdown: String,
    pub title: Option<String>,
}

#[derive(Debug, PartialEq)]
enum Block {
    Heading(usize, String),
    Paragraph(String),
    ListItem(String),
}

const MAX_HEADING_CHARS: usize = 90;
const BULLETS: &[char] = &['•', '●', '▪', '◦', '■', '□', '–', '—', '·', '*', '-', '‣'];

pub fn pages_to_markdown(pages: &[String], fallback_title: &str) -> Converted {
    let page_lines: Vec<Vec<String>> = pages.iter().map(|p| normalize_lines(p)).collect();
    let furniture = page_furniture(&page_lines);

    let mut blocks: Vec<Block> = Vec::new();
    for lines in &page_lines {
        let mut page_blocks = Vec::new();
        for raw_block in split_blocks(lines, &furniture) {
            page_blocks.extend(classify(&raw_block));
        }
        // Абзац, разорванный границей страницы: предыдущая страница кончилась
        // не концом предложения, а эта начинается со строчной буквы.
        if let (Some(Block::Paragraph(prev)), Some(Block::Paragraph(next))) = (blocks.last_mut(), page_blocks.first()) {
            if !ends_sentence(prev) && starts_lowercase(next) {
                let next = next.clone();
                join_line(prev, &next);
                page_blocks.remove(0);
            }
        }
        blocks.extend(page_blocks);
    }

    // Первый заголовок — название документа.
    let mut title = None;
    if let Some(Block::Heading(level, text)) = blocks.first_mut() {
        *level = 1;
        title = Some(text.clone());
    }

    let mut markdown = String::new();
    if title.is_none() {
        markdown.push_str(&format!("# {fallback_title}\n\n"));
    }
    let mut prev_was_item = false;
    for block in &blocks {
        match block {
            Block::Heading(level, text) => {
                if !markdown.is_empty() && prev_was_item {
                    markdown.push('\n');
                }
                markdown.push_str(&format!("{} {text}\n\n", "#".repeat(*level)));
                prev_was_item = false;
            }
            Block::Paragraph(text) => {
                if prev_was_item {
                    markdown.push('\n');
                }
                markdown.push_str(text);
                markdown.push_str("\n\n");
                prev_was_item = false;
            }
            Block::ListItem(text) => {
                markdown.push_str(&format!("- {text}\n"));
                prev_was_item = true;
            }
        }
    }
    Converted { markdown: markdown.trim_end().to_string() + "\n", title }
}

/// Строки страницы без лишних пробелов; пустые строки сохраняются как
/// разделители блоков.
fn normalize_lines(page: &str) -> Vec<String> {
    page.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// Колонтитулы: непустые строки, которые встречаются на большей части страниц
/// (от трёх страниц). Сравниваются без цифр, чтобы «Отчёт · стр. 3» и
/// «Отчёт · стр. 4», «02 Отчёт» и «Отчёт 03» считались одной строкой.
fn page_furniture(pages: &[Vec<String>]) -> Vec<String> {
    if pages.len() < 3 {
        return Vec::new();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for lines in pages {
        let mut seen: Vec<String> = lines.iter().filter(|l| !l.is_empty()).map(|l| without_digits(l)).collect();
        seen.sort();
        seen.dedup();
        for key in seen {
            *counts.entry(key).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(key, n)| !key.trim().is_empty() && *n * 2 > pages.len())
        .map(|(key, _)| key)
        .collect()
}

fn without_digits(line: &str) -> String {
    let letters: String = line.chars().filter(|c| !c.is_ascii_digit()).collect();
    letters.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_page_number(line: &str) -> bool {
    let core = line.trim_matches(|c: char| c == '-' || c == '—' || c.is_whitespace());
    let core = core.strip_prefix("стр.").or_else(|| core.strip_prefix("Page")).unwrap_or(core).trim();
    !core.is_empty() && core.len() <= 4 && core.chars().all(|c| c.is_ascii_digit())
}

/// Блоки страницы (группы строк между пустыми строками) без колонтитулов и
/// номеров страниц.
fn split_blocks(lines: &[String], furniture: &[String]) -> Vec<Vec<String>> {
    let mut blocks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in lines {
        if line.is_empty() {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
            continue;
        }
        if is_page_number(line) || furniture.contains(&without_digits(line)) {
            continue;
        }
        current.push(line.clone());
    }
    if !current.is_empty() {
        blocks.push(current);
    }
    blocks
}

fn classify(lines: &[String]) -> Vec<Block> {
    if lines.len() == 1 {
        if let Some(level) = heading_level(&lines[0]) {
            return vec![Block::Heading(level, lines[0].clone())];
        }
    }

    let widest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut current_is_item = false;
    let flush = |current: &mut String, is_item: bool, blocks: &mut Vec<Block>| {
        if !current.is_empty() {
            let text = std::mem::take(current);
            blocks.push(if is_item { Block::ListItem(text) } else { Block::Paragraph(text) });
        }
    };

    for (i, line) in lines.iter().enumerate() {
        // «3.3 Context Engineering» посреди блока: pdftotext не всегда отделяет
        // заголовок пустой строкой. Здесь — только номер с точкой внутри
        // (`3.3`, `2.3.1`): «1 Способность» бывает и пунктом списка.
        //
        // И вообще заголовок внутри блока: короткая строка, похожая на
        // заголовок, в начале блока или после конца предложения, и за ней
        // строка с заглавной (или ничего).
        let next = lines.get(i + 1);
        let at_boundary = current.is_empty() || ends_sentence(&current);
        let short = line.chars().count() * 10 < widest * 8;
        let inline_heading = (short || lines.len() == 1)
            && at_boundary
            && next.is_none_or(|n| starts_uppercase(n))
            && heading_level(line).is_some();
        if let Some(level) = subsection_heading_level(line).or(inline_heading.then(|| heading_level(line)).flatten()) {
            flush(&mut current, current_is_item, &mut blocks);
            current_is_item = false;
            blocks.push(Block::Heading(level, line.clone()));
            continue;
        }
        if let Some(item) = strip_bullet(line) {
            flush(&mut current, current_is_item, &mut blocks);
            current = item.to_string();
            current_is_item = true;
        } else if current.is_empty() {
            current = line.clone();
        } else {
            join_line(&mut current, line);
        }
        // Короткая строка с концом предложения, за которой строка с заглавной, —
        // конец абзаца (или пункта списка без маркера).
        if let Some(next) = next {
            if short && ends_sentence(line) && starts_uppercase(next) && strip_bullet(next).is_none() {
                flush(&mut current, current_is_item, &mut blocks);
                current_is_item = false;
            }
        }
    }
    flush(&mut current, current_is_item, &mut blocks);
    blocks
}

/// Уровень заголовка для одиночной строки или `None`, если это не заголовок.
fn heading_level(line: &str) -> Option<usize> {
    let chars = line.chars().count();
    if chars > MAX_HEADING_CHARS || ends_sentence(line) || line.ends_with(',') || line.ends_with(';') {
        return None;
    }
    // Хотя бы три буквы: одиночная «Т» из инфографики — не заголовок.
    if line.chars().filter(|c| c.is_alphabetic()).count() < 3 || strip_bullet(line).is_some() {
        return None;
    }
    // «2.», «2.1», «2.1.3» в начале — глубина по числу частей.
    let first = line.split_whitespace().next().unwrap_or("");
    let number = first.trim_end_matches('.');
    if !number.is_empty() && number.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())) {
        let depth = number.split('.').count();
        return Some((depth + 1).min(6));
    }
    let first_char = line.chars().next()?;
    (first_char.is_uppercase() && line.split_whitespace().count() <= 12).then_some(2)
}

fn subsection_heading_level(line: &str) -> Option<usize> {
    let first = line.split_whitespace().next()?;
    let number = first.trim_end_matches('.');
    let parts: Vec<&str> = number.split('.').collect();
    let numbered = parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.len() <= 2 && p.chars().all(|c| c.is_ascii_digit()));
    let rest = line[first.len()..].trim_start();
    (numbered && rest.starts_with(char::is_uppercase)).then_some(())?;
    heading_level(line)
}

fn strip_bullet(line: &str) -> Option<&str> {
    let mut chars = line.chars();
    let first = chars.next()?;
    if !BULLETS.contains(&first) {
        return None;
    }
    let rest = chars.as_str();
    // «-5 °C» или «—» посреди текста — не маркер: после маркера нужен пробел.
    rest.starts_with(' ').then(|| rest.trim_start()).filter(|r| !r.is_empty())
}

fn join_line(current: &mut String, next: &str) {
    let before_hyphen = current.strip_suffix('-').and_then(|rest| rest.chars().last());
    match before_hyphen {
        // «пере-\nход» — перенос со слогом.
        Some(c) if c.is_lowercase() && starts_lowercase(next) => {
            current.pop();
        }
        // «PDF-\nдокумент» — дефис в самом слове.
        Some(c) if c.is_alphanumeric() && next.starts_with(char::is_alphanumeric) => {}
        _ => current.push(' '),
    }
    current.push_str(next);
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '!', '?', ':', '…', ';'])
}

fn starts_uppercase(text: &str) -> bool {
    text.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())
}

fn starts_lowercase(text: &str) -> bool {
    text.chars().next().is_some_and(char::is_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(pages: &[&str]) -> String {
        let pages: Vec<String> = pages.iter().map(|p| p.to_string()).collect();
        pages_to_markdown(&pages, "fallback").markdown
    }

    #[test]
    fn headings_and_paragraphs() {
        let out = md(&["\n\nОтчёт о переходе\n\n1. Зачем\n\nПервая строка абзаца, которая\nпродолжается здесь.\n\n2.1 Подраздел\n\nТекст."]);
        assert_eq!(
            out,
            "# Отчёт о переходе\n\n## 1. Зачем\n\nПервая строка абзаца, которая продолжается здесь.\n\n### 2.1 Подраздел\n\nТекст.\n"
        );
    }

    #[test]
    fn lost_list_markers_split_into_paragraphs() {
        let out = md(&["Заголовок\n\nКлиент MCP в ядре: подключение по stdio и Streamable HTTP, включение и выключение\nсерверов без перезапуска.\nСобственный сервер профилирования Java: снимок потоков, гистограмма кучи, запись\nJava Flight Recorder."]);
        assert!(out.contains("выключение серверов без перезапуска.\n\nСобственный сервер"), "{out}");
    }

    #[test]
    fn bullets_become_list_items() {
        let out = md(&["Список\n\n• первый пункт\n• второй пункт,\nс переносом\n\nПосле списка."]);
        assert!(out.contains("- первый пункт\n- второй пункт, с переносом\n\nПосле списка."), "{out}");
    }

    #[test]
    fn numbered_subsection_inside_block() {
        let out = md(&["Т\n\nКонец прошлого раздела.\n3.3 Context Engineering\nКонтекст — ограниченный ресурс.\n1 Способность организации"]);
        assert!(out.contains("раздела.\n\n### 3.3 Context Engineering\n\nКонтекст — ограниченный ресурс."), "{out}");
        assert!(!out.contains("## 1 Способность"), "{out}");
    }

    #[test]
    fn furniture_with_page_number_on_either_side() {
        let out = md(&["02 Отчёт · Whitepaper\n\nПервая страница.", "Отчёт · Whitepaper 03\n\nВторая страница.", "04 Отчёт · Whitepaper\n\nТретья."]);
        assert!(!out.contains("Whitepaper"), "{out}");
    }

    #[test]
    fn hyphenation() {
        let out = md(&["Т\n\nСлово пере-\nнесено, а PDF-\nдокумент нет."]);
        assert!(out.contains("Слово перенесено, а PDF-документ нет."), "{out}");
    }

    #[test]
    fn page_furniture_and_numbers_removed_and_paragraph_joined_across_pages() {
        let out = md(&[
            "Отчёт ООО Ромашка\n\nНачало абзаца, который\nпереходит\n\n1",
            "Отчёт ООО Ромашка\n\nна следующую страницу.\n\n2",
            "Отчёт ООО Ромашка\n\nКонец.\n\n3",
        ]);
        assert!(!out.contains("Ромашка"), "{out}");
        assert!(out.contains("Начало абзаца, который переходит на следующую страницу."), "{out}");
        assert!(!out.contains("\n2\n"), "{out}");
    }

    #[test]
    fn fallback_title_when_no_heading() {
        let converted = pages_to_markdown(&["Просто текст без заголовка.".into()], "report");
        assert!(converted.title.is_none());
        assert!(converted.markdown.starts_with("# report\n\nПросто текст"));
    }
}
