//! Четыре стратегии разбиения документа на чанки.
//!
//! Чанк — это не копия текста, а диапазон байт исходного документа
//! ([`Piece`]): так у каждого чанка есть точные строки начала и конца, а
//! текст чанка — всегда дословный кусок документа.
//!
//! - [`Strategy::Fixed`] — окно фиксированного размера (в символах) с
//!   перекрытием; граница сдвигается назад к пробелу, чтобы не резать слово.
//!   О структуре документа ничего не знает.
//! - [`Strategy::Structure`] — по структуре: Markdown (и PDF, который
//!   сначала становится Markdown) — по заголовкам, Rust — по элементам
//!   верхнего уровня (`fn`, `impl`, `struct`, …) вместе с их doc-комментариями,
//!   большие `impl`/`mod` — ещё и по вложенным элементам. Короткие соседние
//!   разделы сливаются, длинные делятся по абзацам.
//! - [`Strategy::Sentence`] — скользящее окно из предложений: N предложений,
//!   шаг S (перекрытие N−S). Заголовки, пункты списков и строки таблиц — сами
//!   себе «предложения»; внутри блоков кода и в коде Rust единица — строка.
//!   Окно не переходит границу раздела.
//! - [`Strategy::Parent`] — parent-child: векторизуется маленький
//!   «ребёнок» (N предложений, в коде — N строк), а в контекст отдаётся его
//!   «родитель» — абзац целиком вместе с заголовком над ним ([`Piece::parent`]).
//!   Поиск точен, как по одному предложению, а модель видит связный абзац.
//!
//! У каждого чанка есть `section` — путь раздела (`MCP › Own MCP server:
//! documents`, `impl LlmClient › fn chat`). Структурная стратегия режет
//! ровно по разделам; остальные берут раздел, в котором чанк начинается.

use serde::{Deserialize, Serialize};

/// Стратегия разбиения.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    Fixed,
    Structure,
    Sentence,
    Parent,
}

impl Strategy {
    pub const ALL: [Strategy; 4] = [Strategy::Fixed, Strategy::Structure, Strategy::Sentence, Strategy::Parent];

    pub fn name(self) -> &'static str {
        match self {
            Strategy::Fixed => "fixed",
            Strategy::Structure => "structure",
            Strategy::Sentence => "sentence",
            Strategy::Parent => "parent",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fixed" => Some(Strategy::Fixed),
            "structure" => Some(Strategy::Structure),
            "sentence" | "sentence-window" => Some(Strategy::Sentence),
            "parent" | "parent-child" => Some(Strategy::Parent),
            _ => None,
        }
    }

    /// Параметры стратегии по умолчанию одной строкой — для отчётов.
    pub fn describe(self) -> String {
        ChunkParams::default().describe(self)
    }
}

/// Вид документа — от него зависит, как искать разделы и предложения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Markdown, а также PDF после конвертации в Markdown.
    Markdown,
    /// Исходный код на Rust.
    Rust,
    /// Простой текст без разметки.
    Text,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Markdown => "markdown",
            Kind::Rust => "rust",
            Kind::Text => "text",
        }
    }
}

/// Чанк: диапазон байт `start..end` в тексте документа и путь раздела.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub start: usize,
    pub end: usize,
    pub section: Option<String>,
    /// Родитель (parent-child): диапазон, который отдаётся в контекст вместо
    /// самого чанка. У остальных стратегий контекст — сам чанк.
    pub parent: Option<(usize, usize)>,
}

/// Значения параметров по умолчанию (см. [`ChunkParams`]).
pub const FIXED_SIZE: usize = 1200;
pub const FIXED_OVERLAP: usize = 200;
pub const STRUCT_MIN: usize = 300;
pub const STRUCT_MAX: usize = 2000;
pub const SENT_WINDOW: usize = 5;
pub const SENT_STRIDE: usize = 3;
pub const SENT_MAX: usize = 1500;
pub const CODE_WINDOW: usize = 16;
pub const CODE_STRIDE: usize = 12;
pub const CHILD_SENTENCES: usize = 1;
pub const CHILD_LINES: usize = 3;
pub const PARENT_MAX: usize = 2000;

/// Разделитель уровней в пути раздела.
pub const PATH_SEP: &str = " › ";

/// Предел длины чанка в символах: у моделей эмбеддингов конечный контекст
/// (bge-m3 — 8192 токена), а поиску нужны куски, а не главы.
pub const MAX_CHUNK_CHARS: usize = 8000;

/// Параметры стратегий. Выбираются при индексации документа (загрузка в
/// вебе, `add … size=800`) и хранятся в индексе вместе с его чанками —
/// у каждой нарезки видно, с какими параметрами она сделана.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkParams {
    /// fixed: длина окна и перекрытие, символов.
    pub size: usize,
    pub overlap: usize,
    /// structure: раздел короче `min_chars` сливается со следующим, длиннее
    /// `max_chars` — делится по абзацам.
    pub min_chars: usize,
    pub max_chars: usize,
    /// sentence: окно из `window` предложений со сдвигом `stride`, не
    /// длиннее `window_chars`; в коде — `code_window` строк со сдвигом
    /// `code_stride`.
    pub window: usize,
    pub stride: usize,
    pub window_chars: usize,
    pub code_window: usize,
    pub code_stride: usize,
    /// parent: ребёнок — `child_sentences` предложений (в коде —
    /// `child_lines` строк); родитель — абзац, не длиннее `parent_chars`.
    pub child_sentences: usize,
    pub child_lines: usize,
    pub parent_chars: usize,
}

impl Default for ChunkParams {
    fn default() -> Self {
        Self {
            size: FIXED_SIZE,
            overlap: FIXED_OVERLAP,
            min_chars: STRUCT_MIN,
            max_chars: STRUCT_MAX,
            window: SENT_WINDOW,
            stride: SENT_STRIDE,
            window_chars: SENT_MAX,
            code_window: CODE_WINDOW,
            code_stride: CODE_STRIDE,
            child_sentences: CHILD_SENTENCES,
            child_lines: CHILD_LINES,
            parent_chars: PARENT_MAX,
        }
    }
}

/// Параметр стратегии — для форм и подсказок: имя (как в JSON и в
/// `add … имя=значение`), подпись, допустимый диапазон.
#[derive(Debug, Clone, Serialize)]
pub struct ParamSpec {
    pub name: &'static str,
    pub label: &'static str,
    pub min: usize,
    pub max: usize,
    pub default: usize,
}

impl ChunkParams {
    /// Параметры, которые влияют на стратегию.
    pub fn specs(strategy: Strategy) -> Vec<ParamSpec> {
        let d = Self::default();
        let spec = |name, label, min, max, default| ParamSpec { name, label, min, max, default };
        match strategy {
            Strategy::Fixed => vec![
                spec("size", "длина окна, символов", 100, MAX_CHUNK_CHARS, d.size),
                spec("overlap", "перекрытие, символов", 0, MAX_CHUNK_CHARS / 2, d.overlap),
            ],
            Strategy::Structure => vec![
                spec("min_chars", "раздел короче — слить со следующим, символов", 0, MAX_CHUNK_CHARS, d.min_chars),
                spec("max_chars", "раздел длиннее — делить по абзацам, символов", 200, MAX_CHUNK_CHARS, d.max_chars),
            ],
            Strategy::Sentence => vec![
                spec("window", "предложений в окне", 1, 50, d.window),
                spec("stride", "сдвиг окна, предложений", 1, 50, d.stride),
                spec("window_chars", "окно не длиннее, символов", 200, MAX_CHUNK_CHARS, d.window_chars),
                spec("code_window", "в коде: строк в окне", 1, 200, d.code_window),
                spec("code_stride", "в коде: сдвиг окна, строк", 1, 200, d.code_stride),
            ],
            Strategy::Parent => vec![
                spec("child_sentences", "ребёнок: предложений", 1, 20, d.child_sentences),
                spec("child_lines", "ребёнок в коде: строк", 1, 100, d.child_lines),
                spec("parent_chars", "родитель (абзац) не длиннее, символов", 200, MAX_CHUNK_CHARS, d.parent_chars),
            ],
        }
    }

    fn get(&self, name: &str) -> Option<usize> {
        Some(match name {
            "size" => self.size,
            "overlap" => self.overlap,
            "min_chars" => self.min_chars,
            "max_chars" => self.max_chars,
            "window" => self.window,
            "stride" => self.stride,
            "window_chars" => self.window_chars,
            "code_window" => self.code_window,
            "code_stride" => self.code_stride,
            "child_sentences" => self.child_sentences,
            "child_lines" => self.child_lines,
            "parent_chars" => self.parent_chars,
            _ => return None,
        })
    }

    /// Задаёт параметр по имени (как в [`ParamSpec::name`]).
    pub fn set(&mut self, name: &str, value: usize) -> Result<(), String> {
        let slot = match name {
            "size" => &mut self.size,
            "overlap" => &mut self.overlap,
            "min_chars" => &mut self.min_chars,
            "max_chars" => &mut self.max_chars,
            "window" => &mut self.window,
            "stride" => &mut self.stride,
            "window_chars" => &mut self.window_chars,
            "code_window" => &mut self.code_window,
            "code_stride" => &mut self.code_stride,
            "child_sentences" => &mut self.child_sentences,
            "child_lines" => &mut self.child_lines,
            "parent_chars" => &mut self.parent_chars,
            _ => {
                return Err(format!(
                    "неизвестный параметр «{name}» — есть size, overlap, min_chars, max_chars, window, stride, window_chars, \
                     code_window, code_stride, child_sentences, child_lines, parent_chars"
                ))
            }
        };
        *slot = value;
        Ok(())
    }

    /// Проверка параметров стратегии: диапазоны и связи между ними.
    pub fn validate(&self, strategy: Strategy) -> Result<(), String> {
        for spec in Self::specs(strategy) {
            let v = self.get(spec.name).unwrap_or_default();
            if v < spec.min || v > spec.max {
                return Err(format!("{} ({}) — от {} до {}, задано {v}", spec.name, spec.label, spec.min, spec.max));
            }
        }
        match strategy {
            Strategy::Fixed if self.overlap * 2 > self.size => {
                Err(format!("overlap ({}) — не больше половины size ({})", self.overlap, self.size))
            }
            Strategy::Structure if self.min_chars > self.max_chars => {
                Err(format!("min_chars ({}) больше max_chars ({})", self.min_chars, self.max_chars))
            }
            Strategy::Sentence if self.stride > self.window => {
                Err(format!("stride ({}) больше window ({}) — между окнами были бы пропуски", self.stride, self.window))
            }
            Strategy::Sentence if self.code_stride > self.code_window => Err(format!(
                "code_stride ({}) больше code_window ({}) — между окнами были бы пропуски",
                self.code_stride, self.code_window
            )),
            _ => Ok(()),
        }
    }

    /// Параметры стратегии — то, что сохраняется у нарезки.
    pub fn of(&self, strategy: Strategy) -> serde_json::Map<String, serde_json::Value> {
        Self::specs(strategy)
            .into_iter()
            .map(|spec| (spec.name.to_string(), serde_json::Value::from(self.get(spec.name).unwrap_or_default())))
            .collect()
    }

    /// Параметры стратегии одной строкой.
    pub fn describe(&self, strategy: Strategy) -> String {
        match strategy {
            Strategy::Fixed => format!("окно {} символов, перекрытие {}", self.size, self.overlap),
            Strategy::Structure => format!(
                "разделы по заголовкам / элементам Rust; короче {} символов — слияние, длиннее {} — деление по абзацам",
                self.min_chars, self.max_chars
            ),
            Strategy::Sentence => format!(
                "окно {} предложений, шаг {} (в коде — {} строк, шаг {}), не длиннее {} символов",
                self.window, self.stride, self.code_window, self.code_stride, self.window_chars
            ),
            Strategy::Parent => format!(
                "ребёнок — {} предл. (в коде — {} строк), в контекст — абзац с заголовком, не длиннее {} символов",
                self.child_sentences, self.child_lines, self.parent_chars
            ),
        }
    }
}

pub fn chunk(text: &str, kind: Kind, strategy: Strategy, params: &ChunkParams) -> Vec<Piece> {
    let sections = outline(text, kind, params);
    let pieces = match strategy {
        Strategy::Fixed => fixed(text, &sections, params),
        Strategy::Structure => structure(text, kind, &sections, params),
        Strategy::Sentence => sentence(text, kind, &sections, params),
        Strategy::Parent => parent_child(text, kind, &sections, params),
    };
    pieces.into_iter().filter_map(|p| trimmed(text, p)).collect()
}

// ---------------------------------------------------------------------------
// Разделы документа

/// Раздел: `start..end` в тексте и путь (`None` — текст до первого
/// заголовка или элемента).
#[derive(Debug, Clone)]
struct Section {
    start: usize,
    end: usize,
    path: Option<String>,
}

/// Документ, разбитый на подряд идущие разделы (без пропусков).
fn outline(text: &str, kind: Kind, params: &ChunkParams) -> Vec<Section> {
    let starts = match kind {
        Kind::Markdown => markdown_headings(text),
        Kind::Rust => rust_outline(text, params.max_chars),
        Kind::Text => Vec::new(),
    };
    let mut sections = Vec::new();
    let first = starts.first().map(|(s, _)| *s).unwrap_or(text.len());
    if first > 0 {
        sections.push(Section { start: 0, end: first, path: None });
    }
    for (i, (start, path)) in starts.iter().enumerate() {
        let end = starts.get(i + 1).map(|(s, _)| *s).unwrap_or(text.len());
        sections.push(Section { start: *start, end, path: Some(path.clone()) });
    }
    sections
}

/// Строки с байтовым смещением начала (без `\n`).
fn lines_with_offsets(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut offset = 0;
    text.split_inclusive('\n').map(move |line| {
        let start = offset;
        offset += line.len();
        (start, line.trim_end_matches(['\n', '\r']))
    })
}

fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// Уровень и текст заголовка `#`…`######`.
fn heading(line: &str) -> Option<(usize, &str)> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &line[hashes..];
    rest.starts_with(' ').then(|| (hashes, rest.trim().trim_end_matches('#').trim()))
}

/// Начала разделов Markdown и их пути (`Глава › Раздел › Подраздел`).
/// `#` внутри блоков кода — не заголовки.
fn markdown_headings(text: &str) -> Vec<(usize, String)> {
    let mut result = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut in_fence = false;
    for (offset, line) in lines_with_offsets(text) {
        if is_fence(line) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some((level, title)) = heading(line) {
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, title.to_string()));
            let path = stack.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join(PATH_SEP);
            result.push((offset, path));
        }
    }
    result
}

const RUST_ITEMS: &[&str] =
    &["fn", "struct", "enum", "impl", "trait", "mod", "const", "static", "type", "union", "macro_rules!"];

/// Имя элемента Rust по первой строке (`pub async fn chat(` → `fn chat`,
/// `impl<T> Foo for Bar {` → `impl Foo for Bar`), либо `None`, если строка
/// не начинает элемент.
fn rust_item_name(line: &str) -> Option<String> {
    let mut words: Vec<&str> = line.split_whitespace().collect();
    // Модификаторы видимости и прочие перед ключевым словом.
    while let Some(first) = words.first() {
        let modifier = first.starts_with("pub") && (first.len() == 3 || first[3..].starts_with('('))
            || matches!(*first, "async" | "unsafe" | "extern" | "default" | "\"C\"")
            || (*first == "const" && words.get(1) == Some(&"fn"));
        if !modifier {
            break;
        }
        words.remove(0);
    }
    let keyword = *words.first()?;
    let keyword = RUST_ITEMS.iter().find(|k| keyword == **k || keyword.starts_with(&format!("{k}<")))?;
    if *keyword == "impl" {
        // Всё до `{` или `where`, без параметров обобщения у самого impl.
        let rest = line.trim_start();
        let rest = rest.split(" where").next().unwrap_or(rest);
        let rest = rest.split('{').next().unwrap_or(rest).trim();
        let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);
        let body = rest.strip_prefix("impl").unwrap_or(rest);
        let body = if body.starts_with('<') { skip_generics(body) } else { body };
        let name = format!("impl {}", body.split_whitespace().collect::<Vec<_>>().join(" "));
        return Some(cap(&name, 60));
    }
    let name = words.get(1)?;
    let name: String = name.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    (!name.is_empty()).then(|| format!("{keyword} {name}"))
}

fn skip_generics(s: &str) -> &str {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return &s[i + 1..];
                }
            }
            _ => {}
        }
    }
    s
}

fn cap(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// Элементы Rust с отступом `indent` в `text[from..to]`: начало (с учётом
/// doc-комментариев и атрибутов над элементом) и имя.
fn rust_items(text: &str, from: usize, to: usize, indent: usize) -> Vec<(usize, String)> {
    let prefix = " ".repeat(indent);
    let mut items = Vec::new();
    // Начало «шапки» — непрерывных строк `///`, `//`, `#[` над элементом.
    let mut header_start: Option<usize> = None;
    for (offset, line) in lines_with_offsets(&text[from..to]) {
        let offset = from + offset;
        let at_level = line.starts_with(&prefix) && !line[indent..].starts_with([' ', '\t']);
        let body = if at_level { &line[indent..] } else { "" };
        if at_level && (body.starts_with("///") || body.starts_with("#[") || (body.starts_with("//") && !body.starts_with("//!"))) {
            header_start.get_or_insert(offset);
            continue;
        }
        if at_level {
            if let Some(name) = rust_item_name(body) {
                items.push((header_start.unwrap_or(offset), name));
            }
        }
        header_start = None;
    }
    items
}

/// Разделы файла Rust: элементы верхнего уровня; большие `impl`, `mod`,
/// `trait` — ещё и по вложенным элементам (`impl Foo › fn bar`).
fn rust_outline(text: &str, max_chars: usize) -> Vec<(usize, String)> {
    let top = rust_items(text, 0, text.len(), 0);
    let mut result = Vec::new();
    for (i, (start, name)) in top.iter().enumerate() {
        let end = top.get(i + 1).map(|(s, _)| *s).unwrap_or(text.len());
        result.push((*start, name.clone()));
        let container = name.starts_with("impl ") || name.starts_with("mod ") || name.starts_with("trait ");
        if container && end - start > max_chars {
            for (inner_start, inner) in rust_items(text, *start, end, 4) {
                // Первый вложенный элемент, начавшийся сразу за заголовком
                // блока, — всё равно отдельный раздел: заголовок блока короткий
                // и сольётся с ним при слиянии коротких разделов.
                if inner_start > *start {
                    result.push((inner_start, format!("{name}{PATH_SEP}{inner}")));
                }
            }
        }
    }
    result
}

/// Раздел, в котором находится байт `pos`.
fn section_at(sections: &[Section], pos: usize) -> Option<String> {
    sections.iter().rev().find(|s| s.start <= pos).and_then(|s| s.path.clone())
}

// ---------------------------------------------------------------------------
// Вспомогательное: символы и границы

/// Байтовая позиция через `n` символов после `from` (или конец текста).
fn advance(text: &str, from: usize, n: usize) -> usize {
    text[from..].char_indices().nth(n).map(|(i, _)| from + i).unwrap_or(text.len())
}

/// Байтовая позиция за `n` символов до `from`.
fn back(text: &str, from: usize, n: usize) -> usize {
    text[..from].char_indices().rev().nth(n.saturating_sub(1)).map(|(i, _)| i).unwrap_or(0)
}

fn char_len(text: &str, start: usize, end: usize) -> usize {
    text[start..end].chars().count()
}

/// Чанк без пробелов по краям; пустой — `None`.
fn trimmed(text: &str, piece: Piece) -> Option<Piece> {
    let slice = &text[piece.start..piece.end];
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    if lead == slice.len() {
        return None;
    }
    Some(Piece { start: piece.start + lead, end: piece.end - trail, ..piece })
}

/// Режет `start..end` на куски не длиннее `max` символов без перекрытия,
/// сдвигая границу назад к переводу строки или пробелу.
fn hard_split(text: &str, start: usize, end: usize, max: usize, overlap: usize) -> Vec<(usize, usize)> {
    let mut parts = Vec::new();
    let mut pos = start;
    while pos < end {
        let mut cut = advance(text, pos, max).min(end);
        if cut < end {
            // Не дальше чем на пятую часть окна назад: лучше разрезать слово,
            // чем сделать чанк заметно короче остальных.
            let floor = advance(text, pos, max * 4 / 5).min(cut);
            let window = &text[floor..cut];
            if let Some(i) = window.rfind('\n').or_else(|| window.rfind(char::is_whitespace)) {
                cut = floor + i;
            }
            if cut <= pos {
                cut = advance(text, pos, max).min(end);
            }
        }
        parts.push((pos, cut));
        if cut >= end {
            break;
        }
        let mut next = if overlap > 0 { back(text, cut, overlap).max(pos + 1) } else { cut };
        // Начало перекрытия — с начала слова.
        if overlap > 0 {
            if let Some(i) = text[next..cut].find(char::is_whitespace) {
                next += i;
            }
        }
        while !text.is_char_boundary(next) {
            next += 1;
        }
        pos = next.max(pos + 1);
    }
    parts
}

// ---------------------------------------------------------------------------
// Fixed

fn fixed(text: &str, sections: &[Section], p: &ChunkParams) -> Vec<Piece> {
    hard_split(text, 0, text.len(), p.size, p.overlap)
        .into_iter()
        .map(|(start, end)| {
            let first = start + (text[start..end].len() - text[start..end].trim_start().len());
            Piece { start, end, section: section_at(sections, first), parent: None }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Structure

fn structure(text: &str, kind: Kind, sections: &[Section], p: &ChunkParams) -> Vec<Piece> {
    let merged = merge_small(text, sections, p.min_chars, p.max_chars);
    let mut pieces = Vec::new();
    for section in merged {
        if char_len(text, section.start, section.end) <= p.max_chars {
            pieces.push(Piece { start: section.start, end: section.end, section: section.path, parent: None });
            continue;
        }
        for (start, end) in split_paragraphs(text, section.start, section.end, kind, p.max_chars) {
            pieces.push(Piece { start, end, section: section.path.clone(), parent: None });
        }
    }
    pieces
}

/// Раздел короче `min` сливается со следующим, пока вместе они не длиннее
/// `max`; раздел из одной первой строки (заголовок главы сразу перед
/// подглавой, `impl Foo {` перед длинным методом) — со следующим в любом
/// случае, длинное потом делится по абзацам. Путь объединения: у родителя и
/// потомка — путь потомка (в нём есть и родитель), у соседей — общая часть
/// пути и их имена через запятую.
fn merge_small(text: &str, sections: &[Section], min: usize, max: usize) -> Vec<Section> {
    let mut result: Vec<Section> = Vec::new();
    for section in sections {
        if let Some(last) = result.last_mut() {
            let last_len = char_len(text, last.start, last.end);
            let both = char_len(text, last.start, section.end);
            if (last_len < min && both <= max) || header_only(text, last) {
                last.end = section.end;
                last.path = merged_path(last.path.as_deref(), section.path.as_deref());
                continue;
            }
        }
        result.push(section.clone());
    }
    // Короткий хвост — к предыдущему разделу.
    if result.len() >= 2 {
        let last = result.last().unwrap();
        let prev = &result[result.len() - 2];
        if char_len(text, last.start, last.end) < min && char_len(text, prev.start, last.end) <= max {
            let last = result.pop().unwrap();
            let prev = result.last_mut().unwrap();
            prev.end = last.end;
            prev.path = merged_path(prev.path.as_deref(), last.path.as_deref());
        }
    }
    result
}

/// В разделе нет ничего, кроме первой строки (заголовка).
fn header_only(text: &str, section: &Section) -> bool {
    let body = &text[section.start..section.end];
    body.split_once('\n').is_none_or(|(_, rest)| rest.trim().is_empty())
}

fn merged_path(a: Option<&str>, b: Option<&str>) -> Option<String> {
    match (a, b) {
        (None, b) => b.map(str::to_string),
        (a, None) => a.map(str::to_string),
        (Some(a), Some(b)) => {
            if b.starts_with(&format!("{a}{PATH_SEP}")) || a == b {
                return Some(b.to_string());
            }
            let pa: Vec<&str> = a.split(PATH_SEP).collect();
            let pb: Vec<&str> = b.split(PATH_SEP).collect();
            let common = pa.iter().zip(&pb).take_while(|(x, y)| x == y).count();
            // Соседи: общая часть пути и их имена через запятую
            // (`impl Foo › fn a, fn b`), не больше трёх.
            let tail_a = pa[common..].join(PATH_SEP);
            let tail_b = pb[common..].join(PATH_SEP);
            let names: Vec<&str> = tail_a.split(", ").filter(|n| *n != "…").chain(std::iter::once(tail_b.as_str())).collect();
            let list = if names.len() > 3 { format!("{}, …", names[..3].join(", ")) } else { names.join(", ") };
            Some(if common == 0 { list } else { format!("{}{PATH_SEP}{list}", pa[..common].join(PATH_SEP)) })
        }
    }
}

/// Абзацы раздела (через пустую строку; блок кода в Markdown — целиком),
/// упакованные в куски не длиннее `max`. Абзац длиннее — режется
/// по строкам/пробелам.
fn split_paragraphs(text: &str, start: usize, end: usize, kind: Kind, max: usize) -> Vec<(usize, usize)> {
    let paragraphs = paragraphs(text, start, end, kind);

    let mut parts: Vec<(usize, usize)> = Vec::new();
    let mut pack: Option<(usize, usize)> = None;
    for (p_start, p_end) in paragraphs {
        if char_len(text, p_start, p_end) > max {
            parts.extend(pack.take());
            parts.extend(hard_split(text, p_start, p_end, max, 0));
            continue;
        }
        pack = Some(match pack {
            Some((s, _)) if char_len(text, s, p_end) <= max => (s, p_end),
            Some(done) => {
                parts.push(done);
                (p_start, p_end)
            }
            None => (p_start, p_end),
        });
    }
    parts.extend(pack);
    parts
}

/// Абзацы `start..end`: строки между пустыми строками; блок кода в Markdown
/// — целиком, даже с пустыми строками внутри.
fn paragraphs(text: &str, start: usize, end: usize, kind: Kind) -> Vec<(usize, usize)> {
    let mut paragraphs: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    let mut in_fence = false;
    for (offset, line) in lines_with_offsets(&text[start..end]) {
        let (line_start, line_end) = (start + offset, start + offset + line.len());
        if kind == Kind::Markdown && is_fence(line) {
            in_fence = !in_fence;
        }
        if line.trim().is_empty() && !in_fence {
            if let Some(p) = current.take() {
                paragraphs.push(p);
            }
            continue;
        }
        current = Some(match current {
            Some((s, _)) => (s, line_end),
            None => (line_start, line_end),
        });
    }
    paragraphs.extend(current);
    paragraphs
}

// ---------------------------------------------------------------------------
// Sentence window

/// Сокращения, после точки в которых предложение не кончается.
const ABBREVIATIONS: &[&str] = &[
    "т.е", "т.д", "т.п", "т.к", "т.н", "и.т.д", "др", "см", "стр", "рис", "напр", "г", "гг", "им", "англ", "проч", "ок",
    "e.g", "i.e", "etc", "vs", "cf", "mr", "dr", "no", "fig",
];

/// Единицы текста для окна (`start..end`) — предложения или строки.
fn units(text: &str, kind: Kind, sections: &[Section], max: usize) -> Vec<(usize, usize)> {
    let mut units = Vec::new();
    for section in sections {
        let (s, e) = (section.start, section.end);
        match kind {
            Kind::Rust => units.extend(line_units(text, s, e)),
            Kind::Markdown | Kind::Text => units.extend(prose_units(text, s, e, kind, max)),
        }
    }
    units
}

fn line_units(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    lines_with_offsets(&text[start..end])
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(offset, line)| (start + offset, start + offset + line.len()))
        .collect()
}

/// Предложения прозы; заголовки, пункты списков и строки таблиц — отдельные
/// единицы; блок кода — одна единица, если помещается в окно, иначе строки.
fn prose_units(text: &str, start: usize, end: usize, kind: Kind, max: usize) -> Vec<(usize, usize)> {
    let mut units = Vec::new();
    let mut paragraph: Option<(usize, usize)> = None;
    let mut fence: Option<usize> = None;
    let flush = |paragraph: &mut Option<(usize, usize)>, units: &mut Vec<(usize, usize)>| {
        if let Some((s, e)) = paragraph.take() {
            units.extend(sentences(text, s, e));
        }
    };
    for (offset, line) in lines_with_offsets(&text[start..end]) {
        let (ls, le) = (start + offset, start + offset + line.len());
        if kind == Kind::Markdown && is_fence(line) {
            match fence.take() {
                None => {
                    flush(&mut paragraph, &mut units);
                    fence = Some(ls);
                }
                Some(fs) => {
                    if char_len(text, fs, le) <= max {
                        units.push((fs, le));
                    } else {
                        units.extend(line_units(text, fs, le));
                    }
                }
            }
            continue;
        }
        if fence.is_some() {
            continue;
        }
        let t = line.trim_start();
        let standalone = kind == Kind::Markdown
            && (heading(line).is_some()
                || t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with('|')
                || t.starts_with('>')
                || numbered_item(t));
        if line.trim().is_empty() || standalone {
            flush(&mut paragraph, &mut units);
            if standalone {
                // Длинный пункт списка — тоже по предложениям.
                paragraph = Some((ls, le));
                flush(&mut paragraph, &mut units);
            }
            continue;
        }
        paragraph = Some(match paragraph {
            Some((s, _)) => (s, le),
            None => (ls, le),
        });
    }
    flush(&mut paragraph, &mut units);
    if let Some(fs) = fence {
        // Незакрытый блок кода — до конца раздела, по строкам.
        units.extend(line_units(text, fs, end));
    }
    units
}

fn numbered_item(t: &str) -> bool {
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && (t[digits..].starts_with(". ") || t[digits..].starts_with(") "))
}

/// Предложения в `start..end`: граница — после `.`, `!`, `?`, `…` (и
/// закрывающих кавычек/скобок за ними), если дальше пробел и заглавная буква,
/// цифра или открывающий знак; точка после сокращения — не граница.
pub(crate) fn sentences(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let slice = &text[start..end];
    let chars: Vec<(usize, char)> = slice.char_indices().collect();
    let mut result = Vec::new();
    let mut sentence_start = 0;
    let mut i = 0;
    while i < chars.len() {
        let (_, c) = chars[i];
        if matches!(c, '.' | '!' | '?' | '…') {
            let mut j = i + 1;
            while j < chars.len() && matches!(chars[j].1, '.' | '!' | '?' | '…' | '"' | '»' | ')' | '\'' | '”' | '*' | '`') {
                j += 1;
            }
            let boundary = j >= chars.len()
                || (chars[j].1.is_whitespace() && {
                    let next = chars[j..].iter().find(|(_, ch)| !ch.is_whitespace()).map(|(_, ch)| *ch);
                    next.is_some_and(|n| n.is_uppercase() || n.is_ascii_digit() || "«\"(*`[".contains(n))
                });
            if boundary && !(c == '.' && is_abbreviation(slice, chars[i].0)) {
                let end_byte = chars.get(j).map(|(b, _)| *b).unwrap_or(slice.len());
                if slice[sentence_start..end_byte].trim().is_empty() {
                    sentence_start = end_byte;
                } else {
                    result.push((start + sentence_start, start + end_byte));
                    sentence_start = end_byte;
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if !slice[sentence_start..].trim().is_empty() {
        result.push((start + sentence_start, end));
    }
    result.into_iter().filter_map(|(s, e)| trimmed(text, Piece { start: s, end: e, section: None, parent: None })).map(|p| (p.start, p.end)).collect()
}

/// Слово перед точкой в позиции `dot` — сокращение или инициал.
fn is_abbreviation(slice: &str, dot: usize) -> bool {
    let before = &slice[..dot];
    let word_start = before.rfind(|c: char| c.is_whitespace() || c == '(').map(|i| i + 1).unwrap_or(0);
    let word = before[word_start..].to_lowercase();
    if word.is_empty() {
        return false;
    }
    let single_letter = word.chars().count() == 1 && word.chars().all(char::is_alphabetic);
    single_letter || ABBREVIATIONS.contains(&word.as_str())
}

fn sentence(text: &str, kind: Kind, sections: &[Section], p: &ChunkParams) -> Vec<Piece> {
    let (window, stride) = if kind == Kind::Rust { (p.code_window, p.code_stride) } else { (p.window, p.stride) };
    let max = p.window_chars;
    let overlap = window - stride;
    // В коде окно строк идёт через весь файл (элементы вроде `mod x;` — в
    // одну строку, окно на каждый было бы крошечным), раздел — по началу
    // окна. В тексте окно не переходит границу раздела, но раздел из одного
    // заголовка присоединяется к следующему.
    let groups = match kind {
        Kind::Rust => vec![Section { start: 0, end: text.len(), path: None }],
        _ => merge_small(text, sections, 0, 0),
    };
    let label = |group: &Section, start: usize| match kind {
        Kind::Rust => section_at(sections, start),
        _ => group.path.clone(),
    };
    let mut pieces = Vec::new();
    for group in &groups {
        let units = units(text, kind, std::slice::from_ref(group), max);
        let mut i = 0;
        while i < units.len() {
            let mut j = i + 1;
            while j < units.len() && j - i < window && char_len(text, units[i].0, units[j].1) <= max {
                j += 1;
            }
            let (start, end) = (units[i].0, units[j - 1].1);
            if char_len(text, start, end) > max {
                // Одна единица длиннее окна — режется по размеру.
                for (s, e) in hard_split(text, start, end, max, 0) {
                    pieces.push(Piece { start: s, end: e, section: label(group, s), parent: None });
                }
            } else {
                pieces.push(Piece { start, end, section: label(group, start), parent: None });
            }
            if j >= units.len() {
                break;
            }
            i += (j - i).saturating_sub(overlap).max(1);
        }
    }
    pieces
}

// ---------------------------------------------------------------------------
// Parent-child

/// Диапазон без пробелов по краям.
fn trim_range(text: &str, start: usize, end: usize) -> (usize, usize) {
    let slice = &text[start..end];
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    if lead == slice.len() {
        (start, start)
    } else {
        (start + lead, end - trail)
    }
}

/// Блок из одних заголовков Markdown.
fn heading_block(text: &str, start: usize, end: usize) -> bool {
    text[start..end].lines().filter(|l| !l.trim().is_empty()).all(|l| heading(l).is_some())
}

/// Parent-child: родитель — абзац (заголовки над ним — его часть), дети —
/// по `child_sentences` предложений (в коде — по `child_lines` строк) без
/// перекрытия. Абзац длиннее `parent_chars` — несколько родителей: его
/// предложения пакуются подряд, пока влезают. Заголовок сам по себе
/// ребёнком не становится (разве что в абзаце больше ничего нет).
fn parent_child(text: &str, kind: Kind, sections: &[Section], p: &ChunkParams) -> Vec<Piece> {
    let per_child = if kind == Kind::Rust { p.child_lines } else { p.child_sentences }.max(1);
    let max = p.parent_chars;
    let mut pieces = Vec::new();
    // Раздел из одного заголовка (глава сразу перед подглавой) — к следующему.
    for section in &merge_small(text, sections, 0, 0) {
        let mut parents: Vec<(usize, usize)> = Vec::new();
        let mut heading_start: Option<usize> = None;
        for (s, e) in paragraphs(text, section.start, section.end, kind) {
            if kind == Kind::Markdown && heading_block(text, s, e) {
                heading_start.get_or_insert(s);
                continue;
            }
            parents.push((heading_start.take().unwrap_or(s), e));
        }
        if let Some(h) = heading_start {
            parents.push((h, section.end));
        }

        for (ps, pe) in parents {
            let units = match kind {
                Kind::Rust => line_units(text, ps, pe),
                _ => prose_units(text, ps, pe, kind, max),
            };
            let content: Vec<(usize, usize)> = units
                .iter()
                .copied()
                .filter(|&(a, b)| !(kind == Kind::Markdown && heading(&text[a..b]).is_some()))
                .collect();
            let units = if content.is_empty() { units } else { content };

            let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
            for unit in units {
                match groups.last_mut() {
                    Some(group) if char_len(text, group[0].0, unit.1) <= max => group.push(unit),
                    _ => groups.push(vec![unit]),
                }
            }
            let count = groups.len();
            for (gi, group) in groups.iter().enumerate() {
                // Первый родитель абзаца начинается с его заголовка, последний
                // кончается там же, где абзац.
                let from = if gi == 0 { ps } else { group[0].0 };
                let to = if gi + 1 == count { pe } else { group[group.len() - 1].1 };
                let parent = trim_range(text, from, to);
                for kids in group.chunks(per_child) {
                    let (cs, ce) = (kids[0].0, kids[kids.len() - 1].1);
                    // Одно предложение длиннее родителя — режется по размеру.
                    let spans = if char_len(text, cs, ce) > max { hard_split(text, cs, ce, max, 0) } else { vec![(cs, ce)] };
                    for (s, e) in spans {
                        pieces.push(Piece { start: s, end: e, section: section.path.clone(), parent: Some(parent) });
                    }
                }
            }
        }
    }
    pieces
}

// ---------------------------------------------------------------------------
// Качество границ — для сравнения стратегий

/// Чанк кончается на границе предложения/абзаца: последний знак — конец
/// предложения (или `}`/`;` в коде), либо дальше в документе пустая строка
/// или конец текста.
pub fn clean_end(text: &str, piece: &Piece) -> bool {
    let rest = &text[piece.end..];
    if rest.trim().is_empty() {
        return true;
    }
    let next_line_blank = rest.strip_prefix('\n').is_some_and(|r| r.lines().next().is_none_or(|l| l.trim().is_empty() || heading(l).is_some()));
    let last = text[piece.start..piece.end].chars().last().unwrap_or(' ');
    next_line_blank || matches!(last, '.' | '!' | '?' | '…' | ':' | ';' | '}' | '|' | ')' | '»' | '`' | '"')
}

/// Чанк разрезает блок кода: в Markdown — нечётное число ```` ``` ````,
/// в Rust — несбалансированные фигурные скобки.
pub fn splits_code(text: &str, kind: Kind, piece: &Piece) -> bool {
    let slice = &text[piece.start..piece.end];
    match kind {
        Kind::Markdown => slice.lines().filter(|l| is_fence(l)).count() % 2 == 1,
        Kind::Rust => slice.matches('{').count() != slice.matches('}').count(),
        Kind::Text => false,
    }
}

/// Номер строки (с 1) байта `pos`.
pub fn line_of(text: &str, pos: usize) -> usize {
    text[..pos.min(text.len())].matches('\n').count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts<'a>(text: &'a str, pieces: &[Piece]) -> Vec<&'a str> {
        pieces.iter().map(|p| &text[p.start..p.end]).collect()
    }

    fn long_paragraph(words: usize) -> String {
        (0..words).map(|i| format!("слово{i}")).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn fixed_windows_overlap_and_cover_text() {
        let text = long_paragraph(800);
        let pieces = chunk(&text, Kind::Text, Strategy::Fixed, &ChunkParams::default());
        assert!(pieces.len() > 3);
        assert_eq!(pieces[0].start, 0);
        assert_eq!(pieces.last().unwrap().end, text.len());
        for pair in pieces.windows(2) {
            assert!(pair[1].start < pair[0].end, "окна перекрываются");
            let overlap = char_len(&text, pair[1].start, pair[0].end);
            assert!((150..=FIXED_OVERLAP).contains(&overlap), "перекрытие {overlap}");
        }
        for p in &pieces {
            assert!(char_len(&text, p.start, p.end) <= FIXED_SIZE);
            // Слова не режутся.
            assert!(text[p.start..p.end].starts_with("слово"));
            assert!(text[p.start..p.end].chars().last().unwrap().is_ascii_digit());
        }
    }

    #[test]
    fn fixed_takes_section_where_chunk_starts() {
        let text = format!("# Док\n\n## Первый\n\n{}\n\n## Второй\n\n{}", long_paragraph(300), long_paragraph(300));
        let pieces = chunk(&text, Kind::Markdown, Strategy::Fixed, &ChunkParams::default());
        assert_eq!(pieces[0].section.as_deref(), Some("Док"));
        assert_eq!(pieces.last().unwrap().section.as_deref(), Some("Док › Второй"));
    }

    #[test]
    fn markdown_headings_build_paths_and_ignore_code() {
        let text = "# A\n\n## B\n\n```\n# не заголовок\n```\n\n### C\n\n## D\n";
        let paths: Vec<String> = markdown_headings(text).into_iter().map(|(_, p)| p).collect();
        assert_eq!(paths, ["A", "A › B", "A › B › C", "A › D"]);
    }

    #[test]
    fn structure_merges_short_and_splits_long_sections() {
        let text = format!(
            "# Док\n\nВступление.\n\n## Короткий\n\nОдна строка.\n\n## Длинный\n\n{}\n\n{}\n\n{}\n",
            long_paragraph(150),
            long_paragraph(150),
            long_paragraph(150)
        );
        let pieces = chunk(&text, Kind::Markdown, Strategy::Structure, &ChunkParams::default());
        // «Док» и его короткий подраздел слились — путь подраздела (в нём
        // есть и родитель).
        assert!(texts(&text, &pieces)[0].contains("Вступление.") && texts(&text, &pieces)[0].contains("Одна строка."));
        assert_eq!(pieces[0].section.as_deref(), Some("Док › Короткий"));
        // «Длинный» поделён по абзацам, у всех частей его путь.
        let long: Vec<&Piece> = pieces.iter().filter(|p| p.section.as_deref() == Some("Док › Длинный")).collect();
        assert!(long.len() >= 2, "{pieces:?}");
        for p in &pieces {
            assert!(char_len(&text, p.start, p.end) <= STRUCT_MAX);
        }
    }

    #[test]
    fn structure_keeps_code_block_whole() {
        let code = (0..40).map(|i| format!("let x{i} = {i};")).collect::<Vec<_>>().join("\n");
        let text = format!("# Док\n\n{}\n\n```rust\n{code}\n\nlet after_blank = 1;\n```\n\n{}", long_paragraph(200), long_paragraph(200));
        let pieces = chunk(&text, Kind::Markdown, Strategy::Structure, &ChunkParams::default());
        assert!(pieces.iter().all(|p| !splits_code(&text, Kind::Markdown, p)), "{:#?}", texts(&text, &pieces));
    }

    #[test]
    fn rust_items_with_docs_and_nested_impl() {
        let methods = (0..30)
            .map(|i| format!("    /// Метод {i}.\n    pub fn method_{i}(&self) -> usize {{\n        {i} + {i} * 2 + self.base\n    }}\n"))
            .collect::<String>();
        let text = format!(
            "//! Модуль.\n\nuse std::fmt;\n\n/// Структура.\n#[derive(Debug)]\npub struct Foo {{\n    base: usize,\n}}\n\nimpl<T: Clone> fmt::Display for Foo {{\n    fn fmt(&self) {{}}\n}}\n\nimpl Foo {{\n{methods}}}\n\npub(crate) async fn run() {{}}\n"
        );
        let outline: Vec<String> = rust_outline(&text, STRUCT_MAX).into_iter().map(|(_, p)| p).collect();
        assert_eq!(outline[0], "struct Foo");
        assert_eq!(outline[1], "impl fmt::Display for Foo");
        assert_eq!(outline[2], "impl Foo");
        assert_eq!(outline[3], "impl Foo › fn method_0");
        assert_eq!(outline.last().unwrap(), "fn run");
        // Doc-комментарий и атрибут — в начале раздела структуры.
        let (start, _) = rust_outline(&text, STRUCT_MAX)[0].clone();
        assert!(text[start..].starts_with("/// Структура.\n#[derive(Debug)]"));

        let pieces = chunk(&text, Kind::Rust, Strategy::Structure, &ChunkParams::default());
        assert!(pieces.iter().any(|p| p.section.as_deref().is_some_and(|s| s.starts_with("impl Foo › fn method_"))));
        assert!(pieces.iter().all(|p| char_len(&text, p.start, p.end) <= STRUCT_MAX));
    }

    #[test]
    fn rust_item_names() {
        assert_eq!(rust_item_name("pub async fn chat(&self) {").as_deref(), Some("fn chat"));
        assert_eq!(rust_item_name("pub(crate) const fn x() {").as_deref(), Some("fn x"));
        assert_eq!(rust_item_name("const LIMIT: usize = 3;").as_deref(), Some("const LIMIT"));
        assert_eq!(rust_item_name("impl<'a> From<&'a str> for Name {").as_deref(), Some("impl From<&'a str> for Name"));
        assert_eq!(rust_item_name("macro_rules! foo {").as_deref(), Some("macro_rules! foo"));
        assert_eq!(rust_item_name("let x = 1;"), None);
        assert_eq!(rust_item_name("use std::fmt;"), None);
    }

    #[test]
    fn sentence_split_respects_abbreviations() {
        let text = "Первое предложение, т.е. с сокращением. Второе! Версия 1.2 не граница? Да… Итог: см. рис. 3 и т.д. Конец.";
        let s: Vec<&str> = sentences(text, 0, text.len()).into_iter().map(|(a, b)| &text[a..b]).collect();
        assert_eq!(
            s,
            ["Первое предложение, т.е. с сокращением.", "Второе!", "Версия 1.2 не граница?", "Да…", "Итог: см. рис. 3 и т.д. Конец."]
        );
    }

    #[test]
    fn sentence_window_slides_with_overlap() {
        let body = (1..=12).map(|i| format!("Предложение номер {i}.")).collect::<Vec<_>>().join(" ");
        let text = format!("# Док\n\n{body}\n");
        let pieces = chunk(&text, Kind::Markdown, Strategy::Sentence, &ChunkParams::default());
        let t = texts(&text, &pieces);
        // Заголовок — первая единица окна: [#, 1..4], [3..7], [6..10], [9..12].
        assert!(t[0].starts_with("# Док") && t[0].ends_with("номер 4."), "{t:?}");
        assert!(t[1].starts_with("Предложение номер 3.") && t[1].ends_with("номер 7."), "{t:?}");
        assert!(t.last().unwrap().ends_with("номер 12."));
        assert!(pieces.iter().all(|p| p.section.as_deref() == Some("Док")));
    }

    #[test]
    fn sentence_window_stays_inside_section() {
        let text = "# A\n\nРаз. Два.\n\n## B\n\nТри. Четыре.\n";
        let pieces = chunk(text, Kind::Markdown, Strategy::Sentence, &ChunkParams::default());
        let t = texts(text, &pieces);
        assert_eq!(t, ["# A\n\nРаз. Два.", "## B\n\nТри. Четыре."]);
        assert_eq!(pieces[1].section.as_deref(), Some("A › B"));
    }

    #[test]
    fn sentence_heading_only_section_joins_next() {
        let text = "# Профиль\n\n## Правила\n\nРаз. Два.\n";
        let pieces = chunk(text, Kind::Markdown, Strategy::Sentence, &ChunkParams::default());
        assert_eq!(texts(text, &pieces), [text.trim_end()]);
        assert_eq!(pieces[0].section.as_deref(), Some("Профиль › Правила"));
    }

    #[test]
    fn sentence_code_window_crosses_one_line_items() {
        let text = "mod a;\nmod b;\nmod c;\n\npub fn run() {\n    work();\n}\n";
        let pieces = chunk(text, Kind::Rust, Strategy::Sentence, &ChunkParams::default());
        assert_eq!(pieces.len(), 1, "{pieces:?}");
        assert_eq!(pieces[0].section.as_deref(), Some("mod a"));
    }

    #[test]
    fn sentence_list_items_are_units() {
        let text = "Список:\n- первый пункт\n- второй пункт\n\nПосле.";
        let units = prose_units(text, 0, text.len(), Kind::Markdown, SENT_MAX);
        let u: Vec<&str> = units.iter().map(|(a, b)| &text[*a..*b]).collect();
        assert_eq!(u, ["Список:", "- первый пункт", "- второй пункт", "После."]);
    }

    #[test]
    fn sentence_code_uses_lines() {
        let text = (0..40).map(|i| format!("let v{i} = {i};")).collect::<Vec<_>>().join("\n");
        let pieces = chunk(&text, Kind::Rust, Strategy::Sentence, &ChunkParams::default());
        let first = &text[pieces[0].start..pieces[0].end];
        assert_eq!(first.lines().count(), CODE_WINDOW);
        assert!(text[pieces[1].start..].starts_with(&format!("let v{CODE_STRIDE} ")));
    }

    #[test]
    fn boundary_quality_checks() {
        let text = "Раз два три. Четыре пять";
        let whole = Piece { start: 0, end: text.len(), section: None, parent: None };
        assert!(clean_end(text, &whole));
        let mid = Piece { start: 0, end: text.find(" два").unwrap(), section: None, parent: None };
        assert!(!clean_end(text, &mid));
        let first = Piece { start: 0, end: text.find(" Четыре").unwrap(), section: None, parent: None };
        assert!(clean_end(text, &first));

        let md = "```\ncode\n```\nтекст";
        assert!(splits_code(md, Kind::Markdown, &Piece { start: 0, end: 8, section: None, parent: None }));
        assert_eq!(merged_path(Some("impl Foo › fn a"), Some("impl Foo › fn b")).as_deref(), Some("impl Foo › fn a, fn b"));
        assert_eq!(merged_path(Some("fn a, fn b, fn c"), Some("fn d")).as_deref(), Some("fn a, fn b, fn c, …"));
        assert!(!splits_code(md, Kind::Markdown, &Piece { start: 0, end: md.len(), section: None, parent: None }));
        assert_eq!(line_of(md, 5), 2);
    }

    #[test]
    fn parent_child_embeds_sentences_and_returns_paragraph() {
        let text = "# Отпуск\n\n## Подача\n\nЗаявление подаётся за две недели. Руководитель согласует его за три дня.\n\nПеренос возможен раз в год.\n";
        let pieces = chunk(text, Kind::Markdown, Strategy::Parent, &ChunkParams::default());
        let kids: Vec<&str> = texts(text, &pieces);
        // Ребёнок — одно предложение, заголовки детьми не становятся.
        assert_eq!(kids, ["Заявление подаётся за две недели.", "Руководитель согласует его за три дня.", "Перенос возможен раз в год."]);
        let parent = |p: &Piece| {
            let (a, b) = p.parent.unwrap();
            &text[a..b]
        };
        // Родитель — абзац вместе с заголовками над ним.
        assert_eq!(parent(&pieces[0]), "# Отпуск\n\n## Подача\n\nЗаявление подаётся за две недели. Руководитель согласует его за три дня.");
        assert_eq!(parent(&pieces[1]), parent(&pieces[0]));
        assert_eq!(parent(&pieces[2]), "Перенос возможен раз в год.");
        assert_eq!(pieces[0].section.as_deref(), Some("Отпуск › Подача"));

        // Два предложения на ребёнка; длинный абзац — несколько родителей.
        let two = ChunkParams { child_sentences: 2, ..ChunkParams::default() };
        assert_eq!(chunk(text, Kind::Markdown, Strategy::Parent, &two).len(), 2);
        let long = (1..=40).map(|i| format!("Предложение номер {i} о правилах.")).collect::<Vec<_>>().join(" ");
        let small = ChunkParams { parent_chars: 300, ..ChunkParams::default() };
        let pieces = chunk(&long, Kind::Text, Strategy::Parent, &small);
        assert_eq!(pieces.len(), 40);
        let parents: std::collections::BTreeSet<(usize, usize)> = pieces.iter().map(|p| p.parent.unwrap()).collect();
        assert!(parents.len() > 1);
        assert!(parents.iter().all(|&(a, b)| char_len(&long, a, b) <= 300));
        assert!(pieces.iter().all(|p| p.parent.unwrap().0 <= p.start && p.end <= p.parent.unwrap().1));
    }

    #[test]
    fn parent_child_in_code_uses_lines() {
        let text = "fn a() {\n    one();\n    two();\n    three();\n}\n\nfn b() {}\n";
        let pieces = chunk(text, Kind::Rust, Strategy::Parent, &ChunkParams::default());
        assert_eq!(texts(text, &pieces)[0], "fn a() {\n    one();\n    two();");
        assert_eq!(&text[pieces[0].parent.unwrap().0..pieces[0].parent.unwrap().1], "fn a() {\n    one();\n    two();\n    three();\n}");
    }

    #[test]
    fn params_change_chunking_and_are_validated() {
        let text = long_paragraph(800);
        let small = ChunkParams { size: 400, overlap: 50, ..ChunkParams::default() };
        let pieces = chunk(&text, Kind::Text, Strategy::Fixed, &small);
        assert!(pieces.len() > chunk(&text, Kind::Text, Strategy::Fixed, &ChunkParams::default()).len());
        assert!(pieces.iter().all(|p| char_len(&text, p.start, p.end) <= 400));

        let body = (1..=12).map(|i| format!("Предложение {i}.")).collect::<Vec<_>>().join(" ");
        let two = ChunkParams { window: 2, stride: 2, ..ChunkParams::default() };
        assert_eq!(chunk(&body, Kind::Text, Strategy::Sentence, &two).len(), 6);

        assert!(ChunkParams { overlap: 700, ..ChunkParams::default() }.validate(Strategy::Fixed).is_err());
        assert!(ChunkParams { stride: 6, ..ChunkParams::default() }.validate(Strategy::Sentence).is_err());
        assert!(ChunkParams { min_chars: 3000, ..ChunkParams::default() }.validate(Strategy::Structure).is_err());
        assert!(ChunkParams { size: 50, ..ChunkParams::default() }.validate(Strategy::Fixed).is_err());
        // Параметры другой стратегии не мешают.
        assert!(ChunkParams { stride: 99, ..ChunkParams::default() }.validate(Strategy::Fixed).is_ok());
        assert_eq!(serde_json::to_string(&small.of(Strategy::Fixed)).unwrap(), r#"{"size":400,"overlap":50}"#);
        let mut p = ChunkParams::default();
        assert!(p.set("window", 7).is_ok() && p.window == 7);
        assert!(p.set("nope", 1).is_err());
    }

    #[test]
    fn unicode_is_never_split_inside_a_char() {
        let text = "ёжик ".repeat(700);
        for strategy in Strategy::ALL {
            for p in chunk(&text, Kind::Text, strategy, &ChunkParams::default()) {
                assert!(text.is_char_boundary(p.start) && text.is_char_boundary(p.end));
            }
        }
    }
}
