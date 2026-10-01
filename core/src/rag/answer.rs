//! Ответ RAG-режима: обязательные блоки «Ответ / Источники / Цитаты» и их
//! проверка кодом — инструкции модели для этого мало.
//!
//! Цитаты включаются на агента ([`super::RagSettings::quotes`]). Без них
//! модель отвечает своими словами со ссылками на фрагменты `[n]`
//! ([`ANSWER_RULES`]), формат не навязывается и не проверяется — код лишь
//! добавляет блок «Источники» по фрагментам, на которые она сослалась.
//! «Не знаю» ниже порога — в обоих режимах.
//!
//! ```text
//! поиск → ничего не прошло порог → refusal: «Не знаю» + просьба уточнить (без модели)
//!       → фрагменты → модель (FORMAT_RULES) → check → ошибки? → fix_request, один повтор
//!       → finalize: источники — из данных поиска, цитаты — с отметкой проверки и строками
//! ```
//!
//! Номера: `[n]` — фрагмент (источник), `[n.m]` — m-я цитата из фрагмента n.
//! Утверждение ответа ссылается на цитату, которая его подтверждает (`[1.2]`),
//! или на весь фрагмент (`[1]`), — так видно, какая цитата к чему.
//!
//! [`finalize`] гарантирует формат при любом ответе модели: блок «Источники»
//! строится по номерам фрагментов, на которые сослалась модель, из самих
//! данных поиска (документ, раздел, `chunk_id` не переписываются с её слов);
//! у цитаты код сам находит строки документа; цитата, которой нет во
//! фрагменте дословно, помечается, а источнику без подтверждённой цитаты
//! подставляется начало его фрагмента с пометкой.

use serde::Serialize;

use super::display_source;
use super::retrieve::{RagContext, RagSource};

/// Правила свободного ответа (цитаты выключены) — в начале запроса с
/// фрагментами (см. [`super::retrieve::augment_prompt`]).
pub(crate) const ANSWER_RULES: &str = "\
[Режим RAG] Ниже — фрагменты из базы документов, найденные поиском по вопросу. Фрагменты пронумерованы: [1], [2], …

Ответь на вопрос, опираясь на них: своими словами, на языке вопроса, так, как объяснил бы эксперт, — суть, как это работает или зачем нужно, вывод.
- После фактов из фрагментов ставь номер фрагмента в квадратных скобках: [1], [2]. Источники и цитаты отдельным списком не пиши — их добавят автоматически.
- Фрагменты, не относящиеся к вопросу, игнорируй; не добавляй факты из общих знаний, не выдумывай цифры и названия.
- Если во фрагментах ответа нет — начни ответ со слов «Не знаю», одним предложением скажи, чего не хватает, и попроси уточнить вопрос.";

/// Правила ответа с цитатами — в начале запроса с фрагментами (см.
/// [`super::retrieve::augment_prompt`]).
pub(crate) const FORMAT_RULES: &str = "\
[Режим RAG] Ниже — фрагменты из базы документов, найденные поиском по вопросу. Фрагменты пронумерованы: [1], [2], … Отвечай только на их основе.

Как отвечать:
1. Сначала пойми, что из фрагментов отвечает на вопрос, и напиши ответ так, как эксперт объяснил бы это человеку: своими словами, на языке вопроса — суть, как это работает или зачем нужно, вывод. Ответ должен быть понятен без цитат.
2. Затем выбери 2–4 коротких дословных цитаты, которые подтверждают ключевые утверждения ответа, и поставь номер цитаты после такого утверждения. Не каждое предложение нуждается в цитате.

Пример (на вопрос «Зачем сервису кэш?» по фрагментам о кэше):
Плохо — ответ из пересказанных по очереди цитат:
«Кэш — это промежуточное хранилище [1.1]. Он хранит ответы базы [1.2]. Записи живут 60 секунд [2.1].»
Хорошо — объяснение, цитаты только подтверждают:
«Кэш нужен, чтобы не ходить в базу за одним и тем же: повторный запрос отдаётся из памяти [1.2]. Цена этого — данные могут отставать от базы до минуты, поэтому для свежих значений кэш не подходит [2.1].»

Формат ответа — строго три блока с этими заголовками:
Ответ: <объяснение своими словами с номерами цитат после ключевых утверждений>
Источники:
- [n] <документ › раздел> · chunk <chunk_id> — строка на каждый фрагмент, из которого есть цитаты
Цитаты:
- [n.m] «<дословный кусок текста фрагмента n>» — m — номер цитаты внутри фрагмента n: [1.1], [1.2], [2.1]

Правила:
- не переводи и не пересказывай цитаты фраза за фразой и не вставляй их в «Ответ» — цитаты стоят только в своём блоке;
- у каждой цитаты свой номер [n.m]: две цитаты из фрагмента 1 — это [1.1] и [1.2], не две [1];
- цитату копируй из текста фрагмента символ в символ — не пересказывай и не переводи; длинное место сократи многоточием «…» между дословными частями;
- фрагменты, не относящиеся к вопросу, игнорируй и в источники не включай;
- не добавляй факты из общих знаний, не выдумывай источники, номера, цифры;
- если во фрагментах ответа нет — напиши «Ответ: Не знаю.», одним предложением скажи, чего не хватает, и попроси уточнить вопрос; блоки «Источники» и «Цитаты» тогда не нужны.";

/// Цитата из ответа и что о ней известно.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quote {
    /// Фрагмент, из которого цитата.
    pub n: usize,
    /// Номер цитаты внутри фрагмента: `[n.m]`.
    pub m: usize,
    pub text: String,
    /// Найдена во фрагменте `n` дословно (без учёта регистра, пробелов,
    /// кавычек и концевой пунктуации; «…» — пропуск между частями).
    pub verified: bool,
    /// Строки документа, где цитата стоит, — у найденной.
    pub lines: Option<(i64, i64)>,
    /// Подставлена кодом — модель для источника цитату не привела или
    /// привела не дословную.
    pub auto: bool,
}

impl Quote {
    pub fn label(&self) -> String {
        format!("[{}.{}]", self.n, self.m)
    }
}

/// Итог проверки ответа модели.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AnswerCheck {
    /// Ответ с цитатами (строгий формат) — иначе свободный ответ со ссылками [n].
    pub with_quotes: bool,
    /// Модель сама сказала «не знаю» — во фрагментах ответа нет.
    pub unknown: bool,
    /// Фрагменты, на которые ссылается ответ, — по ним собран блок «Источники».
    pub cited: Vec<usize>,
    pub quotes: Vec<Quote>,
    /// Что было не так в ответе модели (после повтора — в повторном).
    pub problems: Vec<String>,
    /// Модель просили исправить ответ.
    pub retried: bool,
}

impl AnswerCheck {
    /// Строка для интерфейсов: `ответ: ✓ источники [1][2], цитаты 2/2 дословно`.
    pub fn summary(&self) -> String {
        if self.unknown {
            return "ответ: модель не нашла ответа во фрагментах — «не знаю»".into();
        }
        let refs: String = self.cited.iter().map(|n| format!("[{n}]")).collect();
        let mark = if self.problems.is_empty() { "✓" } else { "⚠" };
        if !self.with_quotes {
            return format!("ответ: {mark} источники {refs}");
        }
        let model_quotes: Vec<&Quote> = self.quotes.iter().filter(|q| !q.auto).collect();
        let verified = model_quotes.iter().filter(|q| q.verified).count();
        let mut s = format!("ответ: {mark} источники {refs}, цитаты {verified}/{} дословно", model_quotes.len());
        let auto = self.quotes.iter().filter(|q| q.auto).count();
        if auto > 0 {
            s.push_str(&format!(", {auto} подставлено из фрагментов"));
        }
        if self.retried {
            s.push_str(", после повтора");
        }
        s
    }
}

/// Ответ без запроса к модели, если ни один фрагмент не прошёл порог
/// релевантности; `None` — фрагменты есть или поиск не удался (тогда ответ
/// идёт без контекста, как раньше, и интерфейсы говорят почему). Причина и
/// ближайшие кандидаты — в строке RAG над ответом
/// ([`RagContext::summary_lines`]), не в самом ответе.
pub fn refusal(ctx: &RagContext) -> Option<String> {
    (ctx.error.is_none() && ctx.sources.is_empty()).then(|| REFUSAL.to_string())
}

/// Ответ «не знаю».
pub const REFUSAL: &str = "Не знаю: в базе документов нет фрагментов, достаточно близких к вопросу. Уточните, пожалуйста, вопрос.";

/// Просьба исправить ответ — повторный запрос, если проверка нашла ошибки.
pub fn fix_request(problems: &[String]) -> String {
    let list: String = problems.iter().map(|p| format!("\n- {p}")).collect();
    format!(
        "Ответ не прошёл проверку:{list}\n\nПерепиши его в формате «Ответ / Источники / Цитаты» по правилам выше. \
         «Ответ» — объяснение своими словами, цитаты — только в своём блоке; у каждой цитаты свой номер [n.m], \
         цитаты бери из текста фрагментов дословно. \
         Если во фрагментах ответа нет — «Ответ: Не знаю.» и просьба уточнить вопрос."
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Block {
    Answer,
    Sources,
    Quotes,
}

/// Ответ модели по блокам; строки до первого заголовка — `preamble`.
#[derive(Debug, Default)]
struct Parsed {
    preamble: Vec<String>,
    answer: Option<Vec<String>>,
    sources: Option<Vec<String>>,
    quotes: Option<Vec<String>>,
}

impl Parsed {
    fn parse(text: &str) -> Self {
        let mut parsed = Parsed::default();
        let mut current: Option<Block> = None;
        for line in text.lines() {
            if let Some((block, rest)) = header(line) {
                current = Some(block);
                let body = parsed.block_mut(block);
                if !rest.is_empty() {
                    body.push(rest.to_string());
                }
                continue;
            }
            match current {
                Some(block) => parsed.block_mut(block).push(line.to_string()),
                None => parsed.preamble.push(line.to_string()),
            }
        }
        parsed
    }

    fn block_mut(&mut self, block: Block) -> &mut Vec<String> {
        match block {
            Block::Answer => self.answer.get_or_insert_with(Vec::new),
            Block::Sources => self.sources.get_or_insert_with(Vec::new),
            Block::Quotes => self.quotes.get_or_insert_with(Vec::new),
        }
    }

    /// Текст ответа: блок «Ответ», а без заголовка — всё до первого блока.
    fn answer_text(&self) -> String {
        let lines = self.answer.as_ref().unwrap_or(&self.preamble);
        lines.join("\n").trim().to_string()
    }
}

/// `Ответ: …`, `**Источники:**`, `### Цитаты` — заголовок блока и текст после
/// него в той же строке.
fn header(line: &str) -> Option<(Block, &str)> {
    let markup = |c: char| c == '#' || c == '*' || c == '_' || c.is_whitespace();
    let t = line.trim_start_matches(markup);
    const NAMES: [(&str, Block); 5] = [
        ("ответ", Block::Answer),
        ("источники", Block::Sources),
        ("источник", Block::Sources),
        ("цитаты", Block::Quotes),
        ("цитата", Block::Quotes),
    ];
    for (name, block) in NAMES {
        let len = name.chars().count();
        let head: String = t.chars().take(len).collect();
        if head.to_lowercase() != name {
            continue;
        }
        let rest = t[head.len()..].trim_start_matches(markup);
        // «Ответ» — заголовок, только если за ним двоеточие или конец строки:
        // «Ответственность …» и «Ответ дан выше» — обычный текст.
        let rest = match rest.strip_prefix(':') {
            Some(r) => r,
            None if rest.is_empty() => rest,
            None => continue,
        };
        return Some((block, rest.trim_start_matches(markup).trim_end()));
    }
    None
}

/// Ссылка в квадратных скобках: `[2]` — фрагмент, `[2.1]` — цитата из него.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ref {
    n: usize,
    m: Option<usize>,
}

impl Ref {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().split_once('.') {
            Some((n, m)) => Some(Ref { n: n.trim().parse().ok()?, m: Some(m.trim().parse().ok()?) }),
            None => Some(Ref { n: s.trim().parse().ok()?, m: None }),
        }
    }

    fn label(&self) -> String {
        match self.m {
            Some(m) => format!("[{}.{m}]", self.n),
            None => format!("[{}]", self.n),
        }
    }
}

/// Ссылки по порядку: `[1]`, `[2.1]`, `[2, 3]`, `[1.1, 1.2]`.
fn refs(s: &str) -> Vec<Ref> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(open) = rest.find('[') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find(']') else { break };
        let parsed: Option<Vec<Ref>> = rest[..close].split(',').map(Ref::parse).collect();
        if let Some(parsed) = parsed {
            out.extend(parsed);
        }
        rest = &rest[close + 1..];
    }
    out
}

/// Строка блока «Цитаты» → номер и текст цитаты.
fn parse_quote(line: &str) -> Option<(Option<Ref>, String)> {
    let t = line.trim().trim_start_matches(|c: char| matches!(c, '-' | '*' | '•') || c.is_whitespace());
    if t.is_empty() {
        return None;
    }
    let r = refs(t).first().copied();
    // Текст — между первой открывающей и последней закрывающей кавычкой, без
    // кавычек — всё после номера.
    let opening = t.find(['«', '“', '„', '"']);
    let text = match opening {
        Some(open) => {
            let after = &t[open + t[open..].chars().next().unwrap().len_utf8()..];
            match after.rfind(['»', '”', '"', '“']) {
                Some(close) => after[..close].to_string(),
                None => after.to_string(),
            }
        }
        None => {
            let after = t.find(']').map(|i| &t[i + 1..]).unwrap_or(t);
            after.trim_start_matches(|c: char| matches!(c, ':' | '-' | '—' | '–') || c.is_whitespace()).to_string()
        }
    };
    let text = text.trim().to_string();
    (!text.is_empty()).then_some((r, text))
}

/// Текст для сравнения цитаты с фрагментом — с номером строки (от 0) каждого
/// символа: нижний регистр, `ё` → `е`, одни дефисы, без кавычек и
/// разметки, пробелы схлопнуты.
fn normalize_lines(s: &str) -> Vec<(char, usize)> {
    let mut out = Vec::new();
    let mut line = 0;
    let mut space = false;
    for c in s.chars() {
        if c == '\n' {
            line += 1;
        }
        if matches!(c, '*' | '`' | '«' | '»' | '“' | '”' | '„' | '"') {
            continue;
        }
        if c.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        let c = match c {
            '—' | '–' | '−' => '-',
            '’' | '‘' => '\'',
            'ё' | 'Ё' => 'е',
            c => c,
        };
        if space {
            out.push((' ', line));
            space = false;
        }
        out.extend(c.to_lowercase().map(|lc| (lc, line)));
    }
    out
}

fn normalize(s: &str) -> String {
    normalize_lines(s).into_iter().map(|(c, _)| c).collect()
}

/// Где цитата во фрагменте: строки (от 0) её начала и конца, если её части
/// между «…» идут во фрагменте по порядку.
fn locate(quote: &str, fragment: &str) -> Option<(usize, usize)> {
    let chars = normalize_lines(fragment);
    let text: String = chars.iter().map(|(c, _)| *c).collect();
    let quote = normalize(quote).replace("...", "…");
    let trim = |p: &str| p.trim_matches(|c: char| c.is_whitespace() || ".,;:!?()[]".contains(c)).to_string();
    let parts: Vec<String> = quote.split('…').map(trim).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() || parts.iter().map(|p| p.chars().count()).sum::<usize>() < 3 {
        return None;
    }
    let line_at = |byte: usize| chars[text[..byte].chars().count()].1;
    let mut from = 0;
    let mut first = None;
    let mut last = 0;
    for part in parts {
        let at = from + text[from..].find(&part)?;
        first.get_or_insert(line_at(at));
        last = line_at(at + part.len() - part.chars().last().map_or(1, char::len_utf8));
        from = at + part.len();
    }
    first.map(|f| (f, last))
}

/// Строки документа, где цитата стоит во фрагменте `src`.
fn quote_lines(quote: &str, src: &RagSource) -> Option<(i64, i64)> {
    locate(quote, &src.text).map(|(a, b)| (src.start_line + a as i64, src.start_line + b as i64))
}

fn says_unknown(answer: &str) -> bool {
    let head: String = normalize(answer).chars().take(60).collect();
    head.contains("не знаю")
}

/// Проверка ответа модели на фрагментах `ctx`.
pub fn check(text: &str, ctx: &RagContext) -> AnswerCheck {
    let parsed = Parsed::parse(text);
    let answer = parsed.answer_text();
    let total = ctx.sources.len();
    let mut check = AnswerCheck { with_quotes: ctx.quotes, ..AnswerCheck::default() };
    if ctx.quotes && parsed.answer.is_none() {
        check.problems.push("нет блока «Ответ:»".into());
    }
    if answer.is_empty() {
        check.problems.push("ответ пуст".into());
    }
    if says_unknown(&answer) {
        check.unknown = true;
        return check;
    }

    let answer_refs = refs(&answer);
    let valid = |n: usize| (1..=total).contains(&n);
    // Без цитат — только ссылки: источники строятся по ним, несуществующий
    // номер — замечание в строке RAG, повтора ради него нет.
    if !ctx.quotes {
        let mut bad: Vec<usize> = answer_refs.iter().map(|r| r.n).filter(|&n| !valid(n)).collect();
        bad.dedup();
        for n in bad {
            check.problems.push(format!("ссылка на несуществующий фрагмент [{n}] — их {total}"));
        }
        check.cited = answer_refs.iter().map(|r| r.n).filter(|&n| valid(n)).collect();
        check.cited.sort_unstable();
        check.cited.dedup();
        return check;
    }

    let source_refs: Vec<Ref> = parsed.sources.iter().flatten().flat_map(|l| refs(l)).collect();
    let raw_quotes: Vec<(Option<Ref>, String)> =
        parsed.quotes.iter().flatten().filter_map(|l| parse_quote(l)).collect();

    let mut bad: Vec<usize> = answer_refs
        .iter()
        .chain(&source_refs)
        .chain(raw_quotes.iter().filter_map(|(r, _)| r.as_ref()))
        .map(|r| r.n)
        .filter(|&n| !valid(n))
        .collect();
    bad.sort_unstable();
    bad.dedup();
    for n in bad {
        check.problems.push(format!("ссылка на несуществующий фрагмент [{n}] — их {total}"));
    }
    if answer_refs.is_empty() {
        check.problems.push("в ответе нет ссылок на цитаты [n.m]".into());
    }
    if source_refs.is_empty() {
        check.problems.push("нет блока «Источники:» со строками «- [n] …»".into());
    }
    if raw_quotes.is_empty() {
        check.problems.push("нет блока «Цитаты:» с дословными цитатами".into());
    }

    // Номер цитаты внутри фрагмента: свой у модели, если он есть и не
    // повторяется; иначе — следующий свободный (две «[1]» → [1.1] и [1.2]).
    let mut repeated = false;
    for (r, text) in &raw_quotes {
        let Some(r) = r.filter(|r| valid(r.n)) else {
            if r.is_none() {
                check.problems.push(format!("у цитаты «{}» нет номера [n.m]", short(text, 80)));
            }
            continue;
        };
        let taken = |m: usize| check.quotes.iter().any(|q| q.n == r.n && q.m == m);
        let m = match r.m {
            Some(m) if m > 0 && !taken(m) => m,
            _ => {
                repeated |= r.m.is_none() && check.quotes.iter().any(|q| q.n == r.n);
                (1..).find(|&m| !taken(m)).unwrap()
            }
        };
        let lines = quote_lines(text, &ctx.sources[r.n - 1]);
        if lines.is_none() {
            check.problems.push(format!("цитаты [{}.{m}] «{}» нет дословно во фрагменте [{}]", r.n, short(text, 80), r.n));
        }
        check.quotes.push(Quote { n: r.n, m, text: text.clone(), verified: lines.is_some(), lines, auto: false });
    }
    // «Ответ» — объяснение, а не цитаты: длинная цитата в нём дословно —
    // ошибка (перевод цитат код не распознаёт — его держат правила).
    let mut answer_plain = answer.clone();
    for r in &answer_refs {
        answer_plain = answer_plain.replace(&r.label(), "");
    }
    let answer_norm = normalize(&answer_plain);
    for q in check.quotes.iter().filter(|q| q.verified) {
        let quote = normalize(&q.text);
        let quote = quote.trim_matches(|c: char| c.is_whitespace() || ".,;:!?".contains(c));
        if quote.split(' ').count() >= 6 && answer_norm.contains(quote) {
            check.problems.push(format!(
                "«Ответ» повторяет цитату {} дословно — объясни своими словами, цитата остаётся в блоке «Цитаты»",
                q.label()
            ));
        }
    }
    if repeated {
        check.problems.push("у нескольких цитат одного фрагмента один номер — нужны [n.1], [n.2]".into());
    }
    for r in answer_refs.iter().filter(|r| valid(r.n)) {
        if let Some(m) = r.m {
            if !check.quotes.iter().any(|q| q.n == r.n && q.m == m) {
                check.problems.push(format!("в ответе ссылка {} — такой цитаты нет", r.label()));
            }
        }
    }

    let mut cited: Vec<usize> = answer_refs.iter().chain(&source_refs).map(|r| r.n).filter(|&n| valid(n)).collect();
    cited.extend(check.quotes.iter().map(|q| q.n));
    cited.sort_unstable();
    cited.dedup();
    if !raw_quotes.is_empty() {
        for &n in &cited {
            if !check.quotes.iter().any(|q| q.n == n) {
                check.problems.push(format!("нет цитаты из фрагмента [{n}]"));
            }
        }
    }
    check.cited = cited;
    check
}

/// Итоговый текст ответа: блоки в обязательном виде при любом ответе модели.
/// Дополняет `check` подставленными цитатами.
pub fn finalize(text: &str, ctx: &RagContext, check: &mut AnswerCheck) -> String {
    let parsed = Parsed::parse(text);
    let answer = parsed.answer_text();
    let head = if check.with_quotes { "Ответ: " } else { "" };
    if check.unknown {
        let mut s = format!("{head}{answer}");
        if !answer.contains('?') && !answer.to_lowercase().contains("уточн") {
            s.push_str("\n\nУточните, пожалуйста, вопрос: о каком документе или разделе речь, какие термины там используются?");
        }
        return s;
    }
    // Модель не сослалась ни на один фрагмент — источниками считаются все,
    // что ушли ей: ответ построен на них.
    if check.cited.is_empty() {
        check.cited = ctx.sources.iter().map(|s| s.n).collect();
    }
    if !check.with_quotes {
        return format!("{answer}\n\nИсточники:{}", sources_block(ctx, &check.cited));
    }
    for n in check.cited.clone() {
        if check.quotes.iter().any(|q| q.n == n && q.verified) {
            continue;
        }
        let src = &ctx.sources[n - 1];
        let text = excerpt(src);
        let m = (1..).find(|&m| !check.quotes.iter().any(|q| q.n == n && q.m == m)).unwrap();
        let lines = quote_lines(&text, src);
        check.quotes.push(Quote { n, m, text, verified: true, lines, auto: true });
    }
    check.quotes.sort_by_key(|q| (q.n, q.m));

    let mut s = format!("Ответ: {answer}\n\nИсточники:{}", sources_block(ctx, &check.cited));
    s.push_str("\n\nЦитаты:");
    for q in &check.quotes {
        let place = match q.lines {
            Some((a, b)) if a == b => format!(" строка {a}:"),
            Some((a, b)) => format!(" строки {a}–{b}:"),
            None => String::new(),
        };
        s.push_str(&format!("\n- {}{place} «{}»", q.label(), q.text));
        if q.auto {
            s.push_str(" (начало фрагмента — модель дословную цитату не привела)");
        } else if !q.verified {
            s.push_str(&format!(" ⚠ нет дословно во фрагменте [{}]", q.n));
        }
    }
    s
}

/// Строки блока «Источники» — из данных поиска.
fn sources_block(ctx: &RagContext, cited: &[usize]) -> String {
    let mut s = String::new();
    for &n in cited {
        let src = &ctx.sources[n - 1];
        // Имя и chunk_id — кодом: ответ рендерится как Markdown, и `_` в
        // «AI_SDLC.txt … AI_SDLC.txt@v1» стал бы курсивом.
        let location = src.location().replacen(&src.name, &format!("`{}`", src.name), 1);
        s.push_str(&format!("\n- [{n}] {location} · chunk `{}`", display_source(&src.chunk_id)));
    }
    s
}

/// Начало фрагмента одной строкой: первое предложение, не длиннее 200 символов.
fn excerpt(src: &RagSource) -> String {
    let flat = src.text.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence = match flat.find(". ") {
        Some(i) if i >= 20 => &flat[..=i],
        _ => flat.as_str(),
    };
    short(sentence, 200)
}

fn short(s: &str, max: usize) -> String {
    super::retrieve::short(s, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RagContext {
        let src = |n: usize, section: &str, text: &str| RagSource {
            n,
            chunk_id: format!("rag/uploads/guide.md@v1/structure/{n:04}"),
            source: "rag/uploads/guide.md".into(),
            name: "guide.md".into(),
            version: 1,
            section: Some(section.into()),
            start_line: n as i64 * 10,
            end_line: n as i64 * 10 + 5,
            score: 0.7,
            rerank_score: None,
            rank_before: n,
            text: text.into(),
        };
        RagContext {
            strategy: "structure".into(),
            sources: vec![
                src(1, "Настройка", "Порт сервера задаётся переменной APP_PORT,\nпо умолчанию 8080.\nЖурналы — отдельно."),
                src(2, "Журналы", "Журналы пишутся в каталог logs."),
            ],
            candidates: 2,
            quotes: true,
            ..RagContext::default()
        }
    }

    #[test]
    fn free_answer_keeps_its_text_and_gets_sources() {
        let ctx = RagContext { quotes: false, ..ctx() };
        let text = "Порт берётся из переменной APP_PORT, а без неё сервер слушает 8080 [1]. Журналы — отдельно [7].";
        let mut check = check(text, &ctx);
        assert!(!check.with_quotes && check.cited == vec![1]);
        assert_eq!(check.problems, vec!["ссылка на несуществующий фрагмент [7] — их 2".to_string()]);
        let out = finalize(text, &ctx, &mut check);
        assert_eq!(
            out,
            format!("{text}\n\nИсточники:\n- [1] `guide.md` › Настройка · строки 10–15 · chunk `guide.md@v1/structure/0001`")
        );
        assert!(check.quotes.is_empty());
        assert_eq!(check.summary(), "ответ: ⚠ источники [1]");

        // Без формата нет и замечаний о нём; «не знаю» — с просьбой уточнить.
        let mut unknown = super::check("Не знаю — во фрагментах нет версии протокола.", &ctx);
        assert!(unknown.unknown && unknown.problems.is_empty());
        let out = finalize("Не знаю — во фрагментах нет версии протокола.", &ctx, &mut unknown);
        assert!(out.starts_with("Не знаю") && out.contains("Уточните"));
    }

    #[test]
    fn well_formed_answer_passes_and_sources_come_from_search() {
        let text = "**Ответ:** Порт задаётся переменной APP_PORT [1.1], по умолчанию 8080 [1.2].\n\n\
                    **Источники:**\n- [1] guide.md › что-то выдуманное\n\n\
                    **Цитаты:**\n- [1.1] «Порт сервера задаётся переменной APP_PORT»\n- [1.2] «по умолчанию 8080.»";
        let ctx = ctx();
        let mut check = check(text, &ctx);
        assert!(check.problems.is_empty(), "{:?}", check.problems);
        assert_eq!(check.cited, vec![1]);
        assert_eq!(check.quotes.iter().map(|q| (q.m, q.lines)).collect::<Vec<_>>(), vec![(1, Some((10, 10))), (2, Some((11, 11)))]);
        let out = finalize(text, &ctx, &mut check);
        assert!(out.starts_with("Ответ: Порт задаётся переменной APP_PORT [1.1], по умолчанию 8080 [1.2]."));
        // Источник — из данных поиска, не с ответа модели; без папки загрузок.
        assert!(out.contains("- [1] `guide.md` › Настройка · строки 10–15 · chunk `guide.md@v1/structure/0001`"), "{out}");
        assert!(!out.contains("выдуманное") && !out.contains("rag/uploads"));
        assert!(out.contains("- [1.1] строка 10: «Порт сервера задаётся переменной APP_PORT»"), "{out}");
        assert!(out.contains("- [1.2] строка 11: «по умолчанию 8080.»"), "{out}");
        assert!(!out.contains("⚠"));
        assert_eq!(check.summary(), "ответ: ✓ источники [1], цитаты 2/2 дословно");
    }

    #[test]
    fn answer_copying_a_quote_is_a_problem() {
        let text = "Ответ: Порт сервера задаётся переменной APP_PORT, по умолчанию 8080 [1.1].\nИсточники:\n- [1]\n\
                    Цитаты:\n- [1.1] «Порт сервера задаётся переменной APP_PORT, по умолчанию 8080.»";
        let check = check(text, &ctx());
        assert_eq!(check.problems.len(), 1, "{:?}", check.problems);
        assert!(check.problems[0].contains("повторяет цитату [1.1] дословно"));
        // Короткий термин из цитаты в ответе — не копирование.
        let text = "Ответ: Порт берётся из APP_PORT, а если её нет — 8080 [1.1].\nИсточники:\n- [1]\n\
                    Цитаты:\n- [1.1] «Порт сервера задаётся переменной APP_PORT, по умолчанию 8080.»";
        assert!(super::check(text, &ctx()).problems.is_empty());
    }

    #[test]
    fn quotes_with_one_fragment_number_get_their_own() {
        // Как ответила qwen: две цитаты из фрагмента 1 обе под [1].
        let text = "Ответ: Порт — APP_PORT [1], журналы отдельно [1].\nИсточники:\n- [1]\n\
                    Цитаты:\n- [1] «Порт сервера задаётся переменной APP_PORT»\n- [1] «Журналы — отдельно.»";
        let ctx = ctx();
        let mut check = check(text, &ctx);
        assert!(check.problems.iter().any(|p| p.contains("один номер")), "{:?}", check.problems);
        let out = finalize(text, &ctx, &mut check);
        assert!(out.contains("- [1.1] строка 10: «Порт сервера"), "{out}");
        assert!(out.contains("- [1.2] строка 12: «Журналы — отдельно.»"), "{out}");
    }

    #[test]
    fn quote_spanning_lines_and_dangling_ref() {
        let text = "Ответ: По умолчанию 8080 [1.1], журналы в logs [2.3].\nИсточники:\n- [1]\n- [2]\n\
                    Цитаты:\n- [1.1] «APP_PORT, по умолчанию 8080»\n- [2.1] «Журналы пишутся в каталог logs»";
        let ctx = ctx();
        let check = check(text, &ctx);
        assert_eq!(check.quotes[0].lines, Some((10, 11)));
        assert_eq!(check.quotes[1].lines, Some((20, 20)));
        assert_eq!(check.problems, vec!["в ответе ссылка [2.3] — такой цитаты нет".to_string()]);
    }

    #[test]
    fn invented_quote_and_unknown_ref_are_problems() {
        let text = "Ответ: Порт — 9090 [1.1][5].\nИсточники:\n- [1]\nЦитаты:\n- [1.1] «порт по умолчанию 9090»";
        let ctx = ctx();
        let mut check = check(text, &ctx);
        assert!(check.problems.iter().any(|p| p.contains("[5]")), "{:?}", check.problems);
        assert!(check.problems.iter().any(|p| p.contains("нет дословно")), "{:?}", check.problems);
        assert!(!check.quotes[0].verified);
        let out = finalize(text, &ctx, &mut check);
        assert!(out.contains("- [1.1] «порт по умолчанию 9090» ⚠ нет дословно во фрагменте [1]"), "{out}");
        // Подтверждённой цитаты нет — подставлено начало фрагмента.
        assert!(out.contains("- [1.2] строки 10–11: «Порт сервера задаётся переменной APP_PORT, по умолчанию 8080.» (начало фрагмента"), "{out}");
        assert!(check.summary().starts_with("ответ: ⚠ источники [1], цитаты 0/1 дословно, 1 подставлено"));
    }

    #[test]
    fn free_text_answer_gets_all_blocks() {
        let text = "Журналы лежат в каталоге logs.";
        let ctx = ctx();
        let mut check = check(text, &ctx);
        assert!(check.problems.iter().any(|p| p.contains("«Ответ:»")));
        assert!(check.problems.iter().any(|p| p.contains("«Источники:»")));
        assert!(check.problems.iter().any(|p| p.contains("«Цитаты:»")));
        let out = finalize(text, &ctx, &mut check);
        assert!(out.starts_with("Ответ: Журналы лежат в каталоге logs.\n\nИсточники:\n- [1]"), "{out}");
        assert!(out.contains("- [2] `guide.md` › Журналы"));
        assert!(out.contains("- [2.1] строка 20: «Журналы пишутся в каталог logs.» (начало фрагмента"), "{out}");
    }

    #[test]
    fn quote_matching_tolerates_case_spacing_and_ellipsis() {
        let f = "Порт сервера задаётся переменной APP_PORT,\n  по умолчанию 8080.";
        assert_eq!(locate("порт сервера задаётся переменной APP_PORT, по умолчанию 8080", f), Some((0, 1)));
        assert_eq!(locate("Порт сервера … по умолчанию 8080.", f), Some((0, 1)));
        assert_eq!(locate("Порт сервера ... APP_PORT", f), Some((0, 0)));
        assert_eq!(locate("по умолчанию 8080", f), Some((1, 1)));
        assert!(locate("по умолчанию 8080 … Порт сервера", f).is_none(), "части не по порядку");
        assert!(locate("по умолчанию 9090", f).is_none());
        assert!(locate("…", f).is_none());
    }

    #[test]
    fn model_unknown_asks_to_clarify() {
        let text = "Ответ: Не знаю — во фрагментах нет ничего о версии протокола.";
        let ctx = ctx();
        let mut check = check(text, &ctx);
        assert!(check.unknown && check.problems.is_empty(), "{check:?}");
        let out = finalize(text, &ctx, &mut check);
        assert!(out.contains("Уточните, пожалуйста"), "{out}");
        assert!(!out.contains("Источники"));
    }

    #[test]
    fn headers_and_refs_parse() {
        assert_eq!(header("### Цитаты").map(|(b, r)| (b, r.to_string())), Some((Block::Quotes, String::new())));
        assert_eq!(header("Ответ: да [1]").map(|(b, r)| (b, r.to_string())), Some((Block::Answer, "да [1]".into())));
        assert!(header("Ответственность лежит на сервере").is_none());
        assert!(header("Ответ дан выше").is_none());
        let labels: Vec<String> = refs("a [1] b [2, 3] [x] [10] [1.2] [2.1, 2.2] [0.5x]").iter().map(Ref::label).collect();
        assert_eq!(labels, ["[1]", "[2]", "[3]", "[10]", "[1.2]", "[2.1]", "[2.2]"]);
        assert_eq!(parse_quote("- [2.1] — Журналы пишутся"), Some((Some(Ref { n: 2, m: Some(1) }), "Журналы пишутся".into())));
    }

    #[test]
    fn refusal_only_without_fragments() {
        let empty = RagContext { strategy: "structure".into(), candidates: 3, below_threshold: 3, ..RagContext::default() };
        assert_eq!(
            refusal(&empty).as_deref(),
            Some("Не знаю: в базе документов нет фрагментов, достаточно близких к вопросу. Уточните, пожалуйста, вопрос.")
        );
        assert!(refusal(&ctx()).is_none());
        assert!(refusal(&RagContext { error: Some("нет связи".into()), ..RagContext::default() }).is_none());
    }
}
