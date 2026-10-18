# Саммари для Telegram библиотекой telegramify-markdown. Вход — Markdown в
# stdin, выход — JSON в stdout:
#   paste      — текст для вставки в приложение Telegram руками: разметка,
#                которую приложение само понимает при отправке (**жирный**,
#                __курсив__, ~~зачёркнутый~~, `код`, ||спойлер||), без
#                экранирования; заголовки и списки — как у библиотеки (✏, ⦁);
#   markdownv2 — MarkdownV2 для Bot API (parse_mode="MarkdownV2"); вставленный
#                руками, он показывается как есть, с обратными слэшами;
#   text_utf16 — длина сообщения после отправки в единицах UTF-16 (предел 4096).
# Запускает media-mcp (src/telegram.rs); установка — pip install -r media/requirements.txt.
import json
import sys
from importlib.metadata import version

try:
    import telegramify_markdown
except ModuleNotFoundError:
    sys.stderr.write("нет пакета telegramify-markdown — pip install -r media/requirements.txt\n")
    sys.exit(3)

# Разметка, которую приложения Telegram разбирают при отправке сообщения.
# Подчёркивания и цитат в ней нет — такие сущности остаются простым текстом.
PASTE_MARKERS = {
    "bold": "**",
    "italic": "__",
    "strikethrough": "~~",
    "code": "`",
    "spoiler": "||",
}


def paste_markup(text: str, entities) -> str:
    """Текст + сущности (смещения в UTF-16) → текст с разметкой приложения Telegram."""
    units = text.encode("utf-16-le")
    opens: dict[int, list[str]] = {}
    closes: dict[int, list[str]] = {}
    links: dict[int, list[str]] = {}
    # Внешние раньше внутренних: открываются по порядку, закрываются в обратном.
    for e in sorted(entities, key=lambda e: (e.offset, -e.length)):
        end = e.offset + e.length
        if e.type == "pre":
            opens.setdefault(e.offset, []).append("```\n")
            closes.setdefault(end, []).insert(0, "\n```")
        elif e.type == "text_link" and e.url:
            links.setdefault(end, []).append(f" ({e.url})")
        elif e.type in PASTE_MARKERS:
            marker = PASTE_MARKERS[e.type]
            # Заголовок — жирный и подчёркнутый на одном отрезке: маркер один.
            if marker in opens.get(e.offset, []) and marker in closes.get(end, []):
                continue
            opens.setdefault(e.offset, []).append(marker)
            closes.setdefault(end, []).insert(0, marker)
    out = []
    for i in range(0, len(units) // 2 + 1):
        out.extend(closes.get(i, []))
        out.extend(links.get(i, []))
        out.extend(opens.get(i, []))
        if i < len(units) // 2:
            out.append(units[2 * i : 2 * i + 2])
    # Склеить: маркеры — str, текст — по две байта UTF-16 (суррогатные пары вместе).
    result, buf = [], bytearray()
    for part in out:
        if isinstance(part, bytes):
            buf += part
        else:
            if buf:
                result.append(buf.decode("utf-16-le"))
                buf = bytearray()
            result.append(part)
    if buf:
        result.append(buf.decode("utf-16-le"))
    return "".join(result)


markdown = sys.stdin.buffer.read().decode("utf-8")
text, entities = telegramify_markdown.convert(markdown)
mdv2 = telegramify_markdown.markdownify(markdown)
json.dump(
    {
        "paste": paste_markup(text, entities),
        "markdownv2": mdv2,
        "text_utf16": telegramify_markdown.utf16_len(text),
        "version": version("telegramify-markdown"),
    },
    sys.stdout,
    ensure_ascii=False,
)
