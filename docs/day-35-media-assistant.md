# День 35: ассистент для обработки записей встреч

`media-assistant` — ассистент, который доводит запись встречи книжного клуба от видео до саммари: WebM или MP4 → `output.mp4` → `output.mp3` → MP3-чанки → транскрипт → саммари в Markdown → версия саммари для Telegram (для вставки руками и MarkdownV2 для бота). Раньше это был скилл `convert-summarize` для Claude Code или Codex. Больше половины его текста уговаривала агента не ошибаться: дождаться `ffmpeg`, а не верить появившемуся файлу; не пропускать чанки с готовыми TXT; сверять длительность MP4 с WebM. Теперь эти правила — код:

- **MCP `media`** — свой сервер `mcp/media-mcp`. Он разбирает каталог записи, находит последний годный чекпоинт и называет следующий шаг. Шаги вне очереди отклоняет. Долгую работу запускает заданиями и сам ждёт выхода процесса;
- **модель** (общая для проекта, `LLM_MODEL`) ведёт процесс по подсказкам сервера, задаёт вопросы, где нужен выбор («какой WebM?», «Сделать саммари? (да/нет)»), и пишет отчёт;
- **ожидание заданий** делает код ассистента, а не модель: вызов шага возвращается модели, только когда задание закончилось, а ход работы виден в интерфейсе;
- **диалог** сохраняется в `media/runs/<id>/`.

## Как выполняется обработка

```text
«обработай» (или ответ на вопрос ассистента)
  → модель, до MEDIA_MAX_ROUNDS кругов на реплику:
      media · inspect                — чекпоинты и next: какой шаг и с какими доводами
      media · convert_to_mp4         ┐
      media · extract_mp3            │ задание сервера; код ассистента ждёт его через
      media · split_audio            │ wait_job по 5 с и показывает ход; модель получает
      media · transcribe             │ итог: succeeded (result) или failed (error)
      media · summarize              ┘
      media · telegram_summary       — output-summary.md → output-summary-telegram.md (вставить в Telegram)
                                       и output-summary-telegram-bot.md (MarkdownV2 для бота)
      media · list_llm_models, write_summary_prompt, probe, list_recordings
  → вопрос пользователю (choose_source, ask_summary) — диалог продолжается следующей репликой
  → или отчёт: что переиспользовано, что сделано, длительности, итоговые файлы, модель LM Studio
```

Порядок чекпоинтов тот же, что в скилле. Первым годится самый поздний:

| Чекпоинт | Годен, когда | Следующий шаг |
|---|---|---|
| `output-transcript.md` | обычный непустой файл | `ask_summary`; если `output-summary.md` уже есть — `telegram_summary`, а когда есть и `output-summary-telegram.md` не старше саммари — `done` |
| `transcript-chunks/chunk-*.mp3` | хотя бы один, все непустые, имена `chunk-NNN.mp3` | `transcribe` — **все** чанки, TXT перезаписываются |
| `output.mp3` | непустой | `split_audio` |
| `output.mp4` | непустой и совпал по длительности (целые секунды) с WebM; если WebM рядом нет — без сверки | `extract_mp3`. При несовпадении — `convert_to_mp4`: старый файл переезжает в `diagnostics/` |
| исходный `.mp4` | один (несколько — `choose_source`) | `extract_mp3` |
| `.webm` | один (несколько — `choose_source`) | `convert_to_mp4` |

TXT чанков в выборе не участвуют: это не чекпоинт.

## Что гарантирует сервер, а не модель

| Правило скилла | Где в коде |
|---|---|
| Дождаться выхода `ffmpeg`, файл или его рост — не признак готовности | `media.rs`: сервер сам держит процесс (`kill_on_drop`, `-nostdin`), успех — только код 0. `jobs.rs`: в каталоге одновременно идёт одно задание, второе и зависимые шаги получают отказ |
| Шаги по порядку, без лишней работы | `recording.rs::inspect` выбирает шаг. Каждый инструмент из `steps.rs` сверяется с ним и отказывает: «следующий шаг — transcribe (…)» |
| Сверка WebM и MP4 по длительности | `ffprobe` обоих файлов, округление как у `printf "%.0f"`. При несовпадении задание завершается ошибкой, оба файла сохраняются |
| Негодный артефакт не затирать | старый `output.mp4` или негодный набор чанков переезжают в `diagnostics/<имя>-<время>` |
| Прерванная запись не выглядит готовой | MP3 пишется в `.output.mp3.partial`, чанки — во временный каталог, TXT, транскрипт и саммари — через временный файл с переименованием. Недописанный `output.mp4` ловит сверка длительности |
| Все чанки, не больше двух запросов, повторы | `transcribe.rs`: 30 с на соединение, 1800 с на ответ, ещё 2 попытки через 5 с при любой ошибке. Транскрипт собирается только из чанков, расшифрованных в **этом** запуске: старый TXT не делает неполную транскрипцию полной |
| Склейка без правок | байты TXT по порядку номеров, без заголовков |
| Саммари: шаблон, без обрезки, без другого LLM | `summary.rs`: шаблон из скилла (`summary-prompt.md`), полный транскрипт. Сохраняется только ответ модели, без рассуждений `<think>` и без обёртки в блок кода. Готовое саммари перезаписывается только с `force` |

## Саммари для Telegram

Шаблон просит модель ответить в Markdown: `## Краткое резюме` (абзац) и `## Ключевые мысли` (список, главное в пункте — **жирным**). Шаблон из скилла изменён только в части «Формат».

`telegram_summary` разбирает `output-summary.md` Python-библиотекой [telegramify-markdown](https://github.com/sudoskys/telegramify-markdown) и пишет два файла. **Сервер ничего не отправляет.**

| Файл | Для чего | Как выглядит |
|---|---|---|
| `output-summary-telegram.md` | Вставить в приложение Telegram руками и отправить | Разметка, которую приложение само разбирает при отправке: `**жирный**`, `__курсив__`, `~~зачёркнутый~~`, `` `код` ``, `\|\|спойлер\|\|`. Без экранирования. Ссылка — «текст (url)», подчёркивания нет |
| `output-summary-telegram-bot.md` | Bot API: `sendMessage` с `parse_mode="MarkdownV2"` | MarkdownV2 (`markdownify`): `__*заголовок*__`, экранирование `\.`, `\(`, `\!`. Вставленный руками, показывается как есть, с обратными слэшами |

Заголовки в обоих — жирные с эмодзи `✏`, пункты — `⦁`. Версия для вставки собирается из сущностей `convert()` (текст + смещения в UTF-16): маркеры приложения ставятся по их позициям, эмодзи позиции не сбивают. В ответе инструмента — длина отправленного сообщения в единицах UTF-16 (так считает Telegram) и `fits_one_message`: помещается ли оно в одно сообщение (4096).

**Ограничение библиотеки.** `telegramify-markdown` 1.4.0 вместе с `pyromark` 0.10 молча возвращает пустой текст, если во входе есть `~` или `$` («около ~20 человек», «$10»). В `pyromark` 0.10 события — одноразовый итератор, а библиотека проходит по ним дважды. Поэтому в `media/requirements.txt` закреплён `pyromark==0.9.14`, а сервер не записывает пустой результат: он завершается ошибкой с подсказкой про версии.

Шаг встроен в пайплайн: после `summarize` `inspect` называет `telegram_summary`. Если саммари пересоздали (`force`), прежняя версия для Telegram считается устаревшей (`stale`) и делается заново.

Скрипт `telegram.py` встроен в сервер и запускается Python из `MEDIA_PYTHON`. По умолчанию это `media/.venv/bin/python`, если он есть, иначе `python3`. Установка:

```bash
python3 -m venv media/.venv
media/.venv/bin/pip install -r media/requirements.txt      # telegramify-markdown==1.4.0, pyromark==0.9.14
```

Модель LM Studio для саммари выбирается так. Если модель не указана и пригодная для чата всего одна или загружена в память всего одна, сервер берёт её сам (`obvious` в `list_llm_models`). Иначе ассистент спрашивает пользователя. Это расширение правила скилла: на прогонах в LM Studio было 9 чат-моделей и одна загруженная, а незагруженная отвечает «Model is unloaded».

## Где набирать

| Интерфейс | Как |
|---|---|
| TUI (основной) | `cargo run -p media-assistant [каталог]`. Реплики — просто текстом: «обработай», «да», «возьми meeting-2.webm». В шапке: сервер `media`, каталог записи, следующий шаг. Внизу — ход идущего задания (`convert_to_mp4 j1 · 12:01 из 58:10 (20%)`, `transcribe j4 · чанки: 7 из 14 готово`) |
| CLI | `media-assistant run [каталог] [--summary yes\|no] [--quiet]`. Ответ — в stdout, вызовы и ход заданий — в stderr. Без `--summary` прогон остановится на вопросе о саммари |
| Claude Code / Codex и другие клиенты MCP | `cargo run -p media-mcp -- --http` → `http://127.0.0.1:8098/mcp`. Клиент сам вызывает `wait_job` (инструкции сервера это объясняют) |

### Команды TUI

| Команда | Что делает |
|---|---|
| `<текст>` | Реплика ассистенту в текущем диалоге |
| `/dir [каталог]` | Выбрать запись (путь от `MEDIA_ROOT`) и начать новый диалог. Без аргумента — где есть записи и докуда они обработаны |
| `/status` | Чекпоинты каталога и следующий шаг — кодом, без модели |
| `/jobs` | Задания сервера: идущие и завершённые, с ошибками |
| `/new` | Новый диалог о той же записи |
| `/calls [off\|brief\|debug]` | Как показывать вызовы MCP (то же, что **Ctrl+O**) |
| `/clear`, `/?`, `/quit` | Очистить ленту, справка, выход |

Выход во время задания останавливает и его, и `ffmpeg`: сервер работает по stdio и живёт, пока жив ассистент. Поэтому в это время Esc нужно нажать дважды. Начатое не пропадёт: при следующем запуске `inspect` найдёт годные чекпоинты, недописанный `output.mp4` не пройдёт сверку и будет пересоздан, а MP3 и чанки недописанными не бывают. Чтобы задания переживали перезапуск ассистента, запустите сервер отдельно с `--http` и укажите в `media/mcp.json` `"url": "http://127.0.0.1:8098/mcp"`.

## Прогон на реальных сервисах

Запись: 24 секунды русской речи (`say -v Milena`) на тестовом видео, `MEDIA_CHUNK_SECONDS=10`, модель `qwen3.8-27b-fable-distill` в LM Studio, транскрибатор `192.168.0.44:8000`.

1. `run 2026-10-15 --summary no`: inspect → convert_to_mp4 (WebM 24 с = MP4 24 с, Duration check: OK) → extract_mp3 → split_audio (3 чанка) → transcribe (3 из 3 с первой попытки) → write_summary_prompt. 12 вызовов, 1 мин 50 с.
2. `run 2026-10-15 --summary yes` на том же каталоге: inspect сразу отдал `ask_summary` (транскрипт переиспользован) → list_llm_models (9 чат-моделей, загружена одна — `obvious`) → summarize (39 с) → отчёт с моделью.

## Сохранённые диалоги

`media/runs/<UTC-время>-<каталог записи>/` (`MEDIA_RUNS_DIR`, в `.gitignore`). Один диалог — один каталог, файлы переписываются после каждой реплики. Хранятся последние `MEDIA_RUNS_KEEP` (50).

| Файл | Что внутри |
|---|---|
| `dialog.md` | Реплики и ответы, замечания (предел шагов, сбой модели) |
| `tools.json` | Все вызовы: доводы и результаты; у заданий — итог задания |
| `run.json` | Каталог записи, модель, температура, реплики, шаги, токены, задания (инструмент, состояние, длительность, ошибка) |

## MCP-сервер `media` (`mcp/media-mcp`)

По умолчанию работает через stdio: ассистент сам запускает `cargo run -q -p media-mcp` (`media/mcp.json`, `timeout_sec: 900`). Корень записей — `MEDIA_ROOT`, каталоги записей указываются от него, выйти за корень нельзя. С `--http` сервер слушает Streamable HTTP на `127.0.0.1:8098/mcp` (`MEDIA_MCP_ADDR`); тогда задания общие для всех клиентов.

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `list_recordings` | `dir?`, `depth?` (3) | `root`, `recordings[]` {dir, stage: source/mp4/mp3/chunks/transcript/summary} |
| `inspect` | `dir?`, `source?` (выбор при `choose_source`) | `transcript`, `chunks` {state, files, problems}, `mp3`, `mp4` {state: valid/mismatch/unverified/unreadable/ambiguous/busy/…, длительности}, `sources` {webm, mp4}, `summary`, `summary_prompt`, `job`, `next` {step, tool, args, why, choices} |
| `probe` | `dir?`, `file` | `duration`, `seconds`, `clock` |
| `convert_to_mp4` | `dir?`, `source?` | `job`. Итог: `webm_seconds`, `mp4_seconds`, `duration_check`, `moved_to_diagnostics` |
| `extract_mp3` | `dir?`, `source?`, `webm?` | `job`. Итог: `source`, `created` |
| `split_audio` | `dir?` | `job`. Итог: `chunks`, `segment_seconds`, `moved_to_diagnostics` |
| `transcribe` | `dir?` | `job`. Итог: `chunks[]` {chunk, txt, ok, chars, attempts, error}, `created`, `transcript_bytes` |
| `list_llm_models` | — | `models`, `chat_models`, `loaded_chat_models`, `obvious` |
| `summarize` | `dir?`, `model?`, `force?` | `job`. Итог: `created`, `model`, `chars` |
| `write_summary_prompt` | `dir?` | `created`, `chars`, `lm_studio_called: false` |
| `telegram_summary` | `dir?` | `created[]`, `paste` {file, use}, `bot` {file, use}, `utf16_len`, `fits_one_message`, `library`, `sent: false` |
| `wait_job` | `id`, `seconds?` (60, до 600) | задание: `state` running/succeeded/failed, `progress`, `elapsed_sec`, `result`/`error`. **Модели ассистента не выдаётся**: ждёт код |
| `job_status` | `id?` | задание или `jobs[]`. Модели ассистента тоже не выдаётся |

## Устройство

| Где | Что |
|---|---|
| `mcp/media-mcp/src/recording.rs` | Песочница корня, состояния файлов, чанки, исходники, сверка `output.mp4`, выбор следующего шага, `diagnostics/` |
| `mcp/media-mcp/src/jobs.rs` | Задания: одно на каталог, ход, ожидание завершения, паника не оставляет задание «идущим» |
| `mcp/media-mcp/src/media.rs` | `ffprobe`, `ffmpeg` с `-progress` и последними строками stderr при ошибке |
| `mcp/media-mcp/src/steps.rs` | Шаги: проверка очереди, задания конвертации, извлечения, нарезки, транскрипции, саммари |
| `mcp/media-mcp/src/transcribe.rs`, `summary.rs` | Транскрибатор (multipart, 2 параллельно, повторы) и LM Studio (модели, шаблон, ответ) |
| `mcp/media-mcp/src/telegram.rs`, `telegram.py` | Саммари для Telegram через `telegramify-markdown`: версия для вставки (из сущностей `convert`) и MarkdownV2; `media/requirements.txt` — версии библиотек |
| `media/assistant/src/dialog.rs` | Диалог: промпт, цикл «модель → инструменты», ожидание заданий кодом, подстановка `dir` |
| `media/assistant/src/session.rs` | Модель и MCP. Инструменты модели — без `wait_job` и `job_status` |
| `media/assistant/src/runs.rs`, `tui.rs`, `main.rs` | Сохранение диалогов, TUI и CLI |

## Настройки

| Переменная | По умолчанию | Что |
|---|---|---|
| `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL` | — | Модель ассистента, общая для проекта. Нужна модель, которая умеет вызывать инструменты |
| `MEDIA_MCP_CONFIG` / `MEDIA_SERVER` | `media/mcp.json` / `media` | Конфигурация MCP ассистента и имя сервера в ней |
| `MEDIA_DIR` | `.` | Каталог записи при запуске (от `MEDIA_ROOT`) |
| `MEDIA_MAX_ROUNDS` | `25` | Кругов «модель → инструменты» на реплику; ожидание задания — внутри одного круга |
| `MEDIA_TEMPERATURE` | `0` | Температура |
| `MEDIA_LLM_TIMEOUT` | `180` | Сколько секунд ждать ответа модели ассистента на один запрос |
| `MEDIA_RUNS_DIR` / `MEDIA_RUNS_KEEP` | `media/runs` / `50` | Куда сохранять диалоги и сколько хранить |
| `MEDIA_ROOT` | рабочая директория | Корень записей (сервер) |
| `MEDIA_MCP_ADDR` | `127.0.0.1:8098` | Адрес сервера с `--http` |
| `MEDIA_TRANSCRIBE_URL` | `http://192.168.0.44:8000` | Сервис распознавания (`/v1/transcribe`) |
| `MEDIA_LANGUAGE` | `ru` | Язык распознавания |
| `MEDIA_TRANSCRIBE_PARALLEL` | `2` | Одновременных запросов на распознавание (1–2) |
| `MEDIA_TRANSCRIBE_MAX_TIME` | `1800` | Секунд на ответ по одному чанку |
| `MEDIA_CHUNK_SECONDS` | `300` | Длина чанка |
| `MEDIA_LMSTUDIO_URL` | `http://192.168.0.44:1234` | LM Studio для саммари |
| `MEDIA_SUMMARY_TIMEOUT` | `7200` | Секунд на ответ LM Studio — без короткого предела |
| `MEDIA_SUMMARY_TEMPLATE` | встроенный | Свой шаблон промпта саммари (с `<TRANSCRIPT>`) |
| `MEDIA_FFMPEG` / `MEDIA_FFPROBE` | `ffmpeg` / `ffprobe` | Пути к программам |
| `MEDIA_PYTHON` | `media/.venv/bin/python`, иначе `python3` | Python с `telegramify-markdown` для `telegram_summary` |

## Проверка

```bash
cargo test -p media-mcp                         # чекпоинты, задания, вся цепочка на настоящем ffmpeg,
                                                #   поддельные транскрибатор и LM Studio
cargo test -p media-assistant                   # подстановка dir, ожидание заданий, диалоги, TUI
cargo test -p media-assistant -- --ignored      # живой media-mcp по stdio через McpManager
MEDIA_ROOT=~/Recordings cargo run -p media-assistant -- run 2026-10-15 --summary no
```

Тесты с `ffmpeg` сами генерируют короткие WebM (`lavfi`). Если `ffmpeg` не установлен, они пропускаются с сообщением. Так же пропускается тест Telegram-версии, если нет Python с `telegramify-markdown` (`MEDIA_PYTHON` или `media/.venv`).

Прогон Telegram-версии на реальных сервисах: `run 2026-10-15 --summary yes` после удаления старого саммари — summarize (38 с, Markdown по новому шаблону) → inspect назвал `telegram_summary` → `output-summary-telegram.md`, 1286 единиц UTF-16, одно сообщение. Вставленный руками в приложение, MarkdownV2 показывался с обратными слэшами. Отсюда отдельная версия для вставки, а MarkdownV2 переехал в `output-summary-telegram-bot.md`.
