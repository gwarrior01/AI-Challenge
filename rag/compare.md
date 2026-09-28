# Сравнение стратегий chunking

Сформировано 2026-09-28 15:23 UTC · модель эмбеддингов `text-embedding-bge-m3` (размерность 1024) · документов в индексе: 38 (из них с версией > 1: 0).

## Стратегии

- **fixed** — окно 1200 символов, перекрытие 200.
- **structure** — разделы по заголовкам / элементам Rust; короче 300 символов — слияние, длиннее 2000 — деление по абзацам.
- **sentence** — окно 5 предложений, шаг 3 (в коде — 16 строк, шаг 12), не длиннее 1500 символов.

## Чанки

| стратегия | чанков | символов min/медиана/p95/max | токенов на чанк | текст: обрыв фразы | текст: разрез ``` | код: разрез элемента | с разделом | эмбеддингов / время |
|---|---|---|---|---|---|---|---|---|
| fixed | 867 | 6 / 1167 / 1196 / 1199 | ≈285 | 52% | 14% | 60% | 94% | 867 / 30.5 с |
| structure | 965 | 6 / 661 / 1945 / 1999 | ≈214 | 8% | 2% | 21% | 96% | 958 / 29.1 с |
| sentence | 1529 | 6 / 735 / 1104 / 1496 | ≈184 | 7% | 2% | 62% | 96% | 1504 / 46.2 с |

Доли — среди чанков текста (Markdown, PDF) и кода Rust отдельно. «Обрыв фразы» — чанк текста кончается не на конце предложения, абзаца или блока. «Разрез ```» — в чанке нечётное число ```` ``` ````. «Разрез элемента» — в чанке кода не сходятся `{` и `}`. «Эмбеддингов / время» — последняя полная сборка стратегии, с учётом кэша векторов.

## Поиск: 14 контрольных вопросов

Чанк считается найденным, если содержит фрагмент ответа целиком (из любого документа — тот же факт в doc-комментарии кода тоже ответ). Крупный чанк чаще вмещает фрагмент целиком, мелкий — точнее по смыслу: смотрите вместе с размерами выше. Вопросы — `rag/eval.json`, «где ответ» — документ, по которому вопрос составлен.

| стратегия | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|
| fixed | 6/14 (43%) | 7/14 (50%) | 7/14 (50%) | 0.45 |
| structure | 3/14 (21%) | 8/14 (57%) | 8/14 (57%) | 0.37 |
| sentence | 4/14 (29%) | 6/14 (43%) | 7/14 (50%) | 0.35 |

| # | вопрос | где ответ | fixed | structure | sentence |
|---|---|---|---|---|---|
| 1 | Что происходит с плановым запуском агента, если агент занят другим запросом или у него активная задача? | `README.md` | ✗ | ✗ | ✗ |
| 2 | На каком адресе по умолчанию слушает MCP-сервер планировщика? | `README.md` | ✓1 | ✓1 | ✓1 |
| 3 | Почему у сервера documents в конфигурации MCP такой большой таймаут? | `README.md` | ✗ | ✗ | ✗ |
| 4 | Какие представления jfr view профилировщик Java возвращает по умолчанию? | `README.md` | ✗ | ✗ | ✗ |
| 5 | Как documents-mcp делает краткое содержание длинного документа? | `README.md` | ✓1 | ✓3 | ✓3 |
| 6 | Выполняются ли инструменты MCP, пока задача на этапе планирования? | `README.md` | ✗ | ✗ | ✗ |
| 7 | Сколько последних сообщений держит агент со стратегией sliding-window, если размер окна не задан? | `core/src/context.rs` | ✓1 | ✓2 | ✗ |
| 8 | Какой минимальный интервал у планового запуска агента? | `core/src/automation.rs` | ✗ | ✗ | ✓4 |
| 9 | Сколько ждать ответа инструмента MCP, прежде чем вызов считается неудачным? | `core/src/mcp.rs` | ✗ | ✗ | ✗ |
| 10 | Что будет, если не задать адрес LLM API в переменных окружения? | `core/src/lib.rs` | ✓1 | ✓2 | ✓3 |
| 11 | Как при конвертации PDF в Markdown определяется, что строка — заголовок? | `core/src/pdf.rs` | ✓3 | ✓1 | ✓1 |
| 12 | Насколько сократилось время разбора инцидента после перехода на MCP-инструменты? | `mcp/documents-mcp/demo/mcp-report.pdf` | ✓1 | ✓2 | ✓1 |
| 13 | Какую базу данных требует инвариант хранения? | `invariants/storage.md` | ✗ | ✓3 | ✗ |
| 14 | Насколько качество среды влияет на успешность одной и той же модели? | `documents/inbox/new.pdf` | ✓1 | ✓1 | ✓1 |

## Где стратегии разошлись

### 5. Как documents-mcp делает краткое содержание длинного документа?

Ответ: «~12 000-character chunks» в `README.md`.

- **fixed** (✓1): топ-1 — `README.md` · LLM Agent in Rust › MCP: external tools for agents › Own MCP server: documents (tool composition) · строки 290–292 · score 0.677
  > the gist in one sentence, then key points — facts, figures, decisions, deadlines — only from the text. A long document is split at section and paragraph boundar…
- **structure** (✓3): топ-1 — `README.md` · LLM Agent in Rust › MCP: external tools for agents › Own MCP server: documents (tool composition) · строки 270–284 · score 0.701
  > #### Own MCP server: documents (tool composition) `mcp/documents-mcp/` is the third own server: three separate tools that form one chain — the first gets the da…
- **sentence** (✓3): топ-1 — `README.md` · LLM Agent in Rust › MCP: external tools for agents › Own MCP server: documents (tool composition) · строки 270–278 · score 0.674
  > #### Own MCP server: documents (tool composition) `mcp/documents-mcp/` is the third own server: three separate tools that form one chain — the first gets the da…

### 7. Сколько последних сообщений держит агент со стратегией sliding-window, если размер окна не задан?

Ответ: «DEFAULT_SLIDING_WINDOW_SIZE: usize = 6» в `core/src/context.rs`.

- **fixed** (✓1): топ-1 — `core/src/context.rs` · fn context_summary_chunk · строки 70–94 · score 0.672
  > } /// Имя переменной окружения, которой можно переопределить [`sliding_window_size`]. const SLIDING_WINDOW_ENV: &str = "LLM_SLIDING_WINDOW_SIZE"; /// Значение […
- **structure** (✓2): топ-1 — `core/src/agent.rs` · impl Agent › fn sliding_window_status · строки 1927–1943 · score 0.668
  > /// Считает [`SlidingWindowInfo`] — `None`, если активна не `SlidingWindow`/`Facts`. fn sliding_window_status(&self, config: &AgentConfig) -> Option<SlidingWind…
- **sentence** (✗): топ-1 — `README.md` · LLM Agent in Rust › Environment variables · строки 439–440 · score 0.710
  > - `LLM_SLIDING_WINDOW_SIZE` — optional. How many of the most recent user messages (exchanges) named agents keep when their `context_strategy` is `sliding-window…

### 8. Какой минимальный интервал у планового запуска агента?

Ответ: «MIN_INTERVAL: Duration = Duration::from_secs(30)» в `core/src/automation.rs`.

- **fixed** (✗): топ-1 — `core/src/automation.rs` · без раздела · строки 1–19 · score 0.648
  > //! Плановые запуски агентов — агент «на дежурстве», который работает сам, //! без реплик человека. //! //! Плановый запуск ([`ScheduledRun`]) — это агент, пром…
- **structure** (✗): топ-1 — `core/src/agent.rs` · fn scheduled_run_for · строки 3677–3709 · score 0.697
  > /// Плановый запуск агента `agent` каждые `every` (ещё не сохранённый, `id` — 0). /// Без `prompt` — сводка планировщика ([`crate::automation::DEFAULT_PROMPT`])…
- **sentence** (✓4): топ-1 — `core/src/agent.rs` · fn register_agent · строки 3673–3689 · score 0.707
  > agents.insert(name, agent.clone()); Ok(agent) } /// Плановый запуск агента `agent` каждые `every` (ещё не сохранённый, `id` — 0). /// Без `prompt` — сводка план…

### 10. Что будет, если не задать адрес LLM API в переменных окружения?

Ответ: «не задана переменная окружения LLM_API_URL» в `core/src/lib.rs`.

- **fixed** (✓1): топ-1 — `core/src/lib.rs` · const DEFAULT_CONTEXT_WINDOW · строки 318–344 · score 0.652
  > провайдеров /// (OpenAI, Ollama, LM Studio, OpenRouter...) нет единого способа узнать его программно — /// поэтому точное значение для своей модели нужно указыв…
- **structure** (✓2): топ-1 — `README.md` · LLM Agent in Rust › Environment variables · строки 421–429 · score 0.635
  > ## Environment variables Required: - `LLM_API_URL` — base URL of the LLM API, without a trailing `/chat/completions`. Examples: `https://api.openai.com/v1`, `ht…
- **sentence** (✓3): топ-1 — `core/src/rag/embed.rs` · impl Embedder › fn from_env · строки 59–75 · score 0.694
  > "не задана переменная окружения LLM_EMBEDDING_MODEL — модель эмбеддингов (например text-embedding-bge-m3)", )?; let base_url = env("LLM_EMBEDDING_API_URL").or_e…
