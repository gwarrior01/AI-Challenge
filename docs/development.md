# Разработка

## Сборка, тесты, запуск

```bash
cargo build                      # весь workspace
cargo test                       # все тесты (сети не требуют: LLM и эмбеддинги — фиктивные серверы в тестах)
cargo test -p llm-core devhelp   # тесты одного модуля
cargo test -p git-mcp            # MCP-сервер git, в т.ч. на временном настоящем репозитории

cargo run -p llm-tui             # TUI — основной интерфейс
cargo run -p llm-web             # веб, http://localhost:8080 (PORT)
cargo run -p llm-cli             # REPL; llm-cli agent … / mcp … / rag … / help …
```

Свои MCP-серверы: `cargo run -p java-profiler-mcp` (8091), `-p scheduler-mcp` (8092), `-p documents-mcp` (8093) — Streamable HTTP, запускаются отдельно. `git-mcp` — stdio: клиент запускает его сам по записи в `mcp.json` (`cargo run -q -p git-mcp`), отдельно запускать не нужно; `cargo run -p git-mcp -- --http` — HTTP на 8094.

Релизные бинарники — `cargo build --release`; шлюз для VPS собирается `cargo zigbuild --target x86_64-unknown-linux-musl -p llm-gateway` (см. `gateway/deploy/push.sh`).

## Переменные окружения

Шаблон — `.env.example`.

| Переменная | Для чего |
|---|---|
| `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-совместимый API модели (обязательны для обращений к модели) |
| `LLM_CONTEXT_WINDOW` | Размер контекстного окна для индикатора (по умолчанию 262000) |
| `LLM_ANALYSIS_MODEL` | Модель для анализа решений во вкладке «Задача» |
| `LLM_CONTEXT_SUMMARY_CHUNK`, `LLM_SLIDING_WINDOW_SIZE` | Параметры стратегий контекста |
| `AGENTS_STORE_PATH` | SQLite агентов (`agents.db`) |
| `LLM_PROFILES_DIR`, `LLM_INVARIANTS_DIR` | Каталоги профилей и инвариантов |
| `LLM_MCP_CONFIG` | Файл MCP-серверов (`mcp.json`) |
| `LLM_EMBEDDING_MODEL` (+ `_API_URL`, `_API_KEY`, `_BATCH`, `_DOC_PREFIX`, `_QUERY_PREFIX`) | Модель эмбеддингов для RAG и `/help`; без неё поиск по документам не работает |
| `RAG_CORPUS`, `RAG_DB_PATH`, `RAG_UPLOADS`, `RAG_EVAL`, `RAG_REPORT` | Общий RAG-индекс агентов |
| `DEVHELP_DOCS`, `DEVHELP_DB`, `DEVHELP_GIT_SERVER` | Ассистент разработчика: корпус (`README.md,docs`), индекс (`rag/project.db`), имя MCP-сервера git (`git`) |
| `GIT_MCP_REPO`, `GIT_MCP_ADDR` | Сервер git: репозиторий (по умолчанию рабочая директория), адрес для `--http` |
| `TUI_CALLS` | Начальный режим показа вызовов инструментов во всех TUI: `off`, `brief`, `debug` (по умолчанию `brief`) |
| `JAVA_PROFILER_MCP_ADDR`, `SCHEDULER_MCP_ADDR`, `SCHEDULER_DB`, `DOCUMENTS_MCP_ADDR`, `DOCUMENTS_DIR` | Адреса и данные своих MCP-серверов |

## Соглашения

- **Комментарии и тексты для пользователя — на русском**; идентификаторы — на английском. Doc-комментарии (`//!`, `///`) объясняют «зачем», а не пересказывают код.
- **Ядро не знает об интерфейсах.** Всё, что можно сделать в одном интерфейсе, делается через `llm-core` и доступно во всех; тексты ядра не называют клавиши и команды конкретного интерфейса. Приоритет проверки изменений: TUI, затем веб, затем CLI.
- **Решает код, а не промпт**, там, где правило можно проверить: «не знаю» ниже порога RAG, источники из данных поиска, этапы задачи и утверждение переходов, проверка цитат.
- **Ошибки — значения, а не паника**: недоступный сервер эмбеддингов, MCP-сервер или неверные доводы модели отражаются в ответе и в строке статуса, обмен не обрывается.
- **Конфигурация — в файлах и окружении, секреты — только в окружении** (`${VAR}` в `mcp.json`, `.env` в git не попадает).
- Тесты — рядом с кодом (`#[cfg(test)] mod tests`); MCP-клиент тестируется сервером в том же процессе (`mcp::test_support`), RAG — фиктивным сервером эмбеддингов (`rag::embed::test_support`).

## Как добавить свой MCP-сервер

1. Каталог `mcp/<имя>-mcp/` с `Cargo.toml` (зависимость `rmcp` с фичами `server`, `macros` и транспортом: `transport-io` для stdio, `transport-streamable-http-server` для HTTP) и добавить его в `members` корневого `Cargo.toml`.
2. Структура сервера с `#[tool_router]`: каждый инструмент — метод с `#[tool(name, description)]`, параметры — `Parameters<T>` с `T: Deserialize + JsonSchema`, результат — `Json<T>` (структурированный ответ). Описание инструмента — это и есть инструкция модели: что делает и когда его звать.
3. `#[tool_handler] impl ServerHandler` с `get_info`: имя, версия и `with_instructions` — как связывать инструменты между собой; клиент кладёт эти инструкции в системный промпт.
4. Запись в `mcp.example.json`: `command`/`args` для stdio или `url` для HTTP, `description`, при долгих вызовах — `timeout_sec`.
5. Строка в таблице крейтов в `docs/architecture.md` и инструменты в `docs/api.md` — тогда `/help` будет о нём знать.

Образцы: `mcp/git-mcp` (stdio и HTTP, только чтение), `mcp/documents-mcp` (цепочка инструментов, связанных идентификаторами), `mcp/scheduler-mcp` (фоновая работа и SQLite).

## Ветки и коммиты

Работа ведётся по дням в ветках `day-NN`, которые сливаются в `main`. Сообщение коммита — одна строка о результате («Private AI service: local LLM on a VPS behind an OpenAI-compatible gateway»).
