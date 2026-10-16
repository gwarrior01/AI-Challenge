# Архитектура

Проект — Cargo workspace на Rust: LLM-агент с тремя интерфейсами (TUI, веб, CLI), общим ядром, собственными MCP-серверами и приватным шлюзом к локальной модели. Модель вызывается по **OpenAI-совместимому** HTTP API (`/chat/completions`), адрес и ключ — из переменных окружения (`LLM_API_URL`, `LLM_API_KEY`).

## Крейты

| Каталог | Крейт | Что это |
|---|---|---|
| `core/` | `llm-core` | Общее ядро: клиент LLM, агенты, память, контекст, MCP-клиент, RAG, ассистент разработчика. Все интерфейсы — тонкие оболочки над ним |
| `tui/` | `llm-tui` | Терминальный интерфейс (ratatui + crossterm) — **основной интерфейс** |
| `web/` | `llm-web` | Веб-интерфейс (axum), порт 8080; страница — один файл `web/src/index.html`, вкладка RAG — `web/src/rag.rs` |
| `cli/` | `llm-cli` | Консольный интерфейс: разовый запрос, REPL, подкоманды `agent`, `mcp`, `rag`, `jobs`, `help` |
| `gateway/` | `llm-gateway` | Приватный AI-сервис: OpenAI-совместимый шлюз с ключами и лимитами перед llama.cpp на VPS |
| `mcp/java-profiler-mcp/` | `java-profiler-mcp` | MCP-сервер: профилирование JVM на этой машине (`jps`, `jcmd`, `jfr`), Streamable HTTP, порт 8091 |
| `mcp/scheduler-mcp/` | `scheduler-mcp` | MCP-сервер: отложенные и периодические задания, хранение в SQLite, сводки; порт 8092 |
| `mcp/documents-mcp/` | `documents-mcp` | MCP-сервер: PDF → Markdown → краткое содержание → файл, три отдельных инструмента; порт 8093 |
| `mcp/git-mcp/` | `git-mcp` | MCP-сервер: состояние git-репозитория (ветка, статус, файлы, дифф, журнал), только чтение; stdio (или HTTP 8094) |
| `mcp/fs-mcp/` | `fs-mcp` | MCP-сервер: файлы проекта — список, поиск, чтение, правки через оверлей с диффом, запись на диск по подтверждению; stdio (или HTTP 8097) |
| `assistant-tui/` | `assistant-tui` | Общие части TUI ассистентов: лента с Markdown и плашками MCP (режимы показа вызовов: кратко/отладка/скрыты), прокрутка колесом мыши, строка ввода с историей |
| `files/assistant/` | `file-assistant` | Ассистент файлов: цель словами → поиск, чтение, правки через `fs-mcp` → отчёт, дифф, `files/runs/` |

Приоритет интерфейсов: **TUI** — главный, затем **веб**, затем **CLI**. Тексты, которые формирует `core` (ошибки, результаты инструментов для модели, блоки промптов, отказы), не называют команды конкретного интерфейса — подсказки вида «нажмите F4» живут в крейте интерфейса.

## Модули `llm-core` (`core/src/`)

| Файл | Что делает |
|---|---|
| `lib.rs` | `LlmClient` — HTTP-клиент `/chat/completions` (`chat`, `chat_with_options`, `chat_with_model`, `chat_with_tools`), `ChatMessage`, `Usage`, `ToolDefinition`, `RequestMessage` |
| `agent.rs` | Именованные агенты: `Agent`, `AgentConfig`, `AgentManager` (реестр, SQLite `agents.db`), `handle_request` — обработка вопроса, цикл вызова инструментов `run_tool_loop`, конечный автомат задачи (`move_stage`/`update_step`), `ToolCallRecord`, `CallsView` (режимы показа вызовов инструментов) |
| `context.rs` | Стратегии управления контекстом: `full`, `summary`, `sliding-window`, `facts`, `branching` |
| `memory.rs` | Модель памяти: краткосрочная (диалог), рабочая (общая задача `TaskState`, этапы `Stage`), долговременная (`LongTermMemory`) |
| `profile.rs` | Профили персонализации — markdown-файлы в `profiles/` |
| `invariants.rs` | Жёсткие инварианты — markdown-файлы в `invariants/`, которые ассистент не нарушает |
| `task_tracking.rs` | Агент сам ведёт цель и заметки задачи (`track_task`) |
| `automation.rs` | Плановые запуски агентов («на дежурстве») и доставка событий MCP-серверов |
| `pricing.rs` | Стоимость запросов: от провайдера (`usage.cost`) или оценка по ставкам |
| `mcp.rs` | MCP-клиент на `rmcp`: `McpManager` читает `mcp.json`, подключается (stdio или Streamable HTTP), отдаёт инструменты как `mcp__<сервер>__<инструмент>` и вызывает их |
| `pdf.rs` | PDF → Markdown (общий для `documents-mcp` и RAG) |
| `rag/` | Индекс документов для RAG — см. ниже |
| `devhelp.rs` | Ассистент разработчика: `/help` — ответ о проекте по README и `docs/` (RAG) с контекстом git (MCP) |

### `core/src/rag/`

| Файл | Что делает |
|---|---|
| `mod.rs` | `RagConfig`, индексация (`index`, `index_file`), поиск (`search`, `search_queries`), сравнение стратегий (`compare`), команды `run_command` для TUI/CLI |
| `corpus.rs` | Какие файлы индексировать (`.md`, `.txt`, `.rs`, `.pdf`), их текст, SHA-256 и git-коммит |
| `chunking.rs` | Стратегии нарезки: `fixed`, `structure`, `sentence`, `parent`; параметры `ChunkParams` |
| `embed.rs` | Клиент `/embeddings` (модель — `LLM_EMBEDDING_MODEL`), нормировка векторов |
| `store.rs` | SQLite-индекс: документы, версии, чанки, векторы |
| `retrieve.rs` | RAG-режим: вопрос → поиск → порог → реранкинг → фрагменты для модели (`retrieve`, `retrieve_in`, `RagContext`) |
| `rerank.rs` | Переписывание запроса и реранкинг (эвристика по терминам или оценка моделью) |
| `answer.rs` | Правила ответа, проверка ссылок и цитат, блок «Источники» |
| `browse.rs` | Просмотр индекса для веб-вкладки RAG |

## Путь вопроса к агенту

```text
интерфейс (TUI / веб / CLI)
  → Agent::handle_request(вопрос)
      → [RAG] retrieve: фрагменты документов, если у агента включён rag
      → сборка сообщений: системный промпт + профиль + инварианты + рабочая/долговременная память
                          + история по стратегии контекста + вопрос (с фрагментами)
      → run_tool_loop: LlmClient::chat_with_tools с инструментами MCP (+ автомат задачи)
           ↺ вызовы инструментов → McpManager::call → результат модели (до предела кругов)
      → ответ + ToolCallRecord + RagContext + токены/стоимость → история в agents.db
  → интерфейс показывает строки RAG, вызовы инструментов и ответ
```

Без активной задачи инструменты MCP выполняются сразу. С активной задачей — только на этапах `execution` и `validation`; на `planning` они предлагаются модели, но не выполняются (модель вносит их в план, человек его утверждает).

## Путь `/help`

```text
/help <вопрос>
  → devhelp::ask
      → MCP git: git_branch (кодом, всегда)
      → индекс документации rag/project.db: сверка README.md + docs/ с файлами → retrieve_in
      → модель с инструментами git (list_files, git_status, git_diff, git_log)
      → ответ + «Источники»
```

Подробно — [developer-assistant.md](developer-assistant.md).

## Где что лежит

| Что | Где |
|---|---|
| Профили персонализации | `profiles/*.md` (`LLM_PROFILES_DIR`) |
| Инварианты | `invariants/*.md` (`LLM_INVARIANTS_DIR`) |
| Агенты, история, задачи, плановые запуски | `agents.db` (`AGENTS_STORE_PATH`), в git не хранится |
| Конфигурация MCP-серверов | `mcp.json` (`LLM_MCP_CONFIG`); образец — `mcp.example.json` |
| Общий RAG-индекс | `rag/index.db`, загрузки — `rag/uploads/` |
| Индекс документации для `/help` | `rag/project.db` (`DEVHELP_DB`) |
| Контрольные вопросы и отчёт сравнения стратегий | `rag/eval.json`, `rag/compare.md` |
| Данные планировщика | `scheduler.db` (`SCHEDULER_DB`) |
| PDF и результаты documents-mcp | `documents/inbox/`, `documents/out/`, `documents/.artifacts/` |
| Конфигурация шлюза | `gateway/gateway.toml` (образец — `gateway/gateway.example.toml`), развёртывание — `gateway/deploy/` |
