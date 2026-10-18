# API

## HTTP API веб-интерфейса (`llm-web`, порт 8080)

Все ответы — JSON; ошибка — `{"error": "…"}` с кодом 200 (страница проверяет поле `error`). Маршруты — `web/src/main.rs` (функция `main`) и `web/src/rag.rs` (`routes`).

### Чат и ассистент разработчика

| Метод и путь | Тело / параметры | Что делает |
|---|---|---|
| `GET /` | — | Страница `web/src/index.html` |
| `POST /api/ask` | `{prompt, history, json_schema?, max_tokens?, stop?, temperature?, top_p?, reasoning?, analysis?}` | Прямой запрос к модели (вкладка «Чат»); ответ `{answer, usage, requestJson, responseJson, context_window}` |
| `POST /api/summarize` | `{previous_summary, messages}` | Сводка ранней части диалога для чата без сервера-состояния |
| `POST /api/help` | `{question}` | Ассистент разработчика (`/help`): `{question, answer, context[], git, git_note, index_note, rag, tool_calls[], usage, requestJson, responseJson}` |

### Агенты

| Метод и путь | Что делает |
|---|---|
| `GET /api/agents`, `POST /api/agents` | Список агентов; создать агента (тело — поля `AgentConfig`) |
| `DELETE /api/agents/:name` | Удалить агента вместе с историей |
| `POST /api/agents/:name/start`, `/stop` | Запустить / остановить |
| `POST /api/agents/:name/ask` | Вопрос агенту `{prompt, continuation?}` → ответ, `usage`, `tool_calls`, `rag`, `task`, признаки сжатия контекста |
| `GET /api/agents/:name/tool-calls` | Вызовы инструментов текущего обмена — опрашивается, пока идёт `/ask` |
| `GET /api/agents/:name/history` | История диалога с токенами |
| `POST /api/agents/:name/strategy` | Сменить стратегию контекста |
| `POST /api/agents/:name/profile` | Сменить профиль персонализации |
| `POST /api/agents/:name/rag` | RAG-режим: `{enabled, strategy?, k?, min_score?, candidates?, rewrite?, rerank?, rerank_min?, quotes?}` |
| `POST /api/agents/:name/task-mode` | Режим этапов задачи вкл/выкл |
| `POST /api/agents/:name/track` | Агент сам ведёт цель задачи `{enabled}` |
| `POST /api/agents/:name/checkpoint`, `/branch`, `/switch` | Ветвление диалога (стратегия `branching`) |
| `POST /api/agents/:name/remember`, `/forget` | Долговременная память |

### Задачи (рабочая память и конечный автомат)

| Метод и путь | Что делает |
|---|---|
| `GET /api/agents/:name/task` | Состояние задачи агента |
| `POST /api/agents/:name/task/ensure` | Завести задачу, если её нет (перед первым вопросом) |
| `POST …/task/start`, `/join`, `/set`, `/finish` | Создать, присоединиться, записать данные, завершить для всех |
| `POST …/task/advance`, `/step`, `/expect` | Сменить этап, текущий шаг, ожидаемое действие |
| `POST …/task/pause`, `/resume` | Пауза задачи |
| `POST …/task/approve`, `/reject` | Утвердить / отклонить предложенный моделью переход |
| `POST …/task/invariant`, `DELETE …/task/invariant/:id` | Инварианты задачи |
| `POST …/task/forbid`, `/allow`, `/require-approval`, `/unrequire-approval` | Правила переходов |
| `GET /api/tasks` | Все общие задачи и их участники |

### Профили, инварианты, расписание, MCP

| Метод и путь | Что делает |
|---|---|
| `GET /api/profiles`, `POST /api/profiles` | Список профилей; создать файл профиля |
| `GET /api/invariants`, `POST /api/invariants`, `DELETE /api/invariants/:id` | Глобальные инварианты (файлы в `invariants/`) |
| `GET /api/agents/:name/schedule`, `POST /api/agents/:name/schedule` | Плановые запуски агента |
| `POST /api/agents/:name/schedule/:id/:action` | `enable`, `disable`, `run` (запустить сейчас), `delete` |
| `GET /api/schedule` | Все плановые запуски |
| `GET /api/activity?after=N` | Что агенты сказали сами (плановые запуски, события MCP) после номера `N` |
| `GET /api/scheduler/jobs`, `POST /api/scheduler/jobs/:id/:action` | Задания MCP-планировщика |
| `GET /api/mcp` | Серверы из `mcp.json`: статус, инструменты, схемы параметров |
| `POST /api/mcp/reload` | Перечитать `mcp.json` и переподключиться |
| `POST /api/mcp/:name/enable`, `/disable`, `/reconnect` | Управление сервером (флаг пишется в `mcp.json`) |
| `POST /api/mcp/:name/tools/:tool` | Вызвать инструмент вручную `{arguments: {…}}` |

### RAG (`web/src/rag.rs`)

| Метод и путь | Что делает |
|---|---|
| `GET /api/rag/overview` | Сводка индекса: модель, размерность, документы, чанки по стратегиям |
| `GET /api/rag/docs`, `DELETE /api/rag/docs?source=…` | Документы индекса; удалить документ |
| `GET /api/rag/versions`, `/chunks`, `/chunk`, `/map` | Версии документа, чанки, один чанк с вектором, карта документа |
| `POST /api/rag/search` | Поиск по всем стратегиям |
| `POST /api/rag/upload?name=…&strategy=…&<параметры>` | Загрузить файл (тело — сам файл, до 50 МБ) и векторизовать в фоне |
| `POST /api/rag/index` | Проиндексировать корпус в фоне |
| `POST /api/rag/rechunk` | Нарезать версию ещё одной стратегией |
| `GET /api/rag/job` | Ход фоновой индексации |

## Инструменты собственных MCP-серверов

Модели инструмент виден как `mcp__<сервер>__<инструмент>`.

### `git` — `mcp/git-mcp` (stdio; `--http` — `127.0.0.1:8094/mcp`)

Только чтение; пути — относительно корня репозитория.

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `git_branch` | — | `branch` (или `null` при detached HEAD), `head`, `upstream`, `ahead`/`behind`, `last_commit` {hash, subject, author, date}, `changed_files`, `local_branches` |
| `git_status` | — | `branch`, счётчики `staged`/`unstaged`/`untracked`/`conflicts`, `clean`, `files[]` {path, from?, index, worktree, state} |
| `list_files` | `path?`, `pattern?` (`*.rs`), `depth?`, `untracked?`, `limit?` (300) | `total`, `truncated`, `files[]` — или с `depth` `tree[]` {path, files} |
| `git_diff` | `staged?`, `base?` (`main`), `path?`, `stat_only?`, `max_chars?` (15000) | `compared`, `files[]` {path, added, deleted}, `added`, `deleted`, `patch`, `truncated`, `untracked[]` |
| `git_log` | `n?` (10), `path?` | `commits[]` {hash, subject, author, date} |

### `fs` — `mcp/fs-mcp` (stdio; `--http` — `127.0.0.1:8097/mcp`)

Файлы проекта; пути — относительно корня. Правки копятся в оверлее и попадают на диск только через `apply_changes`, который вызывает код ассистента (модели он не выдаётся). Подробно — [day-34-file-assistant.md](day-34-file-assistant.md).

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `list_files` | `path?`, `pattern?` (`*.rs`, `docs/**/*.md`), `depth?`, `limit?` (300) | `total`, `truncated`, `files[]` — или с `depth` `tree[]` {path, files} |
| `read_file` | `path`, `start_line?`, `max_lines?` (400) | `path`, `start_line`, `end_line`, `total_lines`, `text`, `truncated`, `pending` |
| `search` | `query`, `regex?`, `case_sensitive?`, `path?`, `pattern?`, `context?`, `max_results?` (50) | `total_matches`, `files[]` {path, matches}, `matches[]` {path, line, text, before, after}, `truncated`, `files_scanned` |
| `write_file` | `path`, `content` | `path`, `created`, `lines`, `added`, `deleted`, `diff` |
| `edit_file` | `path`, `old_text`, `new_text`, `replace_all?` | то же + `replacements` |
| `pending_changes` | — | `files[]` {path, status, added, deleted}, `added`, `deleted`, `patch` |
| `discard_changes` | `paths?` | `discarded[]` |
| `apply_changes` | `paths?` | `written[]`, `conflicts[]` {path, reason} |

### `media` — `mcp/media-mcp` (stdio; `--http` — `127.0.0.1:8098/mcp`)

Записи встреч; каталоги — от корня `MEDIA_ROOT`. `inspect` называет следующий шаг, остальные шаги вне очереди отказывают; долгие шаги возвращают задание (`job`), итог — `wait_job`. Подробно — [day-35-media-assistant.md](day-35-media-assistant.md).

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `list_recordings` | `dir?`, `depth?` (3) | `root`, `recordings[]` {dir, stage} |
| `inspect` | `dir?`, `source?` | чекпоинты (`transcript`, `chunks`, `mp3`, `mp4`, `sources`, `summary`, `summary_prompt`), `job`, `next` {step, tool, args, why, choices} |
| `probe` | `dir?`, `file` | `duration`, `seconds`, `clock` |
| `convert_to_mp4` / `extract_mp3` / `split_audio` / `transcribe` | `dir?` (+ `source?`, `webm?`) | `job` {id, kind, dir, state, progress, elapsed_sec} |
| `list_llm_models` | — | `models`, `chat_models`, `loaded_chat_models`, `obvious` |
| `summarize` | `dir?`, `model?`, `force?` | `job` |
| `write_summary_prompt` | `dir?` | `created`, `chars`, `lm_studio_called` |
| `wait_job` | `id`, `seconds?` (60, до 600) | задание: `state`, `progress`, `result` / `error` |
| `job_status` | `id?` | задание или `jobs[]` |

### `java-profiler` — `mcp/java-profiler-mcp` (`127.0.0.1:8091/mcp`)

`java_list_processes`, `java_process_info(pid)`, `java_thread_dump(pid, max_threads, include_system)`, `java_heap_histogram(pid, top)`, `java_profile(pid, duration_sec, views)` — всё только чтение, через `jps`/`jcmd`/`jfr`.

### `scheduler` — `mcp/scheduler-mcp` (`127.0.0.1:8092/mcp`)

`schedule_job(action, params, every?/in?/at?, name?, extract?)`, `schedule_reminder(text, in/at, every?)`, `list_jobs`, `cancel_job`, `pause_job`, `resume_job`, `get_runs`, `get_summary(job?, since?)`, `get_events`, `ack_events`. Действия: `reminder`, `http`, `mcp_tool`.

### `documents` — `mcp/documents-mcp` (`127.0.0.1:8093/mcp`)

`list_pdfs`, `pdf_to_markdown(path)` → `document_id`, `save_markdown(name, markdown)` → `document_id`, `summarize_markdown(document_id, max_words?, focus?)` → `summary_id`, `save_summary(summary_id, file_name?)` → путь к файлу.

## `llm-gateway` (порт 8080 на VPS)

| Метод и путь | Что делает |
|---|---|
| `POST /v1/chat/completions` | OpenAI-совместимый чат (`stream: true` — SSE), ключ `Authorization: Bearer <ключ пользователя>` |
| `GET /v1/models` | Единственная модель шлюза |
| `GET /health` | Проверка без ключа |
| `GET /stats` | Счётчики по пользователям (админ видит всех) |
| `GET /` | Минимальная страница чата |

Ошибки — в формате OpenAI: `400 context_length_exceeded`, `429 rate_limit_exceeded` (с `Retry-After`), `503 server_busy`.
