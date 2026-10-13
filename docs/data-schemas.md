# Схемы данных

Все базы — SQLite-файлы в рабочей директории (в git не хранятся). Схема создаётся кодом при открытии (`CREATE TABLE IF NOT EXISTS`), недостающие столбцы старых баз добавляются миграциями `ALTER TABLE` там же.

## `agents.db` — агенты, история, память, задачи

Путь — `AGENTS_STORE_PATH` (по умолчанию `agents.db`). Один файл на все интерфейсы: агент, созданный в TUI, виден в вебе и CLI. Код — `Db` в `core/src/agent.rs`.

| Таблица | Ключ | Что хранит |
|---|---|---|
| `agents` | `name` | `config` — JSON конфигурации агента (`AgentConfig`, ниже), `running` — запущен ли |
| `messages` | `id` | История диалога: `agent_name`, `role` (`user`/`assistant`/`system`), `content`, `created_at`, `branch` (ветка, по умолчанию `main`), токены `prompt_tokens`/`completion_tokens`/`total_tokens`, стоимость `cost`/`cost_input`/`cost_output` |
| `context_summaries` | `agent_name` | Сводка ранней части диалога (стратегия `summary`): `summary`, `summarized_count` |
| `agent_facts` | `agent_name` | Факты стратегии `facts`: `facts_json` |
| `agent_branch_state` | `agent_name` | Текущая ветка (стратегия `branching`) |
| `agent_checkpoints` | `agent_name, name` | Точки ветвления: ветка и число сообщений на момент отметки |
| `long_term_memory` | `key` | Долговременная память, общая для всех агентов: `category`, `value`, `updated_at` |
| `shared_tasks` | `name` | Общая задача (рабочая память): `goal`, этап `stage` (`planning`/`execution`/`validation`/`done`), `current_step`, `expected_action`, `paused`, ожидающий утверждения переход `pending_stage`/`pending_outcome`, `interrupted_prompt`, заметки `notes_json` (уточнения, ограничения, термины) |
| `shared_task_data` | `task_name, key` | Данные задачи «ключ — значение» (`task set`) |
| `shared_task_invariants` | `task_name, id` | Инварианты конкретной задачи |
| `shared_task_transition_rules` | `task_name, from_stage, to_stage, kind` | Правила переходов задачи: запрещён или требует утверждения |
| `shared_task_transitions` | `id` | Журнал переходов: откуда, куда, кто (`actor`), `outcome`, `reason`, когда |
| `agent_current_task` | `agent_name` | К какой задаче присоединён агент |
| `scheduled_runs` | `id` | Плановые запуски: агент, `name`, `prompt`, `interval_ms`, `enabled`, `next_run_at`, `last_run_at`, `last_status` |
| `runtime_leases` | `key` | Аренда фоновых работ, чтобы два процесса не выполняли один плановый запуск дважды |

### Конфигурация агента (`agents.config`, `AgentConfig`)

```json
{
  "name": "Аналитик",
  "system_prompt": "…",
  "model": null,
  "show_tokens": false,
  "max_tokens": null,
  "temperature": null,
  "top_p": null,
  "reasoning": null,
  "context_strategy": "full",
  "window_size": null,
  "profile": "default",
  "rag": { "strategy": "structure", "k": 5, "min_score": 0.5, "candidates": 20,
           "rewrite": false, "rerank": "none", "rerank_min": 0.0, "quotes": false },
  "task_mode": false,
  "track_task": false
}
```

`context_strategy` — `full`, `summary`, `sliding-window`, `facts`, `branching`. `rag: null` — агент отвечает без документов. `profile: "none"` — без персонализации.

## RAG-индекс — `rag/index.db` и `rag/project.db`

Одна схема (`core/src/rag/store.rs`) у двух файлов:

- `rag/index.db` (`RAG_DB_PATH`) — общий индекс агентов: загруженные документы (`rag/uploads/`) и пути из `RAG_CORPUS`;
- `rag/project.db` (`DEVHELP_DB`) — индекс документации проекта для `/help`: `README.md` и `docs/` (`DEVHELP_DOCS`), стратегия `structure`.

| Таблица | Что хранит |
|---|---|
| `documents` | `doc_id`, `source` — путь в корпусе (идентификатор документа), `kind` — `markdown`/`rust`/`text` |
| `versions` | Версии документа: `version` (1, 2, …), `title`, `sha256` файла, `git_commit`, `git_dirty`, `bytes`, `mtime`, `indexed_at`, `status` (`current`/`superseded`/`removed`), полный `text` |
| `chunks` | `chunk_id` (`README.md@v2/structure/0041`), `version_id`, `strategy`, `ord`, `section` (путь разделов через ` › `), строки и байты начала/конца, `char_len`, `tokens_est`, `content_sha256`, `vector_key`, `embedding_model`, флаги `clean_end`/`splits_code`, `text`; у `parent` — `context` (родительский абзац) с его байтами и строками |
| `chunkings` | Параметры нарезки версии стратегией (JSON) и когда |
| `vectors` | `vector_key` (sha256 модели + префикса + текста) → `vector` (BLOB little-endian f32, L2-нормирован), `model`, `dim` |
| `meta` | Модель и размерность индекса, время последней индексации, итоги запусков стратегий |

Изменился файл — новая версия N+1, прежняя становится `superseded` и уходит из поиска. Векторы хранятся по тексту чанка, поэтому эмбеддятся заново только изменившиеся куски. Индекс одной модели эмбеддингов не ищется другой.

## `scheduler.db` — планировщик (`scheduler-mcp`)

Путь — `SCHEDULER_DB`. Код — `mcp/scheduler-mcp/src/store.rs`.

| Таблица | Что хранит |
|---|---|
| `jobs` | Задание: `name`, `action` (`reminder`/`http`/`mcp_tool`), `params_json`, `extract_json` (какие метрики извлекать), расписание `schedule_kind`/`schedule_value`, `state`, `next_run_at`, `last_run_at`, `consecutive_failures`, `pause_reason`, `created_at` |
| `runs` | Запуски: `job_id`, `scheduled_at`, `started_at`, `finished_at`, `status`, `result_json`, `metrics_json`, `error` |
| `events` | События для агента (напоминания, сбои, пауза задания): `kind`, `text`, `created_at`, `acked_at` — непрочитанные, пока не подтверждены |

## `documents/` — documents-mcp

- `documents/inbox/` — входящие PDF;
- `documents/out/` — результаты: `<имя>.md` и `<имя>.summary.md` (с YAML-шапкой: источник, хэши всех звеньев, метод);
- `documents/.artifacts/` — артефакты шагов по содержимому: `doc-<16 hex>.md|.json`, `sum-<16 hex>.md|.json`; при чтении хэш пересчитывается и сверяется.

## Файлы конфигурации

### `mcp.json` — MCP-серверы

Формат `mcpServers`, как у Claude Desktop/Cursor. Путь — `LLM_MCP_CONFIG`.

```json
{
  "mcpServers": {
    "git":       { "command": "cargo", "args": ["run", "-q", "-p", "git-mcp"] },
    "scheduler": { "url": "http://127.0.0.1:8092/mcp" },
    "github":    { "url": "https://api.githubcopilot.com/mcp/", "enabled": false,
                   "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" } }
  }
}
```

Поля сервера: `command` + `args` (+ `env`, `cwd`) — stdio-сервер дочерним процессом; `url` (+ `headers`) — Streamable HTTP; `description`; `timeout_sec` — предел одного вызова (120 с); `enabled: false` — сервер в конфигурации, но не подключается. `${VAR}` подставляется из окружения.

### `profiles/*.md`, `invariants/*.md`

Профиль — markdown о том, как говорить с человеком (тон, язык, формат); агент ссылается на него по имени файла. Инвариант — один файл на правило (имя файла — id), общий для всех агентов; ассистент отказывается его нарушать.

### `gateway/gateway.toml`

Ключи пользователей, `rpm`, лимиты контекста и очереди шлюза; образец — `gateway/gateway.example.toml`.

### `rag/eval.json`

Контрольные вопросы для сравнения стратегий нарезки: вопрос, фрагмент ответа, документ, из которого он взят.
