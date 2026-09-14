# LLM Agent in Rust

A Cargo workspace made up of four crates:

| Crate      | Purpose                                                              |
|------------|-----------------------------------------------------------------------|
| `core`     | `llm-core` — shared LLM client (HTTP request/response), used by the other crates |
| `cli`      | `llm-cli` — minimal console agent (the main deliverable)             |
| `web`      | `llm-web` — web chat UI (axum), port `8080`                          |
| `tui`      | `llm-tui` — terminal chat UI (ratatui)                                |

The client talks to any **OpenAI-compatible** HTTP API (`/chat/completions`): OpenAI, OpenRouter, Ollama, LM Studio, etc. all work. The API address and key are supplied via environment variables rather than hardcoded.

If the API response includes a `usage` field (prompt/completion/total tokens — standard for OpenAI-compatible APIs), all three interfaces show it: per-request in the CLI, under each message bubble in the web UI (request tokens under yours, response tokens under the reply) plus a running session total in the header, and under each message in the TUI plus a session total in the status bar.

Every conversation that carries history now — the web UI's "Чат" tab, the TUI's direct chat screen, and named agents in both the web UI and the TUI (see below) — also shows a **context window fill indicator** next to the chat: the `total_tokens` of the most recent exchange (exactly as reported by the model, not a sum we compute ourselves — that number already reflects the full conversation so far, since the whole history is resent with every request) against the model's context window, colored green→yellow→red as it approaches the limit (a ring with a tooltip in the web UI, a text progress bar in the TUI). Since OpenAI-compatible APIs don't report a model's context window size in the response, that denominator is either looked up from `LLM_CONTEXT_WINDOW` (see below) or assumed to be 262000 tokens if that isn't set.

The plain chat screens (web UI's "Чат" tab, TUI direct chat) can turn on **context management (summarization)** — an opt-in checkbox (web UI) or wizard step (TUI), off by default. With it on, only the last few exchanges are sent to the model as-is; everything older is periodically folded into a running text summary — produced by the same model, via a dedicated request — which is substituted for the full history from then on, recomputed in batches of `LLM_CONTEXT_SUMMARY_CHUNK` user messages (10 by default, see below). Since the plain chat has no server-side state at all, the summary lives in the browser tab and the server only performs the one-off summarization request (`POST /api/summarize`).

### Context-management strategies for named agents

Named agents (see below) go further: each one has a **`context_strategy`**, switchable per agent (a dropdown in the web UI, a wizard step / `--context-strategy` flag in the CLI/TUI, changeable on the fly via `agent strategy <name> <strategy>` or `POST /api/agents/:name/strategy`), independent of the plain chat's summarization toggle above. Five strategies, picked to cover meaningfully different trade-offs:

- **`full`** (default) — no management: the whole history is resent every request, exactly like before this feature existed.
- **`summary`** — the same summarization as the plain chat, but persisted to SQLite so it survives restarts: the unsummarized tail (batches of `LLM_CONTEXT_SUMMARY_CHUNK` messages) is resent as-is, everything older lives only in the running summary.
- **`sliding-window`** — only the last `N` user messages (exchanges) are sent; everything older is simply dropped, no extra LLM call, cheapest per request but the only strategy that can make the agent visibly forget an old fact.
- **`facts`** — Sticky Facts / Key-Value Memory: a separate key→value fact table (goal, constraints, preferences, decisions, agreements, ...) is extracted and updated by a dedicated LLM call after *every* user message, and the request sends that facts block plus the same last-`N`-messages window as `sliding-window`. `N` is shared between `sliding-window` and `facts` — `LLM_SLIDING_WINDOW_SIZE` (6 by default, see below), overridable per agent.
- **`branching`** — the history itself is a set of named branches instead of one line: `agent checkpoint <name> <label>` marks the current point in the current branch, `agent branch <name> <new-branch> [--from <label>]` clones the history up to a checkpoint (or the current tip) into a new, independent branch, and `agent switch <name> <branch>` changes which branch subsequent messages go to — each branch sends its *own* full history with every request (no trimming), so two branches never see each other's later messages, only what came before they diverged. All persisted to SQLite (`AGENTS_STORE_PATH`), including checkpoints and branch contents.

Interfaces show a live status badge next to the context-window indicator matching whichever strategy is active: messages left until the next summary recompute (`summary`), how many of the total messages actually made it into the last request (`sliding-window`), the current fact count (`facts`), or the current branch name (`branching`) — nothing is shown for `full`.

A quick self-test (`llm-cli agent add ... --context-strategy X`, run the same ~15-message spec-gathering conversation, then ask the agent to recall an early detail) is the fastest way to see the difference: `full` and `summary` and `facts` all recall it correctly, `sliding-window` typically can't once the detail has scrolled out of the window, and `branching` recalls it in every branch descended from a checkpoint that includes it, but changes made in one branch never leak into a sibling branch.

### Memory model for named agents

Independently of the `context_strategy` above (which only governs how the *dialogue history* is sent to the model), every named agent has a **three-tier memory model** (`core/src/memory.rs`), each tier stored separately (its own SQLite table) and filled only **explicitly** — nothing is written to any tier by guessing what the model said, only by a direct call:

1. **Short-term memory** — the current dialogue. This is exactly the existing message history/branches above, shaped by whichever `context_strategy` is active. It grows with every exchange and forgets exactly as that strategy decides.
2. **Working memory** (`agent task ...`) — data of a **shared task**, not the agent itself. A task is its own named entity, independent of any one agent: one agent creates it (`agent task <name> start <task> [--goal TEXT]`) and is immediately attached to it, and any number of *other* agents can attach to that same task by name (`agent task <name> join <task>`) — from then on every attached agent reads and writes the same key/value data (`agent task <name> set <key> <value>`), so a task doubles as a shared scratchpad for several agents working toward one goal. Each agent can be attached to at most one task at a time (attach elsewhere only after finishing the current one), but a task itself can have any number of members. `agent task <name> finish` ends the task **for every attached agent at once** and deletes its data irrecoverably (enforced by `ON DELETE CASCADE` in SQLite, not just application logic) — anything worth keeping must be moved into long-term memory first (`agent remember`), and by a member *before* anyone calls finish.
3. **Long-term memory** (`agent remember` / `agent forget`) — profile facts, decisions, and knowledge that don't belong to any one task or conversation. This is a **single store shared by every agent** — there's no per-agent isolation at all: editing it through one agent is immediately visible through any other agent, regardless of which task (if any) it's attached to. Entries are key → (category, value) pairs that persist until explicitly removed with `agent forget`; nothing is added or changed automatically. (The `agent remember Х ...`/`agent forget Х ...` syntax still names an agent because that's how the CLI/TUI/web route the call — but `Х` is only an entry point, not an owner: the data itself has no `agent_name` column.)

Both working and long-term memory (when non-empty) are injected as extra system messages on *every* request, regardless of `context_strategy` — that axis only controls how much of the raw dialogue is resent. `agent memory <name>` prints all three tiers side by side so the boundary between them is visible, not just present in storage. `agent tasks` lists every existing shared task and its current members, independent of any single agent — useful to find the name of a task before `join`ing it.

All three interfaces expose the same explicit read/write points — none of them ever infers what to store: the **CLI** via the `agent remember`/`forget`/`memory`/`task ...`/`tasks` subcommands below, the **TUI** (the primary interface) via a dedicated memory screen (`F3` from an agent's chat, see "Running the TUI" below) using the same command syntax plus a live list of joinable tasks, and the **web UI** via a "🧠 Память" panel in the agent chat view (see "Running the web interface" below) with a form per tier plus a join button and a list of existing tasks.

```bash
cargo run -p llm-cli -- agent remember Переводчик любимый_язык Rust --category profile
cargo run -p llm-cli -- agent forget Переводчик любимый_язык
cargo run -p llm-cli -- agent task Переводчик start проект-X --goal "перевести и проверить отчёт"
cargo run -p llm-cli -- agent task Переводчик set черновик "перевод готов"
cargo run -p llm-cli -- agent task Аналитик join проект-X       # второй агент присоединяется к той же задаче
cargo run -p llm-cli -- agent task Аналитик show                # видит данные, которые сохранил Переводчик
cargo run -p llm-cli -- agent task Аналитик finish               # завершает задачу для ОБОИХ агентов разом
cargo run -p llm-cli -- agent tasks
cargo run -p llm-cli -- agent memory Переводчик
```

This is distinct from the `facts` context strategy above: `facts` auto-extracts a sticky key/value summary of the dialogue via an LLM call after every message and is itself part of *short-term* context management, while long-term/working memory here are never auto-populated — only a human (or a script calling `Agent::remember`/`Agent::task_set`) decides what goes in and which of the two tiers it belongs to.

### Cost tracking for named agents

Named agents can also show **how much money each request cost**, from two sources, preferred in this order:

1. **Real cost from the provider** — some OpenAI-compatible APIs (OpenRouter, notably) include an actual billed amount in the response as `usage.cost` (plus a `usage.cost_details.upstream_inference_prompt_cost` / `..._completions_cost` breakdown). When present, this is what's shown — it's the real number, not an estimate, and it's **persisted** to SQLite alongside the message's token counts (new `cost`/`cost_input`/`cost_output` columns, migrated in automatically for existing `agents.db` files), because unlike an estimate it can't be recomputed later from just the token counts.
2. **An estimate from configured rates** — when the provider doesn't send `usage.cost` (most don't: OpenAI, Ollama, LM Studio...), but both `LLM_PRICE_INPUT_PER_1M` and `LLM_PRICE_OUTPUT_PER_1M` are set (see below), cost is estimated from the `usage` token counts the model returns. Unlike real cost, an estimate is **never persisted** — it's recomputed on the fly from the already-stored token counts using whichever rate is currently configured, so changing the rates (or restarting with different ones) retroactively re-prices the whole visible history rather than leaving stale numbers around. Estimated figures are marked with a leading `≈` wherever they're shown, so they're never mistaken for the real billed amount.

If neither is available (no `usage.cost` from the provider and no rates configured), cost is never computed or shown anywhere — there's no honest price to attach to a free local model, and no default is invented.

Each interface shows it two ways: **per-message**, split the same way tokens already are (input cost under the user's message, output cost under the assistant's reply — when the provider sends `usage.cost` without a prompt/completion breakdown, the whole amount is attributed to the output side rather than split arbitrarily), and as a **running total for the whole dialogue** — not just the messages sent in the current process/tab, but the entire persisted conversation with that agent, so reopening a chat after a restart shows what it has cost so far, not $0.00 (and, since real cost is persisted, a dialogue that mixed a provider with and without `usage.cost` over its lifetime totals both correctly, marking `≈` only if at least one contributing message was an estimate). In the CLI (gated by `--show-tokens`, same as the token summary) it's a `[💰 запрос: $X · весь диалог: $Y]` line after each reply; in the TUI it's appended to the same per-message token line and to the context-fill status line (`💰 $Y`); in the web UI it's appended to each message's token caption and shown as its own `💰 $Y` badge next to the context-window ring in the agent chat header.

## Installing Rust

If Rust isn't installed yet:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

## Environment variables

Required:

- `LLM_API_URL` — base URL of the LLM API, without a trailing `/chat/completions`.
  Examples: `https://api.openai.com/v1`, `http://localhost:11434/v1` (Ollama), `http://localhost:1234/v1` (LM Studio).
- `LLM_API_KEY` — API key. For local servers with no auth, any non-empty value works, e.g. `local`.

Optional:

- `LLM_MODEL` — model name (defaults to `gpt-4o-mini`). Used for every request in the CLI, TUI, and web chat, and as the default for the web UI's "Задача · 4 способа" tab (all four solving methods, plus the analysis step below if `LLM_ANALYSIS_MODEL` isn't set).
- `LLM_ANALYSIS_MODEL` — optional, web UI only. Overrides the model used specifically by the "Задача · 4 способа" tab's "Проверить решения моделью" button (the step that reviews the four solutions against the reference answer). Useful for pointing solving and grading at different models — e.g. a cheaper/faster model solves the task four ways, while a stronger model judges the results. If unset, that request falls back to `LLM_MODEL`.
- `AGENTS_STORE_PATH` — optional. Path to the SQLite file where named agents (see below) and their conversation history are persisted. Defaults to `agents.db` in the current working directory. The CLI, web UI, and TUI can share the same file — an agent added in one is visible in the others, and its dialogue survives restarting the process.
- `LLM_CONTEXT_WINDOW` — optional. Context window size in tokens, used only for the fill indicator described above (never enforced or sent to the API). There's no reliable way to query a model's real context limit from an OpenAI-compatible API, so set this explicitly to match your model/provider (e.g. `128000`). If unset, the indicator assumes a window of `262000` tokens.
- `LLM_CONTEXT_SUMMARY_CHUNK` — optional. How many user messages to accumulate between summary recomputes when context management (summarization, see above) is turned on, for both named agents and the plain web chat. Read once per process and cached, like the other `LLM_*` variables — changing it requires a restart. If unset, empty, or not a positive integer, defaults to `10`.
- `LLM_SLIDING_WINDOW_SIZE` — optional. How many of the most recent user messages (exchanges) named agents keep when their `context_strategy` is `sliding-window` or `facts` (see above) — an individual agent can override this via `--window-size`/the web UI's window-size field. Read once per process and cached; if unset, empty, or not a positive integer, defaults to `6`.
- `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` — optional, but both must be set together to turn on the cost *estimate* for named agents when the provider doesn't send a real `usage.cost` (see "Cost tracking for named agents" above): price per 1,000,000 input/output tokens, in whatever currency `LLM_PRICE_CURRENCY` names. If either is unset or not a non-negative number, no estimate is computed — cost is then shown only where the provider supplies `usage.cost` directly, or not at all.
- `LLM_PRICE_CURRENCY` — optional. Currency symbol/code shown next to cost figures (both real and estimated, see above), e.g. `$`, `€`, `₽`. Defaults to `$`.

Set them directly in your shell:

```bash
export LLM_API_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
export LLM_ANALYSIS_MODEL="gpt-4o"   # optional — see above
export LLM_CONTEXT_WINDOW="128000"   # optional — see above
export LLM_CONTEXT_SUMMARY_CHUNK="10"   # optional — see above
export LLM_PRICE_INPUT_PER_1M="3"    # optional — see above
export LLM_PRICE_OUTPUT_PER_1M="15"  # optional — see above
```

Or copy `.env.example` to `.env` and load it before running (e.g. `set -a && source .env && set +a`, or use `direnv`) — the code itself does not read `.env` files, only process environment variables.

## Running the CLI (main deliverable)

One-shot request:

```bash
cargo run -p llm-cli -- "Hello, how are you?"
```

Interactive mode (REPL) — no arguments, type requests line by line, exit with `Ctrl+D`:

```bash
cargo run -p llm-cli
```

### Named agents

Beyond the direct request/REPL mode above, the CLI can manage **named agents** — each one a standalone entity with its own configuration (system prompt, model override, generation limits, whether to show token usage) and its own lifecycle (started/stopped). A stopped agent refuses requests without contacting the LLM API.

```bash
cargo run -p llm-cli -- agent list
cargo run -p llm-cli -- agent add Переводчик --system "Переводи на английский" --show-tokens
cargo run -p llm-cli -- agent start Переводчик   # enters a chat loop with this agent
cargo run -p llm-cli -- agent stop Переводчик
cargo run -p llm-cli -- agent remove Переводчик
```

`agent add` flags: `--system TEXT`, `--model NAME`, `--show-tokens`, `--max-tokens N`, `--temperature N`, `--top-p N`, `--reasoning on|off`, `--context-strategy full|summary|sliding-window|facts|branching` (see "Context-management strategies for named agents" above; `--compress` still works as a deprecated alias for `--context-strategy summary`), `--window-size N` (override the shared window size for `sliding-window`/`facts`). Inside `agent start`'s chat loop, type `stop` (or `exit`, or `Ctrl+D`) to stop the agent and leave — it also prints a one-line notice whenever a summary is recomputed or facts are updated. Agents and their message history are persisted to `AGENTS_STORE_PATH` (SQLite, see above) and are the same registry the web UI's "Агенты" tab and the TUI's agents screen use. An agent remembers the whole conversation so far — this survives stopping/starting the agent *and* restarting the whole application; `agent start` prints any restored history before opening the chat prompt. Removing an agent (`agent remove`) deletes its stored history (all branches), summary, and facts too. `agent list` also shows each agent's active `context_strategy` (and, for `branching`, its current branch, branch count, and checkpoint count).

Additional `agent` subcommands for managing strategy and branches on an existing agent:

```bash
cargo run -p llm-cli -- agent strategy Переводчик sliding-window       # switch strategy on the fly
cargo run -p llm-cli -- agent checkpoint Переводчик before-redesign     # mark a branch point (strategy: branching)
cargo run -p llm-cli -- agent branch Переводчик idea-a --from before-redesign  # branch off from a checkpoint
cargo run -p llm-cli -- agent branch Переводчик idea-b --from before-redesign  # ...and a second, independent branch
cargo run -p llm-cli -- agent switch Переводчик idea-a                  # switch which branch chat goes to
cargo run -p llm-cli -- agent branches Переводчик                       # list branches and checkpoints
```

`agent checkpoint`/`branch`/`switch` only work while the agent's `context_strategy` is `branching` (they error out otherwise); `agent branch` without `--from` clones the current branch's history as of right now instead of a named checkpoint.

Further subcommands for the three-tier **memory model** (short-term/working/long-term — see "Memory model for named agents" below):

```bash
cargo run -p llm-cli -- agent memory Переводчик                                  # show all 3 tiers at once
cargo run -p llm-cli -- agent remember Переводчик <key> <value> [--category CAT] # long-term memory (shared by all)
cargo run -p llm-cli -- agent forget Переводчик <key>                            # long-term memory (shared by all)
cargo run -p llm-cli -- agent task Переводчик start <task> [--goal TEXT]         # create a shared task + attach
cargo run -p llm-cli -- agent task Аналитик join <task>                          # attach another agent to it
cargo run -p llm-cli -- agent task Переводчик set <key> <value>                  # shared working memory
cargo run -p llm-cli -- agent task Переводчик show                               # shared working memory
cargo run -p llm-cli -- agent task Переводчик finish                             # ends the task for ALL members
cargo run -p llm-cli -- agent tasks                                              # list all shared tasks + members
```

## Running the web interface

```bash
cargo run -p llm-web
```

Starts a server at `http://localhost:8080` — a dark-themed page (`POST /api/ask` backend) with four tabs:

- **Чат** — a regular chat with message bubbles, a typing indicator, and token counts under each message (request tokens under yours, response tokens under the reply), plus a running session total in the header. It remembers the conversation so far — the browser tab keeps the message history and resends it with every request, so the model sees prior turns — until you click "Новая сессия" (or reload the page); nothing is persisted server-side for this tab, unlike named agents. A context-window fill indicator next to the header stats shows the `total_tokens` of the most recent exchange against the model's context window, same as agents (see below). A "Сжимать контекст диалога" checkbox turns on context management (see above) for this session: state (the summary text, how many messages it covers) lives entirely in the browser tab, and each recompute calls `POST /api/summarize`; a row in the chat announces when a summary is (re)computed.
- **Задача · 4 способа** — enter one logical/algorithmic/analytical task and an optional reference answer, then solve it four ways in parallel: a direct answer, a "think step by step" prompt, a two-step "model writes the solving prompt, then it's used" flow, and a multi-expert (analyst/engineer/critic) prompt. Each card renders Markdown and LaTeX (`\(...\)`, `\[...\]`, `$$...$$`) via KaTeX, can be shown/hidden individually or all at once, and — once at least one method has answered — a "Проверить решения моделью" button sends all four solutions (plus the reference answer, if given) to the model for grading: which ones are correct, wrong, or close. That grading call uses `LLM_ANALYSIS_MODEL` if set, otherwise `LLM_MODEL`.
- **Температура** — send the same prompt N times at a chosen temperature to see how much the answers vary.
- **Агенты** — a control panel for named agents. Creating one starts in **быстро** mode (name + system prompt only, everything else defaults) — switch to **расширенно** to also set a model override, generation limits, temperature/top_p/reasoning, a "show token usage" toggle, and a context-strategy dropdown (`full`/`summary`/`sliding-window`/`facts`/`branching`, see "Context-management strategies for named agents" above, plus a window-size field that appears for `sliding-window`/`facts`). Each card shows status (running/stopped) and has Start/Stop/Delete plus an "Открыть чат →" button that opens a full-screen chat view for that agent (its own header with a "← Назад к агентам" button, a Start/Stop toggle for that agent right there in the header, a context-window fill indicator, a large scrollable message area, and a composer). Token counts are shown under each message in that view too (request tokens under the user's message, response tokens under the agent's reply), and the context-window ring next to the header shows the `total_tokens` of the most recent exchange (straight from the model's own response, not something we sum up) against the model's context window, with a tooltip on hover giving the exact numbers — a stopped agent's requests are rejected without an API call, and a running agent remembers the conversation so far, including across a server restart — opening its chat lazily fetches any history (with its per-message token counts) stored in SQLite via `GET /api/agents/:name/history` the first time. A badge next to the context-window ring reflects whichever strategy is active (messages until next summary, window fill, fact count, or current branch — nothing for `full`), and a row in the chat announces a summary recompute or a facts update as it happens — all of it persisted to SQLite alongside the history. A **🧠 Память** button in the header toggles a memory panel (see "Memory model for named agents" above) — independent of `context_strategy` — with a form for the shared task's working memory (create, join by name — an autocomplete list suggests existing tasks and their current members, add key/value data, "Завершить для всех" ends it for every attached agent) and one for long-term memory (add/remove key → category + value entries, explicitly marked "общая для всех агентов" — editing it from any agent's panel changes the same store every other agent sees); every action calls its own endpoint (`POST /api/agents/:name/remember`, `.../forget`, `.../task/start`, `.../task/join`, `.../task/set`, `.../task/finish`, plus `GET /api/tasks` for the list of all shared tasks) and immediately refreshes the panel from the agent's latest state. Backed by `GET/POST /api/agents`, `POST /api/agents/:name/start`, `POST /api/agents/:name/stop`, `DELETE /api/agents/:name` (also deletes its stored history), `POST /api/agents/:name/ask`, `GET /api/agents/:name/history`, `POST /api/agents/:name/strategy` (switch strategy on the fly), `POST /api/agents/:name/checkpoint` / `.../branch` / `.../switch` (branching — a dedicated branch bar appears under the header whenever `context_strategy` is `branching`: a dropdown to switch the active branch, a field + button to drop a checkpoint at the current point, and a field + "from which checkpoint" dropdown + button to branch off a new one, mirroring the CLI's `agent checkpoint`/`branch`/`switch` above), and the memory endpoints above, all sharing the same on-disk registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the TUI's agents screen.

## Running the TUI

```bash
cargo run -p llm-tui
```

A full-screen terminal chat with rounded panels: type text, `Enter` to send the request, `Esc` to quit. Requests run in the background (a spinner shows while waiting, the UI stays responsive), and a status bar shows the token usage of the last request plus the session total. Like the web UI's "Чат" tab, it remembers the conversation — the process keeps the message history in memory and resends it with every request — until you press `Ctrl+N` (the same shortcut clears the session totals and the on-screen transcript too) or quit; nothing is persisted to disk for this screen, unlike named agents. The status bar also shows a context-window fill indicator (a text progress bar: the `total_tokens` of the most recent exchange against the model's context window, colored green→yellow→red).

Press `F2` to switch to the **Агенты** screen — the same named-agent registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the web UI's "Агенты" tab:

- `↑`/`↓` — select an agent, `n` — create one, `s` — start/stop the selected agent, `d` — delete it (`y` to confirm), `Enter` — open a chat with a running agent, `Esc`/`F2` — back to the direct chat screen. A spinner next to an agent's name means it's currently generating a response even if you're not looking at its chat.
- `n` first asks which creation mode to use: **быстро** (`1`/`q`) asks only for a name and a system prompt — every other setting (model, limits, temperature, top_p, reasoning, show-tokens, context strategy) is left at its default (`full`); **расширенно** (`2`/`a`) walks through the full set of fields one at a time, including a final context-strategy step accepting `full`/`summary`/`sliding-window`/`facts`/`branching` (see "Context-management strategies for named agents" above; branch management itself — checkpoints, branching, switching — has no TUI screen yet, use the CLI's `agent checkpoint`/`branch`/`switch` or the web UI's branch bar, see below).
- Inside an agent's chat (`Enter` to send, `PageUp`/`PageDown` to scroll through earlier messages, `End` to jump back to the latest, `Ctrl+S` to start/stop that agent without leaving the chat — plain `s` is left alone since it's a text input here, unlike the agents list), requests go through that agent's `handle_request` — a stopped agent's requests are rejected without an API call. Each agent remembers the conversation so far (system prompt + prior turns are resent with every message) — this survives being stopped/started and even the whole application restarting, since history lives in SQLite (`AGENTS_STORE_PATH`); opening a running agent's chat restores its prior messages onto the screen, each annotated with its token counts. A status line below the transcript shows a text progress bar for the context window fill (the `total_tokens` of the most recent exchange, straight from the model's response, against the model's context window, colored green→yellow→red as it approaches the limit) in place of the old per-request token summary — that detail now lives under each message instead. If context management is on for the agent, the moment a summary is (re)computed shows up as its own line in the transcript.
- `Esc` returns to the agent list **immediately**, even while a response is still being generated — the request keeps running in the background and the reply lands in that agent's history whenever it arrives, so you're free to check on or chat with other agents in the meantime.
- The agent chat's input box is twice as tall as the direct chat's, and pasting (bracketed paste) inserts the clipboard content as-is — multi-line pastes no longer get cut apart and sent early on their embedded line breaks; only pressing `Enter` yourself sends the message.
- `F3` from an agent's chat opens its **memory screen** (see "Memory model for named agents" above) — `Esc` or `F3` again returns to the chat. It shows all three tiers: a one-line summary of the short-term dialogue (record count + active `context_strategy`), the shared task's working memory in full (including its other members, and — when this agent isn't attached to any task — a list of existing tasks available to `join`), and the long-term memory shared by every agent (identical no matter which agent's screen you're looking at). Since this is TUI, not a form, changes go through the same command syntax as `llm-cli agent remember/forget/task` typed into the input box and run on `Enter`: `remember <key> <value> [--category CAT]`, `forget <key>`, `task start <name> [--goal TEXT]` (creates a task and attaches), `task join <name>` (attaches to one created by another agent), `task set <key> <value>`, `task finish` (ends it for every attached agent) — the result (or error, e.g. trying to `task start`/`join` while already attached elsewhere) shows as a status line below the panel. These are synchronous, local calls (no LLM request), so they work immediately even while this or another agent is still generating a chat reply in the background.

## Building release binaries

```bash
cargo build --release
```

Binaries will appear in `target/release/`: `llm-cli`, `llm-web`, `llm-tui`.

## Project structure

```
Cargo.toml       — workspace tying all crates together
.env.example     — environment variable template
core/             — llm-core: LLM client (src/lib.rs) + agent entity and registry (src/agent.rs) + context summarization (src/context.rs) + 3-tier memory model (src/memory.rs)
cli/              — llm-cli: console interface; also `agent` subcommand for managing named agents
web/              — llm-web: web interface (axum), src/index.html: chat/tasks/agents page
tui/              — llm-tui: terminal interface (ratatui)
```
