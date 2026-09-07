# LLM Agent in Rust

A Cargo workspace made up of four crates:

| Crate      | Purpose                                                              |
|------------|-----------------------------------------------------------------------|
| `core`     | `llm-core` — shared LLM client (HTTP request/response), used by the other crates |
| `cli`      | `llm-cli` — minimal console agent (the main deliverable)             |
| `web`      | `llm-web` — web chat UI (axum), port `8080`                          |
| `tui`      | `llm-tui` — terminal chat UI (ratatui)                                |

The client talks to any **OpenAI-compatible** HTTP API (`/chat/completions`): OpenAI, OpenRouter, Ollama, LM Studio, etc. all work. The API address and key are supplied via environment variables rather than hardcoded.

If the API response includes a `usage` field (prompt/completion/total tokens — standard for OpenAI-compatible APIs), all three interfaces show it: per-request in the CLI, in the chat bubble and header stats in the web UI, and in the status bar of the TUI, plus a running total for the session.

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
- `AGENTS_STORE_PATH` — optional. Path to the JSON file where named agents (see below) are persisted. Defaults to `agents.json` in the current working directory. The CLI, web UI, and TUI can share the same file — an agent added in one is visible in the others.

Set them directly in your shell:

```bash
export LLM_API_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
export LLM_ANALYSIS_MODEL="gpt-4o"   # optional — see above
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

`agent add` flags: `--system TEXT`, `--model NAME`, `--show-tokens`, `--max-tokens N`, `--temperature N`, `--top-p N`, `--reasoning on|off`. Inside `agent start`'s chat loop, type `stop` (or `exit`, or `Ctrl+D`) to stop the agent and leave. Agents are persisted to `AGENTS_STORE_PATH` (see above) and are the same registry the web UI's "Агенты" tab and the TUI's agents screen use. While running, an agent remembers the conversation so far (system prompt + prior turns are resent with every message) — starting it again begins a fresh conversation.

## Running the web interface

```bash
cargo run -p llm-web
```

Starts a server at `http://localhost:8080` — a dark-themed page (`POST /api/ask` backend) with two tabs:

- **Чат** — a regular chat with message bubbles, a typing indicator, and a token counter (per request + running session total).
- **Задача · 4 способа** — enter one logical/algorithmic/analytical task and an optional reference answer, then solve it four ways in parallel: a direct answer, a "think step by step" prompt, a two-step "model writes the solving prompt, then it's used" flow, and a multi-expert (analyst/engineer/critic) prompt. Each card renders Markdown and LaTeX (`\(...\)`, `\[...\]`, `$$...$$`) via KaTeX, can be shown/hidden individually or all at once, and — once at least one method has answered — a "Проверить решения моделью" button sends all four solutions (plus the reference answer, if given) to the model for grading: which ones are correct, wrong, or close. That grading call uses `LLM_ANALYSIS_MODEL` if set, otherwise `LLM_MODEL`.
- **Температура** — send the same prompt N times at a chosen temperature to see how much the answers vary.
- **Агенты** — a control panel for named agents. Creating one starts in **быстро** mode (name + system prompt only, everything else defaults) — switch to **расширенно** to also set a model override, generation limits, temperature/top_p/reasoning, and a "show token usage" toggle. Each card shows status (running/stopped) and has Start/Stop/Delete plus an "Открыть чат →" button that opens a full-screen chat view for that agent (its own header with a "← Назад к агентам" button, a large scrollable message area, and a composer) — a stopped agent's requests are rejected without an API call, and a running agent remembers the conversation so far until it's restarted. Backed by `GET/POST /api/agents`, `POST /api/agents/:name/start`, `POST /api/agents/:name/stop`, `DELETE /api/agents/:name`, and `POST /api/agents/:name/ask`, all sharing the same on-disk registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the TUI's agents screen.

## Running the TUI

```bash
cargo run -p llm-tui
```

A full-screen terminal chat with rounded panels: type text, `Enter` to send the request, `Esc` to quit. Requests run in the background (a spinner shows while waiting, the UI stays responsive), and a status bar shows the token usage of the last request plus the session total.

Press `F2` to switch to the **Агенты** screen — the same named-agent registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the web UI's "Агенты" tab:

- `↑`/`↓` — select an agent, `n` — create one, `s` — start/stop the selected agent, `d` — delete it (`y` to confirm), `Enter` — open a chat with a running agent, `Esc`/`F2` — back to the direct chat screen. A spinner next to an agent's name means it's currently generating a response even if you're not looking at its chat.
- `n` first asks which creation mode to use: **быстро** (`1`/`q`) asks only for a name and a system prompt — every other setting (model, limits, temperature, top_p, reasoning, show-tokens) is left at its default; **расширенно** (`2`/`a`) walks through the full set of fields one at a time, same as before.
- Inside an agent's chat (`Enter` to send, `PageUp`/`PageDown` to scroll through earlier messages, `End` to jump back to the latest), requests go through that agent's `handle_request` — a stopped agent's requests are rejected without an API call. Each running agent remembers the conversation so far (system prompt + prior turns are resent with every message) until it's stopped and restarted, which begins a fresh conversation.
- `Esc` returns to the agent list **immediately**, even while a response is still being generated — the request keeps running in the background and the reply lands in that agent's history whenever it arrives, so you're free to check on or chat with other agents in the meantime.
- The agent chat's input box is twice as tall as the direct chat's, and pasting (bracketed paste) inserts the clipboard content as-is — multi-line pastes no longer get cut apart and sent early on their embedded line breaks; only pressing `Enter` yourself sends the message.

## Building release binaries

```bash
cargo build --release
```

Binaries will appear in `target/release/`: `llm-cli`, `llm-web`, `llm-tui`.

## Project structure

```
Cargo.toml       — workspace tying all crates together
.env.example     — environment variable template
core/             — llm-core: LLM client (src/lib.rs) + agent entity and registry (src/agent.rs)
cli/              — llm-cli: console interface; also `agent` subcommand for managing named agents
web/              — llm-web: web interface (axum), src/index.html: chat/tasks/agents page
tui/              — llm-tui: terminal interface (ratatui)
```
