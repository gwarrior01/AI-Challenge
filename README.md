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

## Running the web interface

```bash
cargo run -p llm-web
```

Starts a server at `http://localhost:8080` — a dark-themed page (`POST /api/ask` backend) with two tabs:

- **Чат** — a regular chat with message bubbles, a typing indicator, and a token counter (per request + running session total).
- **Задача · 4 способа** — enter one logical/algorithmic/analytical task and an optional reference answer, then solve it four ways in parallel: a direct answer, a "think step by step" prompt, a two-step "model writes the solving prompt, then it's used" flow, and a multi-expert (analyst/engineer/critic) prompt. Each card renders Markdown and LaTeX (`\(...\)`, `\[...\]`, `$$...$$`) via KaTeX, can be shown/hidden individually or all at once, and — once at least one method has answered — a "Проверить решения моделью" button sends all four solutions (plus the reference answer, if given) to the model for grading: which ones are correct, wrong, or close. That grading call uses `LLM_ANALYSIS_MODEL` if set, otherwise `LLM_MODEL`.

## Running the TUI

```bash
cargo run -p llm-tui
```

A full-screen terminal chat with rounded panels: type text, `Enter` to send the request, `Esc` to quit. Requests run in the background (a spinner shows while waiting, the UI stays responsive), and a status bar shows the token usage of the last request plus the session total.

## Building release binaries

```bash
cargo build --release
```

Binaries will appear in `target/release/`: `llm-cli`, `llm-web`, `llm-tui`.

## Project structure

```
Cargo.toml       — workspace tying all crates together
.env.example     — environment variable template
core/             — llm-core: LLM client
cli/              — llm-cli: console interface
web/              — llm-web: web interface (axum), src/index.html: chat page
tui/              — llm-tui: terminal interface (ratatui)
```
