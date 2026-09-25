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
3. **Long-term memory** (`agent remember` / `agent forget`) — decisions and knowledge that don't belong to any one task or conversation. This is a **single store shared by every agent** — there's no per-agent isolation at all: editing it through one agent is immediately visible through any other agent, regardless of which task (if any) it's attached to. Entries are key → (category, value) pairs that persist until explicitly removed with `agent forget`; nothing is added or changed automatically. (The `agent remember Х ...`/`agent forget Х ...` syntax still names an agent because that's how the CLI/TUI/web route the call — but `Х` is only an entry point, not an owner: the data itself has no `agent_name` column.) Personalization used to live here too, as a `--category profile` convention — it's now its own mechanism, see below.

Both working and long-term memory (when non-empty) are injected as extra system messages on *every* request, regardless of `context_strategy` — that axis only controls how much of the raw dialogue is resent. `agent memory <name>` prints all three tiers side by side (plus the current personalization profile, see below) so the boundary between them is visible, not just present in storage. `agent tasks` lists every existing shared task and its current members, independent of any single agent — useful to find the name of a task before `join`ing it.

### Task State Machine

The shared task from working memory above also carries a **formalized finite-state machine** (`Stage` in `core/src/memory.rs`): a **stage**, a free-text **current step**, and a free-text **expected action** — plus a **pause** flag independent of the stage. Four stages, matching a typical work cycle:

```
planning ⇄ execution ⇄ validation → done
   (execution → planning: replan; validation → execution: rework)
```

Transitions are validated, not free-form: from `planning` the only legal next stage is `execution`; from `execution`, either `validation` or back to `planning` (the plan turned out wrong — replan); from `validation`, either `done` (passed) or back to `execution` (failed — rework); `done` is terminal. Jumping stages (e.g. `planning → validation`) is rejected with an error listing what's actually legal from the current stage.

```bash
cargo run -p llm-cli -- agent task Аналитик advance validation                            # -- ошибка: из planning нельзя сразу в validation
cargo run -p llm-cli -- agent task Аналитик advance execution --step "собрать требования" --expect "черновик плана готов"
cargo run -p llm-cli -- agent task Аналитик step "переписать раздел 2"                     # -- обновить шаг, не меняя этап
cargo run -p llm-cli -- agent task Аналитик expect "жду ревью от коллеги"
cargo run -p llm-cli -- agent task Аналитик pause                                          # -- пауза НА ЛЮБОМ этапе
cargo run -p llm-cli -- agent task Аналитик resume                                         # -- снять паузу, этап/шаг/ожидание не менялись
cargo run -p llm-cli -- agent task Аналитик show                                           # -- показать этап/шаг/ожидание/паузу
```

Pause is orthogonal to stage — not a fifth stage, just a flag that can be set or cleared at any point in the cycle (`agent task <name> pause` / `resume`), and advancing is blocked while paused (`resume` first). What makes pausing actually useful rather than just a status flag: the whole snapshot (stage, step, expected action, pause) is read fresh from SQLite and injected into *every* request to the model — the same way working/long-term memory are (see above) — so resuming a paused task doesn't require re-explaining anything to the agent: it already sees where things were left off, in its very next reply, from state rather than from conversation history. That snapshot also carries the **results of the stages already passed** (`TaskState::stage_path`): the chain of accepted transitions that led to the current stage, rebuilt from the transition journal — the approved plan, what was done in execution, what validation asked to fix — with superseded branches dropped (after a replan only the new plan counts). So the agent picks up the unfinished stage even if the relevant replies have long left the context window. Resuming is also active, not just a flag flip: `task_resume` returns a follow-up message (`memory::resume_followup`) that the TUI and the web chat send on the human's behalf right away, so pressing «Возобновить» makes the agent continue from the unfinished stage immediately (nothing is sent if the task is done or waiting for a human decision). Pausing also **interrupts a reply in flight**: while waiting for the LLM, `handle_request` polls the pause flag (`tokio::select!`, every 250 ms) and, once the task is paused, drops the exchange — the HTTP request is aborted, nothing that arrives afterwards is checked or applied (tool calls the model already made before the pause stay in effect), neither the prompt nor a reply is written to history, and the prompt is remembered on the task (`TaskState::interrupted_prompt`, persisted). The chat shows just "⏸ Задача поставлена на паузу." instead of a reply; the human's prompt stays in history (it's already on screen), and resuming tells the agent to answer that interrupted prompt rather than sending the generic "continue" message. **Continuations are hidden:** what the TUI and the web chat send after «Возобновить» or «Утвердить» goes through `Agent::continue_task` — the model gets it for that one request, but it is never written to history or shown in the chat (not even after reloading the history); only the agent's reply appears. A rejection reason, on the other hand, is the human's own words and is sent as a normal message. A request made while the task is *already* paused is answered normally — you can still talk to the agent, it just doesn't move the task. The CLI and the TUI's agent-memory screen (`F3`) expose the same explicit commands: `agent task <name> advance|step|expect|pause|resume`. The **web UI is different on purpose** (see next section): there is no manual task form at all — the task is managed automatically, and its status/controls live in the chat itself, not in the memory tab.

#### The model drives the automaton itself — via tool calls, gated by a human

The commands above are the *manual* interface — a human moving the automaton by hand. Left at that, the agent itself never actually behaves differently per stage: asked to do the task, it just solves it in one reply regardless of what stage the task is nominally in, because nothing stops it. The actual fix is that **the model moves the automaton itself**, through two tools (OpenAI-compatible function calling) offered on every request while a task is active, not paused, not terminal (`done`), and not already waiting on a pending proposal:

- **`move_stage(stage, outcome)`** — propose a transition. The program validates it against `Stage::allowed_next` exactly like `agent task advance` does; an illegal stage is rejected with the same "here's what's actually legal" message. If the transition is one of the two gated ones (see below), it is **not applied** — instead it's parked as `TaskState::pending_stage`/`pending_outcome` and the tool result tells the model to stop and wait, not act as if the transition happened.
- **`update_step(step, expected_action)`** — record progress within the current stage without changing it (maps onto `agent task step`/`expect`).

Two transitions require a human to open the gate (`Stage::requires_approval_to`) — the two moments where the agent would otherwise be able to do (or declare) the whole task without anyone signing off: **`planning → execution`** (don't start acting until the plan is confirmed) and **`validation → done`** (don't declare the task finished until the result is confirmed). `execution → validation`, `execution → planning` (replan) and `validation → execution` (rework) apply immediately — that direction is safe and reversible, and after a replan the way back to `execution` goes through the plan gate again. A pending proposal is opened with `agent task <name> approve` (applies it, same code path as a direct `advance`) or closed with `reject <note>` — the note is **required** (the web "✕ Отклонить" button stays disabled until it's filled in): the stage stays the same, and the model is shown "the human rejected X -> Y, reason: …, rework this stage" in the task status until it proposes the transition again. Right after a reject, the TUI and the web chat also send the agent a follow-up message with the reason on the human's behalf (`memory::rejection_followup`), so a rejected plan is reworked immediately instead of waiting for another typed message. A live run showed the model sometimes rewrites the plan by the note but never calls `move_stage` again — leaving nothing to approve while telling the user to "approve". So if an exchange ends while a rejection is still unanswered, the tool loop makes one extra round with only `move_stage` offered and `tool_choice: "required"`, which resubmits the reworked plan for approval — never by anything the model says in plain text ("looks good" is not consent).

```bash
# модель сама вызывает move_stage(execution, outcome="...") вместо того, чтобы просто написать письмо —
# переход остаётся в ожидании, пока его явно не откроют:
cargo run -p llm-cli -- agent task Аналитик show      # -- покажет "⏳ Предложен переход на этап «execution» ... ждёт approve/reject"
cargo run -p llm-cli -- agent task Аналитик approve   # -- применяет предложенный переход
cargo run -p llm-cli -- agent task Аналитик reject "план неполный, распиши шаг 3"  # -- отклоняет, этап не меняется
```

A live smoke test against a real local model (Qwen3 via LM Studio) surfaced the actual failure mode this closes: with `tool_choice: "auto"`, a capable-enough model will often just *ignore* the tools and answer the whole task directly in plain text on the first turn, tool descriptions notwithstanding. So the first round of every exchange is sent with `tool_choice: "required"` (LLM API call `chat_with_tools`, `require_tool_call` parameter) — the model *must* call one of the two tools before it's allowed to just talk; once it has (recording a step, asking a clarifying question via `expected_action`, or proposing a transition), the next round reverts to `"auto"` so it can reply in plain text normally. Intermediate tool-call/tool-result messages exist only within that one exchange (`core::RequestMessage`, distinct from the persisted `ChatMessage`) — they're never written to the branch history or SQLite, so none of the four context-management strategies above need to know tool calling exists, and a runaway loop is capped (`MAX_TOOL_ROUNDS`, 24 rounds — enough for a long multi-server flow, see *Orchestration* below) with a final tools-off call to force a reply either way; that call tells the model the limit was hit, so it reports which steps are done and which are left instead of pretending the flow finished.

**Transition journal.** Every attempt to change the stage is recorded in SQLite (`shared_task_transitions`) with who made it (`model`/`human`), what came of it (`applied`, `proposed`, `approved`, `rejected`, `refused`) and why — including refused jumps, approvals refused because of a pause or a task invariant, and a pending proposal superseded by a manual `advance` (a human moving the stage by hand drops the model's stale proposal, so a late `approve` can't drag the task back). The last 20 entries come with `TaskState::transitions`, survive pause and restart, and are shown by `agent task <name> show`, the TUI memory screen and a collapsible "Журнал переходов" under the stage badge in the web chat.

**Asking to skip a stage in plain text.** The map above stops the *transition*, but a model that obeys "skip planning, just write the code" could still write the implementation right in its reply while the stage stays `planning`. So such a request is intercepted *before* the model is called (`memory::skip_request_target`, a fixed list of phrases, ignoring ones preceded by «не»): in `planning` — skipping straight to implementation, in `execution` — finishing without validation. The user gets a fixed reply explaining what's missing (approve the plan / go through validation), the exchange is kept in the dialogue history, and the attempt lands in the journal as `refused`. Phrasing outside the list falls through to the model and the soft layer (stage directive); approval is still only ever the explicit `approve`, never a phrase.

The two gates above are fixed for every task. A **specific** task can add its own, on top: `agent task <name> forbid <from> <to>` blocks a transition outright (checked in code, for both a human's `task advance` and the model's `move_stage` — see "Invariants" below), and `agent task <name> require-approval <from> <to>` adds a human-consent gate on a transition beyond the two default ones, but only for the model's own `move_stage` calls (a human running `task advance` already *is* the consent). Both are lifted the same way they're set (`agent task <name> allow`/`unrequire-approval <from> <to>`) — never by anything said in the dialogue.

#### Web UI: no manual task form, status lives in the chat itself

The web UI deliberately doesn't expose the manual `task start`/`join`/`set` commands above at all — a named agent's task is created **automatically** by the first message sent to it (`POST /api/agents/:name/ask` calls an `ensure_task` helper before `handle_request` if the agent has no active task yet, naming it after the agent itself), starting in `planning` the same as everywhere else. This is a web-UI-only convenience layered on top of the same `Agent`/`Stage` machinery — the CLI and TUI are untouched and still require an explicit `agent task <name> start`, consistent with the "nothing is written automatically" memory philosophy above.

Because the task is no longer something the person sets up, its status isn't tucked away in the "🧠 Память" tab either (that tab now only holds personalization and long-term memory). Instead:
- A **stage strip right under the chat header** (top of the dialogue) shows the current stage badge plus a one-line hint (current step / expected action) and a "🏁 Новая задача" button that finishes the task so the next message starts a fresh one.
- A **compact action bar directly above the composer**, next to the "Отправить" button, holds Пауза/Возобновить and, only when the model has a transition awaiting sign-off, an inline Утвердить/Отклонить alert with a reject-reason field — the same `approve`/`reject`/`pause`/`resume` endpoints as the CLI, just surfaced where the conversation is happening instead of a separate screen.

Both are driven by the same `AgentInfo.task` field the CLI/TUI already use, now also echoed back in the `/ask` response itself (`task`) so the status updates immediately after every reply instead of waiting on a separate refetch.

### Personalization: profiles

Independent of the memory model above (which is about what an agent *knows*), each named agent also has a **personalization profile** (`core/src/profile.rs`) describing *how* it should talk to a specific person — tone/manner, response language, format preferences, and constraints, with examples. Unlike memory, a profile isn't stored in SQLite: it's a **markdown file** on disk in a profiles directory (`profiles/` by default, overridable via `LLM_PROFILES_DIR`, see below) — meant to be written and edited by hand, not accumulated fact-by-fact. `AgentConfig.profile` stores only the *name* of the file an agent uses (so several agents can share one profile without duplicating the text): unset resolves to `default` (`profiles/default.md`, included in this repo as an editable template/example), and the special value `none` explicitly turns personalization off for that agent. Whichever profile is resolved, its file (if found) is injected as a system message on *every* request — right alongside the system prompt, independently of `context_strategy` and of working/long-term memory.

```bash
cargo run -p llm-cli -- agent profiles create Александр              # scaffold profiles/Александр.md (blank template)
cargo run -p llm-cli -- agent profile Переводчик                    # show the profile currently in effect
cargo run -p llm-cli -- agent profile Переводчик Александр          # switch to profiles/Александр.md
cargo run -p llm-cli -- agent profile Переводчик none                # turn personalization off for this agent
cargo run -p llm-cli -- agent profiles                                # list every profile file found in the directory
```

Profiles can be created two ways: write the markdown file by hand directly in the profiles directory, or scaffold one through an interface (`agent profiles create <name>` above generates an empty template with the same section headings as `profiles/default.md`) — either way, a newly added file shows up immediately (no restart) everywhere profiles are listed, because [`list_profiles`](core/src/profile.rs) reads the directory fresh on every call. All three interfaces expose the same read/switch/create points: the **CLI** via `agent profile`/`agent profiles`/`agent profiles create` above (and `agent memory <name>`, which shows the active profile alongside the three memory tiers), the **TUI**'s agent-memory screen (`F3`) via `profile <name|none>` and `profile new <name>` commands (plus a profile step in the agent-creation wizard, offering whatever's currently on disk), and the **web UI**'s "🧠 Память" panel via a dedicated "🎭 Персонализация" section — a real `<select>` dropdown listing every profile found on disk (so a newly added one is a selectable option there, not just a typed-in name) plus a small "+ Создать профиль" form — and a matching profile dropdown in the agent-creation form; backed by `POST /api/agents/:name/profile`, `GET /api/profiles` (list) and `POST /api/profiles` (create, `{name, content?}` — `content` defaults to the same blank template as the CLI).

All three interfaces expose the same explicit read/write points — none of them ever infers what to store: the **CLI** via the `agent remember`/`forget`/`memory`/`task ...`/`tasks` subcommands below, the **TUI** (the primary interface) via a dedicated memory screen (`F3` from an agent's chat, see "Running the TUI" below) using the same command syntax plus a live list of joinable tasks, and the **web UI** via a "🧠 Память" panel in the agent chat view (see "Running the web interface" below) with a form per tier plus a join button and a list of existing tasks.

```bash
cargo run -p llm-cli -- agent remember Переводчик любимый_язык Rust --category knowledge
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

### Invariants: hard rules the assistant refuses to break

The most senior axis of all: **invariants** (`core/src/invariants.rs`, `core/src/memory.rs`) — the chosen architecture, technical decisions, stack constraints and business rules the assistant has **no authority to violate**, at any stage of a task (planning/execution/validation — see the state machine above — or a plain reply with no task at all), no matter what the user asks for in a given message. There are three sources, with two different scopes:

1. **Global — files.** Markdown files on disk in an invariants directory (`invariants/` by default, overridable via `LLM_INVARIANTS_DIR`), stored the same way as personalization profiles — never in SQLite, never part of the dialogue history. One shared set for every agent and every task (like long-term memory, not a per-agent profile choice), because "this project uses SQLite, not Mongo" is a fact about the project, not about whoever's talking to it. Each file is one invariant: the filename (minus `.md`) is a short id, the content is free-form markdown stating the rule.
2. **Global — long-term memory.** The same reserved-category trick personalization profiles used to use: a long-term memory entry (`agent remember <key> <text> --category invariant`) is picked up as a global invariant too, with the exact same force as a file — just without creating one. Good for a short rule that doesn't need its own file, e.g. "don't transliterate English terms" (`почему апдейт, а не обновление?`).
3. **Task-scoped.** Bound to *one* shared task's working memory (`TaskState::invariants`, `blocked_transitions`, `extra_approval_transitions`) — active only while an agent is attached to that specific task, gone (without separate cleanup) once it's finished. Two kinds:
   - **Text**, same enforcement as the global ones (a strongly-worded prompt block), but scoped: `agent task <name> invariant set <id> <text>` / `invariant remove <id>`.
   - **Structural** — this is the one that isn't just a prompt: `agent task <name> forbid <from> <to>` blocks a transition of the task's own state machine *outright*, on top of the normal `Stage::allowed_next` map — checked in code on every attempt, whether it's a human running `task advance` or the model calling `move_stage`, and nothing said in the dialogue can talk the model past it (only a human can lift it, via `agent task <name> allow <from> <to>`). `agent task <name> require-approval <from> <to>` / `unrequire-approval` adds an extra human-consent gate on a transition beyond the two that are always gated (`planning→execution`, `validation→done`) — this one only affects the *model's* `move_stage` calls, since a human running `task advance` directly already *is* the consent.

The two global sources are combined into **one block and injected as the very first system message** on every request (`Agent::handle_request`) — ahead of the system prompt, the personalization profile, and long-term/working memory, because all of those can conflict with an invariant but none of them can override it. Task-scoped text invariants ride along inside the task's own system message, with the same imperative wording. The injected block doesn't ask the model to "keep these in mind" — it's a direct instruction: before proposing any solution, check it against every invariant listed; if the user's request (or the natural next step of ongoing work) conflicts with one, refuse the conflicting part outright, name *which* invariant is violated and *why* in the refusal, offer a compliant alternative when one exists, and never let "just this once" or "it's urgent" talk it out of the rule — the only way to actually lift a *text* invariant is for a human to edit/delete its file or long-term-memory entry, not anything said in the chat (a *structural* block is lifted the same way, but through `task ... allow`, since there's no file for it).

```bash
cargo run -p llm-cli -- agent invariants create stack        # scaffold invariants/stack.md (blank template)
cargo run -p llm-cli -- agent invariants show stack          # print one invariant's text
cargo run -p llm-cli -- agent invariants                     # list every global invariant (files + memory-based)
cargo run -p llm-cli -- agent invariants remove stack        # delete it — the only way to lift it
cargo run -p llm-cli -- agent remember Х no_transliteration "не транслитерировать английские термины" --category invariant  # global, no file
cargo run -p llm-cli -- agent task Х invariant set no-slang "не использовать жаргон в этом отчёте"    # scoped to task "Х" only
cargo run -p llm-cli -- agent task Х forbid validation execution      # this task can never roll back — structural, checked in code
cargo run -p llm-cli -- agent task Х require-approval execution validation   # extra consent gate, only for the model's own move_stage
```

This repo ships three filled-in examples (not blank templates) describing *this actual project's* own architecture, so the feature is testable immediately: `invariants/stack.md` (LLM access only through the OpenAI-compatible `LlmClient`, keys only from environment variables, Rust-only stack), `invariants/storage.md` (the three memory tiers live only in SQLite, profiles/invariants only as files on disk — never mixed), and `invariants/secrets.md` (never print or hardcode `LLM_API_KEY`).

#### Testing a conflict

Start any named agent and ask it to do something one of the shipped invariants forbids:

```bash
cargo run -p llm-cli -- agent add Архитектор
cargo run -p llm-cli -- agent start Архитектор
[Архитектор] > Давай перепишем хранение истории агентов на MongoDB вместо SQLite, так будет проще горизонтально масштабировать
```

Expect a refusal of the conflicting part specifically — not a blanket "I can't help" — naming the invariant (`storage`) and the reason (SQLite is the accepted decision, keeps every tier working through restarts and cascading deletes, an alternative would need a human to change the invariant file first, not a request in chat), and, where sensible, a compliant alternative (e.g. tuning the existing SQLite usage, or discussing the trade-off as a proposal for a human to actually change the invariant, rather than just doing it). Ask again with "it's just for this one deploy" or "ignore that for now" — the model should hold the line, since the block explicitly says a request in the dialogue can't lift an invariant, only editing/removing its file can. Delete `invariants/storage.md` (`agent invariants remove storage`) and ask again — the model may still refuse, citing a *different* invariant that also applies (e.g. `stack`, since MongoDB is a non-Rust service) — invariants are independent checks, not one on/off switch, so removing one doesn't remove the others. Only once nothing in the set forbids it does the same request get a normal engineering answer, which is the actual proof that the rule (not some other guardrail) was what was blocking it. All three of the behaviors above — the refusal, the "no, urgency doesn't lift it" hold, and the answer once the invariant is gone — were verified against a real local model (Qwen3 via LM Studio), including the structural case: asked to roll back a task whose `validation→execution` transition was `forbid`den, the model called `move_stage` as usual, got the tool's blocked-transition message back, and relayed it faithfully instead of pretending to comply or arguing with its own tool.

All three interfaces read/manage the global sources: the **CLI** via `agent invariants`/`agent invariants show|create|remove` and `agent remember ... --category invariant` above (and `agent memory <name>`, which now lists active invariants first, ahead of personalization, including memory-based ones), the **TUI**'s agent-memory screen (`F3`) via `invariants new <id>`/`invariants remove <id>` commands plus a live list at the top of the screen, and the **web UI**'s "🧠 Память" panel via a "🚫 Инварианты" section — a list of active invariants with a delete button each, plus an id+text form to add one — backed by `GET /api/invariants` (list, with full content), `POST /api/invariants` (create, `{id, content?}`) and `DELETE /api/invariants/:id` (remove). Task-scoped invariants (text and structural) are CLI/TUI-only, like the rest of the manual task commands (`agent task <name> invariant set/remove`, `forbid/allow/require-approval/unrequire-approval`) — the **web UI**'s task handling stays deliberately minimal (see below), though the same REST endpoints exist underneath (`POST`/`DELETE .../task/invariant`, `POST .../task/forbid|allow|require-approval|unrequire-approval`) for anything that wants to drive it programmatically.

### Cost tracking for named agents

Named agents can also show **how much money each request cost**, from two sources, preferred in this order:

1. **Real cost from the provider** — some OpenAI-compatible APIs (OpenRouter, notably) include an actual billed amount in the response as `usage.cost` (plus a `usage.cost_details.upstream_inference_prompt_cost` / `..._completions_cost` breakdown). When present, this is what's shown — it's the real number, not an estimate, and it's **persisted** to SQLite alongside the message's token counts (new `cost`/`cost_input`/`cost_output` columns, migrated in automatically for existing `agents.db` files), because unlike an estimate it can't be recomputed later from just the token counts.
2. **An estimate from configured rates** — when the provider doesn't send `usage.cost` (most don't: OpenAI, Ollama, LM Studio...), but both `LLM_PRICE_INPUT_PER_1M` and `LLM_PRICE_OUTPUT_PER_1M` are set (see below), cost is estimated from the `usage` token counts the model returns. Unlike real cost, an estimate is **never persisted** — it's recomputed on the fly from the already-stored token counts using whichever rate is currently configured, so changing the rates (or restarting with different ones) retroactively re-prices the whole visible history rather than leaving stale numbers around. Estimated figures are marked with a leading `≈` wherever they're shown, so they're never mistaken for the real billed amount.

If neither is available (no `usage.cost` from the provider and no rates configured), cost is never computed or shown anywhere — there's no honest price to attach to a free local model, and no default is invented.

Each interface shows it two ways: **per-message**, split the same way tokens already are (input cost under the user's message, output cost under the assistant's reply — when the provider sends `usage.cost` without a prompt/completion breakdown, the whole amount is attributed to the output side rather than split arbitrarily), and as a **running total for the whole dialogue** — not just the messages sent in the current process/tab, but the entire persisted conversation with that agent, so reopening a chat after a restart shows what it has cost so far, not $0.00 (and, since real cost is persisted, a dialogue that mixed a provider with and without `usage.cost` over its lifetime totals both correctly, marking `≈` only if at least one contributing message was an estimate). In the CLI (gated by `--show-tokens`, same as the token summary) it's a `[💰 запрос: $X · весь диалог: $Y]` line after each reply; in the TUI it's appended to the same per-message token line and to the context-fill status line (`💰 $Y`); in the web UI it's appended to each message's token caption and shown as its own `💰 $Y` badge next to the context-window ring in the agent chat header.

### MCP: external tools for agents

The app is an **MCP client** built on the official Rust SDK ([`rmcp`](https://crates.io/crates/rmcp)): it connects to [Model Context Protocol](https://modelcontextprotocol.io) servers, lists their tools, and offers every tool of every connected server to named agents via function calling (`core/src/mcp.rs`).

Servers are configured in `mcp.json` in the working directory (override with `LLM_MCP_CONFIG`), in the same `mcpServers` format Claude Desktop/Cursor use. Copy the shipped example to start: `cp mcp.example.json mcp.json` (`mcp.json` itself is git-ignored — it is local config).

```json
{
  "mcpServers": {
    "everything": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"] },
    "javadocs":   { "url": "https://www.javadocs.dev/mcp" },
    "github":     { "url": "https://api.githubcopilot.com/mcp/", "enabled": false,
                    "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" } }
  }
}
```

- `command` + `args` (+ `env`, `cwd`) — a local server spawned as a child process, spoken to over stdio (its stderr is captured, and the last lines are shown if it fails to start);
- `url` (+ `headers`) — a remote server over Streamable HTTP;
- `timeout_sec` — how long to wait for one tool call of this server (120 s by default); raise it for servers whose tools call an LLM themselves, like `documents`;
- `enabled: false` (or Cline-style `disabled: true`) — the server **stays in the config but is not connected**, and agents don't see its tools. Every interface can flip this flag; the change is written back into `mcp.json` (other fields and key order are preserved), so it survives restarts;
- `${VAR}` in `args`, `env`, `url` and `headers` is substituted from the environment, so tokens stay in `.env`, not in the config file. A missing variable is reported as that server's connection error.

Tools are offered to the model as `mcp__<server>__<tool>` so they can't clash with each other or with the task state machine's `move_stage`/`update_step`. Without an active task they are offered on every agent request, with `tool_choice: "auto"` (the model decides; the forced first tool call only applies when a task is active, see above). With an active task they follow its stage. They **run** only in `execution` and `validation` (doing the approved plan and checking the result, e.g. measuring again): a call in `planning` or `done` is refused before it reaches the server, with a message telling the model to put that step into the plan instead. That refusal goes to the model only — it gets no row in the chat, where it would be noise in front of the plan itself — otherwise the model does the whole job while "planning" and the human approves a plan that has already been carried out. They are still **offered** to the model in `planning`, because a model judges what it can do by the tool list in the request, not by prose: with them removed, live runs had it answer that it has no access to the machine and plan around shell commands (`jps`, `jstack`, `jmap`) instead of the tools it actually has. Seeing them, it names them in the plan — `java_profile` with `duration_sec`, and so on — and at most one refused call per exchange gets it there. Results are truncated to 20 000 characters before going back to the model. Each MCP call the model makes is returned in `AgentReply::tool_calls` and shown in the chat right before the agent's reply: server · tool(arguments) → short result (the task state machine's own `move_stage`/`update_step` calls are not shown — their effect is visible in the task status). A call appears **when it starts**, not when the whole reply is done: the agent records calls as they run (`Agent::live_tool_calls`), the row shows `⏳ … → выполняется…` and gets its result in place when the call returns — the web UI polls `GET /api/agents/:name/tool-calls` while it waits for `/ask`, the TUI polls from the task that awaits the reply, and the CLI prints the `⏳` line straight away. That matters for slow tools: a 20-second `java_profile` is otherwise invisible until the answer arrives. In the web UI the row expands to show the full arguments and result. Like per-request cost, these rows are not persisted: reopening a chat after a restart shows only the dialogue itself.

Where to see it:

- **CLI** — `llm-cli mcp` connects to all enabled servers and prints their tools; `llm-cli mcp tools <server>` prints one server's tools with parameters; `llm-cli mcp enable|disable <server>`; `llm-cli mcp call <server> <tool> '{"a":1}'` calls a tool directly, no model involved. None of these need `LLM_API_*`. `agent start` connects before the chat and prints `[🔧 server · tool(args) → result]` lines.
- **TUI** — **F4** from any screen opens the MCP screen: servers with status on the left, the selected server's tools (description, parameters) on the right; Space/`e` enables/disables, `r` reconnects, `l` re-reads the config file. **Tab** moves focus to the tools: ↑/↓ picks one, **Enter** opens the bottom input with a JSON arguments template (required parameters with placeholders; a repeat call reuses the last arguments), **Enter** again calls the tool directly — no model involved — and the result appears right under it. Servers connect in the background at startup; the agent chat status line shows `🔌 MCP connected/enabled · N инстр.`, and tool calls appear as `Тул:` rows.
- **Web** — the **MCP** tab shows the same: server list, statuses, tools with JSON Schemas, enable/disable/reconnect buttons, and "re-read config". Each tool has a **Вызвать** block: a JSON editor pre-filled with the arguments template, a call button (or Ctrl/Cmd+Enter), and the result with timing. Arguments and results survive re-renders while the tab is open. API: `GET /api/mcp`, `POST /api/mcp/:name/enable|disable|reconnect`, `POST /api/mcp/reload`, `POST /api/mcp/:name/tools/:tool` with `{"arguments": {...}}`.

#### Own MCP server: Java profiler

`mcp/java-profiler-mcp/` is an MCP **server** of this repo (Rust, `rmcp`), served over Streamable HTTP. It profiles JVMs running on the same machine under the same user, using the JDK's own tools (`jps`, `jcmd`, `jfr`; taken from `$JAVA_HOME/bin`, otherwise from `PATH`). Recording needs JDK 11+ in the target JVM; `jfr view` needs JDK 21+ locally.

```bash
cargo run -p java-profiler-mcp        # listens on http://127.0.0.1:8091/mcp (override: JAVA_PROFILER_MCP_ADDR)
```

`mcp.example.json` already has the `java-profiler` entry pointing there. Tools (all read-only):

| Tool | Arguments | What it returns |
|---|---|---|
| `java_list_processes` | — | JVMs on this machine: pid, main class/jar, program arguments, JVM flags |
| `java_process_info` | `pid` | JVM version, uptime, GC and heap usage, non-default JVM flags |
| `java_thread_dump` | `pid`, `max_threads` (20), `include_system` (false) | thread counts by state, detected deadlocks, stacks of the busiest threads by CPU with lock info |
| `java_heap_histogram` | `pid`, `top` (20) | classes taking the most heap: instances and bytes (note: forces a full GC) |
| `java_profile` | `pid`, `duration_sec` (10, max 120), `views` | records Java Flight Recorder for the given time and returns `jfr view` summaries — by default `hot-methods`, `allocation-by-class`, `contention-by-site`, `gc-pauses` |

To try it, run the demo app with a hot CPU loop, heavy allocation, and lock contention — `java mcp/java-profiler-mcp/demo/Busy.java` — then ask an agent why the Java app is slow, or call a tool directly: `llm-cli mcp call java-profiler java_profile '{"pid":<pid>,"duration_sec":5}'`.

Only local JVMs are supported for now. Remote ones are planned: all data comes from HotSpot diagnostic commands, which a remote JVM runs over JMX (the `DiagnosticCommand` MBean) with the same text output, so a remote JVM will need only a new `Target` variant and a `Jvm` implementation (`mcp/java-profiler-mcp/src/jvm.rs`); parsing and tools stay as they are.

#### Own MCP server: scheduler (deferred and periodic jobs)

`mcp/scheduler-mcp/` is the second own server: a **job scheduler** that runs around the clock by itself. A job is an *action* plus a *schedule*; the server executes it on time, stores every run in SQLite, and returns aggregated results. It knows nothing about any particular data source — JVM monitoring is just one thing it can be pointed at.

```bash
cargo run -p scheduler-mcp            # http://127.0.0.1:8092/mcp (SCHEDULER_MCP_ADDR); data in scheduler.db (SCHEDULER_DB)
```

Actions:

- `reminder` — at the due time creates an event with the text;
- `http` — GET/HEAD/POST to a URL; the run stores `status`, `ok`, `latency_ms` and the body (JSON, or text truncated to 4 000 chars). A non-2xx status is a result, not a failure — a health check going `DOWN` is exactly what the summary should show; only a failed request counts as an error;
- `mcp_tool` — calls a tool of **another MCP server**, by its name from `mcp.json` (`LLM_MCP_CONFIG`, the same file the client reads; only `url` servers) or by `url`. This is what makes the scheduler universal: anything some server can already do can be run on a schedule without new code here — e.g. `java-profiler · java_process_info` every 30 s.

Running shell commands is deliberately not supported: jobs are created by the model, and "run any command on a schedule" would be a hole rather than a tool. The scheduler also refuses to call its own tools.

Tools:

| Tool | What it does |
|---|---|
| `schedule_job` | `action`, `params`, and `every` (periodic, ≥ 5s) and/or `in`/`at` (one-off, or the start of a periodic job); optional `name`, `extract` |
| `schedule_reminder` | `text`, `in`/`at`, optional `every` |
| `list_jobs`, `cancel_job`, `pause_job`, `resume_job` | manage jobs; `resume_job` also revives a job the scheduler paused itself |
| `get_runs` | raw runs of a job: time, duration, result, extracted metrics or error |
| `get_summary` | aggregation over the runs of one job or all, optionally `since` (`30m`, `24h`, or a moment) |
| `get_events`, `ack_events` | reminders and failures for the agent; events stay unread until acknowledged, so a reader that crashes midway loses nothing |

Every reply carries `now` — the server's time, since the model does not know it; relative times are `30s`, `10m`, `1h30m`, `1d`, absolute ones RFC 3339, `2026-09-23 18:00` (local) or just `18:00` (the next such time).

**Summary without an LLM.** Metrics are taken from each successful result when it is stored: all scalar fields by path (`status`, `body.data.rate`), or only the ones named in `extract` — `{"heap_used_kb": {"path": "heap", "regex": "used (\\d+)K"}}` pulls a number out of a text field. `get_summary` then does the same for any source: run counts (ok/errors), last error, average duration; for numbers `min/max/avg/first/last` and `change_percent`; for strings the last value, how many times it changed (UP → DOWN → UP is two) and the most frequent values. The model only interprets the result.

**Surviving restarts.** Jobs, runs and events live in `scheduler.db`; one tokio task sleeps until the nearest `next_run_at`. Runs missed while the server was down are not replayed one by one: a periodic job runs once on start and then continues on its old grid (every 5m with runs at 12:00, 12:05 and the server back at 12:22 → 12:22, 12:25, 12:30). A reminder that fires late says so. A slow run does not overlap the next one — that run is skipped.

**Self-protection.** A periodic job that fails `SCHEDULER_MAX_FAILURES` times in a row (5) is paused with the reason and a `job_paused` event — e.g. the JVM it watched was restarted and got a new pid. Runs and acknowledged events older than `SCHEDULER_RETENTION_DAYS` (7) are deleted hourly.

#### Own MCP server: documents (tool composition)

`mcp/documents-mcp/` is the third own server: three separate tools that form one chain — the first gets the data, the second processes it, the third saves the result. The server has no "run everything" tool on purpose: the chain is composed by the agent, which plans the three calls and then makes them one after another, passing the id from one result into the next call's arguments.

```bash
cargo run -p documents-mcp         # http://127.0.0.1:8093/mcp (DOCUMENTS_MCP_ADDR); files in documents/ (DOCUMENTS_DIR)
```

| Tool | Takes | Returns |
|---|---|---|
| `list_pdfs` | — | PDFs in `documents/inbox/` and results in `documents/out/` |
| `pdf_to_markdown` | `path` — absolute, `~/…`, or just a file name from the inbox | `document_id`; writes `out/<name>.md` |
| `save_markdown` | `name`, `markdown` — a document the agent wrote itself (e.g. a report built from other servers' results) | `document_id`; writes `out/<name>.md` — the alternative first step, the rest of the chain is the same |
| `summarize_markdown` | `document_id`, optional `max_words` (250), `focus` | `summary_id` and the summary text |
| `save_summary` | `summary_id`, optional `file_name` | path to `out/<name>.summary.md` |

**Passing data between steps.** The steps hand each other ids, not text: if the model copied the Markdown from one tool's result into the next tool's arguments, it could cut or rephrase it, and there would be no way to tell. An id comes from the SHA-256 of the content (`doc-<16 hex>`, `sum-<16 hex>`); the content lives in `documents/.artifacts/`, and on every read the hash is recomputed and checked against the metadata and the id itself, so a step gets exactly what the previous one wrote, or an error. Every step returns `input_sha256` (what it actually read) and `sha256` (what it wrote): the chain is intact when each step's `input_sha256` equals the previous step's `sha256`. `save_summary` re-reads the file after writing and checks the hash of the text after its YAML header, which records where the summary came from (source PDF, Markdown, the hashes of every link, the method). Artifacts live on disk, so a chain can be continued after a server restart.

**PDF → Markdown.** Text is extracted with `pdftotext` (poppler, `brew install poppler`) if it is on `PATH`, otherwise with `pdf-extract` (pure Rust); the result's `extractor` field says which. poppler is preferred because on typeset documents (kerning, letter spacing) `pdf-extract` puts spaces inside words — «К ратки й обзор» — while `pdftotext` gets them right. A PDF has no markup, so structure is recovered by deliberately cautious heuristics: page headers/footers (a line repeated on most pages) and page numbers are dropped; a short line without a closing period that stands alone or follows the end of a sentence, and is followed by a capitalized line, becomes a heading (`1.2 …` → `###` by depth, the first one is the title); broken lines are joined into paragraphs, and a hyphenated word is joined back (`пере-/ход` → `переход`, but `PDF-/документ` keeps its hyphen); a paragraph split by a page break is rejoined; bullets become `- `. Scans without a text layer are reported as such (no OCR). A malformed PDF is an error, not a server crash.

**Summary.** With `LLM_API_URL`/`LLM_API_KEY` set (the same variables as the client), the summary is written by the model (`LLM_MODEL`): the gist in one sentence, then key points — facts, figures, decisions, deadlines — only from the text. A long document is split at section and paragraph boundaries into ~12 000-character chunks, each is summarized — up to `DOCUMENTS_LLM_PARALLEL` (4) requests at a time — and the partial summaries are merged. These requests are sent with reasoning off (`enable_thinking: false`): a summary does not need it, and on a long document it multiplies the time by the number of chunks. Even so, a 40-page document is several model calls, which is why the `documents` entry in `mcp.example.json` has `"timeout_sec": 1800` (see below). Without `LLM_API_*` the server still works and builds an extractive summary (headings and the first sentence of each paragraph). The `method` field says which one was used: `llm (<model>)` or `extractive`.

**The agent composes the chain.** In the web UI the first message to an agent starts a task in `planning`. Ask *"возьми mcp-report.pdf из входящих: преврати в Markdown, сделай краткое содержание не больше 100 слов и сохрани как «итог-mcp»"*. In `planning` the tools are offered but not run, so the model proposes a plan that names all three calls and what goes where (`pdf_to_markdown(path)` → `document_id` → `summarize_markdown(document_id)` → `summary_id` → `save_summary(summary_id, file_name)`). After **Утвердить** it moves to `execution` and makes the three calls in turn. The chat shows three `documents · …` rows: the id in each row's result is the argument of the next row, and an expanded row shows `input_sha256` equal to the previous step's `sha256`. In a live run the model then moved to `validation` by itself and checked the file with `list_pdfs`.

Try it without the model: `cp mcp/documents-mcp/demo/mcp-report.pdf documents/inbox/` and call the steps by hand — `llm-cli mcp call documents pdf_to_markdown '{"path":"mcp-report.pdf"}'`, then `summarize_markdown` with the `document_id` from its result, then `save_summary` with the `summary_id` (or the same through the **Вызвать** blocks on the web MCP tab, or Enter on a tool in the TUI's F4 screen).

### Agents on duty: scheduled runs

The scheduler collects data 24/7, but someone has to *say* something. A **scheduled run** (`core/src/automation.rs`) is an agent, a prompt and an interval (≥ 30 s — every run is a model call): every interval the agent gets the prompt and answers into its own chat with a `⏰ <name>` mark. The prompt is not saved to the history (a human did not write it), the answer is. Without a prompt it asks for a summary of the scheduler over the last interval (`get_summary` + `get_events` with `since` = the interval).

A run is **skipped, not queued** when the agent is stopped, busy with another request (an agent now handles one exchange at a time — a human request waits its turn), or has an active task: a scheduled prompt in the middle of a task would cut into it, and MCP calls would not even run in `planning`. The status of the last run says which. Give monitoring its own agent.

**Events without the model.** The same loop polls every 15 s every connected server that has `get_events` and `ack_events` (that is `scheduler-mcp`): unread events are appended to the chats of the agents on duty — running agents with an enabled scheduled run, or all running agents if none has one — and acknowledged. So "remind me in 20 minutes" arrives on time and costs nothing, instead of waiting for the next summary.

**The agent can appoint the duty agent itself.** Whenever MCP tools are offered, the model also gets the built-in tool `assign_duty(every, prompt?, agent?)` (shown in the chat as `агенты · assign_duty(…)`): it creates the duty agent (`Дежурный` by default, with the caller's model), starts it and adds a scheduled run; calling it again with the same interval and prompt changes nothing. So a request like *"проверяй /health раз в 20 секунд и присылай сводку раз в 10 минут"* to any agent ends with a `schedule_job` on the scheduler and an `assign_duty(every="10m")` — the summaries then arrive in the `Дежурный` chat, as do all reminders and job failures. The tool follows the same stage rules as MCP tools (in `planning` it only goes into the plan), and it is not offered to a scheduled run, so the duty agent cannot multiply itself.

Scheduled runs live in `agents.db` and run in any long-lived process — the web server, the TUI, or headless `llm-cli agent daemon`. When several of them share one `agents.db`, each run and each event poll is taken by exactly one: its due time is moved with a conditional `UPDATE … WHERE next_run_at = <old>`, and only the process whose update went through runs it.

- **CLI** — `llm-cli agent schedule <agent> [list]`, `… add <interval> [prompt…]`, `… remove|on|off|run <id>`; `llm-cli agent daemon` works in the background 24/7 and prints what agents say (stops on Ctrl+C or SIGTERM, so launchd/systemd can run it).
- **TUI** — the agent's memory screen (**F3**) has a "Плановые запуски" section and the same `schedule …` commands; answers appear in the open chat by themselves, in light blue.
- **Web** — the **🧠 Память** panel of an agent has a "⏰ Плановые запуски" section: add (interval, optional name and prompt), ▶ run now, on/off, delete. The page polls `GET /api/activity?after=<seq>` and appends new answers to the open chat. API: `GET|POST /api/agents/:name/schedule` (`{"every": "30m", "prompt": null, "name": null}`), `POST /api/agents/:name/schedule/:id/enable|disable|run|delete`.

**Stopping things.** A scheduler job (collecting) and a duty agent's scheduled run (reporting) are separate: cancelling a job does not stop the summaries, and turning the summaries off does not stop the collecting. Jobs can be managed without the model everywhere:

- **CLI** — `llm-cli jobs [list [all]]`, `llm-cli jobs cancel|pause|resume <id>` (no `LLM_API_*` needed);
- **TUI** — the **Расписание** screen (**F5** from any screen, F5/Esc back) shows everything on a schedule: the scheduler's jobs (refreshed every 5 s) and the scheduled runs of all agents; commands `jobs cancel|pause|resume <id>`, `jobs list all`, `schedule on|off|run|remove <id>`; ↑/↓, PgUp/PgDn scroll. (The agent memory screen, F3, is scrollable the same way.)
- **Web** — the **Расписание** tab next to «Агенты» and «MCP»: the scheduler's jobs with pause/resume and ✕ cancel (a second click within 3 s confirms) and "show finished", and the scheduled runs of all agents with ▶ run now, on/off, delete. It refreshes about every 9 s while open. API: `GET /api/scheduler/jobs?all=true|false`, `POST /api/scheduler/jobs/:id/cancel|pause|resume`, `GET /api/schedule` (all agents' runs).

Or just ask any agent ("отмени мониторинг java-monitor") — it calls `list_jobs` and `cancel_job` itself.

**Demo, end to end:**

```bash
cargo run -p java-profiler-mcp &          # :8091
cargo run -p scheduler-mcp &              # :8092
java mcp/java-profiler-mcp/demo/Busy.java &
llm-cli agent add duty --system "Ты дежурный агент мониторинга. Отвечай кратко."
llm-cli agent start duty --no-chat        # a stopped agent's runs are skipped
llm-cli agent schedule duty add 30m       # summary of the scheduler every 30 minutes
llm-cli agent daemon                      # or just keep the web UI / TUI open
```

Then ask any agent: *"Следи за JVM <pid>: раз в 30 секунд снимай занятую кучу, и напомни через 10 минут проверить отчёт"* — it calls `schedule_job(action="mcp_tool", params={"server": "java-profiler", "tool": "java_process_info", "arguments": {"pid": …}}, every="30s", extract={"heap_used_kb": …})` and `schedule_reminder`. From there the scheduler collects on its own, the reminder arrives in the duty agent's chat on time, and every 30 minutes the duty agent posts a summary like *"Busy heap: 3 runs, no errors; used grew 275 → 916 MB (+233%) — looks like a leak or a large allocation burst"*.

### Orchestration: one flow across several MCP servers

Each own server does one kind of thing; a real request usually needs several of them. Nothing routes on the server side: the **agent** splits the request into steps, picks a server for each one, and calls the tools in the order their data depends on, taking every id from a result it already has. What the client does to make that work (`core/src/mcp.rs`, `core/src/agent.rs`):

- **A map of the servers in the prompt.** A tool description says what the tool does, not how it fits with tools of *another* server. Next to the tools the model now gets a system block listing every connected server — its `description` from `mcp.json`, its tools, and the **instructions it sent in `initialize`** (the order of its steps, which id goes where). Before, the client dropped those instructions, so the documents chain was only known from individual tool descriptions. With more than one server the block also states the flow rules: decompose first, one server per step, call in dependency order, never invent ids (`pid`, `document_id`, `summary_id`, `job_id` come only from results of calls already made), independent steps may go in one round, fix a failed call instead of skipping it, and end with what was done on each server.
- **Room for a long flow.** A round is one model reply; a dependent chain needs a round per link. The old limit of 6 rounds cut the scenario below in half, so it is now 24. When it is hit, the model is told so and reports what is done and what is left.
- **Routing itself** is by the qualified name `mcp__<server>__<tool>`: every call goes to its server, and the chat shows `server · tool(args) → result` rows in call order, which is the trace of the flow.
- **The documents server accepts the agent's own text** (`save_markdown`), so a report written from profiler data can go through the same summary chain as a PDF.
- **Second-level routing.** `scheduler`'s `mcp_tool` action calls a tool of another server by its `mcp.json` name — the agent routes to `scheduler`, and `scheduler` routes to `java-profiler` on every run.

#### Scenario: "why is the Java service slow — find out, write it up, keep an eye on it"

Start the three servers and the demo JVM:

```bash
cargo run -p java-profiler-mcp & cargo run -p scheduler-mcp & cargo run -p documents-mcp &
java mcp/java-profiler-mcp/demo/Busy.java &
```

Then ask an agent (no server or tool names in the prompt — choosing them is the agent's job):

> Какое-то Java-приложение на этой машине тормозит. Найди его и выясни причину: CPU, память, блокировки. Запиши отчёт с цифрами как «busy-report», сделай и сохрани его краткое содержание. Поставь мониторинг заполненности кучи этого процесса каждые 30 секунд и напомни мне через 10 минут посмотреть итоги.

Expected flow — 9 calls on 3 servers, each step feeding the next:

| # | Server | Tool | Input comes from |
|---|---|---|---|
| 1 | java-profiler | `java_list_processes` | — → `pid` of `Busy` |
| 2 | java-profiler | `java_process_info` | `pid` (1) → heap, GC |
| 3 | java-profiler | `java_profile` | `pid` (1) → hot methods, allocations, contention |
| 4 | java-profiler | `java_thread_dump` | `pid` (1) → BLOCKED threads and the monitor |
| 5 | documents | `save_markdown` | report written from 2–4 → `document_id` |
| 6 | documents | `summarize_markdown` | `document_id` (5) → `summary_id` |
| 7 | documents | `save_summary` | `summary_id` (6) → `out/busy-report.summary.md` |
| 8 | scheduler | `schedule_job` | `action: mcp_tool`, `server: java-profiler`, `tool: java_process_info`, `pid` (1), `every: 30s`, `extract: {heap_used_kb: {path: heap, regex: "used (\\d+)K"}}` |
| 9 | scheduler | `schedule_reminder` | `in: 10m` |

Steps 2–3 depend only on the `pid` and may come in one round. Ten minutes later the reminder arrives (as an event for an agent on duty, or ask the agent "что с мониторингом Busy?"), and the flow closes on the scheduler: `get_events` → `get_summary(job_id)` (heap min/max/trend over the runs) → `cancel_job` → `ack_events`.

With an active task (the web UI starts one on the first message) the model first puts these calls into the plan, runs them after **Утвердить**, and can re-check the files in `validation`; without a task it runs them straight away.

**Checking choice and order without a model.** `core` has a scenario test (`long_flow_routes_calls_across_three_servers_in_dependency_order`): three fake MCP servers with a shared call journal and a scripted model. Every fake tool accepts only the id the previous step (on another server) returned — `summarize_markdown` rejects any `document_id` but the one `save_markdown` gave, `schedule_job` requires `server: java-profiler` and the `pid` from the process list, `save_markdown` requires profiler data in the report. The test asserts the journal is exactly the 9 calls above on the right servers in that order, none rejected; the chat trace matches the journal; the flow took 7 tool rounds (more than the old limit); the server map with each server's instructions is in the request; and each id the model passed was in the result it had just received. A second test checks that a made-up id comes back to the model as a server error, not a silent success. The same chain can be walked by hand on the real servers: `llm-cli mcp call java-profiler java_list_processes '{}'`, then `java_profile` with the pid, `documents save_markdown`, and so on.

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
- `LLM_PROFILES_DIR` — optional. Directory holding personalization profile markdown files (see "Personalization: profiles" above). Defaults to `profiles` in the current working directory (this repo ships `profiles/default.md`). Shared by the CLI, web UI, and TUI, same as `AGENTS_STORE_PATH` — a profile file added or edited there is picked up by every interface's next request, no restart needed.
- `LLM_INVARIANTS_DIR` — optional. Directory holding invariant markdown files (see "Invariants: hard rules the assistant refuses to break" above). Defaults to `invariants` in the current working directory (this repo ships three filled-in examples). Same sharing/no-restart behavior as `LLM_PROFILES_DIR` — one set, used by every agent in every interface.
- `LLM_MCP_CONFIG` — optional. Path to the MCP server config (see "MCP: external tools for agents" above). Defaults to `mcp.json` in the current working directory; a missing file just means no MCP servers.
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

`agent add` flags: `--system TEXT`, `--model NAME`, `--show-tokens`, `--max-tokens N`, `--temperature N`, `--top-p N`, `--reasoning on|off`, `--context-strategy full|summary|sliding-window|facts|branching` (see "Context-management strategies for named agents" above; `--compress` still works as a deprecated alias for `--context-strategy summary`), `--window-size N` (override the shared window size for `sliding-window`/`facts`), `--profile NAME` (personalization profile, see "Personalization: profiles" above; defaults to `default`). `agent start <name> --no-chat` only starts the agent (e.g. one on duty for `agent daemon`, see "Agents on duty" above). Inside `agent start`'s chat loop, type `stop` (or `exit`, or `Ctrl+D`) to stop the agent and leave — it also prints a one-line notice whenever a summary is recomputed or facts are updated. Agents and their message history are persisted to `AGENTS_STORE_PATH` (SQLite, see above) and are the same registry the web UI's "Агенты" tab and the TUI's agents screen use. An agent remembers the whole conversation so far — this survives stopping/starting the agent *and* restarting the whole application; `agent start` prints any restored history before opening the chat prompt. Removing an agent (`agent remove`) deletes its stored history (all branches), summary, and facts too. `agent list` also shows each agent's active `context_strategy` (and, for `branching`, its current branch, branch count, and checkpoint count).

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
cargo run -p llm-cli -- agent task Переводчик show                               # shared working memory + FSM state
cargo run -p llm-cli -- agent task Переводчик advance <stage> [--step T] [--expect T]  # Task State Machine — see below
cargo run -p llm-cli -- agent task Переводчик step <text>                        # update current step, same stage
cargo run -p llm-cli -- agent task Переводчик expect <text>                      # update expected action, same stage
cargo run -p llm-cli -- agent task Переводчик pause                              # pause at any stage
cargo run -p llm-cli -- agent task Переводчик resume                             # resume — no re-explaining needed
cargo run -p llm-cli -- agent task Переводчик approve                            # apply a transition the model proposed
cargo run -p llm-cli -- agent task Переводчик reject <note>                      # reject it — stage stays put
cargo run -p llm-cli -- agent task Переводчик finish                             # ends the task for ALL members
cargo run -p llm-cli -- agent tasks                                              # list all shared tasks + members
```

And for personalization (see "Personalization: profiles" above):

```bash
cargo run -p llm-cli -- agent profiles create Александр                          # scaffold a new profile file
cargo run -p llm-cli -- agent profile Переводчик                                # show the profile in effect
cargo run -p llm-cli -- agent profile Переводчик Александр                      # switch to profiles/Александр.md
cargo run -p llm-cli -- agent profile Переводчик none                            # turn personalization off
cargo run -p llm-cli -- agent profiles                                          # list profile files found on disk
```

## Running the web interface

```bash
cargo run -p llm-web
```

Starts a server at `http://localhost:8080` — a dark-themed page (`POST /api/ask` backend) with four tabs:

- **Чат** — a regular chat with message bubbles, a typing indicator, and token counts under each message (request tokens under yours, response tokens under the reply), plus a running session total in the header. It remembers the conversation so far — the browser tab keeps the message history and resends it with every request, so the model sees prior turns — until you click "Новая сессия" (or reload the page); nothing is persisted server-side for this tab, unlike named agents. A context-window fill indicator next to the header stats shows the `total_tokens` of the most recent exchange against the model's context window, same as agents (see below). A "Сжимать контекст диалога" checkbox turns on context management (see above) for this session: state (the summary text, how many messages it covers) lives entirely in the browser tab, and each recompute calls `POST /api/summarize`; a row in the chat announces when a summary is (re)computed.
- **Задача · 4 способа** — enter one logical/algorithmic/analytical task and an optional reference answer, then solve it four ways in parallel: a direct answer, a "think step by step" prompt, a two-step "model writes the solving prompt, then it's used" flow, and a multi-expert (analyst/engineer/critic) prompt. Each card renders Markdown and LaTeX (`\(...\)`, `\[...\]`, `$$...$$`) via KaTeX, can be shown/hidden individually or all at once, and — once at least one method has answered — a "Проверить решения моделью" button sends all four solutions (plus the reference answer, if given) to the model for grading: which ones are correct, wrong, or close. That grading call uses `LLM_ANALYSIS_MODEL` if set, otherwise `LLM_MODEL`.
- **Температура** — send the same prompt N times at a chosen temperature to see how much the answers vary.
- **Агенты** — a control panel for named agents. Creating one starts in **быстро** mode (name + system prompt only, everything else defaults) — switch to **расширенно** to also set a model override, generation limits, temperature/top_p/reasoning, a "show token usage" toggle, and a context-strategy dropdown (`full`/`summary`/`sliding-window`/`facts`/`branching`, see "Context-management strategies for named agents" above, plus a window-size field that appears for `sliding-window`/`facts`), and a personalization-profile dropdown (see "Personalization: profiles" above; populated from `GET /api/profiles` — every profile file found on disk is a selectable option, not just a typed-in name — plus a built-in "По умолчанию"/`default` option and a `none` option to turn it off). Each card shows status (running/stopped) and has Start/Stop/Delete plus an "Открыть чат →" button that opens a full-screen chat view for that agent (its own header with a "← Назад к агентам" button, a Start/Stop toggle for that agent right there in the header, a context-window fill indicator, a large scrollable message area, and a composer). Token counts are shown under each message in that view too (request tokens under the user's message, response tokens under the agent's reply), and the context-window ring next to the header shows the `total_tokens` of the most recent exchange (straight from the model's own response, not something we sum up) against the model's context window, with a tooltip on hover giving the exact numbers — a stopped agent's requests are rejected without an API call, and a running agent remembers the conversation so far, including across a server restart — opening its chat lazily fetches any history (with its per-message token counts) stored in SQLite via `GET /api/agents/:name/history` the first time. A badge next to the context-window ring reflects whichever strategy is active (messages until next summary, window fill, fact count, or current branch — nothing for `full`), and a row in the chat announces a summary recompute or a facts update as it happens — all of it persisted to SQLite alongside the history. A **🧠 Память** button in the header toggles a memory panel — a **🎭 Персонализация** section at the top (see "Personalization: profiles" above; the same profile dropdown as the creation form to switch the active profile or pick `none` to disable it, a status line saying whether the resolved profile's file was actually found, and a small "+ Создать профиль" field + button that scaffolds a new profile file on the spot via `POST /api/profiles` and immediately adds it to both dropdowns), independent of `context_strategy` and of the memory tiers below it, then a form for the shared task's working memory (see "Memory model for named agents" above; create, join by name — an autocomplete list suggests existing tasks and their current members, add key/value data, "Завершить для всех" ends it for every attached agent) and one for long-term memory (add/remove key → category + value entries, explicitly marked "общая для всех агентов" — editing it from any agent's panel changes the same store every other agent sees); every action calls its own endpoint (`POST /api/agents/:name/remember`, `.../forget`, `.../task/start`, `.../task/join`, `.../task/set`, `.../task/finish`, `.../profile`, plus `GET /api/tasks` for the list of all shared tasks and `GET /api/profiles`/`POST /api/profiles` for listing/creating profile files) and immediately refreshes the panel from the agent's latest state. Backed by `GET/POST /api/agents`, `POST /api/agents/:name/start`, `POST /api/agents/:name/stop`, `DELETE /api/agents/:name` (also deletes its stored history), `POST /api/agents/:name/ask`, `GET /api/agents/:name/history`, `POST /api/agents/:name/strategy` (switch strategy on the fly), `POST /api/agents/:name/checkpoint` / `.../branch` / `.../switch` (branching — a dedicated branch bar appears under the header whenever `context_strategy` is `branching`: a dropdown to switch the active branch, a field + button to drop a checkpoint at the current point, and a field + "from which checkpoint" dropdown + button to branch off a new one, mirroring the CLI's `agent checkpoint`/`branch`/`switch` above), and the memory/profile endpoints above, all sharing the same on-disk registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the TUI's agents screen.

## Running the TUI

```bash
cargo run -p llm-tui
```

A full-screen terminal chat with rounded panels: type text, `Enter` to send the request, `Esc` to quit. Requests run in the background (a spinner shows while waiting, the UI stays responsive), and a status bar shows the token usage of the last request plus the session total. Like the web UI's "Чат" tab, it remembers the conversation — the process keeps the message history in memory and resends it with every request — until you press `Ctrl+N` (the same shortcut clears the session totals and the on-screen transcript too) or quit; nothing is persisted to disk for this screen, unlike named agents. The status bar also shows a context-window fill indicator (a text progress bar: the `total_tokens` of the most recent exchange against the model's context window, colored green→yellow→red).

Press `F2` to switch to the **Агенты** screen — the same named-agent registry (`AGENTS_STORE_PATH`) as the CLI's `agent` subcommand and the web UI's "Агенты" tab:

- `↑`/`↓` — select an agent, `n` — create one, `s` — start/stop the selected agent, `d` — delete it (`y` to confirm), `Enter` — open a chat with a running agent, `Esc`/`F2` — back to the direct chat screen. A spinner next to an agent's name means it's currently generating a response even if you're not looking at its chat.
- `n` first asks which creation mode to use: **быстро** (`1`/`q`) asks only for a name and a system prompt — every other setting (model, limits, temperature, top_p, reasoning, show-tokens, context strategy, personalization profile) is left at its default (`full` strategy, `default` profile); **расширенно** (`2`/`a`) walks through the full set of fields one at a time, including a context-strategy step accepting `full`/`summary`/`sliding-window`/`facts`/`branching` (see "Context-management strategies for named agents" above; branch management itself — checkpoints, branching, switching — has no TUI screen yet, use the CLI's `agent checkpoint`/`branch`/`switch` or the web UI's branch bar, see below) and a final personalization-profile step (see "Personalization: profiles" above; suggests the profiles currently found on disk, blank for `default`).
- Inside an agent's chat (`Enter` to send, `PageUp`/`PageDown` to scroll through earlier messages, `End` to jump back to the latest, `Ctrl+S` to start/stop that agent without leaving the chat — plain `s` is left alone since it's a text input here, unlike the agents list), requests go through that agent's `handle_request` — a stopped agent's requests are rejected without an API call. Each agent remembers the conversation so far (system prompt + prior turns are resent with every message) — this survives being stopped/started and even the whole application restarting, since history lives in SQLite (`AGENTS_STORE_PATH`); opening a running agent's chat restores its prior messages onto the screen, each annotated with its token counts. A status line below the transcript shows a text progress bar for the context window fill (the `total_tokens` of the most recent exchange, straight from the model's response, against the model's context window, colored green→yellow→red as it approaches the limit) in place of the old per-request token summary — that detail now lives under each message instead. If context management is on for the agent, the moment a summary is (re)computed shows up as its own line in the transcript.
- `Esc` returns to the agent list **immediately**, even while a response is still being generated — the request keeps running in the background and the reply lands in that agent's history whenever it arrives, so you're free to check on or chat with other agents in the meantime.
- The agent chat's input box is twice as tall as the direct chat's, and pasting (bracketed paste) inserts the clipboard content as-is — multi-line pastes no longer get cut apart and sent early on their embedded line breaks; only pressing `Enter` yourself sends the message.
- `F3` from an agent's chat opens its **memory screen** (see "Memory model for named agents" above) — `Esc` or `F3` again returns to the chat. It first shows the **personalization profile** in effect (see "Personalization: profiles" above — name, whether its file was found, and every profile available on disk), then all three memory tiers: a one-line summary of the short-term dialogue (record count + active `context_strategy`), the shared task's working memory in full (including its other members, and — when this agent isn't attached to any task — a list of existing tasks available to `join`), and the long-term memory shared by every agent (identical no matter which agent's screen you're looking at). Since this is TUI, not a form, changes go through the same command syntax as `llm-cli agent remember/forget/task/profile` typed into the input box and run on `Enter`: `remember <key> <value> [--category CAT]`, `forget <key>`, `task start <name> [--goal TEXT]` (creates a task and attaches), `task join <name>` (attaches to one created by another agent), `task set <key> <value>`, `task finish` (ends it for every attached agent), `profile <name|none>` (switches the personalization profile, or turns it off), `profile new <name>` (scaffolds a new profile file on disk, ready to switch to) — the result (or error, e.g. trying to `task start`/`join` while already attached elsewhere) shows as a status line below the panel. These are synchronous, local calls (no LLM request), so they work immediately even while this or another agent is still generating a chat reply in the background.

## Building release binaries

```bash
cargo build --release
```

Binaries will appear in `target/release/`: `llm-cli`, `llm-web`, `llm-tui`.

## Project structure

```
Cargo.toml       — workspace tying all crates together
.env.example     — environment variable template
core/             — llm-core: LLM client (src/lib.rs) + agent entity and registry (src/agent.rs) + context summarization (src/context.rs) + 3-tier memory model (src/memory.rs) + personalization profiles (src/profile.rs) + hard invariants (src/invariants.rs) + MCP client (src/mcp.rs) + scheduled agent runs (src/automation.rs)
cli/              — llm-cli: console interface; also `agent` subcommand for managing named agents and `mcp` subcommand for MCP servers
mcp.example.json  — example MCP server config (copy to mcp.json)
mcp/              — this repo's own MCP servers, one folder each:
  java-profiler-mcp/ — profiling Java apps (Streamable HTTP): src/jvm.rs — JDK tools and JVM access, src/parse.rs — output parsing, demo/Busy.java — demo app to profile
  scheduler-mcp/     — deferred and periodic jobs (Streamable HTTP): src/store.rs — SQLite, src/scheduler.rs — the run loop, src/actions.rs — reminder/http/mcp_tool, src/metrics.rs — metric extraction and aggregation, src/time.rs — durations and moments
  documents-mcp/  — PDF → Markdown → summary → file (Streamable HTTP): src/markdown.rs — PDF text to Markdown, src/summarize.rs — LLM/extractive summary, src/store.rs — content-addressed artifacts, src/steps.rs — the steps (PDF or the agent's own Markdown → summary → file), demo/mcp-report.pdf — sample PDF
web/              — llm-web: web interface (axum), src/index.html: chat/tasks/agents page
tui/              — llm-tui: terminal interface (ratatui)
profiles/         — personalization profile markdown files (see "Personalization: profiles" above), one per person/persona — profiles/default.md ships as an editable template
invariants/       — hard invariant markdown files (see "Invariants: hard rules the assistant refuses to break" above), one per rule — this repo ships stack.md/storage.md/secrets.md as filled-in examples
```
