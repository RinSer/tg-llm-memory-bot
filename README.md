# tg-llm-memory-bot

A locally-run Telegram bot server for talking to LLMs, with persistent conversation memory.

Each authorized user can run multiple named **sessions**, each with its own message history and its own LLM provider/model. A session's provider and model can be switched at any time without losing its history, since history is stored as plain text, not tied to any provider SDK type. Memory is split into two tiers, keyed only by Telegram user ID:

- **Session memory**: a per-session, windowed message history stored in SQLite (last N messages; N is configurable). Summarizing what falls out of the window instead of dropping it is a planned follow-up, not yet implemented.
- **Global memory**: a cross-session, long-term memory shared across all of a user's sessions. A background worker per user summarizes conversation into concise **facts**, embeds each with a dedicated embedding model, and stores it as a vector in the `global_memory` SQLite table. On each chat turn, the user's message is embedded and the most similar facts (above a similarity threshold) are injected as a system message — so the model gets relevant long-term context without resending whole transcripts. See "Global memory" below for how it's triggered, configured, and compacted.

## Environment variables

Configure these either as real environment variables or by adding a `.env` file in the project root — `.env` is loaded on startup if present, but it's entirely optional (e.g. a built binary deployed standalone can just have its environment set directly instead).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `TG_BOT_API_TOKEN` | yes | — | Telegram bot token from [@BotFather](https://t.me/BotFather). |
| `OPENAI_API_TOKEN` | yes | — | OpenAI API key. Required at startup even if you only ever use Ollama, since a session can be switched to OpenAI via `/model` at any time. |
| `OLLAMA_BASE_URL` | no | `http://localhost:11434` | Base URL of a running Ollama server. |
| `OLLAMA_KEEP_ALIVE` | no | `-1s` | Sent as Ollama's `keep_alive` on every request — how long the model stays loaded in memory/VRAM after a request. Must be a Go duration string with a unit (e.g. `5m`, `1h`) — a bare `-1` fails with "missing unit in duration" (Ollama parses it via Go's `time.ParseDuration`; only a raw JSON number, which this client can't send, accepts a unitless value). A negative duration like `-1s` keeps it loaded indefinitely (no reload latency on the next message, at the cost of holding VRAM even when idle); Ollama's own default if you unset this is `5m`. |
| `SESSION_HISTORY_LIMIT` | no | `20` | Max number of past messages sent as context per session (see "Session memory" above). |
| `DB_PATH` | no | `bot.db` | Path to the SQLite database file. |
| `EMBEDDING_PROVIDER` | no | `ollama` | Provider for the embedding model (`ollama` or `openai`). |
| `EMBEDDING_MODEL` | no | `embeddinggemma` | Embedding model used for global memory. **Fixed per database** — see "Global memory" below. |
| `GLOBAL_MEMORY_MESSAGE_THRESHOLD` | no | `20` | Summarize a session into global memory once this many new (un-summarized) messages accumulate. |
| `GLOBAL_MEMORY_MIN_SIMILARITY` | no | `0.4` | Cosine-similarity floor a fact must clear to be injected into a chat. The real relevance filter; token cost scales with how many facts actually match. Model-dependent — tune by watching real retrievals against your embedding model. |
| `GLOBAL_MEMORY_TOP_K` | no | `25` | Safety ceiling on facts injected per turn (only bounds the rare case where many clear the threshold at once). |
| `GLOBAL_MEMORY_MAX_ROWS` | no | `10000` | Hard cap on a user's stored facts; the `/compact` target is half this. Reaching it pauses (never drops) new summarization until `/compact` runs. |

Additionally, edit `allowedUserIDs` in [auth/auth.go](auth/auth.go) to add the Telegram user IDs allowed to use the bot — this is hardcoded rather than configured, since the bot is meant for a fixed, known set of users. Everyone else is silently ignored.

## Building

The SQLite driver (`modernc.org/sqlite`) is pure Go, so cross-compiling needs no C toolchain — just set `GOOS`/`GOARCH`.

**Windows** (PowerShell, native build):
```powershell
go build -o tg-llm-memory-bot.exe .
```

**macOS, Apple Silicon** (M1 and later) — from Windows or macOS:
```powershell
$env:GOOS='darwin'; $env:GOARCH='arm64'; $env:CGO_ENABLED='0'; go build -o tg-llm-memory-bot .
```
```bash
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -o tg-llm-memory-bot .
```

**macOS, Intel**:
```powershell
$env:GOOS='darwin'; $env:GOARCH='amd64'; $env:CGO_ENABLED='0'; go build -o tg-llm-memory-bot .
```
```bash
GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go build -o tg-llm-memory-bot .
```

**Linux** (amd64):
```powershell
$env:GOOS='linux'; $env:GOARCH='amd64'; $env:CGO_ENABLED='0'; go build -o tg-llm-memory-bot .
```
```bash
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o tg-llm-memory-bot .
```

Each produces a single self-contained binary for that platform — copy it over and run it directly, no Go install needed on the target machine. `.env` and `allowedUserIDs` (see above) still apply wherever it runs.

Note: `go build ./...` (used for verifying the whole module compiles, e.g. in CI) matches multiple packages and therefore discards its output rather than writing a binary — use `go build .` (targeting just the root `main` package) as shown above to actually get an executable.

### Automated releases

Pushing a version tag builds the binaries and publishes them as a GitHub Release automatically, via [.github/workflows/release.yml](.github/workflows/release.yml):

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow runs the test suite, cross-compiles for Windows (amd64) and macOS (Apple Silicon + Intel) on a single Linux runner (possible because the build is pure-Go, `CGO_ENABLED=0`), and attaches the binaries plus a `SHA256SUMS.txt` to the release. To add more targets (e.g. Linux), extend the `for target in ...` list in the workflow.

## Stack

- **Go**, [go-telegram/bot](https://github.com/go-telegram/bot) for the Telegram Bot API.
- **[langchaingo](https://github.com/tmc/langchaingo)** for LLM access — OpenAI and local Ollama models behind a single provider-agnostic interface (`llm.Provider.Generate`).
- **SQLite** via `modernc.org/sqlite` (pure Go, no CGO) for all persistence — sessions, message history, and (eventually) global memory.

## Project layout

- `llm/` — provider-agnostic `Message`/`Provider` types; `openai.go` (hardcoded model whitelist) and `ollama.go` (models listed live from the local server's `/api/tags`, i.e. whatever you've actually pulled).
- `store/` — SQLite schema and repositories for sessions and messages.
- `session/` — `Manager` ties storage and an LLM provider together per request.
- `auth/` — hardcoded Telegram user ID allow-list, enforced as bot middleware.
- `main.go` — Telegram wiring: commands and inline-keyboard handlers.

## Running it

1. Have [Ollama](https://ollama.com) running locally if you want to use it (the default provider for new sessions, since it's free) — pull at least one model, e.g. `ollama pull llama3.2`.
2. Set up `.env` and `allowedUserIDs` as described in "Environment variables" above.
3. `go run .`, or build a binary as described in "Building" and run that.

## Bot commands

- `/new [title]` — start a new session (becomes the active one). Without a title, it's named with the current timestamp.
- `/sessions` — list your sessions as buttons (`*` marks the active one); tap one to switch to it.
- `/model` — pick a provider then a model via inline buttons; applies to the active session only.
- `/summarymodel` — pick the LLM used to summarize conversation into global memory (per user, shared across sessions).
- `/save` — summarize the active session's recent messages into global memory right now, without waiting for the message threshold.
- `/compact` — consolidate global memory when it has grown large (see below). Warns and asks for confirmation first.

Anything else is treated as a message to the active session's model. Text starting with `/` that doesn't match a command above is rejected rather than sent to the model.

While a reply is generating, the bot posts a placeholder message that shows the streamed text-so-far (with a trailing cursor), updated roughly once a second, then replaces it with the final reply — so long generations don't look unresponsive. Edits are batched on a timer rather than sent per-token, since Telegram throttles rapid edits to the same message.

## Global memory

Global memory is populated asynchronously by one background worker per allowed user. A worker summarizes a session's un-summarized messages into facts, embeds them, and stores them — triggered when a session accumulates `GLOBAL_MEMORY_MESSAGE_THRESHOLD` new messages, when you switch away from a session (its unsaved tail is flushed), or when you run `/save`. Progress is checkpointed per session in SQLite, so a crash or a transient LLM/embedding error never loses messages: the failed range is simply retried (against however many messages exist by then) on the next attempt.

**Embedding model is fixed per database.** Cosine similarity is only meaningful when every stored vector came from the same embedding model, so the DB records which `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` it was built with. If you start the bot with a different embedding model than the DB was built with, startup fails with a fatal error — revert the env vars, or point `DB_PATH` at a fresh database for the new model. If the embedding endpoint is simply unreachable at startup, global memory is disabled for that run (a warning, not fatal) and normal chatting continues.

**Compaction.** Facts accumulate additively, so over time duplicates and stale/contradictory facts build up. `/compact` consolidates all of a user's facts down to about `GLOBAL_MEMORY_MAX_ROWS / 2` in one LLM pass. It's manual and never automatic: it can be slow and it blocks that user's chatting until it finishes, so it warns and asks for confirmation first. Reaching `GLOBAL_MEMORY_MAX_ROWS` pauses (never drops) new summarization until you compact, and you're sent a heads-up notification when memory crosses 90% of the cap.
