# tg-llm-memory-bot

A locally-run Telegram bot server for talking to LLMs, with persistent conversation memory.

Each authorized user can run multiple named **sessions**, each with its own message history and its own LLM provider/model. A session's provider and model can be switched at any time without losing its history, since history is stored as plain text, not tied to any provider SDK type. Memory is split into two tiers, keyed only by Telegram user ID:

- **Session memory**: a per-session, windowed message history stored in SQLite (last N messages; N is configurable). Summarizing what falls out of the window instead of dropping it is a planned follow-up, not yet implemented.
- **Global memory**: a cross-session `global_memory` table, scaffolded but not yet populated or queried. The intended design is a RAG-style lookup — content embedded and stored as a `BLOB` in the same SQLite file, retrieved via a brute-force cosine-similarity scan (no separate vector database needed at this scale).

## Environment variables

Configure these either as real environment variables or by adding a `.env` file in the project root — `.env` is loaded on startup if present, but it's entirely optional (e.g. a built binary deployed standalone can just have its environment set directly instead).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `TG_BOT_API_TOKEN` | yes | — | Telegram bot token from [@BotFather](https://t.me/BotFather). |
| `OPENAI_API_TOKEN` | yes | — | OpenAI API key. Required at startup even if you only ever use Ollama, since a session can be switched to OpenAI via `/model` at any time. |
| `OLLAMA_BASE_URL` | no | `http://localhost:11434` | Base URL of a running Ollama server. |
| `SESSION_HISTORY_LIMIT` | no | `20` | Max number of past messages sent as context per session (see "Session memory" above). |
| `DB_PATH` | no | `bot.db` | Path to the SQLite database file. |

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

- `/new [title]` — start a new session (becomes the active one).
- `/sessions` — list your sessions, `*` marks the active one.
- `/switch <id>` — switch the active session.
- `/model` — pick a provider then a model via inline buttons; applies to the active session only.

Anything else is treated as a message to the active session's model.
