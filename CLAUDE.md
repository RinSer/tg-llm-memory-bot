# CLAUDE.md

Guidance for working on this repo. See [README.md](README.md) for what the project does and how to run it.

## Design decisions worth knowing before changing anything

- **Memory is provider-agnostic.** `llm.Message`/`llm.Provider` (in `llm/provider.go`) are plain data, independent of any SDK type. This is what lets a session switch provider/model (`Manager.SetModel`) without touching stored history. Don't let SQLite rows or the session manager depend on an SDK-specific message type.
- **Auth is a hardcoded allow-list**, not config/env — see `auth/auth.go`. This was an explicit choice (only a known, fixed set of Telegram user IDs may use the bot). Don't turn it into an env var or open registration without being asked.
- **OpenAI models are a hardcoded whitelist; Ollama models are queried live.** OpenAI's list lives in `llm/openai.go` (`OpenAIModels`) because there's no cheap way to ask a remote paid API "what should I let the user pick" — Ollama's list is fetched from the local server's `/api/tags` (`llm/ollama.go`, `listOllamaModels`) because it depends on whatever the user actually has pulled. Keep this asymmetry; don't hardcode an Ollama model list.
- **Session history is windowed, not unbounded.** `store.GetHistory(sessionID, limit)` only returns the last `limit` messages (`SESSION_HISTORY_LIMIT` env var). Full history is retained in SQLite regardless — nothing is deleted, it just falls out of what's sent to the model. Summarizing what falls out of the window (instead of silently dropping it) is a known, deliberately deferred follow-up — don't build it unless asked.
- **Global (cross-session) memory is scaffolded, not implemented.** The `global_memory` table (`store/store.go`) has a nullable `embedding BLOB` column. The intended design, already agreed on, is: embed content, store the embedding as a packed `[]float32` BLOB in that same SQLite table, and retrieve via a brute-force cosine-similarity scan in Go — no separate vector database (chromem-go, sqlite-vec, etc.). At this scale (a hardcoded handful of users) a real vector index buys nothing. If asked to implement this phase, follow that design rather than reaching for a new dependency.
- **Everything is keyed by Telegram user ID only** (`int64` — SQLite's `INTEGER` is already 64-bit, no special handling needed), never by chat ID or provider. Sessions and global memory both key off it.

## Testing is mandatory

**Every change to behavior must come with new or updated tests in the same change.** Don't leave test updates for later, and don't treat this repo's existing tests as a ceiling — if you touch a package, its test file should reflect what you changed. If a change genuinely can't be tested (rare), say so explicitly and explain why, rather than skipping silently.

Ground rules for how tests in this repo are written, and why — follow the same approach for new ones:

- **Mock at the real boundary, not by adding seams to production code.** `session/manager_test.go` tests `Manager.Reply` end-to-end against a real `httptest.Server` playing the part of an Ollama server (`/api/chat`, `/api/tags`), because `llm.New("ollama", ...)` just points an HTTP client at a configurable base URL — no fake-provider injection point was added to `Manager` for this. Prefer this over adding test-only fields/factories to production types.
- **`store` tests use a real SQLite in-memory DB** (`store.Open(":memory:")`), not a mocked `database/sql` driver — it's fast and exercises real SQL.
- **`llm` tests use `langchaingo/llms/fake`** for the shared `generate()` helper (role mapping, response/error handling), and `httptest` for the Ollama-specific HTTP calls. Don't call real OpenAI/Ollama endpoints from tests.
- Run `go build ./...`, `go vet ./...`, and `go test ./...` before considering a change done.

## Common commands

```
go build ./...
go vet ./...
go test ./...          # add -v for per-test output
go run .               # run the bot (needs .env, see README)
```
