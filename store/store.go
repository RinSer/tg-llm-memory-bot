package store

import (
	"database/sql"

	_ "modernc.org/sqlite"
)

const schema = `
CREATE TABLE IF NOT EXISTS sessions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL,
	title TEXT NOT NULL,
	provider TEXT NOT NULL,
	model TEXT NOT NULL,
	is_active INTEGER NOT NULL DEFAULT 0,
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS messages (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id INTEGER NOT NULL REFERENCES sessions(id),
	role TEXT NOT NULL,
	content TEXT NOT NULL,
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- Cross-session memory, keyed only by Telegram user id. "embedding" is a
-- packed []float32 (see store.encodeEmbedding). Similarity search is a
-- brute-force cosine scan in Go, not a SQLite vector index -- fine at this
-- row count.
CREATE TABLE IF NOT EXISTS global_memory (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL,
	content TEXT NOT NULL,
	embedding BLOB,
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_global_memory_user ON global_memory(user_id);

-- One row per session that has any summarized history. The durable
-- checkpoint that makes the summarization queue crash-safe:
-- last_summarized_message_id only advances after a batch's facts are
-- committed, so a crash/error before that leaves the next wake to safely
-- redo the same range (idempotent).
CREATE TABLE IF NOT EXISTS summarization_checkpoints (
	session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
	user_id INTEGER NOT NULL,
	last_summarized_message_id INTEGER NOT NULL DEFAULT 0,
	last_error TEXT,
	last_attempt_at DATETIME,
	retry_count INTEGER NOT NULL DEFAULT 0,
	updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_summarization_checkpoints_user ON summarization_checkpoints(user_id);

-- Summarization LLM choice, global per user (not per session).
CREATE TABLE IF NOT EXISTS summarization_models (
	user_id INTEGER PRIMARY KEY,
	provider TEXT NOT NULL,
	model TEXT NOT NULL,
	updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Single-row table recording which embedding provider/model this DB's
-- vectors were produced with, compared against env config at startup so a
-- mismatch can be caught before it silently corrupts similarity search.
CREATE TABLE IF NOT EXISTS embedding_config (
	id INTEGER PRIMARY KEY CHECK (id = 1),
	provider TEXT NOT NULL,
	model TEXT NOT NULL,
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
`

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// Pin to a single connection. SQLite has a single writer anyway, our
	// queries are short, and this both avoids "database is locked" under
	// the concurrent summarization workers and keeps an in-memory (":memory:")
	// DB coherent (otherwise each pooled connection is a separate database).
	db.SetMaxOpenConns(1)
	if err := db.Ping(); err != nil {
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		return nil, err
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}
