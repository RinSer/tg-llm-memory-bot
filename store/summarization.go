package store

import (
	"database/sql"
	"errors"
	"time"
)

// SummarizationCheckpoint tracks how far a session's history has been
// summarized into global memory, plus the last error (if any) so retries
// and diagnostics are durable.
type SummarizationCheckpoint struct {
	SessionID               int64
	UserID                  int64
	LastSummarizedMessageID int64
	LastError               string
	LastAttemptAt           time.Time
	RetryCount              int
	UpdatedAt               time.Time
}

// GetCheckpoint returns the session's checkpoint, or a zero-value one (with
// LastSummarizedMessageID == 0, meaning "nothing summarized yet") if the
// session has never been summarized. A missing row is not an error.
func (s *Store) GetCheckpoint(sessionID int64) (*SummarizationCheckpoint, error) {
	row := s.db.QueryRow(
		`SELECT session_id, user_id, last_summarized_message_id, last_error, last_attempt_at, retry_count, updated_at
		 FROM summarization_checkpoints WHERE session_id = ?`,
		sessionID,
	)
	var c SummarizationCheckpoint
	var lastError sql.NullString
	var lastAttempt sql.NullTime
	err := row.Scan(&c.SessionID, &c.UserID, &c.LastSummarizedMessageID, &lastError, &lastAttempt, &c.RetryCount, &c.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return &SummarizationCheckpoint{SessionID: sessionID}, nil
	}
	if err != nil {
		return nil, err
	}
	c.LastError = lastError.String
	c.LastAttemptAt = lastAttempt.Time
	return &c, nil
}

// SaveSummarizationBatch inserts the extracted facts into global_memory and
// advances the session's checkpoint to upToMessageID -- all in one
// transaction, so facts and progress commit together. Clears any prior
// error/retry state on success.
func (s *Store) SaveSummarizationBatch(userID, sessionID int64, facts []Fact, upToMessageID int64) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	for _, f := range facts {
		if _, err := tx.Exec(
			`INSERT INTO global_memory (user_id, content, embedding) VALUES (?, ?, ?)`,
			userID, f.Content, encodeEmbedding(f.Embedding),
		); err != nil {
			return err
		}
	}

	if _, err := tx.Exec(
		`INSERT INTO summarization_checkpoints (session_id, user_id, last_summarized_message_id, last_error, retry_count, updated_at)
		 VALUES (?, ?, ?, NULL, 0, CURRENT_TIMESTAMP)
		 ON CONFLICT(session_id) DO UPDATE SET
			last_summarized_message_id = excluded.last_summarized_message_id,
			last_error = NULL,
			retry_count = 0,
			updated_at = CURRENT_TIMESTAMP`,
		sessionID, userID, upToMessageID,
	); err != nil {
		return err
	}

	return tx.Commit()
}

// RecordSummarizationError upserts the checkpoint's error fields and bumps
// retry_count, leaving last_summarized_message_id untouched -- so the next
// wake retries the same (plus any newer) messages. This is the retry
// mechanism.
func (s *Store) RecordSummarizationError(userID, sessionID int64, errMsg string) error {
	_, err := s.db.Exec(
		`INSERT INTO summarization_checkpoints (session_id, user_id, last_summarized_message_id, last_error, last_attempt_at, retry_count, updated_at)
		 VALUES (?, ?, 0, ?, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)
		 ON CONFLICT(session_id) DO UPDATE SET
			last_error = excluded.last_error,
			last_attempt_at = CURRENT_TIMESTAMP,
			retry_count = summarization_checkpoints.retry_count + 1,
			updated_at = CURRENT_TIMESTAMP`,
		sessionID, userID, errMsg,
	)
	return err
}
