package store

import (
	"database/sql"
	"errors"
)

// GetEmbeddingConfig returns the embedding provider/model this DB's vectors
// were produced with. ok is false (no error) if none has been recorded yet
// (a fresh DB).
func (s *Store) GetEmbeddingConfig() (provider, model string, ok bool, err error) {
	row := s.db.QueryRow(`SELECT provider, model FROM embedding_config WHERE id = 1`)
	err = row.Scan(&provider, &model)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", false, nil
	}
	if err != nil {
		return "", "", false, err
	}
	return provider, model, true, nil
}

// SetEmbeddingConfig records (once) which embedding provider/model this DB
// is bound to.
func (s *Store) SetEmbeddingConfig(provider, model string) error {
	_, err := s.db.Exec(
		`INSERT INTO embedding_config (id, provider, model) VALUES (1, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET provider = excluded.provider, model = excluded.model`,
		provider, model,
	)
	return err
}
