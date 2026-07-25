package store

import (
	"database/sql"
	"errors"
)

// GetSummarizationModel returns the user's chosen summarization provider/model.
// ok is false (with no error) if the user has never set one.
func (s *Store) GetSummarizationModel(userID int64) (provider, model string, ok bool, err error) {
	row := s.db.QueryRow(`SELECT provider, model FROM summarization_models WHERE user_id = ?`, userID)
	err = row.Scan(&provider, &model)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", false, nil
	}
	if err != nil {
		return "", "", false, err
	}
	return provider, model, true, nil
}

// SetSummarizationModel upserts the user's summarization provider/model.
func (s *Store) SetSummarizationModel(userID int64, provider, model string) error {
	_, err := s.db.Exec(
		`INSERT INTO summarization_models (user_id, provider, model, updated_at)
		 VALUES (?, ?, ?, CURRENT_TIMESTAMP)
		 ON CONFLICT(user_id) DO UPDATE SET
			provider = excluded.provider,
			model = excluded.model,
			updated_at = CURRENT_TIMESTAMP`,
		userID, provider, model,
	)
	return err
}
