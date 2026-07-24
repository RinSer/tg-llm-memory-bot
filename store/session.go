package store

import (
	"database/sql"
	"errors"
	"time"
)

var ErrNoActiveSession = errors.New("no active session")

type Session struct {
	ID        int64
	UserID    int64
	Title     string
	Provider  string
	Model     string
	IsActive  bool
	CreatedAt time.Time
	UpdatedAt time.Time
}

// CreateSession deactivates any other active session for the user and
// inserts a new active one.
func (s *Store) CreateSession(userID int64, title, provider, model string) (*Session, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`UPDATE sessions SET is_active = 0 WHERE user_id = ?`, userID); err != nil {
		return nil, err
	}

	res, err := tx.Exec(
		`INSERT INTO sessions (user_id, title, provider, model, is_active) VALUES (?, ?, ?, ?, 1)`,
		userID, title, provider, model,
	)
	if err != nil {
		return nil, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}

	return s.GetSession(id)
}

func (s *Store) GetSession(id int64) (*Session, error) {
	row := s.db.QueryRow(
		`SELECT id, user_id, title, provider, model, is_active, created_at, updated_at FROM sessions WHERE id = ?`,
		id,
	)
	return scanSession(row)
}

func (s *Store) GetActiveSession(userID int64) (*Session, error) {
	row := s.db.QueryRow(
		`SELECT id, user_id, title, provider, model, is_active, created_at, updated_at
		 FROM sessions WHERE user_id = ? AND is_active = 1`,
		userID,
	)
	session, err := scanSession(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNoActiveSession
	}
	return session, err
}

func (s *Store) ListSessions(userID int64) ([]Session, error) {
	rows, err := s.db.Query(
		`SELECT id, user_id, title, provider, model, is_active, created_at, updated_at
		 FROM sessions WHERE user_id = ? ORDER BY created_at DESC, id DESC`,
		userID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		session, err := scanSession(rows)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, *session)
	}
	return sessions, rows.Err()
}

// SetActiveSession activates the given session and deactivates every other
// session belonging to the same user. It fails if the session does not
// belong to userID.
func (s *Store) SetActiveSession(userID, sessionID int64) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`UPDATE sessions SET is_active = 0 WHERE user_id = ?`, userID); err != nil {
		return err
	}

	res, err := tx.Exec(
		`UPDATE sessions SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?`,
		sessionID, userID,
	)
	if err != nil {
		return err
	}
	if n, err := res.RowsAffected(); err != nil {
		return err
	} else if n == 0 {
		return errors.New("session not found for user")
	}

	return tx.Commit()
}

// SetSessionModel switches which provider/model a session uses going
// forward. Stored message history is untouched.
func (s *Store) SetSessionModel(sessionID int64, provider, model string) error {
	_, err := s.db.Exec(
		`UPDATE sessions SET provider = ?, model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?`,
		provider, model, sessionID,
	)
	return err
}

type rowScanner interface {
	Scan(dest ...any) error
}

func scanSession(row rowScanner) (*Session, error) {
	var s Session
	var isActive int
	if err := row.Scan(&s.ID, &s.UserID, &s.Title, &s.Provider, &s.Model, &isActive, &s.CreatedAt, &s.UpdatedAt); err != nil {
		return nil, err
	}
	s.IsActive = isActive != 0
	return &s, nil
}
