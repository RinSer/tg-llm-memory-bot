package store

import "time"

type Message struct {
	ID        int64
	SessionID int64
	Role      string
	Content   string
	CreatedAt time.Time
}

func (s *Store) AppendMessage(sessionID int64, role, content string) error {
	_, err := s.db.Exec(
		`INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)`,
		sessionID, role, content,
	)
	return err
}

// GetHistory returns up to the last `limit` messages of the session, in
// chronological order. Older messages remain in SQLite but fall out of
// context -- summarizing them instead of dropping them is a later phase.
func (s *Store) GetHistory(sessionID int64, limit int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT id, session_id, role, content, created_at FROM messages
		 WHERE session_id = ? ORDER BY id DESC LIMIT ?`,
		sessionID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Role, &m.Content, &m.CreatedAt); err != nil {
			return nil, err
		}
		messages = append(messages, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Query returns newest-first (required for LIMIT to pick the last N);
	// reverse in place to hand callers chronological order.
	for i, j := 0, len(messages)-1; i < j; i, j = i+1, j-1 {
		messages[i], messages[j] = messages[j], messages[i]
	}
	return messages, nil
}
