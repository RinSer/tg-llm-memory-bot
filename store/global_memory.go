package store

import (
	"encoding/binary"
	"math"
	"time"
)

// Fact is a single distilled memory item plus its embedding vector, ready
// to be persisted into global_memory.
type Fact struct {
	Content   string
	Embedding []float32
}

// GlobalMemoryRow is a stored fact, decoded back from SQLite.
type GlobalMemoryRow struct {
	ID        int64
	UserID    int64
	Content   string
	Embedding []float32
	CreatedAt time.Time
}

// ListGlobalMemory returns every stored fact for the user, decoded. Cosine
// ranking happens in Go over this slice (no SQL vector index).
func (s *Store) ListGlobalMemory(userID int64) ([]GlobalMemoryRow, error) {
	rows, err := s.db.Query(
		`SELECT id, user_id, content, embedding, created_at FROM global_memory WHERE user_id = ? ORDER BY id ASC`,
		userID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []GlobalMemoryRow
	for rows.Next() {
		var r GlobalMemoryRow
		var blob []byte
		if err := rows.Scan(&r.ID, &r.UserID, &r.Content, &blob, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.Embedding = decodeEmbedding(blob)
		out = append(out, r)
	}
	return out, rows.Err()
}

// CountGlobalMemory returns how many facts the user currently has stored.
func (s *Store) CountGlobalMemory(userID int64) (int, error) {
	var n int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM global_memory WHERE user_id = ?`, userID).Scan(&n)
	return n, err
}

// ReplaceGlobalMemory atomically deletes all of the user's existing facts
// and inserts the given consolidated set. Used only by compaction; a crash
// mid-transaction leaves the old rows fully intact.
func (s *Store) ReplaceGlobalMemory(userID int64, facts []Fact) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`DELETE FROM global_memory WHERE user_id = ?`, userID); err != nil {
		return err
	}
	for _, f := range facts {
		if _, err := tx.Exec(
			`INSERT INTO global_memory (user_id, content, embedding) VALUES (?, ?, ?)`,
			userID, f.Content, encodeEmbedding(f.Embedding),
		); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// encodeEmbedding packs a []float32 into a little-endian byte slice for
// storage in a SQLite BLOB.
func encodeEmbedding(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// decodeEmbedding reverses encodeEmbedding. A nil/short blob decodes to nil.
func decodeEmbedding(b []byte) []float32 {
	n := len(b) / 4
	if n == 0 {
		return nil
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
