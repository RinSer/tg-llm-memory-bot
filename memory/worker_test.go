package memory

import (
	"context"
	"fmt"
	"testing"

	"github.com/RinSer/tg-llm-memory-bot/store"
)

// seedRows inserts n placeholder facts for the user directly.
func seedRows(t *testing.T, db *store.Store, userID int64, n int) {
	t.Helper()
	facts := make([]store.Fact, n)
	for i := range facts {
		facts[i] = store.Fact{Content: fmt.Sprintf("seed fact %d", i), Embedding: []float32{1, 0, 0}}
	}
	if err := db.SaveSummarizationBatch(userID, 999, facts, 1); err != nil {
		t.Fatalf("seedRows: %v", err)
	}
}

func TestProcessUserThresholdGating(t *testing.T) {
	fake := newFakeOllama(t)
	m, db := newTestManager(t, fake, 1000, nil) // MessageThreshold = 2
	const userID = int64(1)

	// One pending message: below the threshold of 2.
	sess := seedSession(t, db, userID, "only one")

	// A threshold wake must NOT summarize (1 < 2).
	m.processUser(context.Background(), userID, SignalMessageThreshold)
	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 0 {
		t.Fatalf("threshold wake below N should not summarize, got %d rows", len(rows))
	}

	// A manual/switch wake acts on any pending (>= 1).
	m.processUser(context.Background(), userID, SignalManual)
	if rows, _ := db.ListGlobalMemory(userID); len(rows) == 0 {
		t.Fatal("manual wake should summarize any pending messages")
	}
	c, _ := db.GetCheckpoint(sess.ID)
	if c.LastSummarizedMessageID == 0 {
		t.Fatal("expected checkpoint advanced after manual wake")
	}
}

func TestProcessUserScansAllSessions(t *testing.T) {
	fake := newFakeOllama(t)
	m, db := newTestManager(t, fake, 1000, nil)
	const userID = int64(1)

	// Two sessions each with enough pending messages; a single wake should
	// process both (the full scan is what makes dropped signals safe).
	s1 := seedSession(t, db, userID, "a1", "a2")
	s2 := seedSession(t, db, userID, "b1", "b2")

	m.processUser(context.Background(), userID, SignalMessageThreshold)

	c1, _ := db.GetCheckpoint(s1.ID)
	c2, _ := db.GetCheckpoint(s2.ID)
	if c1.LastSummarizedMessageID == 0 || c2.LastSummarizedMessageID == 0 {
		t.Fatalf("expected both sessions summarized, got c1=%d c2=%d",
			c1.LastSummarizedMessageID, c2.LastSummarizedMessageID)
	}
}

func TestRunCompactionReducesRows(t *testing.T) {
	fake := newFakeOllama(t)
	// chatReply is "fact one\nfact two" -> compaction produces 2 rows.
	m, db := newTestManager(t, fake, 10, nil) // target = 5
	const userID = int64(1)

	// Seed 8 rows (> target of 5).
	seedRows(t, db, userID, 8)
	if n, _ := db.CountGlobalMemory(userID); n != 8 {
		t.Fatalf("expected 8 seeded rows, got %d", n)
	}

	if err := m.runCompaction(context.Background(), userID); err != nil {
		t.Fatalf("runCompaction: %v", err)
	}

	n, _ := db.CountGlobalMemory(userID)
	if n != 2 {
		t.Fatalf("expected compaction down to 2 rows, got %d", n)
	}
}

func TestRunCompactionNoopWhenUnderTarget(t *testing.T) {
	fake := newFakeOllama(t)
	m, db := newTestManager(t, fake, 100, nil) // target = 50
	const userID = int64(1)

	seedRows(t, db, userID, 3)
	if err := m.runCompaction(context.Background(), userID); err != nil {
		t.Fatalf("runCompaction: %v", err)
	}
	if n, _ := db.CountGlobalMemory(userID); n != 3 {
		t.Fatalf("expected no change (3 rows), got %d", n)
	}
}
