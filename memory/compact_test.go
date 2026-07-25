package memory

import (
	"context"
	"testing"
)

func TestStartCompactionViaWorker(t *testing.T) {
	fake := newFakeOllama(t)
	m, db := newTestManager(t, fake, 10, nil) // target = 5
	const userID = int64(1)
	seedRows(t, db, userID, 8)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.Start(ctx, []int64{userID})

	if m.IsCompacting(userID) {
		t.Fatal("should not be compacting before start")
	}

	before, after, err := m.StartCompaction(ctx, userID)
	if err != nil {
		t.Fatalf("StartCompaction: %v", err)
	}
	if before != 8 {
		t.Fatalf("expected before=8, got %d", before)
	}
	if after != 2 { // chatReply "fact one\nfact two" -> 2 consolidated facts
		t.Fatalf("expected after=2, got %d", after)
	}
	if m.IsCompacting(userID) {
		t.Fatal("compaction flag should be cleared once StartCompaction returns")
	}
}

func TestStartCompactionRejectsUnknownUser(t *testing.T) {
	fake := newFakeOllama(t)
	m, _ := newTestManager(t, fake, 10, nil)
	// No Start() called, so no worker/channel for this user.
	if _, _, err := m.StartCompaction(context.Background(), 999); err == nil {
		t.Fatal("expected an error when no worker is running for the user")
	}
}
