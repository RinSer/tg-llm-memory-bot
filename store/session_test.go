package store

import (
	"errors"
	"testing"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestCreateSessionActivatesAndDeactivatesPrevious(t *testing.T) {
	s := newTestStore(t)
	const userID = int64(1)

	first, err := s.CreateSession(userID, "first", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if !first.IsActive {
		t.Fatal("expected first session to be active")
	}

	second, err := s.CreateSession(userID, "second", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if !second.IsActive {
		t.Fatal("expected second session to be active")
	}

	got, err := s.GetSession(first.ID)
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
	if got.IsActive {
		t.Fatal("expected first session to be deactivated once second was created")
	}
}

func TestGetActiveSessionNoneYet(t *testing.T) {
	s := newTestStore(t)

	_, err := s.GetActiveSession(1)
	if !errors.Is(err, ErrNoActiveSession) {
		t.Fatalf("expected ErrNoActiveSession, got %v", err)
	}
}

func TestGetActiveSessionIsPerUser(t *testing.T) {
	s := newTestStore(t)

	sessA, err := s.CreateSession(1, "a", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	sessB, err := s.CreateSession(2, "b", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	gotA, err := s.GetActiveSession(1)
	if err != nil {
		t.Fatalf("GetActiveSession(1): %v", err)
	}
	if gotA.ID != sessA.ID {
		t.Fatalf("expected session %d for user 1, got %d", sessA.ID, gotA.ID)
	}

	gotB, err := s.GetActiveSession(2)
	if err != nil {
		t.Fatalf("GetActiveSession(2): %v", err)
	}
	if gotB.ID != sessB.ID {
		t.Fatalf("expected session %d for user 2, got %d", sessB.ID, gotB.ID)
	}
}

func TestSetActiveSessionSwitchesAndRejectsOtherUsers(t *testing.T) {
	s := newTestStore(t)
	const userID = int64(1)

	first, err := s.CreateSession(userID, "first", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	second, err := s.CreateSession(userID, "second", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	if err := s.SetActiveSession(userID, first.ID); err != nil {
		t.Fatalf("SetActiveSession: %v", err)
	}

	active, err := s.GetActiveSession(userID)
	if err != nil {
		t.Fatalf("GetActiveSession: %v", err)
	}
	if active.ID != first.ID {
		t.Fatalf("expected active session %d, got %d", first.ID, active.ID)
	}

	// second's session belongs to a different (nonexistent) user here.
	if err := s.SetActiveSession(2, second.ID); err == nil {
		t.Fatal("expected error switching to a session owned by another user")
	}
}

func TestSetSessionModel(t *testing.T) {
	s := newTestStore(t)

	sess, err := s.CreateSession(1, "title", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	if err := s.SetSessionModel(sess.ID, "openai", "gpt-4.1-mini"); err != nil {
		t.Fatalf("SetSessionModel: %v", err)
	}

	got, err := s.GetSession(sess.ID)
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
	if got.Provider != "openai" || got.Model != "gpt-4.1-mini" {
		t.Fatalf("expected provider/model to be updated, got %s/%s", got.Provider, got.Model)
	}
}

func TestListSessionsOrderedNewestFirst(t *testing.T) {
	s := newTestStore(t)
	const userID = int64(1)

	first, err := s.CreateSession(userID, "first", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	second, err := s.CreateSession(userID, "second", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	sessions, err := s.ListSessions(userID)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(sessions))
	}
	if sessions[0].ID != second.ID || sessions[1].ID != first.ID {
		t.Fatalf("expected newest-first order [%d, %d], got [%d, %d]",
			second.ID, first.ID, sessions[0].ID, sessions[1].ID)
	}
}
