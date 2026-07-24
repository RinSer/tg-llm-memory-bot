package store

import "testing"

func TestAppendAndGetHistoryChronologicalOrder(t *testing.T) {
	s := newTestStore(t)
	sess, err := s.CreateSession(1, "title", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	turns := []struct{ role, content string }{
		{"user", "hi"},
		{"assistant", "hello"},
		{"user", "how are you"},
		{"assistant", "good"},
	}
	for _, turn := range turns {
		if err := s.AppendMessage(sess.ID, turn.role, turn.content); err != nil {
			t.Fatalf("AppendMessage: %v", err)
		}
	}

	history, err := s.GetHistory(sess.ID, 10)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != len(turns) {
		t.Fatalf("expected %d messages, got %d", len(turns), len(history))
	}
	for i, turn := range turns {
		if history[i].Role != turn.role || history[i].Content != turn.content {
			t.Fatalf("message %d: expected %s/%s, got %s/%s", i, turn.role, turn.content, history[i].Role, history[i].Content)
		}
	}
}

func TestGetHistoryWindowsToLastN(t *testing.T) {
	s := newTestStore(t)
	sess, err := s.CreateSession(1, "title", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	contents := []string{"m1", "m2", "m3", "m4", "m5"}
	for _, c := range contents {
		if err := s.AppendMessage(sess.ID, "user", c); err != nil {
			t.Fatalf("AppendMessage: %v", err)
		}
	}

	history, err := s.GetHistory(sess.ID, 2)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected window of 2 messages, got %d", len(history))
	}
	if history[0].Content != "m4" || history[1].Content != "m5" {
		t.Fatalf("expected [m4, m5] in chronological order, got [%s, %s]", history[0].Content, history[1].Content)
	}
}

func TestGetHistoryEmptySession(t *testing.T) {
	s := newTestStore(t)
	sess, err := s.CreateSession(1, "title", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	history, err := s.GetHistory(sess.ID, 10)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 0 {
		t.Fatalf("expected no messages, got %d", len(history))
	}
}
