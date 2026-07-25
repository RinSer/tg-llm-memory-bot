package store

import "testing"

func TestSummarizationModelGetSet(t *testing.T) {
	s := newTestStore(t)

	if _, _, ok, err := s.GetSummarizationModel(1); err != nil || ok {
		t.Fatalf("expected no model set, got ok=%v err=%v", ok, err)
	}

	if err := s.SetSummarizationModel(1, "ollama", "llama3.2"); err != nil {
		t.Fatalf("SetSummarizationModel: %v", err)
	}
	p, m, ok, err := s.GetSummarizationModel(1)
	if err != nil || !ok || p != "ollama" || m != "llama3.2" {
		t.Fatalf("expected ollama/llama3.2, got %s/%s ok=%v err=%v", p, m, ok, err)
	}

	// Upsert replaces.
	if err := s.SetSummarizationModel(1, "openai", "gpt-4.1-mini"); err != nil {
		t.Fatalf("SetSummarizationModel: %v", err)
	}
	p, m, _, _ = s.GetSummarizationModel(1)
	if p != "openai" || m != "gpt-4.1-mini" {
		t.Fatalf("expected openai/gpt-4.1-mini, got %s/%s", p, m)
	}
}

func TestEmbeddingConfigGetSet(t *testing.T) {
	s := newTestStore(t)

	if _, _, ok, err := s.GetEmbeddingConfig(); err != nil || ok {
		t.Fatalf("expected no embedding config, got ok=%v err=%v", ok, err)
	}

	if err := s.SetEmbeddingConfig("ollama", "embeddinggemma"); err != nil {
		t.Fatalf("SetEmbeddingConfig: %v", err)
	}
	p, m, ok, err := s.GetEmbeddingConfig()
	if err != nil || !ok || p != "ollama" || m != "embeddinggemma" {
		t.Fatalf("expected ollama/embeddinggemma, got %s/%s ok=%v err=%v", p, m, ok, err)
	}
}

func TestGetMessagesSince(t *testing.T) {
	s := newTestStore(t)
	sess, err := s.CreateSession(1, "t", "ollama", "llama3.2")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	for _, c := range []string{"m1", "m2", "m3", "m4"} {
		if err := s.AppendMessage(sess.ID, "user", c); err != nil {
			t.Fatalf("AppendMessage: %v", err)
		}
	}

	all, err := s.GetMessagesSince(sess.ID, 0)
	if err != nil {
		t.Fatalf("GetMessagesSince: %v", err)
	}
	if len(all) != 4 || all[0].Content != "m1" || all[3].Content != "m4" {
		t.Fatalf("expected all 4 in order, got %+v", all)
	}

	// Only messages after the 2nd id.
	since, err := s.GetMessagesSince(sess.ID, all[1].ID)
	if err != nil {
		t.Fatalf("GetMessagesSince: %v", err)
	}
	if len(since) != 2 || since[0].Content != "m3" || since[1].Content != "m4" {
		t.Fatalf("expected [m3, m4], got %+v", since)
	}

	// Nothing after the last id.
	none, err := s.GetMessagesSince(sess.ID, all[3].ID)
	if err != nil {
		t.Fatalf("GetMessagesSince: %v", err)
	}
	if len(none) != 0 {
		t.Fatalf("expected no messages, got %d", len(none))
	}
}
