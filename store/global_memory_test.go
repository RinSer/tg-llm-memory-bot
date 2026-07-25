package store

import "testing"

func TestEncodeDecodeEmbeddingRoundTrip(t *testing.T) {
	in := []float32{0, 1, -1, 3.14159, 1e-9, -2.5e8}
	out := decodeEmbedding(encodeEmbedding(in))
	if len(out) != len(in) {
		t.Fatalf("expected %d floats, got %d", len(in), len(out))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("index %d: expected %v, got %v", i, in[i], out[i])
		}
	}
}

func TestDecodeEmbeddingEmpty(t *testing.T) {
	if decodeEmbedding(nil) != nil {
		t.Fatal("expected nil for nil blob")
	}
	if decodeEmbedding([]byte{}) != nil {
		t.Fatal("expected nil for empty blob")
	}
}

func TestListAndCountGlobalMemory(t *testing.T) {
	s := newTestStore(t)

	if n, err := s.CountGlobalMemory(1); err != nil || n != 0 {
		t.Fatalf("expected 0 count, got %d (err %v)", n, err)
	}

	facts := []Fact{
		{Content: "likes tea", Embedding: []float32{0.1, 0.2}},
		{Content: "lives in Berlin", Embedding: []float32{0.3, 0.4}},
	}
	if err := s.SaveSummarizationBatch(1, 10, facts, 5); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}
	// A different user's fact must not leak into user 1's list.
	if err := s.SaveSummarizationBatch(2, 20, []Fact{{Content: "other user", Embedding: []float32{9, 9}}}, 3); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}

	rows, err := s.ListGlobalMemory(1)
	if err != nil {
		t.Fatalf("ListGlobalMemory: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows for user 1, got %d", len(rows))
	}
	if rows[0].Content != "likes tea" || len(rows[0].Embedding) != 2 || rows[0].Embedding[1] != 0.2 {
		t.Fatalf("unexpected first row: %+v", rows[0])
	}

	if n, err := s.CountGlobalMemory(1); err != nil || n != 2 {
		t.Fatalf("expected count 2, got %d (err %v)", n, err)
	}
}

func TestReplaceGlobalMemory(t *testing.T) {
	s := newTestStore(t)

	if err := s.SaveSummarizationBatch(1, 10, []Fact{
		{Content: "a", Embedding: []float32{1}},
		{Content: "b", Embedding: []float32{2}},
		{Content: "c", Embedding: []float32{3}},
	}, 5); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}

	if err := s.ReplaceGlobalMemory(1, []Fact{{Content: "merged", Embedding: []float32{9}}}); err != nil {
		t.Fatalf("ReplaceGlobalMemory: %v", err)
	}

	rows, err := s.ListGlobalMemory(1)
	if err != nil {
		t.Fatalf("ListGlobalMemory: %v", err)
	}
	if len(rows) != 1 || rows[0].Content != "merged" {
		t.Fatalf("expected a single 'merged' row, got %+v", rows)
	}
}

func TestReplaceGlobalMemoryOnlyAffectsUser(t *testing.T) {
	s := newTestStore(t)
	if err := s.SaveSummarizationBatch(1, 10, []Fact{{Content: "u1", Embedding: []float32{1}}}, 5); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}
	if err := s.SaveSummarizationBatch(2, 20, []Fact{{Content: "u2", Embedding: []float32{2}}}, 5); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}

	if err := s.ReplaceGlobalMemory(1, []Fact{{Content: "u1new", Embedding: []float32{3}}}); err != nil {
		t.Fatalf("ReplaceGlobalMemory: %v", err)
	}

	if n, _ := s.CountGlobalMemory(2); n != 1 {
		t.Fatalf("expected user 2's rows untouched (1), got %d", n)
	}
}
