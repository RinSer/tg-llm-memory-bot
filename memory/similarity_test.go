package memory

import (
	"testing"

	"github.com/RinSer/tg-llm-memory-bot/store"
)

func TestCosineSimilarity(t *testing.T) {
	if got := cosineSimilarity([]float32{1, 0}, []float32{1, 0}); got < 0.999 {
		t.Fatalf("identical vectors should score ~1, got %v", got)
	}
	if got := cosineSimilarity([]float32{1, 0}, []float32{0, 1}); got != 0 {
		t.Fatalf("orthogonal vectors should score 0, got %v", got)
	}
	if got := cosineSimilarity([]float32{1, 0}, []float32{-1, 0}); got > -0.999 {
		t.Fatalf("opposite vectors should score ~-1, got %v", got)
	}
	// Mismatched length and zero vectors are treated as unrelated.
	if got := cosineSimilarity([]float32{1, 2, 3}, []float32{1, 2}); got != 0 {
		t.Fatalf("mismatched length should score 0, got %v", got)
	}
	if got := cosineSimilarity([]float32{0, 0}, []float32{1, 1}); got != 0 {
		t.Fatalf("zero vector should score 0, got %v", got)
	}
}

func TestRankFactsThresholdAndCap(t *testing.T) {
	rows := []store.GlobalMemoryRow{
		{Content: "same", Embedding: []float32{1, 0}},       // sim 1.0
		{Content: "similar", Embedding: []float32{0.9, 0.1}}, // high sim
		{Content: "orthogonal", Embedding: []float32{0, 1}},  // sim 0
		{Content: "opposite", Embedding: []float32{-1, 0}},   // sim -1
	}
	query := []float32{1, 0}

	// Threshold filters out the orthogonal and opposite rows.
	got := rankFacts(query, rows, 0.5, 10)
	if len(got) != 2 || got[0].Content != "same" || got[1].Content != "similar" {
		t.Fatalf("expected [same, similar] above threshold, got %+v", got)
	}

	// Cap truncates to the top-N even when more clear the threshold.
	got = rankFacts(query, rows, 0.5, 1)
	if len(got) != 1 || got[0].Content != "same" {
		t.Fatalf("expected only [same] with cap 1, got %+v", got)
	}
}
