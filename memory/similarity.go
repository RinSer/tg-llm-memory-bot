package memory

import (
	"math"
	"sort"

	"github.com/RinSer/tg-llm-memory-bot/store"
)

// cosineSimilarity returns the cosine similarity of two equal-length
// vectors, in [-1, 1]. Mismatched lengths or a zero-magnitude vector yield
// 0 (treated as "unrelated" rather than an error, so a stray malformed row
// can't break a whole retrieval).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// rankFacts returns the rows whose embedding scores at least minSimilarity
// against query, sorted most-similar first, truncated to cap.
func rankFacts(query []float32, rows []store.GlobalMemoryRow, minSimilarity float32, cap int) []store.GlobalMemoryRow {
	type scored struct {
		row store.GlobalMemoryRow
		sim float32
	}
	matches := make([]scored, 0, len(rows))
	for _, r := range rows {
		sim := cosineSimilarity(query, r.Embedding)
		if sim >= minSimilarity {
			matches = append(matches, scored{r, sim})
		}
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].sim > matches[j].sim })
	if cap > 0 && len(matches) > cap {
		matches = matches[:cap]
	}
	out := make([]store.GlobalMemoryRow, len(matches))
	for i, m := range matches {
		out[i] = m.row
	}
	return out
}
