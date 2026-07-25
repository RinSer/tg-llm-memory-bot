package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

const compactPromptTemplate = `Below is a list of remembered facts about a user, one per line. Consolidate them into at most %d concise, non-redundant facts: merge duplicates and near-duplicates, drop anything obsolete, and when two facts conflict keep the more specific or more recent one. Output the consolidated facts, one per line, with no numbering or bullet characters. Preserve the language each fact is written in.

Facts:
%s`

// runCompaction consolidates all of the user's facts down toward MaxRows/2.
// It reads every row, asks the summarization LLM to merge them, re-embeds
// the result, and atomically replaces the user's rows. A no-op if already
// at/under target. Checkpoints are untouched.
func (m *Manager) runCompaction(ctx context.Context, userID int64) error {
	if !m.available {
		return fmt.Errorf("global memory is unavailable (embedding endpoint unreachable)")
	}

	target := m.cfg.MaxRows / 2
	rows, err := m.cfg.Store.ListGlobalMemory(userID)
	if err != nil {
		return err
	}
	if len(rows) <= target {
		return nil // nothing to do
	}

	provider, model := m.resolveSummarizationModel(userID)
	chat, err := llm.New(m.cfg.summarizationConfig(provider, model))
	if err != nil {
		return fmt.Errorf("build summarization provider: %w", err)
	}

	var sb strings.Builder
	for _, r := range rows {
		sb.WriteString(r.Content)
		sb.WriteByte('\n')
	}
	prompt := fmt.Sprintf(compactPromptTemplate, target, sb.String())

	reply, err := chat.Generate(ctx, []llm.Message{{Role: llm.RoleUser, Content: prompt}})
	if err != nil {
		return fmt.Errorf("compaction generate: %w", err)
	}

	facts := parseFacts(reply)
	if len(facts) == 0 {
		return fmt.Errorf("compaction produced no facts; leaving memory unchanged")
	}

	vecs, err := m.embedder.Embed(ctx, facts)
	if err != nil {
		return fmt.Errorf("embed compacted facts: %w", err)
	}
	if len(vecs) != len(facts) {
		return fmt.Errorf("embed returned %d vectors for %d facts", len(vecs), len(facts))
	}

	storeFacts := make([]store.Fact, len(facts))
	for i := range facts {
		storeFacts[i] = store.Fact{Content: facts[i], Embedding: vecs[i]}
	}

	if err := m.cfg.Store.ReplaceGlobalMemory(userID, storeFacts); err != nil {
		return err
	}
	m.logf("compacted global memory for user %d: %d row(s) -> %d row(s)", userID, len(rows), len(facts))
	return nil
}
