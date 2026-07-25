package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

const summarizePromptTemplate = `Extract the durable, factual information about the user and their situation from the conversation below. Output concise facts, one per line, with no numbering or bullet characters. Keep only information worth remembering long-term (preferences, identity, ongoing projects, stated goals, commitments); skip small talk and transient details. Write each fact in the SAME LANGUAGE as the conversation. If there is nothing worth remembering, output nothing.

Conversation:
%s`

// summarizeSession summarizes the given pending messages of one session
// into facts and persists them, advancing the checkpoint. Any failure is
// recorded against the checkpoint (leaving it unadvanced) so the next wake
// retries -- nothing is ever lost.
func (m *Manager) summarizeSession(ctx context.Context, userID int64, sess store.Session, pending []store.Message) {
	upTo := pending[len(pending)-1].ID

	provider, model := m.resolveSummarizationModel(userID)
	chat, err := llm.New(m.cfg.summarizationConfig(provider, model))
	if err != nil {
		m.recordErr(userID, sess.ID, fmt.Sprintf("build summarization provider: %v", err))
		return
	}

	prompt := fmt.Sprintf(summarizePromptTemplate, formatMessages(pending))
	reply, err := chat.Generate(ctx, []llm.Message{{Role: llm.RoleUser, Content: prompt}})
	if err != nil {
		m.recordErr(userID, sess.ID, fmt.Sprintf("summarization generate: %v", err))
		return
	}

	facts := parseFacts(reply)
	if len(facts) == 0 {
		// Nothing worth remembering in this batch, but the messages are
		// still processed -- advance the checkpoint so we don't reprocess
		// them forever.
		if err := m.cfg.Store.SaveSummarizationBatch(userID, sess.ID, nil, upTo); err != nil {
			m.recordErr(userID, sess.ID, fmt.Sprintf("save empty batch: %v", err))
		}
		return
	}

	// Row-cap check: refuse (as a recoverable error, not a drop) if this
	// batch would push the user past MaxRows. The checkpoint stays put, so
	// these messages are retried automatically after a /compact frees room.
	before, err := m.cfg.Store.CountGlobalMemory(userID)
	if err != nil {
		m.recordErr(userID, sess.ID, fmt.Sprintf("count rows: %v", err))
		return
	}
	if m.cfg.MaxRows > 0 && before+len(facts) > m.cfg.MaxRows {
		m.recordErr(userID, sess.ID, fmt.Sprintf("global memory full (%d/%d rows) -- run /compact", before, m.cfg.MaxRows))
		return
	}

	vecs, err := m.embedder.Embed(ctx, facts)
	if err != nil {
		m.recordErr(userID, sess.ID, fmt.Sprintf("embed facts: %v", err))
		return
	}
	if len(vecs) != len(facts) {
		m.recordErr(userID, sess.ID, fmt.Sprintf("embed returned %d vectors for %d facts", len(vecs), len(facts)))
		return
	}

	storeFacts := make([]store.Fact, len(facts))
	for i := range facts {
		storeFacts[i] = store.Fact{Content: facts[i], Embedding: vecs[i]}
	}

	if err := m.cfg.Store.SaveSummarizationBatch(userID, sess.ID, storeFacts, upTo); err != nil {
		m.recordErr(userID, sess.ID, fmt.Sprintf("save batch: %v", err))
		return
	}

	m.maybeWarnNearFull(userID, before, before+len(facts))
}

// maybeWarnNearFull fires a one-time notification exactly when a batch write
// crosses the 90%-of-MaxRows line (stateless crossing check).
func (m *Manager) maybeWarnNearFull(userID int64, before, after int) {
	if m.cfg.Notifier == nil || m.cfg.MaxRows <= 0 {
		return
	}
	threshold := (m.cfg.MaxRows * 9) / 10
	if before < threshold && after >= threshold {
		m.cfg.Notifier.Notify(userID, fmt.Sprintf(
			"Global memory is at %d%% (%d/%d rows). Consider running /compact.",
			(after*100)/m.cfg.MaxRows, after, m.cfg.MaxRows,
		))
	}
}

func (m *Manager) resolveSummarizationModel(userID int64) (llm.ProviderName, string) {
	provider, model, ok, err := m.cfg.Store.GetSummarizationModel(userID)
	if err != nil || !ok {
		return m.cfg.DefaultSummarizationProvider, m.cfg.DefaultSummarizationModel
	}
	return llm.ProviderName(provider), model
}

func (m *Manager) recordErr(userID, sessionID int64, msg string) {
	m.logf("session %d: %s", sessionID, msg)
	if err := m.cfg.Store.RecordSummarizationError(userID, sessionID, msg); err != nil {
		m.logf("session %d: also failed to record error: %v", sessionID, err)
	}
}

func formatMessages(messages []store.Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		fmt.Fprintf(&sb, "%s: %s\n", msg.Role, msg.Content)
	}
	return sb.String()
}

// parseFacts splits an LLM reply into trimmed, non-empty fact lines,
// stripping a single leading bullet marker if present. It deliberately does
// NOT strip leading digits generally, so a fact like "3 children" survives.
func parseFacts(reply string) []string {
	var facts []string
	for _, line := range strings.Split(reply, "\n") {
		line = strings.TrimSpace(line)
		for _, marker := range []string{"- ", "* ", "• ", "· "} {
			if strings.HasPrefix(line, marker) {
				line = strings.TrimSpace(line[len(marker):])
				break
			}
		}
		if line != "" {
			facts = append(facts, line)
		}
	}
	return facts
}
