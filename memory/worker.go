package memory

import "context"

// processUser scans every session of the user and summarizes those with
// enough pending (un-summarized) messages. The gate depends on why we woke:
// a threshold wake only acts on sessions at/above MessageThreshold pending,
// while switch/manual wakes flush any session with >= 1 pending. The full
// scan (not just the signaled session) is what makes dropped signals
// harmless.
func (m *Manager) processUser(ctx context.Context, userID int64, kind SignalKind) {
	sessions, err := m.cfg.Store.ListSessions(userID)
	if err != nil {
		m.logf("list sessions for user %d: %v", userID, err)
		return
	}

	minPending := 1
	if kind == SignalMessageThreshold {
		minPending = m.cfg.MessageThreshold
	}

	for _, sess := range sessions {
		select {
		case <-ctx.Done():
			return
		default:
		}

		checkpoint, err := m.cfg.Store.GetCheckpoint(sess.ID)
		if err != nil {
			m.logf("get checkpoint for session %d: %v", sess.ID, err)
			continue
		}
		pending, err := m.cfg.Store.GetMessagesSince(sess.ID, checkpoint.LastSummarizedMessageID)
		if err != nil {
			m.logf("get pending messages for session %d: %v", sess.ID, err)
			continue
		}
		if len(pending) < minPending {
			continue
		}
		m.summarizeSession(ctx, userID, sess, pending)
	}
}
