package session

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/memory"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

func openTestStore(t *testing.T) *store.Store {
	t.Helper()
	db, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// fakeGlobalMemory is a test double for the GlobalMemory interface.
type fakeGlobalMemory struct {
	mu sync.Mutex

	facts       []string
	compacting  bool
	signals     []signalRecord
	summaryProv llm.ProviderName
	summaryModel string
}

type signalRecord struct {
	userID    int64
	sessionID int64
	kind      memory.SignalKind
}

func (f *fakeGlobalMemory) RelevantFacts(_ context.Context, _ int64, _ string) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.facts, nil
}

func (f *fakeGlobalMemory) Signal(userID, sessionID int64, kind memory.SignalKind) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.signals = append(f.signals, signalRecord{userID, sessionID, kind})
}

func (f *fakeGlobalMemory) SetSummarizationModel(_ int64, provider llm.ProviderName, model string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.summaryProv, f.summaryModel = provider, model
	return nil
}

func (f *fakeGlobalMemory) IsCompacting(_ int64) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.compacting
}

func (f *fakeGlobalMemory) CompactionStatus(_ int64) (int, int, error) { return 0, 0, nil }
func (f *fakeGlobalMemory) StartCompaction(_ context.Context, _ int64) (int, int, error) {
	return 0, 0, nil
}

func (f *fakeGlobalMemory) lastSignal(t *testing.T) signalRecord {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.signals) == 0 {
		t.Fatal("no signals recorded")
	}
	return f.signals[len(f.signals)-1]
}

func newTestManagerWithMemory(t *testing.T, baseURL string, gm GlobalMemory) *Manager {
	t.Helper()
	db := openTestStore(t)
	return NewManager(db, Config{
		HistoryLimit:    20,
		DefaultProvider: llm.ProviderOllama,
		DefaultModel:    "test-model",
		OllamaBaseURL:   baseURL,
		GlobalMemory:    gm,
	})
}

func TestReplyInjectsRelevantFactsAsSystemMessage(t *testing.T) {
	fake := newFakeOllama(t, "answer")
	gm := &fakeGlobalMemory{facts: []string{"likes tea", "lives in Berlin"}}
	mgr := newTestManagerWithMemory(t, fake.URL, gm)

	if _, err := mgr.Reply(context.Background(), 1, "hi"); err != nil {
		t.Fatalf("Reply: %v", err)
	}

	req := fake.lastRequest(t)
	if len(req.Messages) == 0 || req.Messages[0].Role != "system" {
		t.Fatalf("expected a leading system message, got %+v", req.Messages)
	}
	if !strings.Contains(req.Messages[0].Content, "likes tea") ||
		!strings.Contains(req.Messages[0].Content, "lives in Berlin") {
		t.Fatalf("expected facts in system message, got %q", req.Messages[0].Content)
	}
}

func TestReplyNoSystemMessageWhenNoFacts(t *testing.T) {
	fake := newFakeOllama(t, "answer")
	gm := &fakeGlobalMemory{facts: nil}
	mgr := newTestManagerWithMemory(t, fake.URL, gm)

	if _, err := mgr.Reply(context.Background(), 1, "hi"); err != nil {
		t.Fatalf("Reply: %v", err)
	}

	req := fake.lastRequest(t)
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Fatalf("expected a single user message with no facts, got %+v", req.Messages)
	}
}

func TestReplySignalsThreshold(t *testing.T) {
	fake := newFakeOllama(t, "answer")
	gm := &fakeGlobalMemory{}
	mgr := newTestManagerWithMemory(t, fake.URL, gm)

	if _, err := mgr.Reply(context.Background(), 42, "hi"); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	sig := gm.lastSignal(t)
	if sig.userID != 42 || sig.kind != memory.SignalMessageThreshold {
		t.Fatalf("expected a threshold signal for user 42, got %+v", sig)
	}
}

func TestReplyBlockedDuringCompaction(t *testing.T) {
	fake := newFakeOllama(t, "answer")
	gm := &fakeGlobalMemory{compacting: true}
	mgr := newTestManagerWithMemory(t, fake.URL, gm)

	reply, err := mgr.Reply(context.Background(), 1, "hi")
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if reply != compactionInProgressMsg {
		t.Fatalf("expected compaction-in-progress message, got %q", reply)
	}
	// No LLM call and no persisted history should have happened.
	if len(fake.requests) != 0 {
		t.Fatalf("expected no LLM request during compaction, got %d", len(fake.requests))
	}
}

func TestSwitchSessionSignalsDeactivated(t *testing.T) {
	fake := newFakeOllama(t, "r1", "r2")
	gm := &fakeGlobalMemory{}
	mgr := newTestManagerWithMemory(t, fake.URL, gm)
	const userID = int64(1)

	// Create an initial active session by replying once.
	if _, err := mgr.Reply(context.Background(), userID, "hi"); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	sessions, _ := mgr.ListSessions(userID)
	first := sessions[0]

	// A second session to switch between.
	second, err := mgr.NewSession(userID, "second")
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}

	gm.mu.Lock()
	gm.signals = nil // clear prior signals
	gm.mu.Unlock()

	if err := mgr.SwitchSession(userID, first.ID); err != nil {
		t.Fatalf("SwitchSession: %v", err)
	}
	sig := gm.lastSignal(t)
	if sig.kind != memory.SignalSessionSwitch || sig.sessionID != second.ID {
		t.Fatalf("expected a switch signal for the deactivated session %d, got %+v", second.ID, sig)
	}
}

func TestSetSummarizationModelDelegates(t *testing.T) {
	gm := &fakeGlobalMemory{}
	mgr := NewManager(openTestStore(t), Config{GlobalMemory: gm})

	if err := mgr.SetSummarizationModel(1, llm.ProviderOpenAI, "gpt-4.1-mini"); err != nil {
		t.Fatalf("SetSummarizationModel: %v", err)
	}
	if gm.summaryProv != llm.ProviderOpenAI || gm.summaryModel != "gpt-4.1-mini" {
		t.Fatalf("expected delegation to fake, got %s/%s", gm.summaryProv, gm.summaryModel)
	}
}
