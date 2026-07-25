package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

// fakeOllama serves both /api/chat (summarization) and /api/embed
// (embeddings) so the memory pipeline can be exercised end-to-end without a
// real server.
type fakeOllama struct {
	*httptest.Server

	mu        sync.Mutex
	chatReply string // content returned by /api/chat
	failChat  bool   // when true, /api/chat returns 500
	failEmbed bool   // when true, /api/embed returns 500
	embedCalls int
}

func newFakeOllama(t *testing.T) *fakeOllama {
	t.Helper()
	f := &fakeOllama{chatReply: "fact one\nfact two"}
	f.Server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.Server.Close)
	return f
}

func (f *fakeOllama) setChatReply(s string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.chatReply = s
}

func (f *fakeOllama) handle(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	switch r.URL.Path {
	case "/api/chat":
		if f.failChat {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		fmt.Fprintf(w, `{"model":%q,"created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":%q},"done":true}`+"\n",
			req.Model, f.chatReply)
	case "/api/embed":
		if f.failEmbed {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		f.embedCalls++
		// One deterministic 3-dim vector per call (single input per request).
		json.NewEncoder(w).Encode(map[string]any{"embeddings": [][]float32{{1, 0, 0}}})
	default:
		http.NotFound(w, r)
	}
}

func newTestManager(t *testing.T, fake *fakeOllama, maxRows int, notifier Notifier) (*Manager, *store.Store) {
	t.Helper()
	db, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	m, err := New(Config{
		Store:                        db,
		Embedding:                    llm.Config{Name: llm.ProviderOllama, Model: "embed", BaseURL: fake.URL},
		MessageThreshold:             2,
		TopK:                         10,
		MinSimilarity:                0.5,
		MaxRows:                      maxRows,
		DefaultSummarizationProvider: llm.ProviderOllama,
		DefaultSummarizationModel:    "sum",
		OllamaBaseURL:                fake.URL,
		Notifier:                     notifier,
	})
	if err != nil {
		t.Fatalf("memory.New: %v", err)
	}
	m.available = true
	return m, db
}

func seedSession(t *testing.T, db *store.Store, userID int64, contents ...string) store.Session {
	t.Helper()
	sess, err := db.CreateSession(userID, "t", "ollama", "chat")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	for _, c := range contents {
		if err := db.AppendMessage(sess.ID, "user", c); err != nil {
			t.Fatalf("AppendMessage: %v", err)
		}
	}
	return *sess
}

func pending(t *testing.T, db *store.Store, sess store.Session) []store.Message {
	t.Helper()
	c, _ := db.GetCheckpoint(sess.ID)
	msgs, err := db.GetMessagesSince(sess.ID, c.LastSummarizedMessageID)
	if err != nil {
		t.Fatalf("GetMessagesSince: %v", err)
	}
	return msgs
}

func TestSummarizeSessionSuccess(t *testing.T) {
	fake := newFakeOllama(t)
	m, db := newTestManager(t, fake, 1000, nil)
	const userID = int64(1)
	sess := seedSession(t, db, userID, "hi", "I love hiking")

	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))

	rows, _ := db.ListGlobalMemory(userID)
	if len(rows) != 2 {
		t.Fatalf("expected 2 facts stored, got %d", len(rows))
	}
	c, _ := db.GetCheckpoint(sess.ID)
	if c.LastSummarizedMessageID == 0 {
		t.Fatal("expected checkpoint to advance")
	}
	if c.LastError != "" {
		t.Fatalf("expected no error, got %q", c.LastError)
	}
}

func TestSummarizeSessionChatErrorLeavesCheckpoint(t *testing.T) {
	fake := newFakeOllama(t)
	fake.failChat = true
	m, db := newTestManager(t, fake, 1000, nil)
	const userID = int64(1)
	sess := seedSession(t, db, userID, "hi", "there")

	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))

	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 0 {
		t.Fatalf("expected no facts on error, got %d", len(rows))
	}
	c, _ := db.GetCheckpoint(sess.ID)
	if c.LastSummarizedMessageID != 0 || c.LastError == "" || c.RetryCount != 1 {
		t.Fatalf("expected checkpoint unadvanced with error recorded, got %+v", c)
	}

	// Recovery: chat works on the next attempt and the same range succeeds.
	fake.mu.Lock()
	fake.failChat = false
	fake.mu.Unlock()
	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))
	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 2 {
		t.Fatalf("expected retry to store 2 facts, got %d", len(rows))
	}
	c, _ = db.GetCheckpoint(sess.ID)
	if c.LastError != "" || c.LastSummarizedMessageID == 0 {
		t.Fatalf("expected error cleared and checkpoint advanced, got %+v", c)
	}
}

func TestSummarizeSessionRowCapPauses(t *testing.T) {
	fake := newFakeOllama(t)
	// chatReply yields 2 facts; cap of 1 means the batch can't fit.
	m, db := newTestManager(t, fake, 1, nil)
	const userID = int64(1)
	sess := seedSession(t, db, userID, "hi", "there")

	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))

	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 0 {
		t.Fatalf("expected nothing stored when over cap, got %d", len(rows))
	}
	c, _ := db.GetCheckpoint(sess.ID)
	if c.LastSummarizedMessageID != 0 || c.LastError == "" {
		t.Fatalf("expected checkpoint unadvanced with 'full' error, got %+v", c)
	}
}

func TestSummarizeSessionEmptyReplyAdvancesCheckpoint(t *testing.T) {
	fake := newFakeOllama(t)
	fake.setChatReply("")
	m, db := newTestManager(t, fake, 1000, nil)
	const userID = int64(1)
	sess := seedSession(t, db, userID, "hi", "there")

	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))

	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 0 {
		t.Fatalf("expected no facts, got %d", len(rows))
	}
	c, _ := db.GetCheckpoint(sess.ID)
	if c.LastSummarizedMessageID == 0 {
		t.Fatal("expected checkpoint to advance even with no facts (messages processed)")
	}
}

type capturingNotifier struct {
	mu       sync.Mutex
	messages []string
}

func (n *capturingNotifier) Notify(userID int64, message string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.messages = append(n.messages, message)
}

func (n *capturingNotifier) count() int {
	n.mu.Lock()
	defer n.mu.Unlock()
	return len(n.messages)
}

func TestMaybeWarnNearFullCrossingSemantics(t *testing.T) {
	notifier := &capturingNotifier{}
	m := &Manager{cfg: Config{MaxRows: 100, Notifier: notifier}} // 90% line at 90
	const userID = int64(1)

	// Below -> below: no notify.
	m.maybeWarnNearFull(userID, 80, 85)
	if notifier.count() != 0 {
		t.Fatalf("expected no notify staying below 90%%, got %d", notifier.count())
	}
	// Below -> at/above: notify once (the crossing).
	m.maybeWarnNearFull(userID, 85, 92)
	if notifier.count() != 1 {
		t.Fatalf("expected one notify on crossing, got %d", notifier.count())
	}
	// Above -> further above: no re-notify.
	m.maybeWarnNearFull(userID, 92, 96)
	if notifier.count() != 1 {
		t.Fatalf("expected no re-notify while above, got %d", notifier.count())
	}
	// Dropped back below (e.g. after compaction) -> re-crossing notifies again.
	m.maybeWarnNearFull(userID, 40, 91)
	if notifier.count() != 2 {
		t.Fatalf("expected re-notify after dropping and re-crossing, got %d", notifier.count())
	}
}

func TestSummarizeBatchCrossingNotifies(t *testing.T) {
	notifier := &capturingNotifier{}
	fake := newFakeOllama(t)
	// MaxRows 12 -> 90% line at 10. Preload 9 rows, then a 2-fact batch -> 11.
	m, db := newTestManager(t, fake, 12, notifier)
	const userID = int64(1)

	preload := make([]store.Fact, 9)
	for i := range preload {
		preload[i] = store.Fact{Content: fmt.Sprintf("f%d", i), Embedding: []float32{1, 0, 0}}
	}
	if err := db.SaveSummarizationBatch(userID, 999, preload, 1); err != nil {
		t.Fatalf("seed: %v", err)
	}

	sess := seedSession(t, db, userID, "a", "b")
	m.summarizeSession(context.Background(), userID, sess, pending(t, db, sess))

	if rows, _ := db.ListGlobalMemory(userID); len(rows) != 11 {
		t.Fatalf("expected 11 rows after batch, got %d", len(rows))
	}
	if notifier.count() != 1 {
		t.Fatalf("expected one crossing notification, got %d", notifier.count())
	}
}
