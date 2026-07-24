package session

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

// chatRequest mirrors the subset of Ollama's /api/chat request body this
// test cares about.
type chatRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

// fakeOllama stands in for a real Ollama server: it records every /api/chat
// request it receives and answers with the next canned response, and
// serves a configurable /api/tags model list.
type fakeOllama struct {
	*httptest.Server

	tags []string

	mu        sync.Mutex
	requests  []chatRequest
	responses []string
	next      int
}

func newFakeOllama(t *testing.T, responses ...string) *fakeOllama {
	t.Helper()
	f := &fakeOllama{responses: responses}
	f.Server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.Server.Close)
	return f
}

func (f *fakeOllama) handle(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/chat":
		body, _ := io.ReadAll(r.Body)
		var req chatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		f.mu.Lock()
		f.requests = append(f.requests, req)
		reply := ""
		if f.next < len(f.responses) {
			reply = f.responses[f.next]
			f.next++
		}
		f.mu.Unlock()

		fmt.Fprintf(w, `{"model":%q,"created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":%q},"done":true}`+"\n", req.Model, reply)
	case "/api/tags":
		names := make([]map[string]string, len(f.tags))
		for i, n := range f.tags {
			names[i] = map[string]string{"name": n}
		}
		json.NewEncoder(w).Encode(map[string]any{"models": names})
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeOllama) lastRequest(t *testing.T) chatRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	return f.requests[len(f.requests)-1]
}

func newTestManager(t *testing.T, baseURL string, historyLimit int) (*Manager, *store.Store) {
	t.Helper()
	db, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	mgr := NewManager(db, Config{
		HistoryLimit:    historyLimit,
		DefaultProvider: llm.ProviderOllama,
		DefaultModel:    "test-model",
		OllamaBaseURL:   baseURL,
	})
	return mgr, db
}

func TestReplyCreatesDefaultSessionAndPersistsHistory(t *testing.T) {
	fake := newFakeOllama(t, "hello there")
	mgr, db := newTestManager(t, fake.URL, 20)
	const userID = int64(42)

	reply, err := mgr.Reply(context.Background(), userID, "hi")
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if reply != "hello there" {
		t.Fatalf("expected %q, got %q", "hello there", reply)
	}

	sess, err := db.GetActiveSession(userID)
	if err != nil {
		t.Fatalf("GetActiveSession: %v", err)
	}
	if sess.Provider != string(llm.ProviderOllama) || sess.Model != "test-model" {
		t.Fatalf("expected ollama/test-model, got %s/%s", sess.Provider, sess.Model)
	}

	history, err := db.GetHistory(sess.ID, 10)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 2 || history[0].Role != "user" || history[0].Content != "hi" ||
		history[1].Role != "assistant" || history[1].Content != "hello there" {
		t.Fatalf("unexpected history: %+v", history)
	}

	req := fake.lastRequest(t)
	if req.Model != "test-model" {
		t.Fatalf("expected model %q sent to server, got %q", "test-model", req.Model)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "hi" {
		t.Fatalf("expected a single message [hi] on first turn, got %+v", req.Messages)
	}
}

func TestReplyIncludesPriorHistory(t *testing.T) {
	fake := newFakeOllama(t, "r1", "r2")
	mgr, _ := newTestManager(t, fake.URL, 20)
	const userID = int64(1)

	if _, err := mgr.Reply(context.Background(), userID, "m1"); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if _, err := mgr.Reply(context.Background(), userID, "m2"); err != nil {
		t.Fatalf("Reply: %v", err)
	}

	req := fake.lastRequest(t)
	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 messages (2 history + new), got %d: %+v", len(req.Messages), req.Messages)
	}
	if req.Messages[0].Content != "m1" || req.Messages[1].Content != "r1" || req.Messages[2].Content != "m2" {
		t.Fatalf("unexpected message sequence: %+v", req.Messages)
	}
}

func TestReplyHistoryWindowing(t *testing.T) {
	fake := newFakeOllama(t, "r1", "r2", "r3")
	mgr, _ := newTestManager(t, fake.URL, 2)
	const userID = int64(1)

	for _, text := range []string{"m1", "m2", "m3"} {
		if _, err := mgr.Reply(context.Background(), userID, text); err != nil {
			t.Fatalf("Reply: %v", err)
		}
	}

	req := fake.lastRequest(t)
	if len(req.Messages) != 3 {
		t.Fatalf("expected window of 2 history + new message, got %d: %+v", len(req.Messages), req.Messages)
	}
	if req.Messages[0].Content != "m2" || req.Messages[1].Content != "r2" || req.Messages[2].Content != "m3" {
		t.Fatalf("expected [m2, r2, m3] (m1/r1 dropped by the window), got %+v", req.Messages)
	}
}

func TestSetModelUpdatesSessionWithoutClearingHistory(t *testing.T) {
	fake := newFakeOllama(t, "first", "second")
	mgr, db := newTestManager(t, fake.URL, 20)
	const userID = int64(1)

	if _, err := mgr.Reply(context.Background(), userID, "hi"); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	sess, err := db.GetActiveSession(userID)
	if err != nil {
		t.Fatalf("GetActiveSession: %v", err)
	}

	if err := mgr.SetModel(userID, llm.ProviderOllama, "new-model"); err != nil {
		t.Fatalf("SetModel: %v", err)
	}

	updated, err := db.GetActiveSession(userID)
	if err != nil {
		t.Fatalf("GetActiveSession: %v", err)
	}
	if updated.Model != "new-model" {
		t.Fatalf("expected model to be updated to new-model, got %s", updated.Model)
	}

	history, err := db.GetHistory(sess.ID, 10)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected history to survive the model switch untouched, got %d messages", len(history))
	}

	if _, err := mgr.Reply(context.Background(), userID, "again"); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	req := fake.lastRequest(t)
	if req.Model != "new-model" {
		t.Fatalf("expected subsequent reply to use new-model, got %s", req.Model)
	}
}

func TestNewSessionEmptyTitleUsesTimestamp(t *testing.T) {
	fake := newFakeOllama(t)
	mgr, _ := newTestManager(t, fake.URL, 20)

	sess, err := mgr.NewSession(1, "")
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if _, err := time.Parse(titleTimeFormat, sess.Title); err != nil {
		t.Fatalf("expected title %q to be a timestamp in %q: %v", sess.Title, titleTimeFormat, err)
	}
}

func TestNewSessionAndSwitchSession(t *testing.T) {
	fake := newFakeOllama(t)
	mgr, db := newTestManager(t, fake.URL, 20)
	const userID = int64(1)

	s1, err := mgr.NewSession(userID, "s1")
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s2, err := mgr.NewSession(userID, "s2")
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}

	sessions, err := mgr.ListSessions(userID)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 2 || sessions[0].ID != s2.ID || !sessions[0].IsActive || sessions[1].IsActive {
		t.Fatalf("unexpected session list: %+v", sessions)
	}

	if err := mgr.SwitchSession(userID, s1.ID); err != nil {
		t.Fatalf("SwitchSession: %v", err)
	}
	active, err := db.GetActiveSession(userID)
	if err != nil {
		t.Fatalf("GetActiveSession: %v", err)
	}
	if active.ID != s1.ID {
		t.Fatalf("expected session %d to be active, got %d", s1.ID, active.ID)
	}
}

func TestModelsForOllamaUsesConfiguredServer(t *testing.T) {
	fake := newFakeOllama(t)
	fake.tags = []string{"llama3.2:latest", "gemma4:latest"}
	mgr, _ := newTestManager(t, fake.URL, 20)

	got, err := mgr.ModelsFor(context.Background(), llm.ProviderOllama)
	if err != nil {
		t.Fatalf("ModelsFor: %v", err)
	}
	if len(got) != 2 || got[0] != "llama3.2:latest" || got[1] != "gemma4:latest" {
		t.Fatalf("expected [llama3.2:latest, gemma4:latest], got %v", got)
	}
}
