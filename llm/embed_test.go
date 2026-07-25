package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOllamaEmbed(t *testing.T) {
	var gotModel, gotInput string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("expected /api/embed, got %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
			Input string `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("unmarshal: %v", err)
		}
		gotModel, gotInput = req.Model, req.Input
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"embeddings":[[0.1,0.2,0.3]]}`))
	}))
	defer srv.Close()

	e, err := NewEmbedder(Config{Name: ProviderOllama, Model: "embeddinggemma", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 3 || vecs[0][0] != 0.1 {
		t.Fatalf("unexpected vectors: %v", vecs)
	}
	if gotModel != "embeddinggemma" || gotInput != "hello" {
		t.Fatalf("expected model=embeddinggemma input=hello, got model=%s input=%s", gotModel, gotInput)
	}
}

func TestNewEmbedderUnknownProvider(t *testing.T) {
	if _, err := NewEmbedder(Config{Name: ProviderName("nope"), Model: "x"}); err == nil {
		t.Fatal("expected an error for an unknown embedding provider")
	}
}
