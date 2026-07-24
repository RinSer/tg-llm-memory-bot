package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListOllamaModelsParsesTagsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("expected request to /api/tags, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"models":[{"name":"llama3.2:latest"},{"name":"gemma4:latest"}]}`))
	}))
	defer srv.Close()

	got, err := listOllamaModels(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("listOllamaModels: %v", err)
	}
	want := []string{"llama3.2:latest", "gemma4:latest"}
	if len(got) != len(want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %v, got %v", want, got)
		}
	}
}

func TestListOllamaModelsServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := listOllamaModels(context.Background(), srv.URL)
	if err == nil {
		t.Fatal("expected an error when the server returns a non-200 status")
	}
}

func TestModelsForOllamaDelegatesToServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"models":[{"name":"mistral:latest"}]}`))
	}))
	defer srv.Close()

	got, err := ModelsFor(context.Background(), ProviderOllama, srv.URL)
	if err != nil {
		t.Fatalf("ModelsFor: %v", err)
	}
	if len(got) != 1 || got[0] != "mistral:latest" {
		t.Fatalf("expected [mistral:latest], got %v", got)
	}
}
