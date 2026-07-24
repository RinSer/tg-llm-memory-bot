package llm

import (
	"context"
	"testing"
)

func TestNewUnknownProviderIsAnError(t *testing.T) {
	_, err := New(Config{Name: ProviderName("unknown"), Model: "whatever"})
	if err == nil {
		t.Fatal("expected an error for an unknown provider")
	}
}

func TestNewOpenAIRequiresNoNetworkCall(t *testing.T) {
	// openai.New only builds a client; it must not error out just because
	// there's no network access in the test environment.
	p, err := New(Config{Name: ProviderOpenAI, Model: string(OpenAIModelGPT4oMini), APIToken: "test-token"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("expected a non-nil provider")
	}
}

func TestModelsForOpenAIIsStaticAndNonEmpty(t *testing.T) {
	models, err := ModelsFor(context.Background(), ProviderOpenAI, "")
	if err != nil {
		t.Fatalf("ModelsFor: %v", err)
	}
	if len(models) != len(OpenAIModels) {
		t.Fatalf("expected %d models, got %d", len(OpenAIModels), len(models))
	}
	if models[0] != string(OpenAIModels[0]) {
		t.Fatalf("expected first model %q, got %q", OpenAIModels[0], models[0])
	}
}

func TestModelsForUnknownProviderIsAnError(t *testing.T) {
	_, err := ModelsFor(context.Background(), ProviderName("unknown"), "")
	if err == nil {
		t.Fatal("expected an error for an unknown provider")
	}
}
