package llm

import (
	"context"
	"fmt"
)

// Role identifies the author of a Message. Kept provider-agnostic so
// conversation history can be persisted and replayed against any Provider.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Message is a single turn of conversation, independent of any LLM SDK type.
type Message struct {
	Role    Role
	Content string
}

// Provider generates a reply given the full message history.
type Provider interface {
	Generate(ctx context.Context, messages []Message) (string, error)

	// GenerateStream is like Generate, but invokes onChunk with each piece
	// of text as the model produces it, in addition to returning the full
	// reply once generation finishes.
	GenerateStream(ctx context.Context, messages []Message, onChunk func(chunk string)) (string, error)
}

// ProviderName identifies a supported LLM backend.
type ProviderName string

const (
	ProviderOpenAI ProviderName = "openai"
	ProviderOllama ProviderName = "ollama"
)

// Providers lists every supported backend, in display order.
var Providers = []ProviderName{ProviderOpenAI, ProviderOllama}

// ModelsFor returns the selectable models for a provider. OpenAI's list is
// a hardcoded whitelist; Ollama's is queried live from the server, since it
// depends on whatever the user has actually pulled locally.
func ModelsFor(ctx context.Context, p ProviderName, baseURL string) ([]string, error) {
	switch p {
	case ProviderOpenAI:
		return modelStrings(OpenAIModels), nil
	case ProviderOllama:
		return listOllamaModels(ctx, baseURL)
	default:
		return nil, fmt.Errorf("unknown LLM provider: %v", p)
	}
}

func modelStrings[T ~string](models []T) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = string(m)
	}
	return out
}

// Config configures a Provider. Fields not relevant to the chosen provider
// are ignored.
type Config struct {
	Name      ProviderName
	Model     string
	APIToken  string // required for openai
	BaseURL   string // optional for ollama, defaults to http://localhost:11434
	KeepAlive string // optional for ollama, e.g. "-1" to keep the model loaded indefinitely
}

func New(cfg Config) (Provider, error) {
	switch cfg.Name {
	case ProviderOpenAI:
		return newOpenAI(cfg.Model, cfg.APIToken)
	case ProviderOllama:
		return newOllama(cfg.Model, cfg.BaseURL, cfg.KeepAlive)
	default:
		return nil, fmt.Errorf("unknown LLM provider: %v", cfg.Name)
	}
}
