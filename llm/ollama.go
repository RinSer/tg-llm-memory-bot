package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmc/langchaingo/llms/ollama"
)

const defaultOllamaBaseURL = "http://localhost:11434"

type localOllama struct {
	model *ollama.LLM
}

// listOllamaModels queries the Ollama server's /api/tags endpoint for the
// models currently pulled locally -- the same list `ollama list` prints.
func listOllamaModels(ctx context.Context, baseURL string) ([]string, error) {
	if baseURL == "" {
		baseURL = defaultOllamaBaseURL
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(baseURL, "/")+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama server returned %s", resp.Status)
	}

	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return nil, err
	}

	names := make([]string, len(tags.Models))
	for i, m := range tags.Models {
		names[i] = m.Name
	}
	return names, nil
}

func newOllama(model, baseURL, keepAlive string) (*localOllama, error) {
	if baseURL == "" {
		baseURL = defaultOllamaBaseURL
	}
	opts := []ollama.Option{
		ollama.WithModel(model),
		ollama.WithServerURL(baseURL),
	}
	if keepAlive != "" {
		// e.g. "-1" keeps the model loaded in memory/VRAM indefinitely,
		// avoiding a reload on every request. Flash attention is a
		// separate, server-side-only setting (OLLAMA_FLASH_ATTENTION=1
		// before starting `ollama serve`) -- there's no per-request or
		// client-side equivalent to set it here.
		opts = append(opts, ollama.WithKeepAlive(keepAlive))
	}
	llm, err := ollama.New(opts...)
	return &localOllama{llm}, err
}

func (o *localOllama) Generate(ctx context.Context, messages []Message) (string, error) {
	return generate(ctx, o.model, messages)
}

func (o *localOllama) GenerateStream(ctx context.Context, messages []Message, onChunk func(string)) (string, error) {
	return generateStream(ctx, o.model, messages, onChunk)
}
