package llm

// llm/llm.go
package llm

import (
    "context"
    "fmt"
)

type Provider interface {
    Generate(ctx context.Context, prompt string) (string, error)
}

func New(providerName, apiToken string) (Provider, error) {
    switch providerName {
    case "openai":
        return newOpenAI(apiToken)
    default:
        return nil, fmt.Errorf("unknown LLM provider: %v", providerName)
    }
}
