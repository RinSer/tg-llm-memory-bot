package llm

import (
	"context"
	"fmt"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
)

type Provider interface {
	Talk(ctx context.Context, b *bot.Bot, update *models.Update)
}

func New(providerName, model, apiToken string) (Provider, error) {
	switch providerName {
	case "openai":
		return newOpenAI(model, apiToken)
	default:
		return nil, fmt.Errorf("unknown LLM provider: %v", providerName)
	}
}
