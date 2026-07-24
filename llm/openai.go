package llm

import (
	"context"
	"log"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

type LLM struct {
	model *openai.LLM
}

func New(apiToken string) (*LLM, error) {
	llm, err := openai.New(
		openai.WithModel("gpt-4.1-mini"),
		openai.WithToken(apiToken),
	)
	return &LLM{llm}, err
}

func (llm *LLM) Handler(ctx context.Context, b *bot.Bot, update *models.Update) {
	prompt := update.Message.Text
	completion, err := llms.GenerateFromSinglePrompt(ctx, llm.model, prompt)
	if err != nil {
		log.Println(err)
	}
	b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: update.Message.Chat.ID,
		Text:   completion,
	})
}
