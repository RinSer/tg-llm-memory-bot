package llm

import (
	"context"
	"log"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

type opanai struct {
	model *openai.LLM
}

func newOpenAI(model, apiToken string) (*opanai, error) {
	llm, err := openai.New(
		openai.WithModel(model),
		openai.WithToken(apiToken),
	)
	return &opanai{llm}, err
}

func (llm *opanai) Talk(ctx context.Context, b *bot.Bot, update *models.Update) {
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
