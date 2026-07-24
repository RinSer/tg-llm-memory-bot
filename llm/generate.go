package llm

import (
	"context"
	"errors"

	"github.com/tmc/langchaingo/llms"
)

func generate(ctx context.Context, model llms.Model, messages []Message) (string, error) {
	content := make([]llms.MessageContent, len(messages))
	for i, m := range messages {
		content[i] = llms.MessageContent{
			Role:  chatMessageType(m.Role),
			Parts: []llms.ContentPart{llms.TextContent{Text: m.Content}},
		}
	}

	resp, err := model.GenerateContent(ctx, content)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("no response from model")
	}
	return resp.Choices[0].Content, nil
}

func chatMessageType(role Role) llms.ChatMessageType {
	switch role {
	case RoleAssistant:
		return llms.ChatMessageTypeAI
	case RoleSystem:
		return llms.ChatMessageTypeSystem
	default:
		return llms.ChatMessageTypeHuman
	}
}
