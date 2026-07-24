package llm

import (
	"context"
	"errors"

	"github.com/tmc/langchaingo/llms"
)

func generate(ctx context.Context, model llms.Model, messages []Message, opts ...llms.CallOption) (string, error) {
	content := make([]llms.MessageContent, len(messages))
	for i, m := range messages {
		content[i] = llms.MessageContent{
			Role:  chatMessageType(m.Role),
			Parts: []llms.ContentPart{llms.TextContent{Text: m.Content}},
		}
	}

	resp, err := model.GenerateContent(ctx, content, opts...)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("no response from model")
	}
	return resp.Choices[0].Content, nil
}

// generateStream is generate, but forwards each chunk the model produces to
// onChunk as it arrives. It still returns the full accumulated reply once
// generation finishes -- langchaingo's streaming Model implementations
// assemble the complete text into the final response regardless.
func generateStream(ctx context.Context, model llms.Model, messages []Message, onChunk func(chunk string)) (string, error) {
	return generate(ctx, model, messages, llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
		onChunk(string(chunk))
		return nil
	}))
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
