package llm

import (
	"context"

	"github.com/tmc/langchaingo/llms/openai"
)

// OpenAIModel is a hardcoded selection of chat-capable OpenAI models.
type OpenAIModel string

const (
	OpenAIModelGPT56Sol   OpenAIModel = "gpt-5.6-sol"
	OpenAIModelGPT56Terra OpenAIModel = "gpt-5.6-terra"
	OpenAIModelGPT56Luna  OpenAIModel = "gpt-5.6-luna"
	OpenAIModelGPT4oMini  OpenAIModel = "gpt-4o-mini"
	OpenAIModelGPT4o      OpenAIModel = "gpt-4o"
	OpenAIModelGPT41Nano  OpenAIModel = "gpt-4.1-nano"
	OpenAIModelGPT41Mini  OpenAIModel = "gpt-4.1-mini"
	OpenAIModelGPT41      OpenAIModel = "gpt-4.1"
	OpenAIModelO3Mini     OpenAIModel = "o3-mini"
	OpenAIModelO1Mini     OpenAIModel = "o1-mini"
)

// OpenAIModels lists the selectable models, in display order.
var OpenAIModels = []OpenAIModel{
	OpenAIModelGPT56Sol,
	OpenAIModelGPT56Terra,
	OpenAIModelGPT56Luna,
	OpenAIModelGPT4oMini,
	OpenAIModelGPT4o,
	OpenAIModelGPT41Nano,
	OpenAIModelGPT41Mini,
	OpenAIModelGPT41,
	OpenAIModelO3Mini,
	OpenAIModelO1Mini,
}

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

func (o *opanai) Generate(ctx context.Context, messages []Message) (string, error) {
	return generate(ctx, o.model, messages)
}
