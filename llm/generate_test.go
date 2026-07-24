package llm

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/fake"
)

func TestGenerateMapsRolesAndReturnsFirstChoice(t *testing.T) {
	f := fake.NewFakeLLM([]string{"canned reply"})

	reply, err := generate(context.Background(), f, []Message{
		{Role: RoleSystem, Content: "be nice"},
		{Role: RoleUser, Content: "hi"},
		{Role: RoleAssistant, Content: "hello"},
		{Role: RoleUser, Content: "how are you"},
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if reply != "canned reply" {
		t.Fatalf("expected %q, got %q", "canned reply", reply)
	}
}

func TestGenerateNoChoicesIsAnError(t *testing.T) {
	stub := stubModel{resp: &llms.ContentResponse{}}

	_, err := generate(context.Background(), stub, []Message{{Role: RoleUser, Content: "hi"}})
	if err == nil {
		t.Fatal("expected an error when the model returns no choices")
	}
}

func TestGeneratePropagatesModelError(t *testing.T) {
	wantErr := errors.New("boom")
	stub := stubModel{err: wantErr}

	_, err := generate(context.Background(), stub, []Message{{Role: RoleUser, Content: "hi"}})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, err)
	}
}

func TestChatMessageType(t *testing.T) {
	cases := []struct {
		role Role
		want llms.ChatMessageType
	}{
		{RoleUser, llms.ChatMessageTypeHuman},
		{RoleAssistant, llms.ChatMessageTypeAI},
		{RoleSystem, llms.ChatMessageTypeSystem},
		{Role("unknown"), llms.ChatMessageTypeHuman},
	}
	for _, c := range cases {
		if got := chatMessageType(c.role); got != c.want {
			t.Errorf("chatMessageType(%q) = %q, want %q", c.role, got, c.want)
		}
	}
}

// stubModel is a minimal llms.Model for exercising error/edge-case paths
// that fake.LLM (always one choice, never errors on empty input) can't.
type stubModel struct {
	resp *llms.ContentResponse
	err  error
}

func (s stubModel) GenerateContent(_ context.Context, _ []llms.MessageContent, _ ...llms.CallOption) (*llms.ContentResponse, error) {
	return s.resp, s.err
}

func (s stubModel) Call(_ context.Context, _ string, _ ...llms.CallOption) (string, error) {
	return "", s.err
}
