package session

import (
	"context"
	"errors"
	"time"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

// Config holds everything the Manager needs that isn't per-session state:
// provider credentials/endpoints and defaults for newly created sessions.
type Config struct {
	HistoryLimit    int
	DefaultProvider llm.ProviderName
	DefaultModel    string
	OpenAIAPIToken  string
	OllamaBaseURL   string
}

type Manager struct {
	store *store.Store
	cfg   Config
}

func NewManager(s *store.Store, cfg Config) *Manager {
	return &Manager{store: s, cfg: cfg}
}

func (m *Manager) providerConfig(providerName llm.ProviderName, model string) llm.Config {
	return llm.Config{
		Name:     providerName,
		Model:    model,
		APIToken: m.cfg.OpenAIAPIToken,
		BaseURL:  m.cfg.OllamaBaseURL,
	}
}

func (m *Manager) activeSession(userID int64) (*store.Session, error) {
	sess, err := m.store.GetActiveSession(userID)
	if errors.Is(err, store.ErrNoActiveSession) {
		return m.store.CreateSession(userID, time.Now().Format(time.RFC3339), string(m.cfg.DefaultProvider), m.cfg.DefaultModel)
	}
	return sess, err
}

// Reply appends the user's message and the model's reply to the user's
// active session (creating one if none exists yet) and returns the reply.
func (m *Manager) Reply(ctx context.Context, userID int64, text string) (string, error) {
	sess, err := m.activeSession(userID)
	if err != nil {
		return "", err
	}

	history, err := m.store.GetHistory(sess.ID, m.cfg.HistoryLimit)
	if err != nil {
		return "", err
	}

	if err := m.store.AppendMessage(sess.ID, string(llm.RoleUser), text); err != nil {
		return "", err
	}

	messages := make([]llm.Message, 0, len(history)+1)
	for _, h := range history {
		messages = append(messages, llm.Message{Role: llm.Role(h.Role), Content: h.Content})
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: text})

	provider, err := llm.New(m.providerConfig(llm.ProviderName(sess.Provider), sess.Model))
	if err != nil {
		return "", err
	}

	reply, err := provider.Generate(ctx, messages)
	if err != nil {
		return "", err
	}

	if err := m.store.AppendMessage(sess.ID, string(llm.RoleAssistant), reply); err != nil {
		return "", err
	}

	return reply, nil
}

func (m *Manager) NewSession(userID int64, title string) (*store.Session, error) {
	return m.store.CreateSession(userID, title, string(m.cfg.DefaultProvider), m.cfg.DefaultModel)
}

func (m *Manager) ListSessions(userID int64) ([]store.Session, error) {
	return m.store.ListSessions(userID)
}

func (m *Manager) SwitchSession(userID, sessionID int64) error {
	return m.store.SetActiveSession(userID, sessionID)
}

// ModelsFor returns the selectable models for a provider -- a hardcoded
// whitelist for OpenAI, queried live from the server for Ollama.
func (m *Manager) ModelsFor(ctx context.Context, providerName llm.ProviderName) ([]string, error) {
	return llm.ModelsFor(ctx, providerName, m.cfg.OllamaBaseURL)
}

// SetModel switches the provider/model of the user's active session.
// History is untouched -- it's stored as plain text, not tied to any
// provider SDK type.
func (m *Manager) SetModel(userID int64, providerName llm.ProviderName, model string) error {
	sess, err := m.activeSession(userID)
	if err != nil {
		return err
	}
	return m.store.SetSessionModel(sess.ID, string(providerName), model)
}
