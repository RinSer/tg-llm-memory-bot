package session

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/memory"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

// GlobalMemory is the slice of the memory.Manager that session.Manager
// depends on. Kept as an interface so tests can substitute a fake without
// standing up the real embedding/summarization pipeline.
type GlobalMemory interface {
	RelevantFacts(ctx context.Context, userID int64, queryText string) ([]string, error)
	Signal(userID, sessionID int64, kind memory.SignalKind)
	SetSummarizationModel(userID int64, provider llm.ProviderName, model string) error
	IsCompacting(userID int64) bool
	CompactionStatus(userID int64) (rows, target int, err error)
	StartCompaction(ctx context.Context, userID int64) (before, after int, err error)
}

// Config holds everything the Manager needs that isn't per-session state:
// provider credentials/endpoints and defaults for newly created sessions.
type Config struct {
	HistoryLimit    int
	DefaultProvider llm.ProviderName
	DefaultModel    string
	OpenAIAPIToken  string
	OllamaBaseURL   string
	OllamaKeepAlive string
	GlobalMemory    GlobalMemory
}

type Manager struct {
	store *store.Store
	cfg   Config
}

func NewManager(s *store.Store, cfg Config) *Manager {
	return &Manager{store: s, cfg: cfg}
}

// compactionInProgressMsg is returned to the user if they try to chat while
// their global memory is being compacted.
const compactionInProgressMsg = "Global memory compaction is in progress. Please try again in a moment."

const titleTimeFormat = "2006-01-02 15:04:05"

// DefaultTitle returns a timestamp-based session title, used whenever a
// session is created without an explicit one.
func DefaultTitle() string {
	return time.Now().Format(titleTimeFormat)
}

func (m *Manager) providerConfig(providerName llm.ProviderName, model string) llm.Config {
	return llm.Config{
		Name:      providerName,
		Model:     model,
		APIToken:  m.cfg.OpenAIAPIToken,
		BaseURL:   m.cfg.OllamaBaseURL,
		KeepAlive: m.cfg.OllamaKeepAlive,
	}
}

func (m *Manager) activeSession(userID int64) (*store.Session, error) {
	sess, err := m.store.GetActiveSession(userID)
	if errors.Is(err, store.ErrNoActiveSession) {
		return m.store.CreateSession(userID, DefaultTitle(), string(m.cfg.DefaultProvider), m.cfg.DefaultModel)
	}
	return sess, err
}

// Reply appends the user's message and the model's reply to the user's
// active session (creating one if none exists yet) and returns the reply.
func (m *Manager) Reply(ctx context.Context, userID int64, text string) (string, error) {
	return m.reply(ctx, userID, text, nil)
}

// ReplyStream is like Reply, but invokes onChunk with each piece of the
// reply as the model produces it. The full reply is still persisted and
// returned only once generation finishes.
func (m *Manager) ReplyStream(ctx context.Context, userID int64, text string, onChunk func(chunk string)) (string, error) {
	return m.reply(ctx, userID, text, onChunk)
}

func (m *Manager) reply(ctx context.Context, userID int64, text string, onChunk func(chunk string)) (string, error) {
	// Block chatting for a user whose global memory is being compacted --
	// before any history/provider/DB side effects.
	if m.cfg.GlobalMemory != nil && m.cfg.GlobalMemory.IsCompacting(userID) {
		return compactionInProgressMsg, nil
	}

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

	messages := make([]llm.Message, 0, len(history)+2)
	// Prepend relevant long-term facts as a system message, if any.
	if m.cfg.GlobalMemory != nil {
		facts, ferr := m.cfg.GlobalMemory.RelevantFacts(ctx, userID, text)
		if ferr != nil {
			// Retrieval is best-effort: log-worthy but never fatal to a chat.
			facts = nil
		}
		if len(facts) > 0 {
			messages = append(messages, llm.Message{
				Role:    llm.RoleSystem,
				Content: "Known facts from the user:\n" + strings.Join(facts, "\n"),
			})
		}
	}
	for _, h := range history {
		messages = append(messages, llm.Message{Role: llm.Role(h.Role), Content: h.Content})
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: text})

	provider, err := llm.New(m.providerConfig(llm.ProviderName(sess.Provider), sess.Model))
	if err != nil {
		return "", err
	}

	var reply string
	if onChunk != nil {
		reply, err = provider.GenerateStream(ctx, messages, onChunk)
	} else {
		reply, err = provider.Generate(ctx, messages)
	}
	if err != nil {
		return "", err
	}

	if err := m.store.AppendMessage(sess.ID, string(llm.RoleAssistant), reply); err != nil {
		return "", err
	}

	// Wake the summarization worker; the threshold gate lives in the worker.
	if m.cfg.GlobalMemory != nil {
		m.cfg.GlobalMemory.Signal(userID, sess.ID, memory.SignalMessageThreshold)
	}

	return reply, nil
}

// NewSession creates a session with the given title, or a timestamp-based
// one if title is empty. Creating a session deactivates whatever was
// active, so its unsaved tail is flushed to global memory first.
func (m *Manager) NewSession(userID int64, title string) (*store.Session, error) {
	if title == "" {
		title = DefaultTitle()
	}

	// Inherit the provider/model of the session being left, so a new
	// session doesn't silently revert to the defaults after the user
	// deliberately switched models. Falls back to the configured defaults
	// only on true first use (no prior session).
	provider, model := string(m.cfg.DefaultProvider), m.cfg.DefaultModel
	if prev, err := m.store.GetActiveSession(userID); err == nil {
		provider, model = prev.Provider, prev.Model
		if m.cfg.GlobalMemory != nil {
			m.cfg.GlobalMemory.Signal(userID, prev.ID, memory.SignalSessionSwitch)
		}
	}
	return m.store.CreateSession(userID, title, provider, model)
}

func (m *Manager) ListSessions(userID int64) ([]store.Session, error) {
	return m.store.ListSessions(userID)
}

// SwitchSession activates a different session; the one being switched away
// from has its unsaved tail flushed to global memory.
func (m *Manager) SwitchSession(userID, sessionID int64) error {
	m.signalDeactivated(userID)
	return m.store.SetActiveSession(userID, sessionID)
}

// signalDeactivated flushes the currently-active session (if any) to global
// memory before it's deactivated by a new/switch operation.
func (m *Manager) signalDeactivated(userID int64) {
	if m.cfg.GlobalMemory == nil {
		return
	}
	if sess, err := m.store.GetActiveSession(userID); err == nil {
		m.cfg.GlobalMemory.Signal(userID, sess.ID, memory.SignalSessionSwitch)
	}
}

// SaveGlobalMemory manually triggers summarization of the user's active
// session's unsaved messages (the /save command).
func (m *Manager) SaveGlobalMemory(userID int64) error {
	if m.cfg.GlobalMemory == nil {
		return nil
	}
	sess, err := m.activeSession(userID)
	if err != nil {
		return err
	}
	m.cfg.GlobalMemory.Signal(userID, sess.ID, memory.SignalManual)
	return nil
}

// SetSummarizationModel sets the user's summarization LLM (the
// /summarymodel command).
func (m *Manager) SetSummarizationModel(userID int64, provider llm.ProviderName, model string) error {
	if m.cfg.GlobalMemory == nil {
		return nil
	}
	return m.cfg.GlobalMemory.SetSummarizationModel(userID, provider, model)
}

// CompactionStatus reports the user's current fact-row count and target.
func (m *Manager) CompactionStatus(userID int64) (rows, target int, err error) {
	if m.cfg.GlobalMemory == nil {
		return 0, 0, nil
	}
	return m.cfg.GlobalMemory.CompactionStatus(userID)
}

// StartCompaction runs (and blocks on) compaction for the user.
func (m *Manager) StartCompaction(ctx context.Context, userID int64) (before, after int, err error) {
	if m.cfg.GlobalMemory == nil {
		return 0, 0, nil
	}
	return m.cfg.GlobalMemory.StartCompaction(ctx, userID)
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
