package memory

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/store"
)

// SignalKind identifies why the summarization worker was woken. It gates
// how eager a wake is (threshold vs. any pending), but never serves as the
// sole source of what work exists -- the worker always re-derives that from
// checkpoints, so a dropped signal is harmless.
type SignalKind int

const (
	SignalMessageThreshold SignalKind = iota // N-new-messages wake; only acts past the threshold
	SignalSessionSwitch                       // a session was deactivated; flush any pending tail
	SignalManual                              // user ran /save; flush any pending tail
	SignalCompaction                          // user ran /compact; consolidate this user's rows
)

type signal struct {
	sessionID int64
	kind      SignalKind
	done      chan error // non-nil only for SignalCompaction (synchronous)
}

// Notifier delivers an unprompted message to a user's chat (used for the
// "global memory nearly full" warning). Optional; nil is fine.
type Notifier interface {
	Notify(userID int64, message string)
}

// Config configures the memory Manager. Embedding reuses llm.Config as-is.
type Config struct {
	Store         *store.Store
	Embedding     llm.Config
	MessageThreshold int
	TopK          int
	MinSimilarity float32
	MaxRows       int

	DefaultSummarizationProvider llm.ProviderName
	DefaultSummarizationModel    string

	// Credentials/endpoints for building the summarization chat provider
	// (which can be any chat-capable provider/model the user picks).
	OpenAIAPIToken  string
	OllamaBaseURL   string
	OllamaKeepAlive string

	Notifier Notifier
}

// summarizationConfig builds the llm.Config for a summarization chat call.
func (c Config) summarizationConfig(provider llm.ProviderName, model string) llm.Config {
	return llm.Config{
		Name:      provider,
		Model:     model,
		APIToken:  c.OpenAIAPIToken,
		BaseURL:   c.OllamaBaseURL,
		KeepAlive: c.OllamaKeepAlive,
	}
}

type Manager struct {
	cfg      Config
	embedder llm.Embedder

	available bool // set by CheckAccessible; when false, memory is a no-op

	chans map[int64]chan signal // per-user signal channel

	mu         sync.Mutex
	compacting map[int64]bool
}

func New(cfg Config) (*Manager, error) {
	embedder, err := llm.NewEmbedder(cfg.Embedding)
	if err != nil {
		return nil, err
	}
	return &Manager{
		cfg:        cfg,
		embedder:   embedder,
		chans:      make(map[int64]chan signal),
		compacting: make(map[int64]bool),
	}, nil
}

// CheckEmbeddingModel enforces req #10: the DB is bound to one embedding
// provider/model. A fresh DB is bootstrapped with the configured pair; a
// mismatch is a fatal error the caller must surface, since it would
// silently corrupt cosine similarity across old and new vectors.
func (m *Manager) CheckEmbeddingModel() error {
	dbProvider, dbModel, ok, err := m.cfg.Store.GetEmbeddingConfig()
	if err != nil {
		return err
	}
	write, err := decideEmbeddingConfig(dbProvider, dbModel, ok, string(m.cfg.Embedding.Name), m.cfg.Embedding.Model)
	if err != nil {
		return err
	}
	if write {
		return m.cfg.Store.SetEmbeddingConfig(string(m.cfg.Embedding.Name), m.cfg.Embedding.Model)
	}
	return nil
}

// decideEmbeddingConfig is the pure decision behind CheckEmbeddingModel,
// separated so the fatal-vs-bootstrap branch is unit-testable without a DB.
func decideEmbeddingConfig(dbProvider, dbModel string, dbOK bool, cfgProvider, cfgModel string) (writeBootstrap bool, err error) {
	if !dbOK {
		return true, nil // fresh DB: bind it to the configured model
	}
	if dbProvider != cfgProvider || dbModel != cfgModel {
		return false, fmt.Errorf(
			"embedding model mismatch: database was built with %s/%s but configured with %s/%s; "+
				"global memory requires a matching model (revert the env var, or use a new DB_PATH for the new model)",
			dbProvider, dbModel, cfgProvider, cfgModel,
		)
	}
	return false, nil
}

// CheckAccessible probes the embedding endpoint once. On success memory is
// enabled; on failure it stays disabled (a no-op) for this run and the
// error is returned for the caller to log as a warning (not fatal).
func (m *Manager) CheckAccessible(ctx context.Context) error {
	if _, err := m.embedder.Embed(ctx, []string{"ping"}); err != nil {
		m.available = false
		return err
	}
	m.available = true
	return nil
}

// Available reports whether embedding was reachable at startup.
func (m *Manager) Available() bool { return m.available }

// RelevantFacts embeds queryText and returns the contents of the user's
// most similar stored facts, filtered by the similarity threshold and
// capped. Returns nil (no error) when memory is unavailable.
func (m *Manager) RelevantFacts(ctx context.Context, userID int64, queryText string) ([]string, error) {
	if !m.available {
		return nil, nil
	}
	vecs, err := m.embedder.Embed(ctx, []string{queryText})
	if err != nil || len(vecs) == 0 {
		return nil, err
	}
	rows, err := m.cfg.Store.ListGlobalMemory(userID)
	if err != nil {
		return nil, err
	}
	ranked := rankFacts(vecs[0], rows, m.cfg.MinSimilarity, m.cfg.TopK)
	out := make([]string, len(ranked))
	for i, r := range ranked {
		out[i] = r.Content
	}
	return out, nil
}

// SetSummarizationModel records the user's chosen summarization LLM.
func (m *Manager) SetSummarizationModel(userID int64, provider llm.ProviderName, model string) error {
	return m.cfg.Store.SetSummarizationModel(userID, string(provider), model)
}

// Signal wakes a user's worker. Non-blocking: if the buffer is full the
// signal is dropped, which is safe because the worker re-derives pending
// work from checkpoints regardless.
func (m *Manager) Signal(userID, sessionID int64, kind SignalKind) {
	ch := m.chans[userID]
	if ch == nil {
		return
	}
	select {
	case ch <- signal{sessionID: sessionID, kind: kind}:
	default:
	}
}

// IsCompacting reports whether a compaction is currently running for the
// user (used to block that user's chat).
func (m *Manager) IsCompacting(userID int64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.compacting[userID]
}

// CompactionStatus returns the user's current fact-row count and the
// compaction target (MaxRows/2).
func (m *Manager) CompactionStatus(userID int64) (rows, target int, err error) {
	n, err := m.cfg.Store.CountGlobalMemory(userID)
	if err != nil {
		return 0, 0, err
	}
	return n, m.cfg.MaxRows / 2, nil
}

var errCompactionBusy = errors.New("a compaction is already running for this user")

// StartCompaction runs compaction for the user and blocks until it
// finishes (or ctx is done). Chat for this user is rejected meanwhile via
// IsCompacting. Returns the before/after row counts.
func (m *Manager) StartCompaction(ctx context.Context, userID int64) (before, after int, err error) {
	ch := m.chans[userID]
	if ch == nil {
		return 0, 0, errors.New("memory worker not running for this user")
	}

	m.mu.Lock()
	if m.compacting[userID] {
		m.mu.Unlock()
		return 0, 0, errCompactionBusy
	}
	m.compacting[userID] = true
	m.mu.Unlock()

	before, _ = m.cfg.Store.CountGlobalMemory(userID)

	done := make(chan error, 1)
	select {
	case ch <- signal{kind: SignalCompaction, done: done}:
	case <-ctx.Done():
		m.clearCompacting(userID)
		return 0, 0, ctx.Err()
	}

	select {
	case err = <-done:
	case <-ctx.Done():
		err = ctx.Err()
	}

	after, _ = m.cfg.Store.CountGlobalMemory(userID)
	return before, after, err
}

func (m *Manager) clearCompacting(userID int64) {
	m.mu.Lock()
	m.compacting[userID] = false
	m.mu.Unlock()
}

// Start launches one worker goroutine per user. Each does an immediate
// catch-up pass, then waits for signals until ctx is cancelled.
func (m *Manager) Start(ctx context.Context, userIDs []int64) {
	for _, id := range userIDs {
		ch := make(chan signal, 4)
		m.chans[id] = ch
		go m.runWorker(ctx, id, ch)
	}
}

func (m *Manager) runWorker(ctx context.Context, userID int64, ch chan signal) {
	// Immediate catch-up: summarize anything left pending from a previous
	// run so a restart never strands messages.
	if m.available {
		m.processUser(ctx, userID, SignalManual)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case sig := <-ch:
			if sig.kind == SignalCompaction {
				err := m.runCompaction(ctx, userID)
				m.clearCompacting(userID)
				if sig.done != nil {
					sig.done <- err
				}
				continue
			}
			if m.available {
				m.processUser(ctx, userID, sig.kind)
			}
		}
	}
}

func (m *Manager) logf(format string, args ...any) {
	log.Printf("memory: "+format, args...)
}
