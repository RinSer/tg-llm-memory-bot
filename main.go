package main

import (
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/RinSer/tg-llm-memory-bot/auth"
	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/memory"
	"github.com/RinSer/tg-llm-memory-bot/session"
	"github.com/RinSer/tg-llm-memory-bot/store"
	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/joho/godotenv"
)

// botCommands is registered with Telegram via SetMyCommands so clients show
// them, with descriptions, in the bot's native commands menu.
var botCommands = []models.BotCommand{
	{Command: "new", Description: "Start a new session"},
	{Command: "sessions", Description: "List your sessions and pick one to switch to"},
	{Command: "model", Description: "Choose a provider and model for the active session"},
	{Command: "summarymodel", Description: "Choose the LLM used to summarize into global memory"},
	{Command: "save", Description: "Save recent messages to global (long-term) memory now"},
	{Command: "compact", Description: "Consolidate global memory when it grows large"},
}

// banner greets on startup. Our mascot, MElephant (memory + elephant),
// lives in banner.txt -- kept in a file and embedded rather than a string
// literal because the art contains both backticks and backslashes, which a
// Go raw-string literal can't hold.
//
//go:embed banner.txt
var banner string

// loadingFrames cycle in the placeholder message shown while a reply is
// being generated.
var loadingFrames = []string{
	"\U0001F914 Thinking",
	"\U0001F914 Thinking.",
	"\U0001F914 Thinking..",
	"\U0001F914 Thinking...",
}

const (
	tgApiTokenVar      = "TG_BOT_API_TOKEN"
	openaiApiTokenVar  = "OPENAI_API_TOKEN"
	ollamaBaseURLVar   = "OLLAMA_BASE_URL"
	ollamaKeepAliveVar = "OLLAMA_KEEP_ALIVE"
	ollamaFlashAttnVar = "OLLAMA_FLASH_ATTENTION"
	historyLimitVar    = "SESSION_HISTORY_LIMIT"
	dbPathVar          = "DB_PATH"

	embeddingProviderVar = "EMBEDDING_PROVIDER"
	embeddingModelVar    = "EMBEDDING_MODEL"
	memThresholdVar      = "GLOBAL_MEMORY_MESSAGE_THRESHOLD"
	memMinSimilarityVar  = "GLOBAL_MEMORY_MIN_SIMILARITY"
	memTopKVar           = "GLOBAL_MEMORY_TOP_K"
	memMaxRowsVar        = "GLOBAL_MEMORY_MAX_ROWS"

	defaultHistoryLimit = 20
	defaultDBPath       = "bot.db"

	defaultEmbeddingProvider = llm.ProviderOllama
	defaultEmbeddingModel    = "embeddinggemma"
	defaultMemThreshold      = 20
	defaultMemMinSimilarity  = 0.4
	defaultMemTopK           = 25
	defaultMemMaxRows        = 10000

	// Summarization defaults to the same free local model as chat.
	defaultSummarizationProvider = llm.ProviderOllama
	defaultSummarizationModel    = "gemma4:latest"

	// Ollama costs nothing to run locally, so it's the default for new
	// sessions; /model lets a session switch to OpenAI (or another Ollama
	// model) at any time.
	defaultProvider = llm.ProviderOllama
	defaultModel    = "gemma4:latest"

	// A negative duration keeps the model loaded in memory/VRAM
	// indefinitely instead of Ollama's own default (unload after 5m
	// idle), trading idle VRAM usage for no reload latency on the next
	// message. It must have a unit suffix: langchaingo's KeepAlive field
	// is a plain string, always sent as a quoted JSON string, and Ollama
	// parses that via Go's time.ParseDuration -- a bare "-1" fails with
	// "missing unit in duration", unlike a raw JSON number -1.
	defaultOllamaKeepAlive = "-1s"

	ollamaDownloadURL = "https://ollama.com/download/windows"
)

func main() {
	fmt.Print(banner)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// .env is optional: a built binary running standalone may have its
	// config set directly in the environment instead. Missing required
	// vars are still caught below by mustEnv, with a clear error naming
	// the var.
	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		log.Printf("Warning: error loading .env file: %v", err)
	}

	checkOllamaInstalled()
	checkFlashAttention()

	tgApiToken := mustEnv(tgApiTokenVar)
	openaiApiToken := mustEnv(openaiApiTokenVar)

	dbPath := getEnv(dbPathVar, defaultDBPath)
	db, err := store.Open(dbPath)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ollamaBaseURL := os.Getenv(ollamaBaseURLVar)
	ollamaKeepAlive := getEnv(ollamaKeepAliveVar, defaultOllamaKeepAlive)
	historyLimit := getEnvInt(historyLimitVar, defaultHistoryLimit)
	memThreshold := getEnvInt(memThresholdVar, defaultMemThreshold)
	embeddingModel := getEnv(embeddingModelVar, defaultEmbeddingModel)

	// Notifier for the "global memory nearly full" warning. Its bot handle
	// is filled in once the bot exists (below), before any worker starts.
	notifier := &botNotifier{ctx: ctx}

	mem, err := memory.New(memory.Config{
		Store: db,
		Embedding: llm.Config{
			Name:      llm.ProviderName(getEnv(embeddingProviderVar, string(defaultEmbeddingProvider))),
			Model:     embeddingModel,
			APIToken:  openaiApiToken,
			BaseURL:   ollamaBaseURL,
			KeepAlive: ollamaKeepAlive,
		},
		MessageThreshold:             memThreshold,
		TopK:                         getEnvInt(memTopKVar, defaultMemTopK),
		MinSimilarity:                float32(getEnvFloat(memMinSimilarityVar, defaultMemMinSimilarity)),
		MaxRows:                      getEnvInt(memMaxRowsVar, defaultMemMaxRows),
		DefaultSummarizationProvider: defaultSummarizationProvider,
		DefaultSummarizationModel:    defaultSummarizationModel,
		OpenAIAPIToken:               openaiApiToken,
		OllamaBaseURL:                ollamaBaseURL,
		OllamaKeepAlive:              ollamaKeepAlive,
		Notifier:                     notifier,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Req #10: the DB is bound to one embedding model. A mismatch is fatal.
	if err := mem.CheckEmbeddingModel(); err != nil {
		log.Fatal(err)
	}
	// Reachability is only a warning: memory degrades to a no-op this run.
	if err := mem.CheckAccessible(ctx); err != nil {
		log.Printf("Warning: embedding model not reachable, global memory disabled this run: %v", err)
	}

	manager := session.NewManager(db, session.Config{
		HistoryLimit:    historyLimit,
		DefaultProvider: defaultProvider,
		DefaultModel:    defaultModel,
		OpenAIAPIToken:  openaiApiToken,
		OllamaBaseURL:   ollamaBaseURL,
		OllamaKeepAlive: ollamaKeepAlive,
		GlobalMemory:    mem,
		EmbeddingModel:  embeddingModel,
	})

	allowList := auth.NewAllowList()

	opts := []bot.Option{
		bot.WithMiddlewares(allowList.Middleware),
		bot.WithDefaultHandler(defaultHandler(manager)),
	}

	b, err := bot.New(tgApiToken, opts...)
	if err != nil {
		log.Fatal(err)
	}
	notifier.bot = b

	if _, err := b.SetMyCommands(ctx, &bot.SetMyCommandsParams{Commands: botCommands}); err != nil {
		log.Printf("Warning: failed to register bot commands menu: %v", err)
	}

	b.RegisterHandler(bot.HandlerTypeMessageText, "new", bot.MatchTypeCommand, newSessionHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "sessions", bot.MatchTypeCommand, listSessionsHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "model", bot.MatchTypeCommand, pickerStartHandler(modelPickerCfg))
	b.RegisterHandler(bot.HandlerTypeMessageText, "summarymodel", bot.MatchTypeCommand, pickerStartHandler(summaryPickerCfg))
	b.RegisterHandler(bot.HandlerTypeMessageText, "save", bot.MatchTypeCommand, saveHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "compact", bot.MatchTypeCommand, compactHandler(manager))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "provider:", bot.MatchTypePrefix, pickerProviderHandler(manager, modelPickerCfg))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "model:", bot.MatchTypePrefix, pickerModelHandler(manager, modelPickerCfg))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "sumprovider:", bot.MatchTypePrefix, pickerProviderHandler(manager, summaryPickerCfg))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "summodel:", bot.MatchTypePrefix, pickerModelHandler(manager, summaryPickerCfg))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "switch:", bot.MatchTypePrefix, switchSessionChosenHandler(manager))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "compact:", bot.MatchTypePrefix, compactChosenHandler(manager))

	// Start per-user summarization workers now that the notifier has its
	// bot handle.
	mem.Start(ctx, auth.AllowedUserIDs())

	log.Printf("Bot is listening for requests (db: %s, session history limit: %d messages, global memory threshold: %d messages). Press Ctrl+C to stop.", dbPath, historyLimit, memThreshold)
	b.Start(ctx)
}

// checkOllamaInstalled warns (but doesn't block startup) if the ollama CLI
// isn't on PATH, since it's still possible to run with OpenAI only.
func checkOllamaInstalled() {
	if _, err := exec.LookPath("ollama"); err != nil {
		log.Printf("Warning: ollama does not appear to be installed. Install it from %s to use local models.", ollamaDownloadURL)
	}
}

// checkFlashAttention warns if OLLAMA_FLASH_ATTENTION isn't set to "1" in
// this process's own environment. It's only a hint: flash attention is a
// setting of the Ollama *server* process, picked up from its environment
// only at startup -- this bot has no way to set or verify it there. The
// check is meaningful when the bot and `ollama serve` share the same
// environment (the common local setup), and a no-op warning otherwise.
func checkFlashAttention() {
	if os.Getenv(ollamaFlashAttnVar) != "1" {
		log.Printf("Warning: %s is not set to \"1\" in this process's environment. "+
			"If you want faster Ollama inference, set it (persistently, at the OS level) and restart the Ollama server -- "+
			"this bot cannot set it for you since it doesn't control the Ollama server process.", ollamaFlashAttnVar)
	}
}

func defaultHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		// Guards against stray updates (e.g. a callback query that matched
		// no registered handler) that carry no Message to read from.
		if update.Message == nil {
			return
		}

		text := update.Message.Text
		// Anything that looks like a command -- known, mistyped, or sent
		// with a "@BotName" suffix the router's entity matcher doesn't
		// strip -- should never be forwarded to the model as chat input.
		if strings.HasPrefix(text, "/") {
			send(ctx, b, update, "Unknown command. Tap the menu button next to the message box to see available commands.")
			return
		}

		chatID := update.Message.Chat.ID
		placeholder, phErr := b.SendMessage(ctx, &bot.SendMessageParams{ChatID: chatID, Text: loadingFrames[0]})
		if phErr != nil {
			log.Println(phErr)
		}

		buf := &streamBuffer{}
		var stopStreaming func()
		if placeholder != nil {
			stopStreaming = startStreamingIndicator(ctx, b, chatID, placeholder.ID, buf)
		}

		reply, err := manager.ReplyStream(ctx, update.Message.From.ID, text, buf.Append)
		if stopStreaming != nil {
			stopStreaming()
		}
		if err != nil {
			log.Println(err)
			reply = fmt.Sprintf("Error: %v", err)
		}

		if placeholder != nil {
			b.EditMessageText(ctx, &bot.EditMessageTextParams{ChatID: chatID, MessageID: placeholder.ID, Text: reply})
			return
		}
		send(ctx, b, update, reply)
	}
}

// streamBuffer accumulates streamed chunks. Append runs on the goroutine
// driving the LLM call; String is read concurrently by the ticker in
// startStreamingIndicator, hence the mutex.
type streamBuffer struct {
	mu   sync.Mutex
	text strings.Builder
}

func (s *streamBuffer) Append(chunk string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.text.WriteString(chunk)
}

func (s *streamBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.text.String()
}

// streamCursor marks the tail of text still being generated.
const streamCursor = " ▌"

// streamEditInterval is how often the placeholder message is updated.
// Telegram throttles rapid edits to the same message, so this deliberately
// batches chunks rather than editing on every token.
const streamEditInterval = 500 * time.Millisecond

// streamRevealRunesPerTick caps how much of the model's buffered-so-far
// text is revealed on each tick. Without this, a burst of chunks arriving
// between ticks would get pasted in all at once, looking like a jolt --
// capping it gives a steady, slower typewriter-style reveal instead. If the
// model is producing text slower than this rate, the reveal just tracks
// generation 1:1 with no artificial delay.
const streamRevealRunesPerTick = 6

// startStreamingIndicator edits messageID on a fixed interval: showing the
// rotating loadingFrames until the model's first chunk arrives in buf, then
// a gradually-revealed prefix of the accumulated text-so-far (with a
// trailing cursor) until stopped. This gives visible, smooth progress
// during a (possibly slow) LLM generation instead of the chat appearing
// unresponsive -- or dumping large bursts of text at once.
func startStreamingIndicator(ctx context.Context, b *bot.Bot, chatID int64, messageID int, buf *streamBuffer) func() {
	loopCtx, cancel := context.WithCancel(ctx)
	go func() {
		ticker := time.NewTicker(streamEditInterval)
		defer ticker.Stop()
		frame := 0
		revealed := 0
		lastText := loadingFrames[0]
		for {
			select {
			case <-loopCtx.Done():
				return
			case <-ticker.C:
				available := []rune(buf.String())
				var text string
				if len(available) == 0 {
					frame = (frame + 1) % len(loadingFrames)
					text = loadingFrames[frame]
				} else {
					revealed = min(revealed+streamRevealRunesPerTick, len(available))
					text = string(available[:revealed]) + streamCursor
				}
				if text == lastText {
					continue
				}
				lastText = text
				b.EditMessageText(loopCtx, &bot.EditMessageTextParams{
					ChatID:    chatID,
					MessageID: messageID,
					Text:      text,
				})
			}
		}
	}()
	return cancel
}

func newSessionHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		title := strings.TrimSpace(commandArgs(update.Message.Text))
		sess, err := manager.NewSession(update.Message.From.ID, title)
		if err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		send(ctx, b, update, fmt.Sprintf("Started session #%d: %s (%s/%s)", sess.ID, sess.Title, sess.Provider, sess.Model))
	}
}

// listSessionsHandler shows the user's sessions as inline-keyboard buttons;
// tapping one switches to it via switchSessionChosenHandler.
func listSessionsHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		sessions, err := manager.ListSessions(update.Message.From.ID)
		if err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		if len(sessions) == 0 {
			send(ctx, b, update, "No sessions yet. Use /new to start one.")
			return
		}

		rows := make([][]models.InlineKeyboardButton, 0, len(sessions))
		for _, s := range sessions {
			marker := " "
			if s.IsActive {
				marker = "*"
			}
			label := fmt.Sprintf("%s #%d %s (%s/%s)", marker, s.ID, s.Title, s.Provider, s.Model)
			rows = append(rows, []models.InlineKeyboardButton{
				{Text: label, CallbackData: fmt.Sprintf("switch:%d", s.ID)},
			})
		}
		b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID:      update.Message.Chat.ID,
			Text:        "Choose a session:",
			ReplyMarkup: models.InlineKeyboardMarkup{InlineKeyboard: rows},
		})
	}
}

// switchSessionChosenHandler finishes the /sessions flow: switch to the
// tapped session.
func switchSessionChosenHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		sessionID, err := strconv.ParseInt(strings.TrimPrefix(cq.Data, "switch:"), 10, 64)
		if err != nil {
			editMessage(ctx, b, cq, "Invalid selection.", nil)
			return
		}
		if err := manager.SwitchSession(cq.From.ID, sessionID); err != nil {
			log.Println(err)
			editMessage(ctx, b, cq, fmt.Sprintf("Error: %v", err), nil)
			return
		}
		editMessage(ctx, b, cq, fmt.Sprintf("Switched to session #%d", sessionID), nil)
	}
}

// modelPicker parameterizes a provider→model inline-keyboard flow. /model
// and /summarymodel share the same listing logic and differ only in their
// callback-data prefixes and the action applied to the final choice.
type modelPicker struct {
	providerPrefix string // callback prefix for provider buttons
	modelPrefix    string // callback prefix for model buttons
	apply          func(m *session.Manager, userID int64, provider llm.ProviderName, model string) error
	successFmt     string // e.g. "Active session now using %s/%s"
}

var modelPickerCfg = modelPicker{
	providerPrefix: "provider:",
	modelPrefix:    "model:",
	apply: func(m *session.Manager, userID int64, p llm.ProviderName, model string) error {
		return m.SetModel(userID, p, model)
	},
	successFmt: "Active session now using %s/%s",
}

var summaryPickerCfg = modelPicker{
	providerPrefix: "sumprovider:",
	modelPrefix:    "summodel:",
	apply: func(m *session.Manager, userID int64, p llm.ProviderName, model string) error {
		return m.SetSummarizationModel(userID, p, model)
	},
	successFmt: "Summarization now uses %s/%s",
}

// pickerStartHandler shows one button per supported provider.
func pickerStartHandler(cfg modelPicker) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		rows := make([][]models.InlineKeyboardButton, 0, len(llm.Providers))
		for _, p := range llm.Providers {
			rows = append(rows, []models.InlineKeyboardButton{
				{Text: string(p), CallbackData: cfg.providerPrefix + string(p)},
			})
		}
		b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID:      update.Message.Chat.ID,
			Text:        "Choose a provider:",
			ReplyMarkup: models.InlineKeyboardMarkup{InlineKeyboard: rows},
		})
	}
}

// pickerProviderHandler follows a provider tap: list its models (a static
// whitelist for OpenAI, a live /api/tags query for Ollama) as buttons.
func pickerProviderHandler(manager *session.Manager, cfg modelPicker) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		providerName := llm.ProviderName(strings.TrimPrefix(cq.Data, cfg.providerPrefix))
		modelNames, err := manager.ModelsFor(ctx, providerName)
		if err != nil {
			log.Println(err)
			editMessage(ctx, b, cq, fmt.Sprintf("Error listing %s models: %v", providerName, err), nil)
			return
		}
		if len(modelNames) == 0 {
			editMessage(ctx, b, cq, fmt.Sprintf("No models available for %s.", providerName), nil)
			return
		}

		rows := make([][]models.InlineKeyboardButton, 0, len(modelNames))
		for _, m := range modelNames {
			rows = append(rows, []models.InlineKeyboardButton{
				{Text: m, CallbackData: cfg.modelPrefix + string(providerName) + ":" + m},
			})
		}
		keyboard := models.InlineKeyboardMarkup{InlineKeyboard: rows}
		editMessage(ctx, b, cq, fmt.Sprintf("Choose a model for %s:", providerName), &keyboard)
	}
}

// pickerModelHandler applies the picked model via the config's apply func.
func pickerModelHandler(manager *session.Manager, cfg modelPicker) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		rest := strings.TrimPrefix(cq.Data, cfg.modelPrefix)
		providerRaw, modelName, ok := strings.Cut(rest, ":")
		if !ok {
			editMessage(ctx, b, cq, "Invalid selection.", nil)
			return
		}
		providerName := llm.ProviderName(providerRaw)

		if err := cfg.apply(manager, cq.From.ID, providerName, modelName); err != nil {
			log.Println(err)
			editMessage(ctx, b, cq, fmt.Sprintf("Error: %v", err), nil)
			return
		}
		editMessage(ctx, b, cq, fmt.Sprintf(cfg.successFmt, providerName, modelName), nil)
	}
}

// saveHandler triggers a manual global-memory save of the active session's
// unsaved messages.
func saveHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		if err := manager.SaveGlobalMemory(update.Message.From.ID); err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		send(ctx, b, update, "Saving recent messages to global memory in the background.")
	}
}

// compactHandler starts the /compact flow: nothing to do below target,
// otherwise a warn-then-confirm prompt.
func compactHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		rows, target, err := manager.CompactionStatus(update.Message.From.ID)
		if err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		if rows <= target {
			send(ctx, b, update, fmt.Sprintf("Nothing to compact (%d rows, target ~%d).", rows, target))
			return
		}
		keyboard := models.InlineKeyboardMarkup{InlineKeyboard: [][]models.InlineKeyboardButton{{
			{Text: "Confirm", CallbackData: "compact:confirm"},
			{Text: "Cancel", CallbackData: "compact:cancel"},
		}}}
		b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: update.Message.Chat.ID,
			Text: fmt.Sprintf(
				"This will consolidate your global memory from %d rows down to ~%d and may take a while. "+
					"You won't be able to chat until it finishes. Proceed?", rows, target),
			ReplyMarkup: keyboard,
		})
	}
}

// compactChosenHandler handles the Confirm/Cancel taps of /compact. Confirm
// blocks on the (synchronous) compaction and reports the result.
func compactChosenHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		switch strings.TrimPrefix(cq.Data, "compact:") {
		case "cancel":
			editMessage(ctx, b, cq, "Cancelled.", nil)
		case "confirm":
			editMessage(ctx, b, cq, "Compacting… please wait.", nil)
			before, after, err := manager.StartCompaction(ctx, cq.From.ID)
			if err != nil {
				log.Println(err)
				editMessage(ctx, b, cq, fmt.Sprintf("Error: %v", err), nil)
				return
			}
			editMessage(ctx, b, cq, fmt.Sprintf("Compacted %d rows into %d.", before, after), nil)
		default:
			editMessage(ctx, b, cq, "Invalid selection.", nil)
		}
	}
}

// commandArgs strips the leading "/command" token off a message's text.
func commandArgs(text string) string {
	_, rest, found := strings.Cut(text, " ")
	if !found {
		return ""
	}
	return rest
}

func send(ctx context.Context, b *bot.Bot, update *models.Update, text string) {
	b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: update.Message.Chat.ID,
		Text:   text,
	})
}

// answer acknowledges a callback query so Telegram stops showing the
// tap's loading spinner on the client.
func answer(ctx context.Context, b *bot.Bot, cq *models.CallbackQuery) {
	b.AnswerCallbackQuery(ctx, &bot.AnswerCallbackQueryParams{CallbackQueryID: cq.ID})
}

func editMessage(ctx context.Context, b *bot.Bot, cq *models.CallbackQuery, text string, keyboard *models.InlineKeyboardMarkup) {
	if cq.Message.Type != models.MaybeInaccessibleMessageTypeMessage || cq.Message.Message == nil {
		return
	}
	params := &bot.EditMessageTextParams{
		ChatID:    cq.Message.Message.Chat.ID,
		MessageID: cq.Message.Message.ID,
		Text:      text,
	}
	if keyboard != nil {
		params.ReplyMarkup = *keyboard
	}
	b.EditMessageText(ctx, params)
}

func mustEnv(name string) string {
	v, ok := os.LookupEnv(name)
	if !ok {
		log.Fatalf("Should set env var %v", name)
	}
	return v
}

func getEnv(name, fallback string) string {
	if v, ok := os.LookupEnv(name); ok {
		return v
	}
	return fallback
}

func getEnvInt(name string, fallback int) int {
	v, ok := os.LookupEnv(name)
	if !ok {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		log.Fatalf("Invalid int value for env var %v: %v", name, v)
	}
	return n
}

func getEnvFloat(name string, fallback float64) float64 {
	v, ok := os.LookupEnv(name)
	if !ok {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		log.Fatalf("Invalid float value for env var %v: %v", name, v)
	}
	return f
}

// botNotifier delivers global-memory warnings to a user's private chat. A
// private chat's ID equals the user's ID, matching how the rest of the bot
// treats chat-id and user-id interchangeably. Its bot handle is set after
// the bot is constructed, before any worker starts.
type botNotifier struct {
	ctx context.Context
	bot *bot.Bot
}

func (n *botNotifier) Notify(userID int64, message string) {
	if n.bot == nil {
		return
	}
	n.bot.SendMessage(n.ctx, &bot.SendMessageParams{ChatID: userID, Text: message})
}
