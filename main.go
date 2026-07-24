package main

import (
	"context"
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
}

// loadingFrames cycle in the placeholder message shown while a reply is
// being generated.
var loadingFrames = []string{
	"\U0001F914 Thinking",
	"\U0001F914 Thinking.",
	"\U0001F914 Thinking..",
	"\U0001F914 Thinking...",
}

const (
	tgApiTokenVar     = "TG_BOT_API_TOKEN"
	openaiApiTokenVar = "OPENAI_API_TOKEN"
	ollamaBaseURLVar  = "OLLAMA_BASE_URL"
	historyLimitVar   = "SESSION_HISTORY_LIMIT"
	dbPathVar         = "DB_PATH"

	defaultHistoryLimit = 20
	defaultDBPath       = "bot.db"

	// Ollama costs nothing to run locally, so it's the default for new
	// sessions; /model lets a session switch to OpenAI (or another Ollama
	// model) at any time.
	defaultProvider = llm.ProviderOllama
	defaultModel    = "gemma4:latest"

	ollamaDownloadURL = "https://ollama.com/download/windows"
)

func main() {
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

	tgApiToken := mustEnv(tgApiTokenVar)
	openaiApiToken := mustEnv(openaiApiTokenVar)

	dbPath := getEnv(dbPathVar, defaultDBPath)
	db, err := store.Open(dbPath)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	historyLimit := getEnvInt(historyLimitVar, defaultHistoryLimit)
	manager := session.NewManager(db, session.Config{
		HistoryLimit:    historyLimit,
		DefaultProvider: defaultProvider,
		DefaultModel:    defaultModel,
		OpenAIAPIToken:  openaiApiToken,
		OllamaBaseURL:   os.Getenv(ollamaBaseURLVar),
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

	if _, err := b.SetMyCommands(ctx, &bot.SetMyCommandsParams{Commands: botCommands}); err != nil {
		log.Printf("Warning: failed to register bot commands menu: %v", err)
	}

	b.RegisterHandler(bot.HandlerTypeMessageText, "new", bot.MatchTypeCommand, newSessionHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "sessions", bot.MatchTypeCommand, listSessionsHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "model", bot.MatchTypeCommand, modelPickerHandler())
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "provider:", bot.MatchTypePrefix, providerChosenHandler(manager))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "model:", bot.MatchTypePrefix, modelChosenHandler(manager))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "switch:", bot.MatchTypePrefix, switchSessionChosenHandler(manager))

	log.Printf("Bot is listening for requests (db: %s, session history limit: %d messages). Press Ctrl+C to stop.", dbPath, historyLimit)
	b.Start(ctx)
}

// checkOllamaInstalled warns (but doesn't block startup) if the ollama CLI
// isn't on PATH, since it's still possible to run with OpenAI only.
func checkOllamaInstalled() {
	if _, err := exec.LookPath("ollama"); err != nil {
		log.Printf("Warning: ollama does not appear to be installed. Install it from %s to use local models.", ollamaDownloadURL)
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

// streamEditInterval is how often the placeholder message is updated with
// newly streamed text. Telegram throttles rapid edits to the same message,
// so this deliberately batches chunks rather than editing on every token.
const streamEditInterval = 1000 * time.Millisecond

// startStreamingIndicator edits messageID on a fixed interval: showing the
// rotating loadingFrames until the model's first chunk arrives in buf, then
// the accumulated text-so-far (with a trailing cursor) until stopped. This
// gives visible progress during a (possibly slow) LLM generation instead of
// the chat appearing unresponsive.
func startStreamingIndicator(ctx context.Context, b *bot.Bot, chatID int64, messageID int, buf *streamBuffer) func() {
	loopCtx, cancel := context.WithCancel(ctx)
	go func() {
		ticker := time.NewTicker(streamEditInterval)
		defer ticker.Stop()
		frame := 0
		lastText := loadingFrames[0]
		for {
			select {
			case <-loopCtx.Done():
				return
			case <-ticker.C:
				text := buf.String()
				if text == "" {
					frame = (frame + 1) % len(loadingFrames)
					text = loadingFrames[frame]
				} else {
					text += streamCursor
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

// modelPickerHandler starts the /model flow: show one button per supported
// provider. Tapping one triggers providerChosenHandler.
func modelPickerHandler() bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		rows := make([][]models.InlineKeyboardButton, 0, len(llm.Providers))
		for _, p := range llm.Providers {
			rows = append(rows, []models.InlineKeyboardButton{
				{Text: string(p), CallbackData: "provider:" + string(p)},
			})
		}
		b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID:      update.Message.Chat.ID,
			Text:        "Choose a provider:",
			ReplyMarkup: models.InlineKeyboardMarkup{InlineKeyboard: rows},
		})
	}
}

// providerChosenHandler follows a provider tap: list its models (a static
// whitelist for OpenAI, a live /api/tags query for Ollama) as buttons.
// Tapping one triggers modelChosenHandler.
func providerChosenHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		providerName := llm.ProviderName(strings.TrimPrefix(cq.Data, "provider:"))
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
				{Text: m, CallbackData: "model:" + string(providerName) + ":" + m},
			})
		}
		keyboard := models.InlineKeyboardMarkup{InlineKeyboard: rows}
		editMessage(ctx, b, cq, fmt.Sprintf("Choose a model for %s:", providerName), &keyboard)
	}
}

// modelChosenHandler finishes the /model flow: apply the picked model to
// the user's active session.
func modelChosenHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		cq := update.CallbackQuery
		answer(ctx, b, cq)

		rest := strings.TrimPrefix(cq.Data, "model:")
		providerRaw, modelName, ok := strings.Cut(rest, ":")
		if !ok {
			editMessage(ctx, b, cq, "Invalid selection.", nil)
			return
		}
		providerName := llm.ProviderName(providerRaw)

		if err := manager.SetModel(cq.From.ID, providerName, modelName); err != nil {
			log.Println(err)
			editMessage(ctx, b, cq, fmt.Sprintf("Error: %v", err), nil)
			return
		}
		editMessage(ctx, b, cq, fmt.Sprintf("Active session now using %s/%s", providerName, modelName), nil)
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
