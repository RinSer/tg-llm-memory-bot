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

	"github.com/RinSer/tg-llm-memory-bot/auth"
	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/RinSer/tg-llm-memory-bot/session"
	"github.com/RinSer/tg-llm-memory-bot/store"
	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/joho/godotenv"
)

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

	b.RegisterHandler(bot.HandlerTypeMessageText, "new", bot.MatchTypeCommand, newSessionHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "sessions", bot.MatchTypeCommand, listSessionsHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "switch", bot.MatchTypeCommand, switchSessionHandler(manager))
	b.RegisterHandler(bot.HandlerTypeMessageText, "model", bot.MatchTypeCommand, modelPickerHandler())
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "provider:", bot.MatchTypePrefix, providerChosenHandler(manager))
	b.RegisterHandler(bot.HandlerTypeCallbackQueryData, "model:", bot.MatchTypePrefix, modelChosenHandler(manager))

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
		reply, err := manager.Reply(ctx, update.Message.From.ID, update.Message.Text)
		if err != nil {
			log.Println(err)
			reply = fmt.Sprintf("Error: %v", err)
		}
		send(ctx, b, update, reply)
	}
}

func newSessionHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		title := strings.TrimSpace(commandArgs(update.Message.Text))
		if title == "" {
			title = "New session"
		}
		sess, err := manager.NewSession(update.Message.From.ID, title)
		if err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		send(ctx, b, update, fmt.Sprintf("Started session #%d: %s (%s/%s)", sess.ID, sess.Title, sess.Provider, sess.Model))
	}
}

func listSessionsHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		sessions, err := manager.ListSessions(update.Message.From.ID)
		if err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		if len(sessions) == 0 {
			send(ctx, b, update, "No sessions yet.")
			return
		}
		var sb strings.Builder
		for _, s := range sessions {
			marker := " "
			if s.IsActive {
				marker = "*"
			}
			fmt.Fprintf(&sb, "%s #%d %s (%s/%s)\n", marker, s.ID, s.Title, s.Provider, s.Model)
		}
		send(ctx, b, update, sb.String())
	}
}

func switchSessionHandler(manager *session.Manager) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		arg := strings.TrimSpace(commandArgs(update.Message.Text))
		sessionID, err := strconv.ParseInt(arg, 10, 64)
		if err != nil {
			send(ctx, b, update, "Usage: /switch <session id>")
			return
		}
		if err := manager.SwitchSession(update.Message.From.ID, sessionID); err != nil {
			log.Println(err)
			send(ctx, b, update, fmt.Sprintf("Error: %v", err))
			return
		}
		send(ctx, b, update, fmt.Sprintf("Switched to session #%d", sessionID))
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
