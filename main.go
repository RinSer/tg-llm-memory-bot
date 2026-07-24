package main

import (
	"context"
	"log"
	"os"
	"os/signal"

	"github.com/RinSer/tg-llm-memory-bot/llm"
	"github.com/go-telegram/bot"
	"github.com/joho/godotenv"
)

const (
	openaiApiTokenVar = "OPENAI_API_TOKEN"
	tgApiTokenVar     = "TG_BOT_API_TOKEN"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}

	tgApiToken, ok := os.LookupEnv(tgApiTokenVar)
	if !ok {
		log.Fatalf(
			"Should set env var %v with TG api token",
			tgApiTokenVar,
		)
	}

	openaiApiToken, ok := os.LookupEnv(openaiApiTokenVar)
	if !ok {
		log.Fatalf(
			"Should set env var %v with OpenAI api token",
			openaiApiTokenVar,
		)
	}

	llm, err := llm.New("openai", "gpt-4.1-mini", openaiApiToken)
	if err != nil {
		log.Fatal(err)
	}

	opts := []bot.Option{
		bot.WithDefaultHandler(llm.Talk),
	}

	b, err := bot.New(tgApiToken, opts...)
	if err != nil {
		log.Fatal(err)
	}

	b.Start(ctx)
}
