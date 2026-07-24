package auth

import (
	"context"
	"testing"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
)

func TestIsAllowed(t *testing.T) {
	allowed := NewAllowList()

	for _, id := range allowedUserIDs {
		if !allowed.IsAllowed(id) {
			t.Errorf("expected hardcoded user %d to be allowed", id)
		}
	}
	if allowed.IsAllowed(999) {
		t.Error("expected an arbitrary user id not to be allowed")
	}
}

func TestMiddlewareAllowsMessageFromAllowedUser(t *testing.T) {
	allowed := NewAllowList()
	called := false
	next := func(ctx context.Context, b *bot.Bot, update *models.Update) { called = true }

	update := &models.Update{Message: &models.Message{From: &models.User{ID: allowedUserIDs[0]}}}
	allowed.Middleware(next)(context.Background(), nil, update)

	if !called {
		t.Error("expected the handler to be called for an allowed user")
	}
}

func TestMiddlewareBlocksMessageFromUnknownUser(t *testing.T) {
	allowed := NewAllowList()
	called := false
	next := func(ctx context.Context, b *bot.Bot, update *models.Update) { called = true }

	update := &models.Update{Message: &models.Message{From: &models.User{ID: 999}}}
	allowed.Middleware(next)(context.Background(), nil, update)

	if called {
		t.Error("expected the handler not to be called for an unknown user")
	}
}

func TestMiddlewareChecksCallbackQueryUser(t *testing.T) {
	allowed := NewAllowList()
	called := false
	next := func(ctx context.Context, b *bot.Bot, update *models.Update) { called = true }

	update := &models.Update{CallbackQuery: &models.CallbackQuery{From: models.User{ID: 999}}}
	allowed.Middleware(next)(context.Background(), nil, update)

	if called {
		t.Error("expected the handler not to be called for an unknown callback query user")
	}
}

func TestMiddlewareIgnoresUpdatesWithoutMessageOrCallback(t *testing.T) {
	allowed := NewAllowList()
	called := false
	next := func(ctx context.Context, b *bot.Bot, update *models.Update) { called = true }

	allowed.Middleware(next)(context.Background(), nil, &models.Update{})

	if called {
		t.Error("expected the handler not to be called for an update with neither a message nor a callback query")
	}
}
