package auth

import (
	"context"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
)

// allowedUserIDs is the hardcoded set of Telegram users permitted to use
// the bot.
var allowedUserIDs = []int64{
	342930895, // @rin5er, Serj Dukareff
	624884518, // @IrenDeuxIren, Irina
}

// AllowList authorizes Telegram users by a hardcoded set of user IDs.
type AllowList map[int64]struct{}

func NewAllowList() AllowList {
	allowed := make(AllowList, len(allowedUserIDs))
	for _, id := range allowedUserIDs {
		allowed[id] = struct{}{}
	}
	return allowed
}

func (a AllowList) IsAllowed(userID int64) bool {
	_, ok := a[userID]
	return ok
}

// Middleware drops updates from users not present in the allow list before
// they reach the wrapped handler. Applies to both plain messages and
// callback queries (inline keyboard taps).
func (a AllowList) Middleware(next bot.HandlerFunc) bot.HandlerFunc {
	return func(ctx context.Context, b *bot.Bot, update *models.Update) {
		var userID int64
		switch {
		case update.Message != nil:
			userID = update.Message.From.ID
		case update.CallbackQuery != nil:
			userID = update.CallbackQuery.From.ID
		default:
			return
		}
		if !a.IsAllowed(userID) {
			return
		}
		next(ctx, b, update)
	}
}
