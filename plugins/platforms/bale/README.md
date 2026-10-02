# Bale gateway plugin

A first-class [Bale](https://bale.ai) messenger channel for Hermes Agent, added
as a **plugin** (`plugins/platforms/bale`) — no core module is modified.

## Why a plugin

Bale's Bot API is Telegram-compatible
(`https://tapi.bale.ai/bot<TOKEN>/<method>`), so the adapter reuses the same
`BasePlatformAdapter` contract every built-in platform uses. The plugin calls
`ctx.register_platform(...)`, which makes Bale appear everywhere the built-ins do:
`gateway setup`, `gateway status`, the session/prompt hints, the toolset, cron
delivery and the `send_message` tool.

## Setup

1. Open the Bale app, talk to **BotFather**, create a bot and copy its token.
2. Export the token (or put it in the platform's `extra.token`):

   ```bash
   export BALE_BOT_TOKEN="<token from BotFather>"
   ```

3. Start the gateway as usual:

   ```bash
   hermes gateway
   ```

The platform self-enables from `BALE_BOT_TOKEN` via `env_enablement_fn`; no
config-file edit is strictly required.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `BALE_BOT_TOKEN` | **Required.** Bot token from BotFather. Never logged. |
| `BALE_API_BASE` | API base URL (default `https://tapi.bale.ai`). |
| `BALE_ALLOWED_USERS` | Comma-separated allowlist of Bale user ids. |
| `BALE_ALLOW_ALL_USERS` | Set `1`/`true` to allow everyone (overrides the allowlist). |
| `BALE_HOME_CHANNEL` | Default chat id for cron / out-of-process delivery. |
| `BALE_HOME_CHANNEL_NAME` | Friendly name for `BALE_HOME_CHANNEL`. |
| `BALE_POLL_TIMEOUT` | `getUpdates` long-poll timeout, seconds (default 30). |
| `BALE_MAX_FAILURES` | Hard cap on consecutive poll failures before giving up (default 8). |
| `BALE_BACKOFF_BASE` / `BALE_BACKOFF_MAX` | Retry backoff bounds, seconds. |
| `BALE_PARSE_MODE` | Optional `parse_mode` sent with messages. |
| `BALE_MARKDOWN` | `1`/`true` to pass Markdown through instead of stripping it. |

## Features

- Long-polling `getUpdates` with `offset` advancement and per-message dedup.
- Bounded retry with exponential backoff + jitter; respects `429 Retry-After`.
- Fatal-vs-retryable classification (401/403 stop; 5xx/transport retry).
- Text, photo, document, voice and video sending; automatic 4096-char splitting.
- Group vs private chat identity mapped correctly (`dm` / `group` / `channel`).
- Bot/self messages are ignored to prevent reply loops.
- The bot token never appears in logs (`_redact`).

## Honest limitations

- Bale does not document a Markdown/HTML parse mode, so messages default to
  **plain text** (Markdown markers are stripped). Opt in with `BALE_MARKDOWN=1`
  only if your Bale build renders Markdown.
- The 4096-character ceiling is Telegram-compatible; Bale's own docs do not
  state a limit, so this is the conservative default.
- `get_chat_info` is minimal (Bale exposes no `getChat` in the documented API).

## Tests

```bash
python3 -m pytest tests/plugins/platforms/bale -q
```

The suite mocks the single HTTP seam and never touches the network.
