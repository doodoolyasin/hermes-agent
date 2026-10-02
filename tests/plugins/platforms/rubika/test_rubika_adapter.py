import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import Platform, PlatformConfig
from plugins.platforms.rubika.adapter import (
    RubikaAdapter,
    RubikaClient,
    RubikaAPIError,
    register,
)
from gateway.platforms.base import SendResult


def make_adapter(extra=None, **cfg):
    cfg.setdefault("enabled", True)
    return RubikaAdapter(PlatformConfig(extra=dict(extra or {}), **cfg))


def test_rubika_capabilities_matrix():
    adapter = make_adapter(extra={"token": "fake_token"})
    caps = adapter.capabilities

    # Rubika official API lacks inline keyboards & callbacks, so it MUST fall back to text menu
    assert caps.inline_keyboard is False
    assert caps.callbacks is False
    assert caps.buttons_text_fallback is True
    assert caps.message_edit is True
    assert caps.message_delete is True
    assert caps.typing_indicator is True
    assert caps.photos is True
    assert caps.documents is True


def test_rubika_fallback_menu_rendering():
    adapter = make_adapter(extra={"token": "fake_token"})
    menu = adapter.render_fallback_menu("منوی اصلی", ["تغییر مدل", "گفتگوی جدید", "وضعیت"])

    assert "منوی اصلی" in menu
    assert "1️⃣ تغییر مدل" in menu
    assert "2️⃣ گفتگوی جدید" in menu
    assert "3️⃣ وضعیت" in menu


@pytest.mark.asyncio
async def test_rubika_loop_guard():
    adapter = make_adapter(extra={"token": "fake_token"})
    adapter.bot_id = "bot_999"
    adapter.handle_message = AsyncMock()

    # Message sent BY the bot itself
    self_msg_update = {
        "update_id": 1,
        "message": {
            "message_id": 10,
            "chat_id": "c1",
            "sender_id": "bot_999",
            "text": "Self echo",
        }
    }
    await adapter._handle_update(self_msg_update)
    adapter.handle_message.assert_not_called()

    # Message sent by a human user
    user_msg_update = {
        "update_id": 2,
        "message": {
            "message_id": 11,
            "chat_id": "c1",
            "sender_id": "user_123",
            "text": "Hello bot",
        }
    }
    await adapter._handle_update(user_msg_update)
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_rubika_deduplication():
    adapter = make_adapter(extra={"token": "fake_token"})
    adapter.bot_id = "bot_999"
    adapter.handle_message = AsyncMock()

    update = {
        "update_id": 100,
        "message": {
            "message_id": 55,
            "chat_id": "chat_a",
            "sender_id": "user_b",
            "text": "Test duplicate",
        }
    }
    # First time -> handled
    await adapter._handle_update(update)
    assert adapter.handle_message.call_count == 1

    # Second time (duplicate) -> discarded by deduplicator
    await adapter._handle_update(update)
    assert adapter.handle_message.call_count == 1


@pytest.mark.asyncio
async def test_rubika_missing_token_fatal():
    adapter = make_adapter(extra={})
    adapter.token = ""

    connected = await adapter.connect()
    assert connected is False
    assert adapter._fatal_error_code == "missing_token"


def test_rubika_registration():
    class FakeCtx:
        def __init__(self):
            self.registered = {}
        def register_platform(self, **kwargs):
            self.registered = kwargs

    ctx = FakeCtx()
    register(ctx)
    assert ctx.registered["name"] == "rubika"
    assert ctx.registered["label"] == "Rubika"
    assert "RUBIKA_BOT_TOKEN" in ctx.registered["required_env"]
