import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig
from plugins.platforms.bale.adapter import BaleAdapter, BaleClient, BaleAPIError


@pytest.mark.asyncio
async def test_inbound_update_blitz_deduplication():
    cfg = PlatformConfig(extra={"token": "fake_token"}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter.handle_message = AsyncMock()

    # Blitz: 50 concurrent updates with same update_id
    update = {
        "update_id": 999111,
        "message": {
            "message_id": 1,
            "chat": {"id": 1001, "type": "private"},
            "from": {"id": 2002, "first_name": "TestUser"},
            "text": "Hello concurrent blitz",
            "date": 1700000000,
        },
    }

    # Dispatch in parallel
    tasks = [adapter._handle_update(update) for _ in range(50)]
    await asyncio.gather(*tasks)

    # Exactly 1 message should be dispatched to agent
    assert adapter.handle_message.call_count == 1


@pytest.mark.asyncio
async def test_callback_query_deduplication():
    cfg = PlatformConfig(extra={"token": "fake_token"}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter._handle_callback_query = AsyncMock()

    cb_update = {
        "update_id": 888222,
        "callback_query": {
            "id": "cb_1",
            "from": {"id": 2002},
            "data": "action:help",
        },
    }

    tasks = [adapter._handle_update(cb_update) for _ in range(20)]
    await asyncio.gather(*tasks)

    # Exactly 1 callback query should be handled
    assert adapter._handle_callback_query.call_count == 1


@pytest.mark.asyncio
async def test_markdown_error_auto_fallback():
    cfg = PlatformConfig(extra={"token": "fake_token", "markdown": True}, enabled=True)
    adapter = BaleAdapter(cfg)
    mock_client = AsyncMock()

    # Simulate 400 Bad Request on markdown formatted text
    # then success on plain-text fallback
    attempts = []

    async def fake_send_message(chat_id, text, reply_to, parse_mode):
        attempts.append({"text": text, "parse_mode": parse_mode})
        if parse_mode and parse_mode.lower() == "markdown":
            raise BaleAPIError("Bad Request: can't parse entities", status_code=400, error_code=400)
        return {"message_id": 12345}

    mock_client.send_message.side_effect = fake_send_message
    adapter._client = mock_client

    result = await adapter.send("1001", "This is *unclosed markdown _error")
    assert result.success is True
    assert result.message_id == "12345"
    assert len(attempts) == 2
    # First attempt had markdown
    assert attempts[0]["parse_mode"] == "Markdown"
    # Second fallback attempt had no parse_mode (plain text)
    assert attempts[1]["parse_mode"] is None
