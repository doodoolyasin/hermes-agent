import pytest
from unittest.mock import AsyncMock, MagicMock
from plugins.platforms.bale.adapter import BaleAdapter
from gateway.config import PlatformConfig
from hermes_cli.model_switch import switch_model


@pytest.fixture
def bale_adapter():
    config = PlatformConfig(enabled=True, extra={"token": "test-bale-token"})
    adapter = BaleAdapter(config)
    adapter._client = MagicMock()
    adapter._client.send_message = AsyncMock(return_value={"message_id": 999})
    adapter._client.edit_message_text = AsyncMock(return_value={"message_id": 999})
    adapter._client.answer_callback_query = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_send_slash_confirm_renders_buttons(bale_adapter):
    result = await bale_adapter.send_slash_confirm(
        chat_id="661453305",
        title="Confirm /new",
        message="This starts a fresh session",
        session_key="test_session",
        confirm_id="conf_123",
    )
    assert result.success is True
    assert result.message_id == "999"
    assert bale_adapter._slash_confirm_state["conf_123"] == "test_session"

    call_args = bale_adapter._client.send_message.call_args
    assert call_args[0][0] == "661453305"
    assert call_args[0][1] == "This starts a fresh session"
    markup = call_args[1]["reply_markup"]
    assert "inline_keyboard" in markup
    # Verify button callbacks
    callbacks = [btn["callback_data"] for row in markup["inline_keyboard"] for btn in row]
    assert "sc:once:conf_123" in callbacks
    assert "sc:always:conf_123" in callbacks
    assert "sc:cancel:conf_123" in callbacks


@pytest.mark.asyncio
async def test_send_clarify_renders_buttons(bale_adapter):
    result = await bale_adapter.send_clarify(
        chat_id="661453305",
        question="Which approach?",
        choices=["Option A", "Option B"],
        clarify_id="clar_456",
        session_key="test_session",
    )
    assert result.success is True
    assert result.message_id == "999"
    assert bale_adapter._clarify_state["clar_456"] == "test_session"

    call_args = bale_adapter._client.send_message.call_args
    markup = call_args[1]["reply_markup"]
    callbacks = [btn["callback_data"] for row in markup["inline_keyboard"] for btn in row]
    assert "cl:clar_456:0" in callbacks
    assert "cl:clar_456:1" in callbacks
    assert "cl:clar_456:other" in callbacks


@pytest.mark.asyncio
async def test_model_switch_emergency_free_no_api_key():
    # Switching to openai-fast without any API key or explicit credentials
    res = switch_model("openai-fast", current_provider="", current_model="gpt-4o")
    assert res.success is True
    assert res.new_model == "openai-fast"
    assert res.target_provider == "custom"
    assert "pollinations" in res.base_url
    assert res.api_key == "free-community"
