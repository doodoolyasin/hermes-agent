import pytest
from unittest.mock import AsyncMock, MagicMock
from plugins.platforms.bale.adapter import BaleAdapter
from gateway.config import PlatformConfig


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
async def test_slash_confirm_empty_id_and_session(bale_adapter):
    bale_adapter._slash_confirm_state = {}
    # empty confirm_id
    res = await bale_adapter._handle_slash_confirm_callback("cq1", "sc:once:", "12345", 100, {})
    # Must not crash, must not modify state
    assert len(bale_adapter._slash_confirm_state) == 0
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_slash_confirm_missing_session_key(bale_adapter):
    # Valid structure but missing state
    res = await bale_adapter._handle_slash_confirm_callback("cq2", "sc:once:orphan1", "12345", 101, {})
    assert len(bale_adapter._slash_confirm_state) == 0
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_slash_confirm_nil_choice_filtering(bale_adapter):
    # Fake state entry
    bale_adapter._slash_confirm_state["confX"] = "sess-A"
    # Test invalid choice string (SQL injection attempt simulated as text)
    res = await bale_adapter._handle_slash_confirm_callback("cq3", "sc:DROP TABLE users;--:confX", "12345", 102, {})
    assert len(bale_adapter._slash_confirm_state) == 0
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_clarify_missing_state(bale_adapter):
    bale_adapter._clarify_state = {}
    res = await bale_adapter._handle_clarify_callback("cq4", "cl:clar1:0", "12345", 103, {})
    assert len(bale_adapter._clarify_state) == 0
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_clarify_invalid_index_numeric_with_leading_zero(bale_adapter):
    bale_adapter._clarify_state["clar2"] = "sess-B"
    res = await bale_adapter._handle_clarify_callback("cq5", "cl:clar2:01", "12345", 104, {})
    # "01" is not a valid digit token (leading zero), so state should be preserved
    assert "clar2" in bale_adapter._clarify_state
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_model_picker_emergency_provider_missing_state(bale_adapter):
    bale_adapter._model_picker_state = {}
    res = await bale_adapter._handle_model_picker_callback({"id": "cq6"}, "mp:emergency_free", "12345", 105)
    assert len(bale_adapter._model_picker_state) == 0
    bale_adapter._client.edit_message_text.assert_not_called()
