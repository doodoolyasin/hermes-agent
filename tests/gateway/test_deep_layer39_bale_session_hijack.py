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
async def test_ea_handler_prevents_unauthorized_access(bale_adapter):
    # We register an approval for a different chat
    bale_adapter._approval_state["1"] = "agent:main:bale:dm:999999"
    bale_adapter._approval_counter = iter(range(10))

    # attacker chat_id 888888 clicks the button with approval_id 1
    await bale_adapter._handle_exec_approval_callback(
        cq_id="cq1",
        data="ea:once:1",
        chat_id="888888",
        msg_id=100,
        from_user={"first_name": "Hacker"},
    )

    # It should fail because session_key "agent:main:bale:dm:999999" does not match "888888"
    # and should NOT remove the state entry! The legitimate user (999999) should still have it.
    assert "1" in bale_adapter._approval_state
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_ea_handler_cleanup_on_authorized_access(bale_adapter):
    bale_adapter._approval_state["2"] = "agent:main:bale:dm:888888"
    await bale_adapter._handle_exec_approval_callback(
        cq_id="cq2",
        data="ea:once:2",
        chat_id="888888",
        msg_id=101,
        from_user={"first_name": "Owner"},
    )
    assert "2" not in bale_adapter._approval_state
    bale_adapter._client.edit_message_text.assert_called_once()


@pytest.mark.asyncio
async def test_sc_handler_prevents_unauthorized_access(bale_adapter):
    bale_adapter._slash_confirm_state["3"] = "agent:main:bale:dm:111111"
    await bale_adapter._handle_slash_confirm_callback(
        cq_id="cq3",
        data="sc:once:3",
        chat_id="222222",
        msg_id=102,
        from_user={"first_name": "Hacker"},
    )
    assert "3" in bale_adapter._slash_confirm_state
    bale_adapter._client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_cl_handler_prevents_unauthorized_access(bale_adapter):
    bale_adapter._clarify_state["4"] = "agent:main:bale:dm:333333"
    await bale_adapter._handle_clarify_callback(
        cq_id="cq4",
        data="cl:4:0",
        chat_id="444444",
        msg_id=103,
        from_user={"first_name": "Hacker"},
    )
    assert "4" in bale_adapter._clarify_state
    bale_adapter._client.edit_message_text.assert_not_called()
