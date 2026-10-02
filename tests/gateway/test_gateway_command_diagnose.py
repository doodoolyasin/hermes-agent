import pytest
from unittest.mock import MagicMock
from gateway.slash_commands_status import GatewayStatusCommandsMixin
from gateway.platforms.event import MessageEvent


@pytest.mark.asyncio
async def test_handle_diagnose_command():
    class FakeGateway(GatewayStatusCommandsMixin):
        def __init__(self):
            self.adapters = {"bale": MagicMock(), "rubika": MagicMock()}

    gw = FakeGateway()
    event = MessageEvent(text="/diagnose")
    result = await gw._handle_diagnose_command(event)

    assert "Hermes Diagnostics" in result
    assert "وضعیت شبکه" in result
    assert "مدارشکن‌ها" in result
    assert "bale" in result
    assert "rubika" in result
