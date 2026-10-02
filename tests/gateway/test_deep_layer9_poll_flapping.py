import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig
from plugins.platforms.bale.adapter import BaleAdapter, BaleClient, BaleAPIError


@pytest.mark.asyncio
async def test_poll_loop_survives_network_flapping_and_recovers():
    cfg = PlatformConfig(extra={"token": "fake_token", "backoff_base": 0.01, "backoff_max": 0.05, "max_attempts": 3}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter._running = True

    mock_client = AsyncMock()
    call_count = 0

    async def flapping_get_updates(offset, timeout):
        nonlocal call_count
        call_count += 1
        # First 8 calls fail (simulate severe network dropout exceeding max_attempts)
        if call_count <= 8:
            raise BaleAPIError("Network connection reset / 502 Bad Gateway", status_code=502, retryable=True)
        # 9th call succeeds and returns an update
        if call_count == 9:
            return [{"update_id": 101, "message": {"chat": {"id": 1}, "text": "Recovered!"}}]
        # Stop loop after recovery
        adapter._running = False
        return []

    mock_client.get_updates.side_effect = flapping_get_updates
    adapter._client = mock_client
    adapter._handle_update = AsyncMock()

    # Run poll loop
    await adapter._poll_loop()

    # Must have survived all 8 network errors, recovered on 9th, and dispatched update!
    assert call_count >= 9
    assert adapter._handle_update.call_count == 1
    assert adapter._handle_update.call_args[0][0]["update_id"] == 101


@pytest.mark.asyncio
async def test_poll_loop_terminates_on_fatal_auth_error():
    cfg = PlatformConfig(extra={"token": "fake_token"}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter._running = True

    mock_client = AsyncMock()
    mock_client.get_updates.side_effect = BaleAPIError("Forbidden: bot was blocked or token revoked", status_code=403)
    adapter._client = mock_client

    await adapter._poll_loop()

    # Loop must have terminated immediately
    assert adapter._running is False
    assert adapter.fatal_error_code == "auth"
