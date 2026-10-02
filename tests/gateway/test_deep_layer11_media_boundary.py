import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig
from plugins.platforms.bale.adapter import BaleAdapter, BaleClient
from plugins.platforms.rubika.adapter import RubikaAdapter, RubikaClient


@pytest.mark.asyncio
async def test_bale_media_file_not_found(tmp_path):
    client = BaleClient("token")
    non_existent = str(tmp_path / "does_not_exist.png")

    with pytest.raises(FileNotFoundError, match="Media file not found"):
        await client.send_document("101", non_existent)


@pytest.mark.asyncio
async def test_bale_media_file_exceeds_size_limit(tmp_path, monkeypatch):
    client = BaleClient("token")
    large_file = tmp_path / "large_video.mp4"
    large_file.write_bytes(b"dummy")

    # Mock getsize to simulate 60MB file
    monkeypatch.setattr(os.path, "getsize", lambda p: 60 * 1024 * 1024)

    with pytest.raises(ValueError, match="exceeds maximum upload limit of 50MB"):
        await client.send_document("101", str(large_file))


@pytest.mark.asyncio
async def test_rubika_document_size_limit_rejection(tmp_path, monkeypatch):
    cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    adapter = RubikaAdapter(cfg)
    adapter._client = AsyncMock()

    large_doc = tmp_path / "huge.pdf"
    large_doc.write_bytes(b"pdf header")

    # Mock getsize to simulate 55MB
    monkeypatch.setattr(os.path, "getsize", lambda p: 55 * 1024 * 1024)

    result = await adapter.send_document("chat_1", str(large_doc))
    assert result.success is False
    assert "exceeds maximum upload limit of 50MB" in result.error
    assert adapter._client.send_document.call_count == 0


@pytest.mark.asyncio
async def test_rubika_document_not_found(tmp_path):
    cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    adapter = RubikaAdapter(cfg)
    adapter._client = AsyncMock()

    result = await adapter.send_document("chat_1", str(tmp_path / "missing.txt"))
    assert result.success is False
    assert "file not found" in result.error.lower()
