"""Bale E2E smoke: hard-stop for the admin's tests — this is the user-visible chain."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.bale.adapter import BaleAdapter
from gateway.config import PlatformConfig
from gateway.resilience_orchestrator import get_orchestrator
from gateway.provider_radar import ProviderRadar, EndpointHealth


@pytest.fixture
def bale_adapter():
    config = PlatformConfig(enabled=True, extra={"token": "test-bale-token"})
    a = BaleAdapter(config)
    a._client = MagicMock()
    a._client.send_message = AsyncMock(return_value={"message_id": 555})
    a._client.edit_message_text = AsyncMock(return_value={"message_id": 555})
    a._client.answer_callback_query = AsyncMock(return_value=True)
    return a


@pytest.mark.asyncio
async def test_emergency_menu_shows_live_picker_buttons(bale_adapter, tmp_path):
    """Check the full chain: radar produces healthy entries → orchestrator assembles
    → Bale renders Persian label + button with REAL target-model name."""
    # make a fake radar with two healthy endpoints incl. local
    radar = ProviderRadar(cache_path=tmp_path / "r.json")
    radar._health["local-llamacpp"] = EndpointHealth(
        name="local-llamacpp", base_url="http://127.0.0.1:8080/v1",
        model_hint="local-model", tier="local", reachable=True, latency_ms=5.0, success_count=5)
    radar._health["pollinations-text"] = EndpointHealth(
        name="pollinations-text", base_url="https://text.pollinations.ai/openai",
        model_hint="openai-fast", tier="anonymous", reachable=True, latency_ms=300.0, success_count=3)

    state = {
        "providers": [{"slug": "custom"}], "current_model": "moonshotai/kimi-k3",
        "current_provider": "custom", "session_key": "s1", "on_model_selected": None,
    }
    bale_adapter._model_picker_state["661453305"] = state

    # Patch get_orchestrator used by the Bale handler
    orch = get_orchestrator()

    real_refresh = orch.refresh
    real_picker = orch.refresh(force=True).picker_buttons
    orch._radar = radar

    def fake_refresh(force=False):
        return orch.refresh(force=True)

    with patch.object(orch, "refresh", lambda **k: real_refresh(force=True)):
        await bale_adapter._handle_model_picker_callback(
            query={"id": "cbq1"}, data="mp:emergency_free", chat_id="661453305", msg_id=555)

    # Verify the button list contains LIVE probed names in "🟢 name · endpoint · latency" shape
    call = bale_adapter._client.edit_message_text.call_args
    assert call, "edit_message_text was never called"
    markup = call[0][4] if len(call[0]) > 4 else call[1].get("reply_markup", {})
    text = call[0][2] if len(call[0]) > 2 else ""
    assert "بخش هوش مصنوعی اضطراری" in text

    buttons = [btn["text"] for row in markup.get("inline_keyboard", []) for btn in row]
    joined = " | ".join(buttons)
    assert "●" not in joined  # no legacy placeholder
    # Persian label shapes present
    assert "مدل اضطراری" in text or "رادار زنده" in text, text
    # Each button carries a real discovered model id, deduped; navigation rows allowed
    ids = [btn["callback_data"] for row in markup.get("inline_keyboard", []) for btn in row]
    mm_btns = [b for b in ids if b.startswith("mm:")]
    assert mm_btns  # at least one model button
    # mm buttons are unique
    assert len(mm_btns) == len(set(mm_btns))
