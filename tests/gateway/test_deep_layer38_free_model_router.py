import pytest
from unittest.mock import AsyncMock, MagicMock
from hermes_cli.model_switch import switch_model, ModelSwitchResult


def test_switch_model_emergency_free_names():
    for name in ("openai-fast", "gpt-oss-20b", "deepseek", "openai"):
        res = switch_model(name, current_provider="", current_model="gpt-4o")
        assert res.success is True, f"Failed on {name}"
        assert res.new_model == name
        assert res.base_url == "https://text.pollinations.ai/openai"
        assert res.api_mode == "chat_completions"
        assert res.target_provider == "custom"


def test_switch_model_emergency_fallback_provider_alias():
    # Explicit provider syntax
    res = switch_model("openai-fast", current_provider="", current_model="gpt-4o", explicit_provider="free_fallback")
    assert res.success is True
    assert res.new_model == "openai-fast"
    assert res.target_provider == "custom"

    # Alternative alias
    res2 = switch_model("gpt-oss-20b", current_provider="custom", current_model="gpt-4o", explicit_provider="emergency_free")
    assert res2.success is True
    assert res2.new_model == "gpt-oss-20b"


def test_switch_model_emergency_provider_not_contaminated_by_custom_url():
    # Ensure that if we pass a custom base_url the payload stays configured with emergency_url
    res = switch_model("openai-fast", current_provider="custom", current_model="custom-model", current_base_url="http://localhost:8080/v1")
    assert res.success is True
    assert res.base_url == "https://text.pollinations.ai/openai"
    assert res.provider_changed is False  # current_provider is already custom; same provider target
    assert res.target_provider == "custom"


@pytest.mark.asyncio
async def test_model_picker_emergency_populates_state_correctly():
    from plugins.platforms.bale.adapter import BaleAdapter
    from gateway.config import PlatformConfig
    adapter = BaleAdapter(PlatformConfig(enabled=True, extra={"token": "test"}))
    adapter._client = MagicMock()
    adapter._client.edit_message_text = AsyncMock(return_value={"message_id": 555})
    adapter._client.answer_callback_query = AsyncMock(return_value=True)

    state = {
        "providers": [{"slug": "custom", "name": "Default"}],
        "current_model": "moonshotai/kimi-k3",
        "current_provider": "custom",
        "session_key": "sess_1",
        "on_model_selected": None,
    }
    adapter._model_picker_state["661453305"] = state

    # Simulate clicking on the emergency_free button
    await adapter._handle_model_picker_callback(
        query={"id": "cbq1"},
        data="mp:emergency_free",
        chat_id="661453305",
        msg_id=1042,
    )

    assert adapter._model_picker_state["661453305"]["model_list"][0]["id"] == "openai-fast"
    assert adapter._model_picker_state["661453305"]["selected_provider"] == "custom"

    kwargs = adapter._client.edit_message_text.call_args
    text = kwargs[0][2] if kwargs else ""
    assert "بخش هوش مصنوعی اضطراری" in text
    # Model buttons render in the inline keyboard; radar may add status dot + latency/display
    markup = kwargs[1].get("reply_markup", {})
    labels = [btn["text"] for row in markup.get("inline_keyboard", []) for btn in row]
    assert any("openai-fast" in lbl for lbl in labels), labels
    # correct model list is stored
    assert adapter._model_picker_state["661453305"]["model_list"][0]["id"] == "openai-fast"
