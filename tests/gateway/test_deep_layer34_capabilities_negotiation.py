import pytest
from gateway.platforms.capabilities import PlatformCapabilities
from plugins.platforms.bale.adapter import BaleAdapter
from plugins.platforms.rubika.adapter import RubikaAdapter
from gateway.config import PlatformConfig


def test_bale_vs_rubika_capability_matrix_comparison():
    bale_cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    bale_adapter = BaleAdapter(bale_cfg)
    bale_caps = bale_adapter.capabilities

    rubika_cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    rubika_adapter = RubikaAdapter(rubika_cfg)
    rubika_caps = rubika_adapter.capabilities

    # 1. Bale has native inline keyboard and callbacks
    assert bale_caps.supports("inline_keyboard") is True
    assert bale_caps.supports("callbacks") is True
    assert bale_caps.get_fallback_strategy("inline_keyboard") == "native"
    assert "inline_keyboard" in bale_caps.list_supported()

    # 2. Rubika lacks inline keyboard and callbacks -> must return numbered_text_menu fallback strategy
    assert rubika_caps.supports("inline_keyboard") is False
    assert rubika_caps.supports("callbacks") is False
    assert rubika_caps.get_fallback_strategy("inline_keyboard") == "numbered_text_menu"
    assert rubika_caps.get_fallback_strategy("callbacks") == "numbered_text_menu"
    assert "inline_keyboard" in rubika_caps.list_unsupported()

    # 3. Both support replies and documents
    assert bale_caps.supports("replies") is True
    assert rubika_caps.supports("replies") is True
    assert bale_caps.supports("documents") is True
    assert rubika_caps.supports("documents") is True


def test_custom_capabilities_fallback_fallthrough():
    caps = PlatformCapabilities(
        markdown=False,
        buttons_text_fallback=False,
    )
    assert caps.get_fallback_strategy("markdown") == "plain_text_strip"
    assert caps.get_fallback_strategy("inline_keyboard") == "omit"
    assert caps.get_fallback_strategy("unknown_future_feature") == "unsupported"
