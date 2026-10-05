"""Orchestrator ↔ radar ↔ picker integration invariants."""
from unittest.mock import MagicMock, patch
import pytest

from gateway.resilience_orchestrator import ResilienceOrchestrator
from gateway.provider_radar import ProviderRadar, EndpointHealth


@pytest.fixture
def radar(tmp_path):
    return ProviderRadar(cache_path=tmp_path / "r.json")


def _h(name, reachable=True, hint="m1", lat=100, tier="anonymous"):
    return EndpointHealth(name=name, base_url=f"https://{name}.example", model_hint=hint,
                          tier=tier, reachable=reachable, latency_ms=lat,
                          success_count=3 if reachable else 1, fail_count=0 if reachable else 2)


def test_local_runtime_wins_over_remote(radar):
    radar._health.update({"pollinations-text": _h("pollinations-text")})
    with patch.object(radar, "refresh", lambda **k: None):
        orch = ResilienceOrchestrator(radar=radar)
        fake_ollama = MagicMock()
        fake_ollama.is_available.return_value = True
        fake_ollama.list_models.return_value = ["qwen2.5:0.5b"]
        with patch("hermes_offline.runtimes.LlamaCppRuntime", return_value=MagicMock(is_available=lambda: False)), \
             patch("hermes_offline.runtimes.OllamaRuntime", return_value=fake_ollama):
            chain = orch.refresh(force=True)
    assert chain.offline_runtime_up
    assert chain.local_model == "qwen2.5:0.5b"
    assert chain.active == []  # remote demoted to degraded when local is up
    buttons = chain.picker_buttons()
    assert buttons[0]["id"] == "qwen2.5:0.5b"  # local first


def test_picker_has_no_duplicate_model_ids(radar):
    radar._health.update({
        "a": _h("a", reachable=True, hint="m-dup"),
        "b": _h("b", reachable=True, hint="m-dup"),  # same model id on two endpoints
        "c": _h("c", reachable=False, hint="m-x"),
    })
    with patch.object(radar, "refresh", lambda **k: None):
        orch = ResilienceOrchestrator(radar=radar)
        chain = orch.refresh(force=True)
    ids = [b["id"] for b in chain.picker_buttons()]
    assert len(ids) == len(set(ids))


def test_refresh_is_idempotent_first_call_then_cache(radar):
    with patch.object(radar, "refresh", side_effect=lambda **k: radar._health.setdefault("x", _h("x"))):
        orch = ResilienceOrchestrator(radar=radar)
        c1 = orch.refresh(force=True)
        n = len(radar._health)
        c2 = orch.refresh()  # must NOT re-probe
        assert len(radar._health) == n
        assert c1 is not None and c2 is c1


def test_choose_falls_back_to_last_known_good(radar):
    radar._health.update({"old": _h("old", reachable=False, hint="m-old")})
    with patch.object(radar, "refresh", lambda **k: None):
        orch = ResilienceOrchestrator(radar=radar)
        choice = orch.choose()
    assert choice is not None
    assert choice["model"] == "m-old"  # reachable=False but succeeds last → still offered


def test_choose_returns_none_when_no_history_at_all(radar):
    with patch.object(radar, "refresh", lambda **k: None):
        orch = ResilienceOrchestrator(radar=radar)
        assert orch.choose() is None
