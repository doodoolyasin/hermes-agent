"""Deep tests for ProviderRadar: offline resilience, scoring, picker output."""
import json
from unittest.mock import patch

import pytest

from gateway.provider_radar import EndpointHealth, ProviderRadar


@pytest.fixture
def radar(tmp_path):
    return ProviderRadar(cache_path=tmp_path / "radar.json")


def fake_probe_factory(outcomes: dict[str, bool]):
    def _probe(url: str, timeout: float = 6.0):
        for key, ok in outcomes.items():
            if key in url:
                return ok, 42.0 if ok else 6000.0
        return False, 6000.0
    return _probe


def test_radar_survives_total_outage(radar):
    """Even with everything down, refresh must not raise and caching must work."""
    with patch.object(ProviderRadar, "_probe", staticmethod(lambda u, t=6.0: (False, 0.0))):
        results = radar.refresh()
    assert all(not h.reachable for h in results)
    assert radar.cache_path.exists()


def test_radar_scores_reachable_first(radar):
    outcomes = {"pollinations.ai": True, "openrouter.ai": False}
    with patch.object(ProviderRadar, "_probe", staticmethod(fake_probe_factory(outcomes))):
        radar.refresh()
    chain = radar.best_chain()
    assert chain[0].name == "pollinations-text"
    assert chain[0].reachable


def test_radar_history_persists_across_instances(tmp_path):
    cache = tmp_path / "radar.json"
    r1 = ProviderRadar(cache_path=cache)
    with patch.object(ProviderRadar, "_probe", staticmethod(fake_probe_factory({"pollinations": True}))):
        r1.refresh()
    # simulate restart
    r2 = ProviderRadar(cache_path=cache)
    assert r2._health["pollinations-text"].success_count >= 1


def test_picker_models_show_live_names_and_status(radar):
    outcomes = {"pollinations": True, "ollama": True}
    with patch.object(ProviderRadar, "_probe", staticmethod(fake_probe_factory(outcomes))):
        radar.refresh()
    models = radar.picker_models()
    names = [m["id"] for m in models]
    assert "openai-fast" in names
    displays = " ".join(m["display"] for m in models)
    assert "🟢" in displays and "pollinations" in displays


def test_picker_includes_offline_but_previously_good(radar):
    """MODEL MUST STAY VISIBLE for retry: mark OK once, then fail — should appear with 🟡."""
    with patch.object(ProviderRadar, "_probe", staticmethod(fake_probe_factory({"pollinations": True}))):
        radar.refresh()
    with patch.object(ProviderRadar, "_probe", staticmethod(lambda u, t=6.0: (False, 0.0))):
        radar.refresh()
    models = radar.picker_models()
    poll = next(m for m in models if m["base_url"].endswith("text.pollinations.ai/openai"))
    assert "🟡" in poll["display"]


def test_radar_corrupt_cache_recovers(tmp_path):
    cache = tmp_path / "radar.json"
    cache.write_text("{BR0KEN,,,")
    r = ProviderRadar(cache_path=cache)
    assert r.best_chain() == []  # no crash, empty state
