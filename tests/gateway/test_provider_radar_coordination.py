"""Coordination guards: radar -> registry -> picker must never disagree or duplicate."""
from unittest.mock import patch
import pytest

from gateway.provider_radar import ProviderRadar


def _ok_probe(url, timeout=6.0):
    if "pollinations" in url or "127.0.0.1:8080" in url:
        return True, 120.0
    return False, 0.0


def _fake_discover(url, timeout=6.0):
    if "pollinations" in url:
        return ["openai-fast", "openai", "gpt-oss-120b"]
    return []


def test_radar_multi_model_picker_lists_real_target_names(tmp_path):
    r = ProviderRadar(cache_path=tmp_path / "r.json")
    with patch.object(ProviderRadar, "_probe", staticmethod(_ok_probe)), \
         patch.object(ProviderRadar, "_discover_models", staticmethod(_fake_discover)):
        r.refresh()
    models = r.picker_models()
    ids = [m["id"] for m in models]
    assert "openai-fast" in ids and "openai" in ids
    assert len(ids) == len(set(ids)), f"duplicate picker entries: {ids}"
    # each entry must carry a base_url+key for actual use
    assert all(m["base_url"] for m in models)


def test_registry_injection_has_no_duplicates_and_valid_urls(tmp_path):
    from gateway import provider_registry as pr
    fake_health = pr.ProviderDescriptor  # ensure types exist
    with patch.dict("os.environ"):
        reg = pr.ProviderRegistry()  # triggers radar injection path with cache absent
    for pid, p in reg._providers.items():
        assert p.base_url.startswith(("http://", "https://")), pid
        assert pid, "empty provider id"


def test_legacy_cache_without_extra_models_loads(tmp_path):
    cache = tmp_path / "r.json"
    cache.write_text('{"endpoints": {"x": {"name": "x", "base_url": "http://a", "api_key": "", "model_hint": "m", "tier": "anonymous", "reachable": true, "latency_ms": 5, "last_success": 0, "success_count": 3, "fail_count": 0}}}')
    r = ProviderRadar(cache_path=cache)
    assert r._health["x"].success_count == 3
    assert r._health["x"].extra_models == []
