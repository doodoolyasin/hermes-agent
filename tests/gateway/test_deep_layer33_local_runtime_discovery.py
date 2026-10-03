import pytest
from unittest.mock import patch
from hermes_cli.free_provider_discovery import (
    discover_free_provider,
    probe_endpoint,
    resolve_free_fallback_runtime,
)


def test_discovery_with_malformed_user_candidates():
    # User config with malformed/empty candidates
    malformed_cfg = {
        "free_providers": {
            "enabled": True,
            "auto_discover": True,
            "candidates": [
                {},  # completely empty
                {"name": "broken_1"},  # missing base_url and probe_url
                {"base_url": None, "probe_url": None},
                {"base_url": "http://127.0.0.1:9999", "probe_url": None},  # non-existent port
            ],
        }
    }

    # Should safely skip broken candidates without throwing ValueError or AttributeError
    with patch("hermes_cli.free_provider_discovery.probe_endpoint", return_value=None):
        res = discover_free_provider(malformed_cfg)
        assert res is None


def test_probe_endpoint_html_or_garbage_json(monkeypatch):
    class FakeResponse:
        status = 200
        def read(self, limit):
            return b"<html><body>502 Bad Gateway Nginx</body></html>"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: FakeResponse())

    # When 200 returns HTML instead of JSON, probe_endpoint should return empty dict, not crash
    data = probe_endpoint("http://example.com/models", timeout=1.0)
    assert data == {}


def test_resolve_free_fallback_runtime_structure():
    with patch(
        "hermes_cli.free_provider_discovery.discover_free_provider",
        return_value={
            "name": "Mock Free Provider",
            "provider": "custom",
            "base_url": "https://api.mock.ai/v1",
            "api_key": "free-key",
            "model": "mock-fast",
            "api_mode": "chat_completions",
        },
    ):
        runtime = resolve_free_fallback_runtime()
        assert runtime is not None
        assert runtime["provider"] == "custom"
        assert runtime["base_url"] == "https://api.mock.ai/v1"
        assert runtime["api_key"] == "free-key"
        assert runtime["model"] == "mock-fast"
        assert runtime["_free_provider_name"] == "Mock Free Provider"
