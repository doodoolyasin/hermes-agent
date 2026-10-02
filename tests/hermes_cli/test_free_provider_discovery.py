"""Unit tests for free AI provider auto-discovery and fallback."""

import pytest
from hermes_cli.free_provider_discovery import (
    discover_free_provider,
    resolve_free_fallback_runtime,
    probe_endpoint,
)


def test_free_provider_discovery_disabled():
    cfg = {"free_providers": {"enabled": False}}
    assert discover_free_provider(cfg) is None
    assert resolve_free_fallback_runtime(cfg) is None


def test_free_provider_discovery_mocked(monkeypatch):
    fake_data = {"data": [{"id": "mock-model-v1"}]}

    def mock_probe(probe_url, timeout=2.0):
        if "1234" in probe_url:
            return fake_data
        return None

    monkeypatch.setattr("hermes_cli.free_provider_discovery.probe_endpoint", mock_probe)

    found = discover_free_provider({"free_providers": {"enabled": True}})
    assert found is not None
    assert found["base_url"] == "http://localhost:1234/v1"
    assert found["model"] == "mock-model-v1"

    runtime = resolve_free_fallback_runtime({"free_providers": {"enabled": True}})
    assert runtime is not None
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == "http://localhost:1234/v1"
    assert runtime["model"] == "mock-model-v1"


def test_free_provider_custom_candidates(monkeypatch):
    custom_candidate = {
        "name": "Custom Free LLM",
        "provider": "custom",
        "base_url": "http://10.0.0.5:8000/v1",
        "probe_url": "http://10.0.0.5:8000/v1/models",
        "api_key": "custom-key",
        "default_model": "custom-llm",
    }

    def mock_probe(probe_url, timeout=2.0):
        if "10.0.0.5" in probe_url:
            return {"models": []}
        return None

    monkeypatch.setattr("hermes_cli.free_provider_discovery.probe_endpoint", mock_probe)

    cfg = {"free_providers": {"candidates": [custom_candidate]}}
    found = discover_free_provider(cfg)
    assert found is not None
    assert found["name"] == "Custom Free LLM"
    assert found["base_url"] == "http://10.0.0.5:8000/v1"
    assert found["model"] == "custom-llm"
