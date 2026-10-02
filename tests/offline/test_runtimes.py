"""Tests for hermes_offline.runtimes (local inference runtime management)."""

from __future__ import annotations

from hermes_offline import runtimes
from hermes_offline.models import ModelRecord


def test_manager_exposes_both_adapters():
    m = runtimes.RuntimeManager.from_config({})
    assert m.get("ollama") is not None
    assert m.get("llama.cpp") is not None


def test_openai_base_urls_are_openai_compatible():
    m = runtimes.RuntimeManager.from_config({})
    assert m.get("ollama").openai_base_url().endswith("/v1")
    assert m.get("llama.cpp").openai_base_url().endswith("/v1")


def test_runtime_for_prefers_declared_runtime():
    m = runtimes.RuntimeManager.from_config({})
    assert m.runtime_for(ModelRecord(name="x", runtime="llama.cpp")).name == "llama.cpp"
    assert m.runtime_for(ModelRecord(name="y", runtime="ollama")).name == "ollama"


def test_runtime_for_falls_back_to_available():
    m = runtimes.RuntimeManager.from_config({})
    adapter = m.runtime_for(ModelRecord(name="z"))
    assert adapter is not None


def test_ollama_unavailable_when_binary_and_port_absent(monkeypatch):
    monkeypatch.setattr(runtimes, "_tcp_reachable", lambda *a, **k: False)
    monkeypatch.setattr(runtimes.shutil, "which", lambda name: None)
    rt = runtimes.OllamaRuntime(host="127.0.0.1", port=11434)
    assert rt.is_available() is False


def test_ollama_available_when_port_open(monkeypatch):
    monkeypatch.setattr(runtimes, "_tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr(runtimes, "_http_get_json", lambda *a, **k: {"models": []})
    rt = runtimes.OllamaRuntime(host="127.0.0.1", port=11434)
    assert rt.is_available() is True


def test_provider_profile_shape():
    m = runtimes.RuntimeManager.from_config({})
    prof = m.get("llama.cpp").provider_profile("local-gguf")
    assert set(prof) >= {"name", "base_url", "api_key", "model", "kind"}
    assert prof["base_url"].endswith("/v1")


def test_detect_and_available_are_lists():
    m = runtimes.RuntimeManager.from_config({})
    assert isinstance(m.detect(), list)
    assert isinstance(m.available(), list)
