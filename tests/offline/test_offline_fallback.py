"""Tests for the connectivity-aware local provider fallback integration."""

from __future__ import annotations

import pytest

from hermes_cli import fallback_config
from hermes_offline import fallback as offline_fallback
from hermes_offline.models import ModelRecord, ModelState


class FakeAdapter:
    def __init__(self, *, available=True, base_url="http://127.0.0.1:11434/v1"):
        self._available = available
        self._base_url = base_url

    def is_available(self):
        return self._available

    def openai_base_url(self):
        return self._base_url


class FakeManager:
    def __init__(self, adapter):
        self._adapter = adapter

    def runtime_for(self, record):
        return self._adapter


def _verified(name="llama-3.2-3b"):
    return ModelRecord(name=name, state=ModelState.VERIFIED)


# ── gating ───────────────────────────────────────────────────────────
def test_disabled_by_default():
    assert offline_fallback.is_enabled({}) is False
    assert offline_fallback.local_fallback_entry({}, manager=FakeManager(FakeAdapter()),
                                                 records=[_verified()]) is None


def test_enabled_but_no_models_returns_none():
    cfg = {"offline": {"auto_local_fallback": True}}
    assert offline_fallback.local_fallback_entry(cfg, manager=FakeManager(FakeAdapter()),
                                                 records=[]) is None


def test_enabled_but_runtime_unavailable_returns_none():
    cfg = {"offline": {"auto_local_fallback": True}}
    entry = offline_fallback.local_fallback_entry(
        cfg, manager=FakeManager(FakeAdapter(available=False)), records=[_verified()])
    assert entry is None


# ── entry shape ──────────────────────────────────────────────────────
def test_entry_shape_is_chain_compatible():
    cfg = {"offline": {"auto_local_fallback": "true"}}
    entry = offline_fallback.local_fallback_entry(
        cfg, manager=FakeManager(FakeAdapter()), records=[_verified("qwen2.5-7b")])
    assert entry == {
        "provider": "custom",
        "model": "qwen2.5-7b",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "local",
        "api_mode": "openai",
    }


def test_verified_preferred_over_installed():
    cfg = {"offline": {"auto_local_fallback": True}}
    records = [ModelRecord(name="installed-only", state=ModelState.INSTALLED),
               ModelRecord(name="verified-one", state=ModelState.VERIFIED)]
    entry = offline_fallback.local_fallback_entry(
        cfg, manager=FakeManager(FakeAdapter()), records=records)
    assert entry["model"] == "verified-one"


# ── integration with the real chain normalizer ───────────────────────
def test_get_fallback_chain_appends_local_last(monkeypatch):
    monkeypatch.setattr(
        offline_fallback, "local_fallback_entry",
        lambda config: {"provider": "custom", "model": "local-gguf",
                        "base_url": "http://127.0.0.1:8080/v1"},
    )
    config = {"fallback_providers": [{"provider": "openrouter", "model": "x/y"}]}
    chain = fallback_config.get_fallback_chain(config)
    assert [e["model"] for e in chain] == ["x/y", "local-gguf"]


def test_get_fallback_chain_dedups_local_entry(monkeypatch):
    monkeypatch.setattr(
        offline_fallback, "local_fallback_entry",
        lambda config: {"provider": "custom", "model": "dup",
                        "base_url": "http://127.0.0.1:8080/v1"},
    )
    config = {"fallback_providers": [
        {"provider": "custom", "model": "dup", "base_url": "http://127.0.0.1:8080/v1"}]}
    chain = fallback_config.get_fallback_chain(config)
    assert len(chain) == 1


def test_get_fallback_chain_survives_offline_errors(monkeypatch):
    def boom(config):
        raise RuntimeError("offline subsystem exploded")

    monkeypatch.setattr(offline_fallback, "local_fallback_entry", boom)
    config = {"fallback_providers": [{"provider": "openrouter", "model": "x/y"}]}
    chain = fallback_config.get_fallback_chain(config)  # must not raise
    assert [e["model"] for e in chain] == ["x/y"]


def test_get_fallback_chain_unchanged_when_disabled():
    config = {"fallback_providers": [{"provider": "openrouter", "model": "x/y"}]}
    assert fallback_config.get_fallback_chain(config) == [
        {"provider": "openrouter", "model": "x/y"}]
