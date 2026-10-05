"""Tests: model_switch must honor radar-catalogue model ids (not just pollinations)."""
import os
import pytest

from hermes_cli.model_switch import switch_model


def test_switch_openrouter_free_model_via_radar(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = switch_model("qwen/qwen3.8-27b:free", current_provider="", current_model="x")
    assert res.success
    assert res.new_model == "qwen/qwen3.8-27b:free"
    assert "openrouter" in res.base_url


def test_switch_groq_free_model_via_radar(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = switch_model("llama-3.3-70b-versatile", current_provider="", current_model="x")
    assert res.success
    assert "groq" in res.base_url


def test_switch_anonymous_still_wins_over_radar():
    # openai-fast is in the static free_catalog AND is pollinations' discover; the anonymous
    # catalogue short-circuit must always win over the dynamic one.
    res = switch_model("openai-fast", current_provider="", current_model="x")
    assert res.success
    assert res.base_url == "https://text.pollinations.ai/openai"


def test_switch_unknown_model_falls_through():
    res = switch_model("this-model-does-not-exist-anywhere-xyz", current_provider="", current_model="x")
    # falls through to standard resolution (fails or routes via generic path, but never to radar)
    if res.success:
        assert "pollinations" not in res.base_url
        assert res.provider_label != "رادار: pollinations-text"
