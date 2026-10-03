"""Automatic discovery and fallback for free AI providers (public & local).

Enabled by default. Probes known local runtimes (Ollama, LM Studio, vLLM)
and public free OpenAI-compatible endpoints (Pollinations AI primary and mirrors)
to ensure Hermes works out of the box even without pre-configured API credentials,
and provides a resilient multi-tier fallback ladder during network restrictions or outages.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Multi-tiered candidate list:
# Tier 1: Local runtimes (fastest, private, zero-cost, offline-proof)
# Tier 2: Public free OpenAI-compatible endpoints (Pollinations fast, reasoning, general)
# Tier 3: Secondary mirrors for essential times & censorship resilience
FREE_CANDIDATES: List[Dict[str, Any]] = [
    {
        "name": "Local Ollama",
        "provider": "custom",
        "base_url": "http://localhost:11434/v1",
        "probe_url": "http://localhost:11434/api/tags",
        "api_key": "ollama",
        "model_finder": lambda data: data.get("models", [{}])[0].get("name") if data.get("models") else None,
        "default_model": "llama3.2",
    },
    {
        "name": "Local vLLM / LM Studio / llama-server",
        "provider": "custom",
        "base_url": "http://localhost:1234/v1",
        "probe_url": "http://localhost:1234/v1/models",
        "api_key": "not-needed",
        "model_finder": lambda data: data.get("data", [{}])[0].get("id") if data.get("data") else None,
        "default_model": "default",
    },
    {
        "name": "Local Server (port 8080)",
        "provider": "custom",
        "base_url": "http://localhost:8080/v1",
        "probe_url": "http://localhost:8080/v1/models",
        "api_key": "not-needed",
        "model_finder": lambda data: data.get("data", [{}])[0].get("id") if data.get("data") else None,
        "default_model": "default",
    },
    {
        "name": "Pollinations AI (Fast Free)",
        "provider": "custom",
        "base_url": "https://text.pollinations.ai/openai",
        "probe_url": "https://text.pollinations.ai/openai/models",
        "api_key": "free-community",
        "model_finder": lambda data: "openai-fast",
        "default_model": "openai-fast",
    },
    {
        "name": "Pollinations AI (Reasoning & Code)",
        "provider": "custom",
        "base_url": "https://text.pollinations.ai/openai",
        "probe_url": "https://text.pollinations.ai/openai/models",
        "api_key": "free-community",
        "model_finder": lambda data: "gpt-oss-20b",
        "default_model": "gpt-oss-20b",
    },
    {
        "name": "Pollinations AI (General)",
        "provider": "custom",
        "base_url": "https://text.pollinations.ai/openai",
        "probe_url": "https://text.pollinations.ai/openai/models",
        "api_key": "free-community",
        "model_finder": lambda data: "openai",
        "default_model": "openai",
    },
    {
        "name": "Pollinations AI (Mirror)",
        "provider": "custom",
        "base_url": "https://genai.pollinations.ai/openai",
        "probe_url": "https://genai.pollinations.ai/openai/models",
        "api_key": "free-community",
        "model_finder": lambda data: "openai-fast",
        "default_model": "openai-fast",
    },
]


def probe_endpoint(probe_url: str, timeout: float = 2.0) -> Optional[dict]:
    """Perform a lightweight GET request with short timeout."""
    try:
        req = urllib.request.Request(
            probe_url,
            headers={"User-Agent": "Hermes-Agent-FreeDiscovery/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                raw = resp.read(65536).decode("utf-8", errors="replace")
                try:
                    return json.loads(raw)
                except Exception:
                    return {}
    except Exception:
        return None
    return None


def get_free_models_catalog() -> List[Dict[str, Any]]:
    """Return catalog of available free models for interactive model picker."""
    return [
        {
            "id": "openai-fast",
            "name": "⚡ openai-fast (سریع و هوشمند)",
            "provider": "custom",
            "base_url": "https://text.pollinations.ai/openai",
            "description": "مدل سریع و رایگان برای کارهای روزمره و چت",
        },
        {
            "id": "gpt-oss-20b",
            "name": "🧠 gpt-oss-20b (استدلال و برنامه‌نویسی)",
            "provider": "custom",
            "base_url": "https://text.pollinations.ai/openai",
            "description": "مدل متن‌باز قدرتمند با توانایی استدلال بالا",
        },
        {
            "id": "openai",
            "name": "🌐 openai (مدل عمومی)",
            "provider": "custom",
            "base_url": "https://text.pollinations.ai/openai",
            "description": "مدل عمومی استاندارد جهت گفتگو و پرسش‌وپاسخ",
        },
        {
            "id": "deepseek",
            "name": "🔍 deepseek (دیپ‌سیک)",
            "provider": "custom",
            "base_url": "https://text.pollinations.ai/openai",
            "description": "مدل چت هوشمند دیپ‌سیک",
        },
    ]


def discover_free_provider(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Probe candidate free providers and return first working one."""
    cfg = config or {}
    free_cfg = cfg.get("free_providers") or {}
    if free_cfg.get("enabled") is False or free_cfg.get("auto_discover") is False:
        return None

    # Custom candidate endpoints from user config take precedence
    custom_candidates = free_cfg.get("candidates") or []
    all_candidates = custom_candidates + FREE_CANDIDATES

    for candidate in all_candidates:
        base_url = candidate.get("base_url")
        probe_url = candidate.get("probe_url")
        if not probe_url:
            if not base_url:
                continue
            probe_url = f"{base_url.rstrip('/')}/models"

        data = probe_endpoint(probe_url)
        if data is not None:
            model = candidate.get("default_model", "openai-fast")
            finder = candidate.get("model_finder")
            if callable(finder):
                try:
                    found = finder(data)
                    if found:
                        model = found
                except Exception:
                    pass

            return {
                "name": candidate.get("name", "Free Provider"),
                "provider": candidate.get("provider", "custom"),
                "base_url": candidate.get("base_url"),
                "api_key": candidate.get("api_key", "free"),
                "model": model,
                "api_mode": "chat_completions",
            }
    return None


def resolve_free_fallback_runtime(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return runtime kwargs dict for AIAgent if free provider is found."""
    found = discover_free_provider(config)
    if not found:
        return None

    return {
        "provider": found["provider"],
        "base_url": found["base_url"],
        "api_key": found["api_key"],
        "model": found["model"],
        "api_mode": found.get("api_mode", "chat_completions"),
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": {},
        "_free_provider_name": found["name"],
    }
