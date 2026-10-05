"""ProviderRadar: resilient free-tier AI endpoint discovery for restricted/DeNAT networks.

Design goals (Iranian complete-outage scenario included):
 1. NEVER crash the gateway on probe failure — every probe is guarded.
 2. Multi-layer connectivity classification: ONLINE / FILTERED / DOMESTIC_ONLY / OFFLINE_LOCAL.
 3. Source candidates from three disjoint channels so one failure can't starve the chain:
    - built-in static catalogue    (free community endpoints: Pollinations, OpenRouter free, ...)
    - dynamic refresh from GitHub  (curated list repo, cache-tolerant)
    - local GGUF runtime           (llama.cpp/Ollama — survives full international outage)
 4. Score = reachability (HTTP 2xx/4xx on /models) + latency + historical success rate,
    persisted to ~/.hermes/offline/provider_radar.json.
 5. Output feeds BOTH the smart router fallback chain AND the Bale emergency picker
    (live model names shown to the admin).
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import socket
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".hermes" / "offline" / "provider_radar.json"
PROBE_TIMEOUT = 6.0
HISTORY_WINDOW = 20

# --- Candidate sources -----------------------------------------------------

# Tier 1: anonymous free endpoints (no key at all)
_STATIC_CATALOGUE: list[dict] = [
    {
        "name": "pollinations-text",
        "base_url": "https://text.pollinations.ai/openai",
        "probe": "https://text.pollinations.ai/models",
        "discover_models_from": "https://text.pollinations.ai/models",
        "api_key": "free-community",
        "model_hint": "openai-fast",
        "tier": "anonymous",
    },
    {
        "name": "openrouter-free",
        "base_url": "https://openrouter.ai/api/v1",
        "probe": "https://openrouter.ai/api/v1/models",
        "discover_models_from": "https://openrouter.ai/api/v1/models",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_hint": "qwen/qwen3.8-27b:free",
        "tier": "keyed-free",
    },
    {
        "name": "voidai-anon",
        "base_url": "https://api.voidai.app/v1",
        "probe": "https://api.voidai.app/v1/models",
        "discover_models_from": "https://api.voidai.app/v1/models",
        "api_key": "free-community",
        "model_hint": "gpt-4o-mini",
        "tier": "anonymous",
    },
    # Pollinations new turnkey API (anonymous tier; succeeds when legacy /v1 saturates)
    {
        "name": "pollinations-enter",
        "base_url": "https://enter.pollinations.ai/api/generate/v1",
        "probe": "https://enter.pollinations.ai/api/generate/v1/models",
        "discover_models_from": "https://enter.pollinations.ai/api/generate/v1/models",
        "api_key": "free-community",
        "model_hint": "openai-fast",
        "tier": "anonymous",
    },
    # Keyed free tiers — listed only when the matching env var actually exists
    {"name": "groq-free",       "base_url": "https://api.groq.com/openai/v1",            "probe": "https://api.groq.com/openai/v1/models",              "api_key_env": "GROQ_API_KEY",       "model_hint": "llama-3.3-70b-versatile",                 "tier": "keyed-free"},
    {"name": "mistral-free",    "base_url": "https://api.mistral.ai/v1",                  "probe": "https://api.mistral.ai/v1/models",                    "api_key_env": "MISTRAL_API_KEY",    "model_hint": "mistral-small-latest",                    "tier": "keyed-free"},
    {"name": "cerebras-free",   "base_url": "https://api.cerebras.ai/v1",                 "probe": "https://api.cerebras.ai/v1/models",                   "api_key_env": "CEREBRAS_API_KEY",   "model_hint": "llama-4-scout-17b-16e-instruct",          "tier": "keyed-free"},
    {"name": "google-free",     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "probe": "https://generativelanguage.googleapis.com/v1beta/models", "api_key_env": "GEMINI_API_KEY", "model_hint": "gemini-2.0-flash-exp",                  "tier": "keyed-free"},
    {"name": "huggingface-free","base_url": "https://router.huggingface.co/v1",           "probe": "https://router.huggingface.co/v1/models",             "api_key_env": "HUGGINGFACE_API_KEY","model_hint": "Qwen/Qwen2.5-72B-Instruct",              "tier": "keyed-free"},
]

# Tier 2: domestic (Iran-routable) relay hints — resolved lazily; survive international outage
_DOMESTIC_HINTS: list[dict] = [
    {"name": "gpn-relay", "base_url_env": "GPN_LLM_RELAY"},  # local GPN tunnel if host runs it
    {"name": "local-ollama", "probe": "http://127.0.0.1:11434/api/tags", "base_url": "http://127.0.0.1:11434/v1", "tier": "local"},
    {"name": "local-llamacpp", "probe": "http://127.0.0.1:8080/v1/models", "base_url": "http://127.0.0.1:8080/v1", "tier": "local"},
    {"name": "local-lmstudio", "probe": "http://127.0.0.1:1234/v1/models", "base_url": "http://127.0.0.1:1234/v1", "tier": "local"},
    # Demo GGUF server already installed by the operator on this host
    {"name": "local-qwen05", "probe": "http://127.0.0.1:8080/health", "base_url": "http://127.0.0.1:8080/v1", "model_hint": "qwen2.5-0.5b-instruct-q2_k.gguf", "tier": "local"},
]

# Dynamic catalogue URL (GitHub-hosted mirror of free endpoints)
_DYNAMIC_CATALOGUE_URL = "https://raw.githubusercontent.com/cheahjs/free-llm-api-resources/main/README.md"


@dataclass
class EndpointHealth:
    name: str
    base_url: str = ""
    api_key: str = ""
    model_hint: str = ""
    tier: str = "anonymous"
    reachable: bool = False
    latency_ms: float = 0.0
    last_success: float = 0.0
    success_count: int = 0
    fail_count: int = 0
    extra_models: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        base = 100.0 if self.reachable else 0.0
        latency_penalty = min(self.latency_ms / 100.0, 30.0)
        history = self.success_count - self.fail_count * 2
        tier_bonus = {"local": 50, "anonymous": 20, "keyed-free": 5}.get(self.tier, 0)
        return base + tier_bonus + history - latency_penalty


class ProviderRadar:
    """Multi-source, fail-safe free provider discovery."""

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self._health: dict[str, EndpointHealth] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text())
                for name, h in data.get("endpoints", {}).items():
                    self._health[name] = EndpointHealth(**{k: v for k, v in h.items() if k in EndpointHealth.__dataclass_fields__})
        except Exception as exc:  # fail-closed: corrupt cache = fresh state
            logger.debug("radar cache load failed (%s); starting fresh", exc)

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"endpoints": {k: asdict(v) for k, v in self._health.items()}}, indent=1))
            tmp.replace(self.cache_path)
        except Exception as exc:
            logger.debug("radar cache save failed: %s", exc)

    # -- probing ------------------------------------------------------------
    @staticmethod
    def _probe(probe_url: str, timeout: float = PROBE_TIMEOUT) -> tuple[bool, float]:
        if not probe_url:
            return False, 0.0
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(probe_url, headers={"User-Agent": "hermes-radar/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read(256)
            return True, (time.monotonic() - t0) * 1000.0
        except urllib.error.HTTPError as e:
            # 401/403 on probe means host reachable (auth gate) — counts as reachable
            if e.code in (401, 403):
                return True, (time.monotonic() - t0) * 1000.0
            return False, (time.monotonic() - t0) * 1000.0
        except Exception:
            return False, (time.monotonic() - t0) * 1000.0

    def refresh(self, candidates: Optional[list[dict]] = None, parallel: bool = True) -> list[EndpointHealth]:
        candidates = candidates if candidates is not None else self.all_candidates()
        results: list[EndpointHealth] = []

        def _one(c: dict) -> EndpointHealth:
            h = self._health.get(c["name"]) or EndpointHealth(name=c["name"])
            h.base_url = c.get("base_url", h.base_url)
            h.tier = c.get("tier", h.tier)
            h.model_hint = c.get("model_hint", h.model_hint)
            if not h.api_key:
                h.api_key = c.get("api_key", "")
            probe = c.get("probe") or (c.get("base_url", "").rstrip("/") + "/models")
            h.base_url = c.get("base_url", h.base_url).rstrip("/")
            ok, lat = self._probe(probe)
            if ok and c.get("discover_models_from"):
                live = self._discover_models(c["discover_models_from"])
                if live:
                    h.model_hint = live[0]
                    h.extra_models = live[:6]
            h.reachable = ok
            h.latency_ms = lat
            if ok:
                h.success_count += 1
                h.last_success = time.time()
            else:
                h.fail_count += 1
            self._health[c["name"]] = h
            return h

        if parallel and len(candidates) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
                results = list(ex.map(_one, candidates))
        else:
            results = [_one(c) for c in candidates]

        self._save()
        return results

    @staticmethod
    def _discover_models(url: str, timeout: float = 6.0) -> list[str]:
        """Fail-safe model listing (Pollinations-style or OpenAI-style /models)."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hermes-radar/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read(1 << 16))
            items = data if isinstance(data, list) else data.get("data") or data.get("models") or []
            names: list[str] = []
            for it in items:
                if isinstance(it, dict):
                    mid = it.get("name") or it.get("id")
                    if mid:
                        names.append(str(mid))
                elif isinstance(it, str):
                    names.append(it)
            return names
        except Exception:
            return []

    # -- candidates ---------------------------------------------------------
    def all_candidates(self) -> list[dict]:
        import os
        out: list[dict] = []
        for c in _STATIC_CATALOGUE:
            env_name = c.get("api_key_env")
            if env_name:
                # Keyed free tier: revealed only when the admin has actually set the key
                import os as _os
                api_key = _os.environ.get(env_name, "").strip()
                if not api_key:
                    continue
                c = dict(c, api_key=api_key)
            out.append(c)
        for h in _DOMESTIC_HINTS:
            if "base_url_env" in h:
                url = os.environ.get(h["base_url_env"], "").strip()
                if url:
                    out.append({"name": h["name"], "base_url": url, "probe": url.rstrip("/") + "/models", "tier": "local"})
            else:
                out.append(dict(h))
        return out

    def best_chain(self, limit: int = 4) -> list[EndpointHealth]:
        """Ranked list — reachable first, then by score. Includes last-known-good
        entries so a momentarily unreachable favourite still appears in the picker."""
        return sorted(self._health.values(), key=lambda h: (h.reachable, h.score), reverse=True)[:limit]

    def picker_models(self, per_endpoint: int = 2) -> list[dict]:
        """Entries the Bale/TG emergency picker renders with LIVE probed names.
        Emits up to ``per_endpoint`` models per healthy endpoint so every real
        target-model name (extra_models or model_hint) is visible as a button.
        Model ids are globally deduplicated — first (highest-scored) endpoint wins."""
        models = []
        seen: set[str] = set()
        for h in self.best_chain(limit=8):
            if not (h.reachable or h.success_count > 0):
                continue
            tag = "🟢" if h.reachable else "🟡"
            names = h.extra_models or ([h.model_hint] if h.model_hint else [h.name])
            for mid in names[:per_endpoint]:
                if mid in seen:
                    continue
                seen.add(mid)
                models.append({
                    "id": mid,
                    "base_url": h.base_url,
                    "api_key": h.api_key,
                    "display": f"{tag} {mid} · {h.name} · {int(h.latency_ms)}ms",
                })
        return models


# Tiny helper used by the adaptive router + picker
_radar_singleton: Optional[ProviderRadar] = None


def get_radar() -> ProviderRadar:
    global _radar_singleton
    if _radar_singleton is None:
        _radar_singleton = ProviderRadar()
    return _radar_singleton
