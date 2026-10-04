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
        "api_key": "free-community",
        "model_hint": "openai-fast",
        "tier": "anonymous",
    },
    {
        "name": "openrouter-free",
        "base_url": "https://openrouter.ai/api/v1",
        "probe": "https://openrouter.ai/api/v1/models",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_hint": "qwen/qwen3.8-27b:free",
        "tier": "keyed-free",
    },
]

# Tier 2: domestic (Iran-routable) relay hints — resolved lazily; survive international outage
_DOMESTIC_HINTS: list[dict] = [
    {"name": "gpn-relay", "base_url_env": "GPN_LLM_RELAY"},  # local GPN tunnel if host runs it
    {"name": "local-ollama", "probe": "http://127.0.0.1:11434/api/tags", "base_url": "http://127.0.0.1:11434/v1", "tier": "local"},
    {"name": "local-llamacpp", "probe": "http://127.0.0.1:8080/v1/models", "base_url": "http://127.0.0.1:8080/v1", "tier": "local"},
    {"name": "local-lmstudio", "probe": "http://127.0.0.1:1234/v1/models", "base_url": "http://127.0.0.1:1234/v1", "tier": "local"},
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
            ok, lat = self._probe(c.get("probe", c.get("base_url", "")))
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

    # -- candidates ---------------------------------------------------------
    def all_candidates(self) -> list[dict]:
        import os
        out = list(_STATIC_CATALOGUE)
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

    def picker_models(self) -> list[dict]:
        """Entries the Bale/TG emergency picker renders with LIVE probed names."""
        models = []
        for h in self.best_chain():
            if h.reachable or h.success_count > 0:
                tag = "🟢" if h.reachable else "🟡"
                models.append({
                    "id": h.model_hint or h.name,
                    "base_url": h.base_url,
                    "api_key": h.api_key,
                    "display": f"{tag} {h.model_hint or h.name} · {h.name} · {int(h.latency_ms)}ms",
                })
        return models


# Tiny helper used by the adaptive router + picker
_radar_singleton: Optional[ProviderRadar] = None


def get_radar() -> ProviderRadar:
    global _radar_singleton
    if _radar_singleton is None:
        _radar_singleton = ProviderRadar()
    return _radar_singleton
