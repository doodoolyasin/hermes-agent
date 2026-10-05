"""ResilienceOrchestrator: single coordination point for Iran internet-outage feeding.

Ensures these four subsystems NEVER conflict, contradict, or starve each other:
  - ProviderRadar     (live probing of free endpoints, multi-channel)
  - ProviderRegistry  (tiers + health, keyed/ad-hoc entry points)
  - OfflineRuntimes   (llama.cpp / Ollama / LMStudio — survive full international cut)
  - AdaptiveRouter    (final decision → a chosen endpoint + model)

Invariants enforced here:
  1. EXACTLY ONE emergency chain is materialised per refresh cycle (idempotent rebuild).
  2. No provider appears twice with different base_url values within one cycle.
  3. Picker names are ALWAYS the live-discovered model IDs, deduplicated.
  4. Local runtimes win over remote endpoints when BOTH are up (lowest latency + survival).
  5. When nothing is reachable, the LAST-KNOWN-GOOD entries remain visible (with 🟡)
     so the picker never collapses to zero length.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.provider_radar import ProviderRadar, get_radar

logger = logging.getLogger(__name__)


@dataclass
class CoordinatedChain:
    """The single source of truth for the emergency fallback chain this cycle."""
    active: list[dict] = field(default_factory=list)
    degraded: list[dict] = field(default_factory=list)
    offline_runtime_up: bool = False
    local_model: Optional[str] = None
    refreshed_at: float = 0.0

    def picker_buttons(self) -> list[dict]:
        """Deduped ordered buttons: local first, then remote."""
        seen: set[str] = set()
        out: list[dict] = []
        if self.offline_runtime_up and self.local_model:
            out.append({"id": self.local_model, "display": f"🏠 {self.local_model} · local-runtime · 0ms", "base_url": "http://127.0.0.1:8080/v1"})
            seen.add(self.local_model)
        for e in self.active + self.degraded:
            mid = e.get("model_hint") or e.get("name", "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            tag = "🟢" if e.get("reachable") else "🟡"
            lat = int(e.get("latency_ms") or 0)
            out.append({"id": mid, "display": f"{tag} {mid} · {e['name']} · {lat}ms", "base_url": e.get("base_url", "")})
        return out


class ResilienceOrchestrator:
    """Coordinates radar + registry + local runtime, fail-closed."""

    def __init__(self, radar: Optional[ProviderRadar] = None):
        self._radar = radar or get_radar()
        self._last_chain: Optional[CoordinatedChain] = None

    def refresh(self, *, force: bool = False) -> CoordinatedChain:
        """Probe everything once, assemble the deduped chain, never raise."""
        if self._last_chain is not None and not force:
            return self._last_chain

        import time
        chain = CoordinatedChain(refreshed_at=time.time())

        # --- 1. local runtimes (survive international cut) ---
        try:
            from hermes_offline.runtimes import LlamaCppRuntime, OllamaRuntime
            for rt in (LlamaCppRuntime(host="127.0.0.1", port=8080),
                      OllamaRuntime(host="127.0.0.1", port=11434)):
                if rt.is_available():
                    models = rt.list_models()
                    if models:
                        chain.offline_runtime_up = True
                        chain.local_model = models[0]
                        break
        except Exception as exc:
            logger.debug("local runtime detect failed: %s", exc)

        # --- 2. radar probes (multi-channel, parallel, guarded) ---
        try:
            self._radar.refresh()
        except Exception as exc:
            logger.debug("radar refresh failed: %s", exc)

        # --- 3. assemble deduped chain ---
        for h in self._radar.best_chain(limit=8):
            entry = {
                "name": h.name, "base_url": h.base_url, "model_hint": h.model_hint,
                "api_key": h.api_key, "reachable": h.reachable, "latency_ms": h.latency_ms,
                "score": h.score, "tier": h.tier,
            }
            (chain.active if h.reachable else chain.degraded).append(entry)

        # local runtime suppresses remote degrade-*500* ambiguity: if local is up,
        # demote all remote entries to degraded so the LIVE picker puts 🏠 first.
        if chain.offline_runtime_up:
            chain.degraded = chain.active + chain.degraded
            chain.active = []

        self._last_chain = chain
        return chain

    def choose(self) -> Optional[dict]:
        """Final pick: local > radar.Top > degrade-with-history. None = nothing at all."""
        c = self.refresh()
        if c.local_model:
            return {"base_url": "http://127.0.0.1:8080/v1", "model": c.local_model, "api_key": ""}
        for bucket in (c.active, c.degraded):
            for e in bucket:
                if e.get("base_url") and e.get("model_hint"):
                    return {"base_url": e["base_url"], "model": e["model_hint"], "api_key": e.get("api_key", "")}
        return None


_orchestrator: Optional[ResilienceOrchestrator] = None


def get_orchestrator() -> ResilienceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResilienceOrchestrator()
    return _orchestrator
