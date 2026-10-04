"""Dynamic Multi-Tier Provider Registry and Health System for Hermes.

Implements P1-B3 and architecture sections 10 & 11:
- Multi-tier classification:
    Tier 1: Local (Ollama, LM Studio, vLLM)
    Tier 2: LAN (Peer Hermes nodes, local GPU nodes)
    Tier 3: Remote Hermes Relay
    Tier 4: Public/Community (Pollinations AI, free tiers)
    Tier 5: Paid (OpenAI, Anthropic, OpenRouter)
    Tier 6: Emergency Local (minimal offline engine)
- Standard health endpoints:
    /health -> Full report (connectivity, providers, queues, ledger)
    /live   -> Process liveness probe
    /ready  -> Traffic readiness probe
- Dynamic health-aware selection avoiding failed/tripped circuits
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

from gateway.circuit_breaker import BreakerState, ProviderCircuitBreaker, global_circuit_breaker
from gateway.connectivity_matrix import ConnectivityStatus, global_connectivity_matrix

logger = logging.getLogger(__name__)


class ProviderTier(IntEnum):
    LOCAL = 1
    LAN = 2
    REMOTE_HERMES = 3
    PUBLIC_COMMUNITY = 4
    PAID = 5
    EMERGENCY_LOCAL = 6


@dataclass
class ProviderDescriptor:
    id: str
    name: str
    tier: ProviderTier
    base_url: str
    models: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    enabled: bool = True
    is_available: bool = False
    latency_ms: Optional[float] = None
    last_error: Optional[str] = None
    last_checked_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.name,
            "tier_level": self.tier.value,
            "base_url": self.base_url,
            "models": self.models,
            "enabled": self.enabled,
            "is_available": self.is_available,
            "latency_ms": self.latency_ms,
            "last_error": self.last_error,
            "last_checked_at": self.last_checked_at,
        }


class ProviderRegistry:
    """Manages AI providers across tiers with dynamic health probing and selection."""

    def __init__(self, breaker: Optional[ProviderCircuitBreaker] = None) -> None:
        self.breaker = breaker or global_circuit_breaker
        self._providers: Dict[str, ProviderDescriptor] = {}
        self._boot_time = time.time()
        self._seed_default_providers()

    def register(self, p: ProviderDescriptor) -> None:
        self._providers[p.id] = p

    def get(self, provider_id: str) -> Optional[ProviderDescriptor]:
        return self._providers.get(provider_id)

    def list_all(self, *, enabled_only: bool = True) -> List[ProviderDescriptor]:
        res = list(self._providers.values())
        if enabled_only:
            res = [p for p in res if p.enabled]
        # Sort by tier (lower is better priority) then latency
        res.sort(key=lambda p: (p.tier.value, p.latency_ms or 999999.0))
        return res

    def check_provider(self, provider_id: str, timeout_s: float = 3.0) -> bool:
        """Run health check against a provider endpoint."""
        p = self.get(provider_id)
        if not p or not p.enabled:
            return False

        # If circuit breaker is tripped OPEN, fast fail without network call
        if not self.breaker.can_execute(provider_id):
            p.is_available = False
            p.last_error = "Circuit breaker OPEN"
            return False

        url = f"{p.base_url.rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-ProviderRegistry/1.0"})
        if p.api_key and p.api_key not in ("not-needed", "free-community", "ollama"):
            req.add_header("Authorization", f"Bearer {p.api_key}")

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                p.is_available = (resp.status < 400)
                p.latency_ms = round(elapsed_ms, 1)
                p.last_error = None
                p.last_checked_at = time.time()
                self.breaker.record_success(provider_id)
                return p.is_available
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            p.is_available = False
            p.latency_ms = round(elapsed_ms, 1)
            p.last_error = str(exc)
            p.last_checked_at = time.time()
            self.breaker.record_failure(provider_id, str(exc))
            return False

    def select_best_provider(self, required_model: Optional[str] = None) -> Optional[ProviderDescriptor]:
        """Select the highest-priority operational provider that is not circuit-broken."""
        candidates = self.list_all(enabled_only=True)
        for p in candidates:
            if not self.breaker.can_execute(p.id):
                continue
            if required_model and p.models and required_model not in p.models:
                continue
            # Return first operational candidate in tier priority order
            return p
        return None

    # -- Standard Health Endpoints (/health, /live, /ready) -------------------

    def live_check(self) -> Dict[str, Any]:
        """GET /live: fast liveness probe for orchestrators and systemd watchdogs."""
        return {
            "status": "alive",
            "uptime_seconds": round(time.time() - self._boot_time, 1),
            "pid": os.getpid(),
        }

    def ready_check(self) -> Dict[str, Any]:
        """GET /ready: readiness probe indicating if system can serve requests."""
        # Ready if at least one provider can execute and connectivity is not total OFFLINE
        best = self.select_best_provider()
        matrix_status = getattr(global_connectivity_matrix.last_report, "overall_status", ConnectivityStatus.ONLINE)

        is_ready = bool(best) and matrix_status != ConnectivityStatus.OFFLINE
        return {
            "status": "ready" if is_ready else "not_ready",
            "active_provider": best.id if best else None,
            "active_tier": best.tier.name if best else None,
            "connectivity": matrix_status.value if hasattr(matrix_status, "value") else str(matrix_status),
        }

    def health_report(self) -> Dict[str, Any]:
        """GET /health: comprehensive system status report."""
        matrix = global_connectivity_matrix.evaluate(check_international=False)
        providers_status = [p.to_dict() for p in self.list_all(enabled_only=False)]
        breakers = global_circuit_breaker.list_all()

        return {
            "status": "healthy" if matrix.overall_status != ConnectivityStatus.OFFLINE else "degraded",
            "uptime_seconds": round(time.time() - self._boot_time, 1),
            "connectivity": matrix.to_dict(),
            "providers": providers_status,
            "circuit_breakers": breakers,
        }

    def _seed_default_providers(self) -> None:
        """Seed registry with baseline multi-tier providers."""
        # Tier 1: Local Ollama
        self.register(
            ProviderDescriptor(
                id="ollama",
                name="Local Ollama",
                tier=ProviderTier.LOCAL,
                base_url="http://127.0.0.1:11434/v1",
                models=["llama3.2", "mistral", "qwen2.5"],
            )
        )
        # Tier 1: Local LM Studio / vLLM
        self.register(
            ProviderDescriptor(
                id="local_vllm",
                name="Local vLLM / LM Studio",
                tier=ProviderTier.LOCAL,
                base_url="http://127.0.0.1:1234/v1",
                models=["default"],
            )
        )
        # Tier 4: Public Community (Pollinations)
        self.register(
            ProviderDescriptor(
                id="pollinations",
                name="Pollinations Community AI",
                tier=ProviderTier.PUBLIC_COMMUNITY,
                base_url="https://text.pollinations.ai/openai",
                models=["openai-fast", "gpt-oss-20b", "deepseek", "openai"],
                api_key="free-community",
            )
        )
        # Tier 6: Emergency Local
        self.register(
            ProviderDescriptor(
                id="emergency_local",
                name="Emergency Local Fallback",
                tier=ProviderTier.EMERGENCY_LOCAL,
                base_url="http://127.0.0.1:8080/v1",
                models=["emergency-fast"],
            )
        )

        # Inject ProviderRadar live-probed endpoints as dynamic LAN/PUBLIC entries
        try:
            from gateway.provider_radar import get_radar
            radar = get_radar()
            for e in radar.best_chain(limit=6):
                if e.base_url and e.name not in self._providers:
                    self.register(
                        ProviderDescriptor(
                            id=f"radar:{e.name}",
                            name=f"Radar:{e.name}",
                            tier=ProviderTier.LOCAL if e.tier == "local" else ProviderTier.PUBLIC_COMMUNITY,
                            base_url=e.base_url,
                            models=[e.model_hint or "default"],
                            api_key=e.api_key or None,
                            is_available=e.reachable,
                            latency_ms=e.latency_ms,
                        )
                    )
        except Exception as exc:
            logger.debug("ProviderRadar injection skipped: %s", exc)


global_provider_registry = ProviderRegistry()
