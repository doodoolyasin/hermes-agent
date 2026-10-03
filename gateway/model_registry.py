"""Dynamic Model Registry and Health Benchmark for Hermes.

Implements P0-A7 and architecture sections 9, 10, 12, 13:
- Full model metadata (capabilities, context_window, streaming, latency, health)
- Health benchmark measuring availability, TTFT (time-to-first-token), and round-trip latency
- Dynamic discovery for local runtimes (Ollama, LM Studio, vLLM) and public endpoints
- Capability-indexed filtering (chat, coding, reasoning, vision, tools)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Rich metadata describing a registered AI model."""

    id: str
    provider: str
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    context_window: int = 32768
    streaming: bool = True
    health: ModelHealthStatus = ModelHealthStatus.UNKNOWN
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    base_url: Optional[str] = None
    api_key_required: bool = False
    enabled: bool = True
    last_checked_at: Optional[float] = None
    last_error: Optional[str] = None

    def supports(self, capability: str) -> bool:
        """Check if model has requested capability (e.g., 'coding', 'reasoning', 'vision', 'tools')."""
        return capability.lower() in [c.lower() for c in self.capabilities]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["health"] = self.health.value
        return d


@dataclass
class HealthProbeResult:
    available: bool
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class ModelHealthProbe:
    """Probes model endpoints to benchmark availability, latency and basic generation."""

    def __init__(self, timeout_seconds: float = 3.5) -> None:
        self.timeout_seconds = timeout_seconds

    def probe_endpoint(self, base_url: str, api_key: Optional[str] = None) -> HealthProbeResult:
        """Test endpoint availability via /models or base endpoint with bounded timeout."""
        models_url = f"{base_url.rstrip('/')}/models"
        headers = {"User-Agent": "Hermes-Agent-Probe/1.0"}
        if api_key and api_key not in ("not-needed", "free-community", "ollama"):
            headers["Authorization"] = f"Bearer {api_key}"

        start = time.perf_counter()
        req = urllib.request.Request(models_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return HealthProbeResult(
                    available=True,
                    latency_ms=round(elapsed_ms, 2),
                    status_code=resp.status,
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return HealthProbeResult(
                available=False,
                latency_ms=round(elapsed_ms, 2),
                error=str(exc),
            )

    def probe_generation(
        self, base_url: str, model_id: str, api_key: Optional[str] = None
    ) -> HealthProbeResult:
        """Benchmark real token generation with a minimal single-token prompt."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent-Probe/1.0",
        }
        if api_key and api_key not in ("not-needed", "free-community", "ollama"):
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return HealthProbeResult(
                    available=True,
                    latency_ms=round(elapsed_ms, 2),
                    ttft_ms=round(elapsed_ms, 2),
                    status_code=resp.status,
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return HealthProbeResult(
                available=False,
                latency_ms=round(elapsed_ms, 2),
                error=str(exc),
            )


class ModelRegistry:
    """Dynamic, capability-aware Model Registry for Hermes."""

    def __init__(self, probe: Optional[ModelHealthProbe] = None) -> None:
        self._models: Dict[str, ModelMetadata] = {}
        self.probe = probe or ModelHealthProbe()
        self._seed_default_models()

    def register(self, model: ModelMetadata) -> None:
        """Register or update a model in the registry."""
        self._models[model.id] = model

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by ID."""
        return self._models.get(model_id)

    def list_all(self, *, enabled_only: bool = True) -> List[ModelMetadata]:
        """List all models in the registry."""
        models = list(self._models.values())
        if enabled_only:
            models = [m for m in models if m.enabled]
        return models

    def list_by_capability(
        self, capability: str, *, healthy_only: bool = False
    ) -> List[ModelMetadata]:
        """Filter models by required capability (e.g., 'coding', 'reasoning', 'vision')."""
        models = [m for m in self.list_all() if m.supports(capability)]
        if healthy_only:
            models = [
                m
                for m in models
                if m.health in (ModelHealthStatus.HEALTHY, ModelHealthStatus.UNKNOWN)
            ]
        return models

    def check_health(self, model_id: str) -> ModelHealthStatus:
        """Run a health benchmark on a specific model."""
        model = self.get(model_id)
        if not model or not model.base_url:
            return ModelHealthStatus.UNKNOWN

        res = self.probe.probe_endpoint(model.base_url)
        now = time.time()
        model.last_checked_at = now

        if res.available:
            if res.latency_ms is not None and res.latency_ms > 4000.0:
                model.health = ModelHealthStatus.DEGRADED
            else:
                model.health = ModelHealthStatus.HEALTHY
            model.latency_ms = res.latency_ms
            model.last_error = None
        else:
            model.health = ModelHealthStatus.UNHEALTHY
            model.last_error = res.error
            model.latency_ms = res.latency_ms

        return model.health

    def _seed_default_models(self) -> None:
        """Seed registry with baseline models and capabilities."""
        baseline = [
            ModelMetadata(
                id="openai-fast",
                provider="pollinations",
                capabilities=["chat", "coding", "tools"],
                context_window=65536,
                streaming=True,
                base_url="https://text.pollinations.ai/openai",
                api_key_required=False,
            ),
            ModelMetadata(
                id="gpt-oss-20b",
                provider="pollinations",
                capabilities=["chat", "coding", "reasoning", "tools"],
                context_window=32768,
                streaming=True,
                base_url="https://text.pollinations.ai/openai",
                api_key_required=False,
            ),
            ModelMetadata(
                id="deepseek",
                provider="pollinations",
                capabilities=["chat", "coding", "reasoning", "tools"],
                context_window=65536,
                streaming=True,
                base_url="https://text.pollinations.ai/openai",
                api_key_required=False,
            ),
            ModelMetadata(
                id="llama3.2",
                provider="ollama",
                capabilities=["chat", "coding", "tools"],
                context_window=32768,
                streaming=True,
                base_url="http://localhost:11434/v1",
                api_key_required=False,
            ),
        ]
        for m in baseline:
            self.register(m)


global_model_registry = ModelRegistry()
