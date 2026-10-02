"""Connectivity classification, circuit breaking and bounded retry (stdlib only).

The core idea: **not every failure means "offline"**.  We distinguish

* ``FULL_ONLINE``          — general endpoints fast and healthy
* ``DEGRADED``             — reachable but slow, or only *some* endpoints work
* ``PROVIDER_UNAVAILABLE`` — the internet is fine but a model/provider endpoint is not
* ``OFFLINE``              — every general endpoint failed
* ``EMERGENCY_OFFLINE_READY`` — OFFLINE *and* the local emergency pack is ready

No proxy, host or credential is hardcoded: endpoints come from config/env.
"""

from __future__ import annotations

import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .models import ConnectivityState, FailureClass, ProbeResult

__all__ = [
    "ConnectivityConfig",
    "BreakerState",
    "CircuitBreaker",
    "EndpointHealth",
    "ConnectivityMonitor",
    "classify_failure",
    "probe_endpoint",
    "compute_backoff",
]


@dataclass
class ConnectivityConfig:
    """Tunables for probing / retrying.  Never contains baked-in endpoints."""

    connect_timeout_s: float = 5.0
    read_timeout_s: float = 8.0
    retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 30.0
    jitter_ratio: float = 0.25
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 60.0
    degraded_latency_ms: float = 1500.0
    cache_ttl_s: float = 15.0
    plain_http: bool = False
    probe_endpoints: List[str] = field(default_factory=list)
    provider_endpoints: List[str] = field(default_factory=list)
    proxies: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConnectivityConfig":
        data = dict(data or {})
        proxies = data.pop("proxies", None)
        probe = data.pop("probe_endpoints", None) or data.pop("endpoints", None)
        provider = data.pop("provider_endpoints", None)
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        if probes := _split(probe):
            cfg.probe_endpoints = probes
        if providers := _split(provider):
            cfg.provider_endpoints = providers
        if isinstance(proxies, dict):
            cfg.proxies = proxies
        return cfg

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "ConnectivityConfig":
        import os

        env = env if env is not None else os.environ
        cfg = cls()
        if (probe := env.get("HERMES_OFFLINE_PROBE_ENDPOINTS")):
            cfg.probe_endpoints = _split(probe)
        if (provider := env.get("HERMES_OFFLINE_PROVIDER_ENDPOINTS")):
            cfg.provider_endpoints = _split(provider)
        if (proxy := env.get("HERMES_OFFLINE_PROXY") or env.get("HTTPS_PROXY")
                or env.get("https_proxy")):
            cfg.proxies = {"http": proxy, "https": proxy}
        return cfg


def _split(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).replace(";", ",").split(",") if p.strip()]


def classify_failure(exc: Optional[BaseException] = None,
                     *, status_code: Optional[int] = None) -> FailureClass:
    """Map an exception (and/or HTTP status) to a :class:`FailureClass`."""
    if status_code is not None:
        if status_code == 429:
            return FailureClass.RATE_LIMITED
        if 400 <= status_code < 500:
            return FailureClass.HTTP_4XX
        if status_code >= 500:
            return FailureClass.HTTP_5XX

    if exc is None:
        return FailureClass.UNKNOWN

    # Unwrap urllib's URLError to its underlying reason.
    if isinstance(exc, urllib.error.URLError) and getattr(exc, "reason", None) is not None:
        reason = exc.reason
        if isinstance(reason, BaseException):
            exc = reason
        elif isinstance(reason, str) and "name or service not known" in reason.lower():
            return FailureClass.DNS

    if isinstance(exc, socket.gaierror):
        return FailureClass.DNS
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return FailureClass.TIMEOUT
    if isinstance(exc, ssl.SSLError) or isinstance(exc, ssl.CertificateError):
        return FailureClass.TLS
    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError)):
        return FailureClass.CONNECTION
    if isinstance(exc, OSError):
        # ssl/timeout are OSError subclasses and were handled above.
        return FailureClass.CONNECTION
    return FailureClass.UNKNOWN


def probe_endpoint(url: str, config: Optional[ConnectivityConfig] = None) -> ProbeResult:
    """Probe one endpoint with a real HTTP(S) request.  Never raises."""
    cfg = config or ConnectivityConfig()
    timeout = cfg.connect_timeout_s
    started = time.monotonic()

    def _elapsed_ms() -> float:
        return round((time.monotonic() - started) * 1000.0, 2)

    handlers = []
    if cfg.proxies:
        handlers.append(urllib.request.ProxyHandler(cfg.proxies))
    opener = urllib.request.build_opener(*handlers)
    try:
        request = urllib.request.Request(url, method="HEAD")
        with opener.open(request, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            latency = _elapsed_ms()
            if status is not None and status >= 400:
                return ProbeResult(url, False, latency,
                                   classify_failure(status_code=status), status, f"HTTP {status}")
            return ProbeResult(url, True, latency, FailureClass.NONE, status, "", time.time())
    except urllib.error.HTTPError as exc:
        return ProbeResult(url, False, _elapsed_ms(),
                           classify_failure(status_code=exc.code), exc.code,
                           f"HTTP {exc.code}", time.time())
    except Exception as exc:  # noqa: BLE001 - probing must never raise
        return ProbeResult(url, False, _elapsed_ms(), classify_failure(exc), None,
                           f"{type(exc).__name__}: {exc}", time.time())


def compute_backoff(attempt: int, config: Optional[ConnectivityConfig] = None, *,
                    rng: Optional[random.Random] = None) -> float:
    """Bounded exponential backoff with additive jitter.

    Always within ``[0, backoff_max_s]``; deterministic for a seeded ``rng``;
    grows in expectation with ``attempt``.
    """
    cfg = config or ConnectivityConfig()
    r = rng if rng is not None else random
    exponent = max(0, int(attempt))
    base = min(cfg.backoff_max_s, cfg.backoff_base_s * (2 ** exponent))
    jitter = r.uniform(0.0, base * max(0.0, cfg.jitter_ratio))
    return float(min(cfg.backoff_max_s, base + jitter))


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Consecutive-failure circuit breaker with a cooldown and half-open probe."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> BreakerState:
        # Lazy transition OPEN → HALF_OPEN once the cooldown elapses.
        if self._state is BreakerState.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self.cooldown_s:
                self._state = BreakerState.HALF_OPEN
        return self._state

    @property
    def failures(self) -> int:
        return self._failures

    def allow(self) -> bool:
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self.state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()

    def reset(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = None


@dataclass
class EndpointHealth:
    """Rolling health of a single endpoint."""

    endpoint: str
    breaker: CircuitBreaker
    successes: int = 0
    failures: int = 0
    last_latency_ms: Optional[float] = None
    ewma_latency_ms: Optional[float] = None

    def record(self, result: ProbeResult) -> None:
        if result.ok:
            self.successes += 1
            self.breaker.record_success()
            if result.latency_ms is not None:
                self.last_latency_ms = result.latency_ms
                if self.ewma_latency_ms is None:
                    self.ewma_latency_ms = result.latency_ms
                else:
                    self.ewma_latency_ms = round(
                        0.7 * self.ewma_latency_ms + 0.3 * result.latency_ms, 2)
        else:
            self.failures += 1
            self.breaker.record_failure()


class ConnectivityMonitor:
    """Probes endpoints and derives the coarse :class:`ConnectivityState`."""

    def __init__(self, config: Optional[ConnectivityConfig] = None, *,
                 offline_ready: bool = False, clock: Callable[[], float] = time.monotonic,
                 probe: Callable[[str, ConnectivityConfig], ProbeResult] = probe_endpoint) -> None:
        self.config = config or ConnectivityConfig()
        self.offline_ready = offline_ready
        self._clock = clock
        self._probe = probe
        self.state: ConnectivityState = ConnectivityState.FULL_ONLINE
        self.last_report: List[ProbeResult] = []
        self._health: Dict[str, EndpointHealth] = {}
        self._provider_breakers: Dict[str, CircuitBreaker] = {}
        self._last_check: float = -1e9

    def _health_for(self, endpoint: str) -> EndpointHealth:
        if endpoint not in self._health:
            self._health[endpoint] = EndpointHealth(
                endpoint,
                CircuitBreaker(self.config.breaker_failure_threshold,
                               self.config.breaker_cooldown_s, self._clock),
            )
        return self._health[endpoint]

    def provider_breaker(self, name: str) -> CircuitBreaker:
        if name not in self._provider_breakers:
            self._provider_breakers[name] = CircuitBreaker(
                self.config.breaker_failure_threshold, self.config.breaker_cooldown_s, self._clock)
        return self._provider_breakers[name]

    def check(self, *, force: bool = False) -> ConnectivityState:
        now = self._clock()
        if not force and (now - self._last_check) < self.config.cache_ttl_s:
            return self.state
        self._last_check = now

        report: List[ProbeResult] = []
        general_results: List[ProbeResult] = []
        for endpoint in self.config.probe_endpoints:
            result = self._probe(endpoint, self.config)
            self._health_for(endpoint).record(result)
            report.append(result)
            general_results.append(result)

        provider_results: List[ProbeResult] = []
        for endpoint in self.config.provider_endpoints:
            result = self._probe(endpoint, self.config)
            self._health_for(endpoint).record(result)
            report.append(result)
            provider_results.append(result)

        self.last_report = report
        self.state = self._derive_state(general_results, provider_results)
        return self.state

    def _derive_state(self, general: List[ProbeResult],
                      provider: List[ProbeResult]) -> ConnectivityState:
        state: ConnectivityState
        if not general:
            # No endpoints configured → we have no evidence of trouble.
            state = ConnectivityState.FULL_ONLINE
        else:
            ok = [r for r in general if r.ok]
            if not ok:
                state = ConnectivityState.OFFLINE
            elif len(ok) < len(general) or any(
                (r.latency_ms or 0.0) > self.config.degraded_latency_ms for r in ok
            ):
                state = ConnectivityState.DEGRADED
            else:
                state = ConnectivityState.FULL_ONLINE

        if state in (ConnectivityState.FULL_ONLINE, ConnectivityState.DEGRADED) and provider:
            if not any(r.ok for r in provider):
                state = ConnectivityState.PROVIDER_UNAVAILABLE

        if state is ConnectivityState.OFFLINE and self.offline_ready:
            state = ConnectivityState.EMERGENCY_OFFLINE_READY
        return state
