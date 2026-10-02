"""Tests for hermes_offline.connectivity (classification, breakers, backoff)."""

from __future__ import annotations

import random
import socket
import ssl

import pytest

from hermes_offline import connectivity as conn
from hermes_offline.models import ConnectivityState, FailureClass, ProbeResult


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HERMES_OFFLINE_PROBE_ENDPOINTS", "HERMES_OFFLINE_PROVIDER_ENDPOINTS",
                 "HERMES_OFFLINE_PROXY", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


# ── classify_failure ─────────────────────────────────────────────────
@pytest.mark.parametrize("exc,expected", [
    (socket.gaierror("no"), FailureClass.DNS),
    (socket.timeout("t"), FailureClass.TIMEOUT),
    (TimeoutError("t"), FailureClass.TIMEOUT),
    (ssl.SSLError("bad cert"), FailureClass.TLS),
    (ConnectionRefusedError("nope"), FailureClass.CONNECTION),
    (ConnectionResetError("rst"), FailureClass.CONNECTION),
    (ValueError("??"), FailureClass.UNKNOWN),
])
def test_classify_by_exception(exc, expected):
    assert conn.classify_failure(exc) is expected


@pytest.mark.parametrize("status,expected", [
    (429, FailureClass.RATE_LIMITED),
    (404, FailureClass.HTTP_4XX),
    (503, FailureClass.HTTP_5XX),
])
def test_classify_by_status(status, expected):
    assert conn.classify_failure(status_code=status) is expected


def test_classify_none_is_unknown():
    assert conn.classify_failure(None) is FailureClass.UNKNOWN


# ── compute_backoff ──────────────────────────────────────────────────
def test_backoff_within_bounds_and_grows():
    cfg = conn.ConnectivityConfig()
    early = conn.compute_backoff(0, cfg)
    later = conn.compute_backoff(5, cfg)
    assert 0.0 <= early <= cfg.backoff_max_s
    assert later > early
    assert conn.compute_backoff(50, cfg) <= cfg.backoff_max_s


def test_backoff_never_exceeds_max_even_with_jitter():
    cfg = conn.ConnectivityConfig(backoff_base_s=100.0, backoff_max_s=5.0, jitter_ratio=1.0)
    for attempt in range(0, 10):
        assert conn.compute_backoff(attempt, cfg) <= 5.0


def test_backoff_deterministic_with_seeded_rng():
    cfg = conn.ConnectivityConfig()
    a = [conn.compute_backoff(i, cfg, rng=random.Random(42)) for i in range(5)]
    b = [conn.compute_backoff(i, cfg, rng=random.Random(42)) for i in range(5)]
    assert a == b


# ── CircuitBreaker ───────────────────────────────────────────────────
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_full_transition_sequence():
    clock = FakeClock()
    breaker = conn.CircuitBreaker(failure_threshold=3, cooldown_s=10.0, clock=clock)
    assert breaker.state is conn.BreakerState.CLOSED
    assert breaker.allow() is True

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is conn.BreakerState.CLOSED  # below threshold
    breaker.record_failure()
    assert breaker.state is conn.BreakerState.OPEN
    assert breaker.allow() is False

    clock.t = 10.0
    assert breaker.state is conn.BreakerState.HALF_OPEN
    assert breaker.allow() is True

    breaker.record_success()
    assert breaker.state is conn.BreakerState.CLOSED
    assert breaker.failures == 0


def test_breaker_half_open_failure_reopens():
    clock = FakeClock()
    breaker = conn.CircuitBreaker(failure_threshold=1, cooldown_s=5.0, clock=clock)
    breaker.record_failure()
    assert breaker.state is conn.BreakerState.OPEN
    clock.t = 5.0
    assert breaker.state is conn.BreakerState.HALF_OPEN
    breaker.record_failure()
    assert breaker.state is conn.BreakerState.OPEN


# ── ConnectivityMonitor ──────────────────────────────────────────────
def _monitor(results, **cfg):
    def probe(url, config):
        return results[url]

    config = conn.ConnectivityConfig(cache_ttl_s=0.0, **cfg)
    return conn.ConnectivityMonitor(config, probe=probe)


def test_monitor_all_ok_is_full_online():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0)
    mon = conn.ConnectivityMonitor(cfg, probe=lambda u, c: ProbeResult(u, True, 50.0))
    cfg.probe_endpoints = ["https://a", "https://b"]
    assert mon.check(force=True) is ConnectivityState.FULL_ONLINE


def test_monitor_high_latency_is_degraded():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0, degraded_latency_ms=100.0)
    mon = conn.ConnectivityMonitor(cfg, probe=lambda u, c: ProbeResult(u, True, 5000.0))
    cfg.probe_endpoints = ["https://a"]
    assert mon.check(force=True) is ConnectivityState.DEGRADED


def test_monitor_partial_failure_is_degraded():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0)
    results = {"https://a": ProbeResult("https://a", True, 10.0),
               "https://b": ProbeResult("https://b", False, 10.0, FailureClass.TIMEOUT)}
    mon = conn.ConnectivityMonitor(cfg, probe=lambda u, c: results[u])
    cfg.probe_endpoints = ["https://a", "https://b"]
    assert mon.check(force=True) is ConnectivityState.DEGRADED


def test_monitor_all_fail_is_offline():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0)
    mon = conn.ConnectivityMonitor(
        cfg, probe=lambda u, c: ProbeResult(u, False, 5.0, FailureClass.DNS))
    cfg.probe_endpoints = ["https://a", "https://b"]
    assert mon.check(force=True) is ConnectivityState.OFFLINE


def test_monitor_provider_failure_is_provider_unavailable():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0)
    results = {"https://a": ProbeResult("https://a", True, 10.0),
               "https://api.provider": ProbeResult("https://api.provider", False, 10.0,
                                                   FailureClass.HTTP_5XX)}
    mon = conn.ConnectivityMonitor(cfg, probe=lambda u, c: results[u])
    cfg.probe_endpoints = ["https://a"]
    cfg.provider_endpoints = ["https://api.provider"]
    assert mon.check(force=True) is ConnectivityState.PROVIDER_UNAVAILABLE


def test_monitor_offline_with_pack_ready_is_emergency_offline():
    cfg = conn.ConnectivityConfig(cache_ttl_s=0.0)
    mon = conn.ConnectivityMonitor(
        cfg, offline_ready=True,
        probe=lambda u, c: ProbeResult(u, False, 5.0, FailureClass.CONNECTION))
    cfg.probe_endpoints = ["https://a"]
    assert mon.check(force=True) is ConnectivityState.EMERGENCY_OFFLINE_READY


def test_monitor_caches_until_ttl():
    cfg = conn.ConnectivityConfig(cache_ttl_s=1000.0)
    calls = {"n": 0}

    def probe(u, c):
        calls["n"] += 1
        return ProbeResult(u, True, 1.0)

    mon = conn.ConnectivityMonitor(cfg, probe=probe)
    cfg.probe_endpoints = ["https://a"]
    mon.check(force=True)
    first = calls["n"]
    mon.check()  # cached
    assert calls["n"] == first


def test_provider_breaker_registry_is_stable():
    mon = conn.ConnectivityMonitor(conn.ConnectivityConfig())
    assert mon.provider_breaker("openai") is mon.provider_breaker("openai")


def test_config_from_env_parses_endpoints(monkeypatch):
    monkeypatch.setenv("HERMES_OFFLINE_PROBE_ENDPOINTS", "https://a, https://b")
    monkeypatch.setenv("HERMES_OFFLINE_PROVIDER_ENDPOINTS", "https://p")
    monkeypatch.setenv("HERMES_OFFLINE_PROXY", "http://proxy.local:8080")
    cfg = conn.ConnectivityConfig.from_env()
    assert cfg.probe_endpoints == ["https://a", "https://b"]
    assert cfg.provider_endpoints == ["https://p"]
    assert cfg.proxies == {"http": "http://proxy.local:8080",
                           "https": "http://proxy.local:8080"}
