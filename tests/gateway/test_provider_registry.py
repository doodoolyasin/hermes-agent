import pytest
from gateway.provider_registry import (
    ProviderRegistry,
    ProviderTier,
    ProviderDescriptor,
)
from gateway.circuit_breaker import ProviderCircuitBreaker


def test_provider_registration_and_tier_sorting():
    reg = ProviderRegistry()
    reg.register(
        ProviderDescriptor(
            id="remote_node",
            name="Remote Hermes Relay",
            tier=ProviderTier.REMOTE_HERMES,
            base_url="https://relay.hermes.local/v1",
        )
    )
    providers = reg.list_all()
    # Tier 1 (LOCAL) must precede Tier 3 (REMOTE_HERMES) and Tier 4 (PUBLIC_COMMUNITY)
    tiers = [p.tier for p in providers]
    assert tiers[0] == ProviderTier.LOCAL
    assert ProviderTier.REMOTE_HERMES in tiers


def test_select_best_provider_respects_circuit_breaker():
    breaker = ProviderCircuitBreaker(failure_threshold=1)
    reg = ProviderRegistry(breaker=breaker)

    # Initially select best provider (Ollama or highest priority available)
    best = reg.select_best_provider()
    assert best is not None

    # Trip breaker on that provider
    breaker.record_failure(best.id, "Connection refused")

    # Next call selects an alternative provider whose circuit is not tripped!
    next_best = reg.select_best_provider()
    assert next_best is not None
    assert next_best.id != best.id


def test_health_endpoints():
    reg = ProviderRegistry()

    # /live
    live = reg.live_check()
    assert live["status"] == "alive"
    assert "uptime_seconds" in live
    assert "pid" in live

    # /ready
    ready = reg.ready_check()
    assert ready["status"] in ("ready", "not_ready")

    # /health
    health = reg.health_report()
    assert health["status"] in ("healthy", "degraded")
    assert "connectivity" in health
    assert "providers" in health
