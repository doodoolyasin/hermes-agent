import concurrent.futures
import pytest
from gateway.adaptive_router import AdaptiveRouter
from gateway.circuit_breaker import global_circuit_breaker
from gateway.provider_registry import global_provider_registry, ProviderDescriptor, ProviderTier
from gateway.task_queue import PersistentTaskQueue


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    yield


def test_concurrent_fallback_ladder_and_blackout_stress():
    router = AdaptiveRouter()

    # Register two mock providers with specific tiers
    p_primary = ProviderDescriptor(
        id="mock_primary",
        name="Mock Primary",
        tier=ProviderTier.LOCAL,
        base_url="https://api.primary.mock",
        models=["mock-model-1"],
    )
    p_backup = ProviderDescriptor(
        id="mock_backup",
        name="Mock Backup",
        tier=ProviderTier.LAN,
        base_url="https://api.backup.mock",
        models=["mock-model-1"],
    )

    global_provider_registry.register(p_primary)
    global_provider_registry.register(p_backup)

    # 1. Phase 1: Explicit preferred provider routing under concurrency
    def normal_worker(idx: int):
        dec = router.route_request(
            f"bale:user_{idx}", f"Prompt {idx}", preferred_provider="mock_primary"
        )
        assert dec.is_queued_offline is False
        assert dec.provider is not None
        assert dec.provider.id == "mock_primary"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(normal_worker, range(20)))

    # 2. Phase 2: Trip primary circuit breaker
    global_circuit_breaker.record_failure("mock_primary", "Connection timeout")
    global_circuit_breaker.record_failure("mock_primary", "503 Service Unavailable")
    global_circuit_breaker.record_failure("mock_primary", "503 Service Unavailable")

    # Now verify all concurrent requests switch to mock_backup or registry alternatives
    def fallback_worker(idx: int):
        dec = router.route_request(
            f"bale:user_{idx}", f"Prompt {idx}", preferred_provider="mock_primary"
        )
        assert dec.is_queued_offline is False
        # Must NOT be primary because breaker is open
        assert dec.provider.id != "mock_primary"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fallback_worker, range(20)))

    # 3. Phase 3: Total blackout — trip all remaining providers
    for prov in global_provider_registry.list_all(enabled_only=False):
        for _ in range(3):
            global_circuit_breaker.record_failure(prov.id, "Total power outage")

    # Now all concurrent requests must cleanly queue offline without exception
    def blackout_worker(idx: int):
        dec = router.route_request(f"bale:user_{idx}", f"Emergency prompt {idx}")
        assert dec.is_queued_offline is True
        assert dec.queued_task_id is not None
        assert dec.reason == "offline_queued"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(blackout_worker, range(20)))
