import pytest
from gateway.adaptive_router import AdaptiveRouter
from gateway.circuit_breaker import global_circuit_breaker
from gateway.provider_registry import global_provider_registry, ProviderDescriptor, ProviderTier


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    yield


def test_adaptive_router_normal_and_fallback():
    router = AdaptiveRouter()

    # Preferred provider route
    decision = router.route_request("bale:101", "Hello AI", preferred_provider="pollinations")
    assert decision.is_queued_offline is False
    assert decision.provider.id == "pollinations"

    # Trip breaker on all providers to simulate complete internet blackout
    for p in global_provider_registry.list_all(enabled_only=False):
        global_circuit_breaker.record_failure(p.id, "Internet outage")
        global_circuit_breaker.record_failure(p.id, "Internet outage")
        global_circuit_breaker.record_failure(p.id, "Internet outage")

    # Now routing must gracefully fall back to persistent task queue
    blackout_decision = router.route_request("bale:101", "Hello in blackout")
    assert blackout_decision.is_queued_offline is True
    assert blackout_decision.queued_task_id is not None
    assert blackout_decision.reason == "offline_queued"

    # User message
    msg = router.get_offline_user_message(blackout_decision.queued_task_id)
    assert "درخواست شما با موفقیت در صف پایدار ذخیره شد" in msg
    assert blackout_decision.queued_task_id in msg
