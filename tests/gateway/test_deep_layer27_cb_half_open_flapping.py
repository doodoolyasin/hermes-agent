import pytest
from gateway.circuit_breaker import ProviderCircuitBreaker, BreakerState


@pytest.fixture(autouse=True)
def _isolated_cb_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    yield


def test_circuit_breaker_half_open_transitions_and_flapping():
    current_time = 1000.0

    def clock():
        return current_time

    cb = ProviderCircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=60.0,
        clock=clock,
    )

    p = "flapping_provider"

    # 1. Closed state initially
    assert cb.can_execute(p) is True
    st, fails, _ = cb.get_status(p)
    assert st == BreakerState.CLOSED
    assert fails == 0

    # 2. Record 2 failures -> still closed
    cb.record_failure(p, "Error 1")
    cb.record_failure(p, "Error 2")
    assert cb.can_execute(p) is True
    st, fails, _ = cb.get_status(p)
    assert st == BreakerState.CLOSED
    assert fails == 2

    # 3. Third failure trips to OPEN
    new_st = cb.record_failure(p, "Error 3")
    assert new_st == BreakerState.OPEN
    assert cb.can_execute(p) is False
    st, fails, _ = cb.get_status(p)
    assert st == BreakerState.OPEN

    # 4. Advance time by 30s (cooldown is 60s) -> still OPEN
    current_time += 30.0
    assert cb.can_execute(p) is False
    st, _, _ = cb.get_status(p)
    assert st == BreakerState.OPEN

    # 5. Advance time by another 35s (total 65s > 60s) -> transitions to HALF_OPEN
    current_time += 35.0
    assert cb.can_execute(p) is True
    st, _, _ = cb.get_status(p)
    assert st == BreakerState.HALF_OPEN

    # 6. Branch A: Failure in HALF_OPEN trips immediately back to OPEN!
    cb.record_failure(p, "Immediate failure in half_open")
    assert cb.can_execute(p) is False
    st, _, _ = cb.get_status(p)
    assert st == BreakerState.OPEN

    # 7. Advance another 61s -> HALF_OPEN again
    current_time += 61.0
    assert cb.can_execute(p) is True
    st, _, _ = cb.get_status(p)
    assert st == BreakerState.HALF_OPEN

    # 8. Branch B: Success in HALF_OPEN closes the circuit and resets failures
    cb.record_success(p)
    assert cb.can_execute(p) is True
    st, fails, _ = cb.get_status(p)
    assert st == BreakerState.CLOSED
    assert fails == 0
