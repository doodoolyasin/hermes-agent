import pytest
from gateway.circuit_breaker import ProviderCircuitBreaker, BreakerState


@pytest.fixture(autouse=True)
def _isolated_cb(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    yield


def test_circuit_breaker_lifecycle():
    fake_time = 1000.0

    def clock():
        return fake_time

    cb = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, clock=clock)

    # Initial state
    assert cb.can_execute("openai") is True
    st, fails, err = cb.get_status("openai")
    assert st == BreakerState.CLOSED
    assert fails == 0

    # Failure 1
    cb.record_failure("openai", "Connection refused")
    assert cb.can_execute("openai") is True
    st, fails, _ = cb.get_status("openai")
    assert fails == 1
    assert st == BreakerState.CLOSED

    # Failure 2
    cb.record_failure("openai", "502 Bad Gateway")
    assert cb.can_execute("openai") is True
    assert cb.get_status("openai")[1] == 2

    # Failure 3 -> trips to OPEN!
    new_state = cb.record_failure("openai", "503 Service Unavailable")
    assert new_state == BreakerState.OPEN
    assert cb.can_execute("openai") is False
    st, fails, err = cb.get_status("openai")
    assert st == BreakerState.OPEN
    assert "503" in err

    # Advance clock mid-cooldown (30s) -> still OPEN
    fake_time += 30.0
    assert cb.can_execute("openai") is False

    # Advance clock past cooldown (70s) -> transitions to HALF_OPEN
    fake_time += 40.0
    assert cb.can_execute("openai") is True
    st, _, _ = cb.get_status("openai")
    assert st == BreakerState.HALF_OPEN

    # Success in HALF_OPEN resets circuit to CLOSED
    cb.record_success("openai")
    assert cb.can_execute("openai") is True
    st, fails, _ = cb.get_status("openai")
    assert st == BreakerState.CLOSED
    assert fails == 0


def test_half_open_failure_re_opens():
    fake_time = 1000.0

    def clock():
        return fake_time

    cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=30.0, clock=clock)
    cb.record_failure("pollinations", "err")
    cb.record_failure("pollinations", "err")
    assert cb.get_status("pollinations")[0] == BreakerState.OPEN

    # Cooldown elapses
    fake_time += 35.0
    assert cb.get_status("pollinations")[0] == BreakerState.HALF_OPEN

    # Probe failure in HALF_OPEN must trip immediately back to OPEN
    cb.record_failure("pollinations", "probe failed")
    assert cb.get_status("pollinations")[0] == BreakerState.OPEN
    assert cb.can_execute("pollinations") is False
