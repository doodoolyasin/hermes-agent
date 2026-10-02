import asyncio
import pytest
from gateway.retry_engine import RetryEngine, RetryPolicy, with_retry
from gateway.circuit_breaker import ProviderCircuitBreaker


def test_sync_retry_succeeds_after_transient_failure():
    calls = 0

    def flaky_func():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("Temporary glitch")
        return "success"

    policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter="none")
    result = RetryEngine.execute_sync(flaky_func, policy)
    assert result == "success"
    assert calls == 3


@pytest.mark.asyncio
async def test_async_retry_succeeds_after_transient_failure():
    calls = 0

    async def async_flaky():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise TimeoutError("Async timeout")
        return 42

    policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter="none")
    res = await RetryEngine.execute_async(async_flaky, policy)
    assert res == 42
    assert calls == 2


def test_non_retryable_exception_fails_immediately():
    calls = 0

    def fatal_func():
        nonlocal calls
        calls += 1
        raise ValueError("Fatal invalid arg")

    policy = RetryPolicy(
        max_attempts=5,
        base_delay=0.01,
        non_retryable_exceptions=(ValueError,)
    )

    with pytest.raises(ValueError):
        RetryEngine.execute_sync(fatal_func, policy)

    # Must fail immediately on attempt 1
    assert calls == 1


def test_circuit_breaker_halts_retries_when_open(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)

    cb = ProviderCircuitBreaker(failure_threshold=2)
    cb.record_failure("test_prov", "error 1")
    cb.record_failure("test_prov", "error 2")  # Circuit is now OPEN

    policy = RetryPolicy(
        max_attempts=3,
        circuit_breaker=cb,
        provider_name="test_prov"
    )

    def dummy():
        return "ok"

    with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
        RetryEngine.execute_sync(dummy, policy)
