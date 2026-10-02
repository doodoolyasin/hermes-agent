import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.circuit_breaker import ProviderCircuitBreaker, BreakerState
from gateway.retry_engine import RetryEngine, RetryPolicy
from gateway.task_queue import PersistentTaskQueue, TaskState
from gateway.delivery_ledger import record_obligation, mark_attempting, sweep_recoverable
from plugins.platforms.bale.adapter import BaleAdapter, BaleClient
from gateway.config import PlatformConfig


@pytest.fixture(autouse=True)
def _isolated_chaos_db(tmp_path, monkeypatch):
    test_db = tmp_path / "chaos_state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.delivery_ledger._db_path", lambda: test_db)
    yield


def test_provider_429_and_503_chaos():
    cb = ProviderCircuitBreaker(failure_threshold=2)

    class FakeHTTPStatusError(Exception):
        def __init__(self, code, msg):
            self.status_code = code
            super().__init__(msg)

    def flaky_429():
        raise FakeHTTPStatusError(429, "Rate limited")

    def broken_503():
        raise FakeHTTPStatusError(503, "Service Unavailable")

    policy = RetryPolicy(max_attempts=2, base_delay=0.01, jitter="none", circuit_breaker=cb, provider_name="test_ai")

    # 429 rate limit is retryable
    with pytest.raises(FakeHTTPStatusError):
        RetryEngine.execute_sync(flaky_429, policy)

    # 503 causes failure and trips breaker to OPEN
    with pytest.raises((FakeHTTPStatusError, RuntimeError)):
        RetryEngine.execute_sync(broken_503, policy)

    # Breaker must now be OPEN
    assert cb.can_execute("test_ai") is False


def test_provider_401_auth_failure_is_non_retryable():
    cb = ProviderCircuitBreaker()
    call_count = 0

    class FakeHTTPError(Exception):
        def __init__(self, code, msg):
            self.status_code = code
            super().__init__(msg)

    def auth_failed():
        nonlocal call_count
        call_count += 1
        raise FakeHTTPError(401, "Unauthorized / Invalid Token")

    policy = RetryPolicy(max_attempts=5, base_delay=0.01, circuit_breaker=cb, provider_name="p1")

    with pytest.raises(FakeHTTPError):
        RetryEngine.execute_sync(auth_failed, policy)

    # Must NOT retry 401 5 times! Must fail immediately on attempt 1
    assert call_count == 1


@pytest.mark.asyncio
async def test_bale_malformed_updates_chaos():
    cfg = PlatformConfig(extra={"token": "fake_bale_token"}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter.handle_message = AsyncMock()

    # Series of malformed updates from chaotic or malicious networks
    malformed_updates = [
        {},  # completely empty
        {"update_id": "not_an_int"},  # malformed update_id
        {"message": None},  # None message
        {"message": "string instead of dict"},  # wrong type
        {"message": {"text": None, "chat": None}},  # missing chat
        {"message": {"chat": {"id": 123}, "text": None}},  # None text
    ]

    for u in malformed_updates:
        # None of these must raise an unhandled exception or crash
        await adapter._handle_update(u)

    # None of the malformed payloads should have dispatched a valid message
    assert adapter.handle_message.call_count == 0


def test_task_queue_simulated_process_crash_recovery():
    tq = PersistentTaskQueue()
    task = tq.enqueue("bale:123", "ai_job", {"prompt": "Write code"})
    claimed = tq.claim_next()
    assert claimed is not None

    # Simulate saving checkpoint before sudden SIGKILL
    tq.checkpoint(claimed.task_id, {"checkpoint_step": 3, "partial_tokens": 150})

    # Simulate crash: owner process died (PID 999999999)
    from gateway.task_queue import _transaction, _DB_LOCK
    with _DB_LOCK, _transaction() as conn:
        conn.execute("UPDATE task_queue SET owner_pid = 999999999 WHERE task_id = ?", (claimed.task_id,))

    # On restart, recover_crashed_tasks runs:
    recovered = tq.recover_crashed_tasks()
    assert recovered == 1

    # Next claim retrieves the recovered task with checkpoint intact!
    re_claimed = tq.claim_next()
    assert re_claimed is not None
    assert re_claimed.task_id == claimed.task_id
    assert re_claimed.checkpoint["checkpoint_step"] == 3


def test_delivery_ledger_simulated_crash_recovery():
    record_obligation(
        obligation_id="ob-crash-1",
        session_key="bale:chat:1",
        platform="bale",
        chat_id="1",
        thread_id=None,
        content="Important answer that got interrupted",
    )
    mark_attempting("ob-crash-1")

    # Simulate process death by zeroing owner PID
    from gateway.delivery_ledger import _transaction, _DB_LOCK
    with _DB_LOCK, _transaction() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid = NULL WHERE obligation_id = 'ob-crash-1'")

    # Gateway restart sweeps recoverable obligations
    recoverable = sweep_recoverable(deliverable_platforms={"bale"})
    assert len(recoverable) == 1
    assert recoverable[0]["obligation_id"] == "ob-crash-1"
    assert recoverable[0]["content"] == "Important answer that got interrupted"
