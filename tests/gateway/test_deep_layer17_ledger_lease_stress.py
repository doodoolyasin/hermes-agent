import concurrent.futures
import pytest
from gateway.delivery_ledger import (
    record_obligation,
    mark_attempting,
    mark_delivered,
    mark_failed,
    mark_retrying,
    mark_dead_letter,
    list_dead_letters,
    replay_dead_letter,
    _connect,
)


def _row(oid):
    with _connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content, last_error
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0],
        "attempts": r[1],
        "owner_pid": r[2],
        "content": r[3],
        "last_error": r[4],
    }


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.delivery_ledger._db_path", lambda: test_db)
    yield


def test_concurrent_state_transitions_stress():
    oid = "stress-ob-1"
    record_obligation(
        obligation_id=oid,
        session_key="bale:chat_1",
        platform="bale",
        chat_id="1",
        thread_id=None,
        content="Concurrent test message",
    )

    def worker(i: int):
        if i % 2 == 0:
            mark_attempting(oid)
        else:
            mark_retrying(oid, f"Transient error attempt {i}")

    # Hammer state transitions concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(40)))

    # Finally mark delivered
    mark_delivered(oid)
    row = _row(oid)
    assert row is not None
    assert row["state"] == "delivered"


def test_delivery_ledger_last_error_redacts_tokens():
    oid = "secret-ob-2"
    record_obligation(
        obligation_id=oid,
        session_key="bale:chat_2",
        platform="bale",
        chat_id="2",
        thread_id=None,
        content="Secret failure test",
    )

    sensitive_api_key = "sk-ant-api03-secret1234567890abcdef"
    error_with_key = f"BaleAPIError: Bad Gateway when calling with token={sensitive_api_key}"

    mark_failed(oid, error=error_with_key)
    row = _row(oid)
    assert row is not None
    assert sensitive_api_key not in row["last_error"]

    mark_dead_letter(oid, error=f"Fatal: {sensitive_api_key}")
    row = _row(oid)
    assert row["state"] == "dead_letter"
    assert sensitive_api_key not in row["last_error"]
