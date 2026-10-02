import pytest
from gateway.audit_log import AuditLogger
from gateway.task_queue import PersistentTaskQueue
from gateway.circuit_breaker import ProviderCircuitBreaker


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.audit_log._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    yield


def test_audit_log_redacts_api_keys():
    al = AuditLogger()
    fake_secret = "sk-live-1234567890abcdef1234567890abcdef"
    al.log(
        "auth.failure",
        details={"token_used": fake_secret, "header": f"Bearer {fake_secret}"},
        severity="WARN",
    )
    records = al.query()
    assert len(records) == 1
    # Secret must be redacted
    details_str = str(records[0]["details"])
    assert fake_secret not in details_str
    assert "[REDACTED]" in details_str or "[REDACTED" in details_str or "sk-live-" not in details_str


def test_task_queue_redacts_last_error():
    tq = PersistentTaskQueue()
    t = tq.enqueue("bale:1", "typeA", {"x": 1})
    claimed = tq.claim_next()
    assert claimed is not None

    secret_in_error = "Authorization: Bearer my_secret_jwt_token_999888777"
    tq.fail(claimed.task_id, error=secret_in_error, retryable=False)

    dls = tq.list_dead_letters()
    assert len(dls) == 1
    last_err = dls[0]["last_error"]
    assert "my_secret_jwt_token_999888777" not in last_err


def test_circuit_breaker_redacts_last_error():
    cb = ProviderCircuitBreaker(failure_threshold=1)
    secret_err = "Failed connection with api_key=sk-ant-api03-abcdefghijklmn123456"
    cb.record_failure("anthropic", secret_err)

    status = cb.list_all()
    assert len(status) == 1
    err_recorded = status[0]["last_error"]
    assert "sk-ant-api03-abcdefghijklmn123456" not in err_recorded
