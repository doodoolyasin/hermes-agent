import pytest
from gateway.audit_log import AuditLogger


@pytest.fixture(autouse=True)
def _isolated_audit_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.audit_log._db_path", lambda: test_db)
    yield


def test_audit_log_write_and_query():
    logger = AuditLogger()

    logger.log(
        "auth.unauthorized_access",
        actor="user_999",
        platform="bale",
        details={"attempted_command": "/admin"},
        severity="WARN",
    )
    logger.log(
        "model.fallback_triggered",
        platform="bale",
        details={"from": "openai-fast", "to": "gpt-oss-20b"},
        severity="INFO",
    )

    all_logs = logger.query()
    assert len(all_logs) == 2

    # Filter by severity
    warn_logs = logger.query(severity="WARN")
    assert len(warn_logs) == 1
    assert warn_logs[0]["actor"] == "user_999"
    assert warn_logs[0]["details"]["attempted_command"] == "/admin"

    # Filter by event_type
    model_logs = logger.query(event_type="model.fallback_triggered")
    assert len(model_logs) == 1
    assert model_logs[0]["severity"] == "INFO"
