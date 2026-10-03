import pytest
import sqlite3
from gateway.audit_log import AuditLogger


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.audit_log._db_path", lambda: test_db)
    yield


def test_audit_log_sql_injection_defense():
    logger = AuditLogger()

    # 1. Record legitimate events
    logger.record(
        event_type="auth_success",
        actor="admin",
        session_key="bale:chat_10",
        platform="bale",
        details={"ip": "1.2.3.4"},
        severity="INFO",
    )
    logger.record(
        event_type="auth_failure",
        actor="attacker",
        session_key="bale:chat_11",
        platform="bale",
        details={"ip": "5.6.7.8"},
        severity="WARN",
    )

    # 2. Malicious SQL injection payloads in query filters
    injection_payloads = [
        "' OR 1=1 --",
        "'; DROP TABLE audit_log; --",
        "' UNION SELECT 1, 2, 3, 4, 5, 6, 7, 8 --",
        "admin'--",
    ]

    for payload in injection_payloads:
        # None of these should throw SQL syntax errors or wipe data
        results = logger.query(event_type=payload)
        assert isinstance(results, list)
        assert len(results) == 0

    # Ensure audit_log table is still intact and data exists
    all_events = logger.query(limit=10)
    assert len(all_events) == 2


def test_audit_log_malicious_details_payload():
    logger = AuditLogger()

    # Highly nested or weird JSON / Unicode null bytes
    tricky_details = {
        "token": "sk-secret-key-that-must-be-redacted",
        "weird_chars": "Robert'); DROP TABLE Students;--\x00\r\n",
        "deep": {"nested": {"danger": "<script>alert(1)</script>"}},
    }

    logger.record(
        event_type="probe",
        actor="tester",
        details=tricky_details,
    )

    records = logger.query(event_type="probe")
    assert len(records) == 1
    stored_details = records[0]["details"]
    # Secret must be redacted
    assert "sk-secret-key-that-must-be-redacted" not in str(stored_details)
    assert stored_details["deep"]["nested"]["danger"] == "<script>alert(1)</script>"
