"""Durable Audit Log for Hermes Gateway and Security Operations.

Implements P1-B6 and specification sections 28 & 30:
- Structured audit event logging for auth, admin commands, model fallbacks, security alerts
- Persistent SQLite table in state.db
- Severities: INFO, WARN, CRITICAL
- Bounded retention with automated pruning
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_process_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()
DEFAULT_RETENTION_SECONDS = 30 * 86400.0  # 30 days


def _db_path() -> Path:
    return get_process_hermes_home() / "state.db"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT,
            session_key TEXT,
            platform TEXT,
            details TEXT,
            severity TEXT NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(timestamp DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_event ON audit_log(event_type)")


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    return open_db(
        _db_path(),
        db_label="state.db (audit_log)",
        busy_timeout_ms=10_000,
        row_factory=None,
        initialize=_initialize_schema,
    )


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


class AuditLogger:
    """Production audit logger storing security and operational events."""

    def log(
        self,
        event_type: str,
        *,
        actor: Optional[str] = None,
        session_key: Optional[str] = None,
        platform: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> None:
        """Record an audit log entry."""
        now = time.time()
        sev = severity.upper()
        if sev not in ("INFO", "WARN", "CRITICAL"):
            sev = "INFO"
        details_str = json.dumps(details or {}, ensure_ascii=False)

        try:
            with _DB_LOCK, _transaction() as conn:
                conn.execute(
                    """INSERT INTO audit_log (timestamp, event_type, actor, session_key, platform, details, severity)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (now, event_type, actor, session_key, platform, details_str, sev),
                )
        except Exception as exc:
            logger.error("Failed to write audit log: %s", exc)

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query audit log entries with filters."""
        query_sql = "SELECT id, timestamp, event_type, actor, session_key, platform, details, severity FROM audit_log WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query_sql += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query_sql += " AND severity = ?"
            params.append(severity.upper())
        if since is not None:
            query_sql += " AND timestamp >= ?"
            params.append(float(since))

        query_sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with _DB_LOCK, _transaction() as conn:
            rows = conn.execute(query_sql, params).fetchall()
            results = []
            for r in rows:
                details_dict = {}
                if r[6]:
                    try:
                        details_dict = json.loads(r[6])
                    except Exception:
                        pass
                results.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "event_type": r[2],
                    "actor": r[3],
                    "session_key": r[4],
                    "platform": r[5],
                    "details": details_dict,
                    "severity": r[7],
                })
            return results

    def prune(self, retention_seconds: float = DEFAULT_RETENTION_SECONDS) -> int:
        """Prune records older than retention period."""
        cutoff = time.time() - retention_seconds
        with _DB_LOCK, _transaction() as conn:
            cur = conn.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff,))
            return cur.rowcount


global_audit_logger = AuditLogger()
