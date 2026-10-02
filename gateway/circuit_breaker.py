"""Durable, generic Provider Circuit Breaker for Hermes.

Implements P1-B1 and specification sections 14 & 15:
- Generic ProviderCircuitBreaker usable across any LLM provider
- States: CLOSED (normal), OPEN (fail fast), HALF_OPEN (trial probe)
- Durable SQLite persistence in state.db (survives process restarts)
- Thread-safe transitions with configurable threshold and cooldown
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_process_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _db_path() -> Path:
    return get_process_hermes_home() / "state.db"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS circuit_breakers (
            provider TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            failures INTEGER NOT NULL DEFAULT 0,
            opened_at REAL,
            last_failure_at REAL,
            last_success_at REAL,
            last_error TEXT
        )"""
    )


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    return open_db(
        _db_path(),
        db_label="state.db (circuit_breakers)",
        busy_timeout_ms=10_000,
        row_factory=None,
        initialize=_initialize_schema,
    )


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


class ProviderCircuitBreaker:
    """Production-grade durable circuit breaker for LLM providers."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock

    def can_execute(self, provider: str) -> bool:
        """Return True if requests to this provider are permitted."""
        state, _, _ = self.get_status(provider)
        return state in (BreakerState.CLOSED, BreakerState.HALF_OPEN)

    def get_status(self, provider: str) -> tuple[BreakerState, int, Optional[str]]:
        """Get current (state, failures, last_error) for provider, resolving cooldown lazily."""
        now = self._clock()
        with _DB_LOCK, _transaction() as conn:
            row = conn.execute(
                """SELECT state, failures, opened_at, last_error
                   FROM circuit_breakers WHERE provider = ?""",
                (provider.lower(),),
            ).fetchone()

            if not row:
                return BreakerState.CLOSED, 0, None

            st_str, failures, opened_at, last_err = row
            current_state = BreakerState(st_str) if st_str in BreakerState._value2member_map_ else BreakerState.CLOSED

            # Transition OPEN -> HALF_OPEN when cooldown expires
            if current_state is BreakerState.OPEN and opened_at:
                if now - opened_at >= self.cooldown_seconds:
                    conn.execute(
                        "UPDATE circuit_breakers SET state = 'half_open' WHERE provider = ?",
                        (provider.lower(),),
                    )
                    return BreakerState.HALF_OPEN, failures, last_err

            return current_state, failures, last_err

    def record_success(self, provider: str) -> None:
        """Successful request: resets breaker to CLOSED and clears failure counter."""
        now = self._clock()
        p = provider.lower()
        with _DB_LOCK, _transaction() as conn:
            conn.execute(
                """INSERT INTO circuit_breakers (provider, state, failures, opened_at, last_success_at, last_error)
                   VALUES (?, 'closed', 0, NULL, ?, NULL)
                   ON CONFLICT(provider) DO UPDATE SET
                       state = 'closed',
                       failures = 0,
                       opened_at = NULL,
                       last_success_at = excluded.last_success_at,
                       last_error = NULL""",
                (p, now),
            )

    def record_failure(self, provider: str, error: str = "") -> BreakerState:
        """Failed request: increments failure counter and trips circuit if threshold exceeded."""
        now = self._clock()
        p = provider.lower()
        err_msg = error[:500] if error else "Unspecified error"
        try:
            from agent.redact import redact_sensitive_text
            err_msg = redact_sensitive_text(err_msg, force=True)
        except Exception:
            pass

        with _DB_LOCK, _transaction() as conn:
            row = conn.execute(
                "SELECT state, failures FROM circuit_breakers WHERE provider = ?",
                (p,),
            ).fetchone()

            if not row:
                current_state = BreakerState.CLOSED
                failures = 1
            else:
                current_state = BreakerState(row[0]) if row[0] in BreakerState._value2member_map_ else BreakerState.CLOSED
                failures = row[1] + 1

            # Trip logic: in HALF_OPEN any failure trips to OPEN immediately.
            # In CLOSED, trips if failures >= threshold.
            if current_state is BreakerState.HALF_OPEN or failures >= self.failure_threshold:
                new_state = BreakerState.OPEN
                opened_at = now
            else:
                new_state = BreakerState.CLOSED
                opened_at = None

            conn.execute(
                """INSERT INTO circuit_breakers (provider, state, failures, opened_at, last_failure_at, last_error)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider) DO UPDATE SET
                       state = excluded.state,
                       failures = excluded.failures,
                       opened_at = COALESCE(excluded.opened_at, circuit_breakers.opened_at),
                       last_failure_at = excluded.last_failure_at,
                       last_error = excluded.last_error""",
                (p, new_state.value, failures, opened_at, now, err_msg),
            )

            return new_state

    def reset(self, provider: str) -> None:
        """Explicitly reset a provider's circuit breaker to CLOSED."""
        p = provider.lower()
        with _DB_LOCK, _transaction() as conn:
            conn.execute(
                """UPDATE circuit_breakers
                   SET state = 'closed', failures = 0, opened_at = NULL, last_error = NULL
                   WHERE provider = ?""",
                (p,),
            )

    def list_all(self) -> List[Dict[str, Any]]:
        """List all circuit breakers and their statuses."""
        with _DB_LOCK, _transaction() as conn:
            rows = conn.execute(
                """SELECT provider, state, failures, opened_at, last_failure_at, last_success_at, last_error
                   FROM circuit_breakers ORDER BY provider ASC"""
            ).fetchall()
            return [
                {
                    "provider": r[0],
                    "state": r[1],
                    "failures": r[2],
                    "opened_at": r[3],
                    "last_failure_at": r[4],
                    "last_success_at": r[5],
                    "last_error": r[6],
                }
                for r in rows
            ]


global_circuit_breaker = ProviderCircuitBreaker()
