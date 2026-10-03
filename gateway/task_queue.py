"""Persistent SQLite Task Queue for Hermes.

Provides crash-resilient asynchronous task execution with:
- States: queued, running, completed, failed, cancelled, retrying, dead_letter
- Idempotency via unique idempotency_key
- Task checkpointing for long-running workflows
- Timeout and bounded retry policies
- Recovery of in-flight tasks after process restart
- Dead-letter queue inspection and manual/automated replay
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_process_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueuedTask:
    task_id: str
    idempotency_key: str
    session_key: str
    task_type: str
    payload: Dict[str, Any]
    state: TaskState
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 300.0
    checkpoint: Optional[Dict[str, Any]] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_error: Optional[str] = None
    owner_pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


def _db_path() -> Path:
    return get_process_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    return open_db(
        _db_path(),
        db_label="state.db (task_queue)",
        busy_timeout_ms=10_000,
        row_factory=None,
        initialize=_initialize_schema,
    )


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS task_queue (
            task_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,
            session_key TEXT NOT NULL,
            task_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            state TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            timeout_seconds REAL NOT NULL DEFAULT 300.0,
            checkpoint TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            started_at REAL,
            completed_at REAL,
            last_error TEXT,
            owner_pid INTEGER
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_queue_state ON task_queue(state, priority DESC, created_at ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_queue_idempotency ON task_queue(idempotency_key)"
    )


class PersistentTaskQueue:
    """Production-grade SQLite-backed persistent task queue."""

    def __init__(self) -> None:
        pass

    def enqueue(
        self,
        session_key: str,
        task_type: str,
        payload: Dict[str, Any],
        *,
        idempotency_key: Optional[str] = None,
        priority: int = 0,
        max_attempts: int = 3,
        timeout_seconds: float = 300.0,
    ) -> QueuedTask:
        """Enqueue a task.

        If idempotency_key is specified and exists, returns existing task without duplication.
        """
        now = time.time()
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        idem_key = idempotency_key or f"idem_{uuid.uuid4().hex}"
        payload_json = json.dumps(payload, ensure_ascii=False)

        with _DB_LOCK, _transaction() as conn:
            # Check idempotency first
            cursor = conn.execute(
                """SELECT task_id, idempotency_key, session_key, task_type, payload,
                          state, priority, attempts, max_attempts, timeout_seconds,
                          checkpoint, created_at, updated_at, started_at, completed_at,
                          last_error, owner_pid
                   FROM task_queue WHERE idempotency_key = ?""",
                (idem_key,),
            )
            existing = cursor.fetchone()
            if existing:
                return self._row_to_task(existing)

            try:
                conn.execute(
                    """INSERT INTO task_queue (
                        task_id, idempotency_key, session_key, task_type, payload,
                        state, priority, attempts, max_attempts, timeout_seconds,
                        checkpoint, created_at, updated_at, owner_pid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, ?, NULL)""",
                    (
                        task_id,
                        idem_key,
                        session_key,
                        task_type,
                        payload_json,
                        TaskState.QUEUED.value,
                        priority,
                        max_attempts,
                        timeout_seconds,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                # Concurrent insert beat us to it with the same idempotency key
                cursor = conn.execute(
                    """SELECT task_id, idempotency_key, session_key, task_type, payload,
                              state, priority, attempts, max_attempts, timeout_seconds,
                              checkpoint, created_at, updated_at, started_at, completed_at,
                              last_error, owner_pid
                       FROM task_queue WHERE idempotency_key = ?""",
                    (idem_key,),
                )
                existing = cursor.fetchone()
                if existing:
                    return self._row_to_task(existing)
                raise

        return QueuedTask(
            task_id=task_id,
            idempotency_key=idem_key,
            session_key=session_key,
            task_type=task_type,
            payload=payload,
            state=TaskState.QUEUED,
            priority=priority,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
            created_at=now,
            updated_at=now,
        )

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """Fetch a task by ID."""
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """SELECT task_id, idempotency_key, session_key, task_type, payload,
                          state, priority, attempts, max_attempts, timeout_seconds,
                          checkpoint, created_at, updated_at, started_at, completed_at,
                          last_error, owner_pid
                   FROM task_queue WHERE task_id = ?""",
                (task_id,),
            )
            row = cursor.fetchone()
            return self._row_to_task(row) if row else None

    def claim_next(self, worker_pid: Optional[int] = None) -> Optional[QueuedTask]:
        """Atomically claim the highest-priority queued or retrying task."""
        now = time.time()
        my_pid = worker_pid or os.getpid()

        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """SELECT task_id, idempotency_key, session_key, task_type, payload,
                          state, priority, attempts, max_attempts, timeout_seconds,
                          checkpoint, created_at, updated_at, started_at, completed_at,
                          last_error, owner_pid
                   FROM task_queue
                   WHERE state IN ('queued', 'retrying')
                   ORDER BY priority DESC, created_at ASC
                   LIMIT 1"""
            )
            row = cursor.fetchone()
            if not row:
                return None

            task_id = row[0]
            conn.execute(
                """UPDATE task_queue
                   SET state = 'running', attempts = attempts + 1,
                       started_at = ?, updated_at = ?, owner_pid = ?
                   WHERE task_id = ? AND state IN ('queued', 'retrying')""",
                (now, now, my_pid, task_id),
            )

            # Re-read claimed task
            cursor = conn.execute(
                """SELECT task_id, idempotency_key, session_key, task_type, payload,
                          state, priority, attempts, max_attempts, timeout_seconds,
                          checkpoint, created_at, updated_at, started_at, completed_at,
                          last_error, owner_pid
                   FROM task_queue WHERE task_id = ?""",
                (task_id,),
            )
            claimed = cursor.fetchone()
            return self._row_to_task(claimed) if claimed else None

    def checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """Save progress checkpoint for a running task."""
        now = time.time()
        cp_json = json.dumps(checkpoint_data, ensure_ascii=False)
        with _DB_LOCK, _transaction() as conn:
            conn.execute(
                """UPDATE task_queue
                   SET checkpoint = ?, updated_at = ?
                   WHERE task_id = ?""",
                (cp_json, now, task_id),
            )

    def complete(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark task as successfully completed."""
        now = time.time()
        with _DB_LOCK, _transaction() as conn:
            conn.execute(
                """UPDATE task_queue
                   SET state = 'completed', completed_at = ?, updated_at = ?
                   WHERE task_id = ?""",
                (now, now, task_id),
            )

    def fail(self, task_id: str, error: str = "", *, retryable: bool = True) -> None:
        """Fail task: transitions to retrying or dead_letter depending on attempts/retryability."""
        now = time.time()
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                "SELECT attempts, max_attempts FROM task_queue WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if not row:
                return
            attempts, max_attempts = row[0], row[1]
            if retryable and attempts < max_attempts:
                next_state = TaskState.RETRYING.value
            else:
                next_state = TaskState.DEAD_LETTER.value

            safe_error = error[:1000] if error else None
            if safe_error:
                try:
                    from agent.redact import redact_sensitive_text
                    safe_error = redact_sensitive_text(safe_error, force=True)
                except Exception:
                    pass

            conn.execute(
                """UPDATE task_queue
                   SET state = ?, last_error = ?, updated_at = ?
                   WHERE task_id = ?""",
                (next_state, safe_error, now, task_id),
            )

    def cancel(self, task_id: str, reason: str = "") -> bool:
        """Cancel a queued or running task."""
        now = time.time()
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """UPDATE task_queue
                   SET state = 'cancelled', last_error = ?, updated_at = ?
                   WHERE task_id = ? AND state NOT IN ('completed', 'cancelled')""",
                (reason[:500] if reason else "Cancelled by caller", now, task_id),
            )
            return bool(cursor.rowcount)

    def recover_crashed_tasks(self) -> int:
        """Inspect running tasks whose process crashed or timed out; re-queue or DLQ them."""
        now = time.time()
        recovered = 0
        with _DB_LOCK, _transaction() as conn:
            rows = conn.execute(
                """SELECT task_id, attempts, max_attempts, timeout_seconds, started_at, owner_pid
                   FROM task_queue
                   WHERE state = 'running'"""
            ).fetchall()

            for tid, attempts, max_attempts, timeout_s, started_at, pid in rows:
                is_dead_owner = False
                if pid:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        is_dead_owner = True
                else:
                    is_dead_owner = True

                is_timed_out = (started_at and (now - started_at) > timeout_s)
                if is_dead_owner or is_timed_out:
                    if attempts < max_attempts:
                        conn.execute(
                            """UPDATE task_queue
                               SET state = 'retrying', updated_at = ?,
                                   last_error = 'Recovered from ungraceful shutdown/timeout'
                               WHERE task_id = ?""",
                            (now, tid),
                        )
                    else:
                        conn.execute(
                            """UPDATE task_queue
                               SET state = 'dead_letter', updated_at = ?,
                                   last_error = 'Exhausted attempts during process restart'
                               WHERE task_id = ?""",
                            (now, tid),
                        )
                    recovered += 1

        return recovered

    def list_dead_letters(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List tasks in dead_letter state."""
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """SELECT task_id, idempotency_key, session_key, task_type, payload,
                          state, priority, attempts, max_attempts, timeout_seconds,
                          checkpoint, created_at, updated_at, started_at, completed_at,
                          last_error, owner_pid
                   FROM task_queue
                   WHERE state = 'dead_letter'
                   ORDER BY updated_at DESC LIMIT ?""",
                (max(1, int(limit)),),
            )
            return [self._row_to_task(r).to_dict() for r in cursor.fetchall()]

    def replay_dead_letter(self, task_id: str) -> bool:
        """Replay a dead letter task by resetting attempts to 0 and state to queued."""
        now = time.time()
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """UPDATE task_queue
                   SET state = 'queued', attempts = 0, last_error = NULL, updated_at = ?
                   WHERE task_id = ? AND state = 'dead_letter'""",
                (now, task_id),
            )
            return bool(cursor.rowcount)

    def prune(self, retention_seconds: float = 7 * 86400.0) -> int:
        """Prune completed, cancelled, or dead_letter tasks older than retention_seconds."""
        cutoff = time.time() - retention_seconds
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """DELETE FROM task_queue
                   WHERE state IN ('completed', 'cancelled', 'dead_letter')
                     AND updated_at < ?""",
                (cutoff,),
            )
            return cursor.rowcount

    def _row_to_task(self, row: tuple) -> QueuedTask:
        (
            task_id,
            idem_key,
            sess_key,
            t_type,
            payload_str,
            st_str,
            prio,
            attempts,
            max_att,
            timeout_s,
            cp_str,
            created,
            updated,
            started,
            completed,
            err,
            owner_pid,
        ) = row

        payload = json.loads(payload_str) if payload_str else {}
        checkpoint = json.loads(cp_str) if cp_str else None
        state = TaskState(st_str) if st_str in TaskState._value2member_map_ else TaskState.QUEUED

        return QueuedTask(
            task_id=task_id,
            idempotency_key=idem_key,
            session_key=sess_key,
            task_type=t_type,
            payload=payload,
            state=state,
            priority=prio,
            attempts=attempts,
            max_attempts=max_att,
            timeout_seconds=timeout_s,
            checkpoint=checkpoint,
            created_at=created,
            updated_at=updated,
            started_at=started,
            completed_at=completed,
            last_error=err,
            owner_pid=owner_pid,
        )

    claim_next_task = claim_next
    complete_task = complete


global_task_queue = PersistentTaskQueue()
