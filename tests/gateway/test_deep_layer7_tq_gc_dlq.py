import time
import pytest
from gateway.task_queue import PersistentTaskQueue, TaskState
from gateway.audit_log import AuditLogger


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.audit_log._db_path", lambda: test_db)
    yield


def test_task_queue_prune_and_retention():
    tq = PersistentTaskQueue()

    # 1. Enqueue and complete a task
    t1 = tq.enqueue("bale:1", "type1", {"k": 1})
    claimed1 = tq.claim_next()
    tq.complete(claimed1.task_id)

    # 2. Enqueue and dead_letter a task
    t2 = tq.enqueue("bale:2", "type2", {"k": 2})
    claimed2 = tq.claim_next()
    tq.fail(claimed2.task_id, "fatal fail", retryable=False)

    # 3. Enqueue an active task (should NEVER be pruned)
    t3 = tq.enqueue("bale:3", "type3", {"k": 3})

    # Fake age of completed/dead_letter tasks to 10 days ago
    from gateway.task_queue import _transaction, _DB_LOCK
    old_time = time.time() - (10 * 86400)
    with _DB_LOCK, _transaction() as conn:
        conn.execute("UPDATE task_queue SET updated_at = ? WHERE task_id IN (?, ?)",
                     (old_time, t1.task_id, t2.task_id))

    # Prune with 7-day retention
    deleted = tq.prune(retention_seconds=7 * 86400)
    assert deleted == 2

    # Active task must still be queued and claimable!
    claimed3 = tq.claim_next()
    assert claimed3 is not None
    assert claimed3.task_id == t3.task_id


def test_task_queue_dead_letter_replay():
    tq = PersistentTaskQueue()
    t = tq.enqueue("bale:99", "job", {"x": 10})
    claimed = tq.claim_next()
    tq.fail(claimed.task_id, "broken", retryable=False)

    assert len(tq.list_dead_letters()) == 1

    # Replay
    ok = tq.replay_dead_letter(t.task_id)
    assert ok is True
    assert len(tq.list_dead_letters()) == 0

    # Task is queued again
    reclaimed = tq.claim_next()
    assert reclaimed is not None
    assert reclaimed.task_id == t.task_id
    assert reclaimed.attempts == 1


def test_audit_log_prune():
    al = AuditLogger()
    al.log("evt1", severity="INFO")
    al.log("evt2", severity="WARN")

    # Age events
    from gateway.audit_log import _transaction, _DB_LOCK
    old_time = time.time() - (40 * 86400)
    with _DB_LOCK, _transaction() as conn:
        conn.execute("UPDATE audit_log SET timestamp = ?", (old_time,))

    # Prune events older than 30 days
    pruned = al.prune(retention_seconds=30 * 86400)
    assert pruned == 2
    assert len(al.query()) == 0
