import os
import pytest
from gateway.task_queue import PersistentTaskQueue, TaskState, QueuedTask
from hermes_constants import get_process_hermes_home


@pytest.fixture(autouse=True)
def _isolated_task_queue(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    yield


def test_enqueue_and_idempotency():
    queue = PersistentTaskQueue()
    task1 = queue.enqueue(
        session_key="bale:1234",
        task_type="ai_generation",
        payload={"prompt": "Explain quantum computing"},
        idempotency_key="request_xyz_1",
        priority=5
    )
    assert task1.state == TaskState.QUEUED
    assert task1.priority == 5
    assert task1.payload["prompt"] == "Explain quantum computing"

    # Enqueue same idempotency key again
    task2 = queue.enqueue(
        session_key="bale:1234",
        task_type="ai_generation",
        payload={"prompt": "Different prompt"},
        idempotency_key="request_xyz_1"
    )
    # Must return original task1 without creating a second row
    assert task2.task_id == task1.task_id
    assert task2.payload["prompt"] == "Explain quantum computing"


def test_claim_priority_and_checkpoint():
    queue = PersistentTaskQueue()
    queue.enqueue("bale:1", "typeA", {"x": 1}, priority=1)
    queue.enqueue("bale:2", "typeB", {"x": 2}, priority=10)

    # Highest priority task (typeB) claimed first
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.task_type == "typeB"
    assert claimed.state == TaskState.RUNNING
    assert claimed.attempts == 1

    # Checkpoint
    queue.checkpoint(claimed.task_id, {"step": 2, "processed_items": 50})

    # Complete
    queue.complete(claimed.task_id)

    # Next task claimed
    next_task = queue.claim_next()
    assert next_task is not None
    assert next_task.task_type == "typeA"


def test_fail_retry_and_dead_letter():
    queue = PersistentTaskQueue()
    task = queue.enqueue("bale:1", "typeA", {"x": 1}, max_attempts=2)

    # Claim attempt 1
    t1 = queue.claim_next()
    assert t1 is not None
    queue.fail(t1.task_id, error="Transient error 1", retryable=True)

    # Claim attempt 2 (retrying)
    t2 = queue.claim_next()
    assert t2 is not None
    assert t2.task_id == t1.task_id
    assert t2.attempts == 2

    # Second failure exceeds max_attempts=2 -> moves to DEAD_LETTER
    queue.fail(t2.task_id, error="Transient error 2", retryable=True)

    # No more available tasks
    assert queue.claim_next() is None

    # Inspect dead letters
    dls = queue.list_dead_letters()
    assert len(dls) == 1
    assert dls[0]["task_id"] == t1.task_id
    assert dls[0]["state"] == "dead_letter"

    # Replay dead letter
    replayed = queue.replay_dead_letter(t1.task_id)
    assert replayed is True

    # Now claimable again as queued
    t3 = queue.claim_next()
    assert t3 is not None
    assert t3.task_id == t1.task_id
    assert t3.attempts == 1


def test_crashed_task_recovery():
    queue = PersistentTaskQueue()
    task = queue.enqueue("bale:1", "offline_work", {"x": 1}, max_attempts=3)
    claimed = queue.claim_next()
    assert claimed is not None

    # Simulate fake crashed owner PID (PID 999999999)
    from gateway.task_queue import _transaction, _DB_LOCK
    with _DB_LOCK, _transaction() as conn:
        conn.execute("UPDATE task_queue SET owner_pid = 999999999 WHERE task_id = ?", (claimed.task_id,))

    # Recover crashed tasks
    recovered_count = queue.recover_crashed_tasks()
    assert recovered_count == 1

    # Should be back in retrying state and claimable again
    re_claimed = queue.claim_next()
    assert re_claimed is not None
    assert re_claimed.task_id == claimed.task_id
    assert re_claimed.attempts == 2
