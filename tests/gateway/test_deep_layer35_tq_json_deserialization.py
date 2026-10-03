import sqlite3
import pytest
from gateway.task_queue import PersistentTaskQueue, TaskState, _connect


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    yield


def test_corrupted_json_in_payload_and_checkpoint():
    queue = PersistentTaskQueue()

    # 1. Manually insert rows with truncated/corrupted JSON
    with _connect() as conn:
        conn.execute(
            """INSERT INTO task_queue (
                task_id, idempotency_key, session_key, task_type, payload,
                state, priority, attempts, max_attempts, timeout_seconds,
                checkpoint, created_at, updated_at, owner_pid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 3, 300.0, ?, 1000.0, 1000.0, NULL)""",
            (
                "task-corrupt-1",
                "idem-corrupt-1",
                "bale:chat_bad",
                "ai_agent",
                '{"prompt": "unclosed json string...',  # broken JSON!
                TaskState.QUEUED.value,
                10,
                '{"step": broken checkpoint',          # broken JSON!
            ),
        )

    # 2. get_task must NOT crash with JSONDecodeError
    task = queue.get_task("task-corrupt-1")
    assert task is not None
    assert task.task_id == "task-corrupt-1"
    assert "_corrupted_raw_payload" in task.payload
    assert "_corrupted_raw_checkpoint" in task.checkpoint

    # 3. claim_next must claim it smoothly without crashing
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.task_id == "task-corrupt-1"
    assert claimed.state == TaskState.RUNNING


def test_empty_string_json_in_payload():
    queue = PersistentTaskQueue()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO task_queue (
                task_id, idempotency_key, session_key, task_type, payload,
                state, priority, attempts, max_attempts, timeout_seconds,
                checkpoint, created_at, updated_at, owner_pid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 3, 300.0, NULL, 1000.0, 1000.0, NULL)""",
            (
                "task-empty-2",
                "idem-empty-2",
                "bale:chat_empty",
                "ping",
                "",  # empty string
                TaskState.QUEUED.value,
                5,
            ),
        )

    task = queue.get_task("task-empty-2")
    assert task is not None
    assert task.payload == {}
    assert task.checkpoint is None
