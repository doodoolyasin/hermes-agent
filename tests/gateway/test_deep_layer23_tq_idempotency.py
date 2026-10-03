import concurrent.futures
import pytest
from gateway.task_queue import PersistentTaskQueue, TaskState


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    yield


def test_concurrent_idempotency_key_collision_safety():
    queue = PersistentTaskQueue()
    shared_key = "idem-fixed-shared-key-12345"

    tasks_returned = []

    def worker(i: int):
        t = queue.enqueue(
            session_key="bale:chat_99",
            task_type="ai_generation",
            payload={"prompt": f"test prompt from worker {i}"},
            idempotency_key=shared_key,
        )
        return t.task_id

    # 20 concurrent threads trying to enqueue with the SAME idempotency key
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        tasks_returned = list(pool.map(worker, range(20)))

    # All 20 threads must receive the EXACT same task_id!
    unique_task_ids = set(tasks_returned)
    assert len(unique_task_ids) == 1

    first_task_id = list(unique_task_ids)[0]
    task = queue.get_task(first_task_id)
    assert task is not None
    assert task.idempotency_key == shared_key


def test_task_queue_lifecycle_transitions():
    queue = PersistentTaskQueue()
    t = queue.enqueue(
        session_key="bale:chat_100",
        task_type="batch_export",
        payload={"export_id": 42},
    )

    # 1. Claim
    claimed = queue.claim_next_task(worker_pid=12345)
    assert claimed is not None
    assert claimed.task_id == t.task_id
    assert claimed.state == TaskState.RUNNING

    # 2. Checkpoint
    queue.checkpoint(t.task_id, {"step": 1, "progress": 0.5})
    t_checkpoint = queue.get_task(t.task_id)
    assert t_checkpoint.checkpoint == {"step": 1, "progress": 0.5}

    # 3. Complete
    queue.complete_task(t.task_id, result={"file": "/tmp/out.csv"})
    t_done = queue.get_task(t.task_id)
    assert t_done.state == TaskState.COMPLETED
    assert t_done.completed_at is not None
