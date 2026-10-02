import concurrent.futures
import sqlite3
import pytest
from gateway.task_queue import PersistentTaskQueue
from gateway.circuit_breaker import ProviderCircuitBreaker
from gateway.audit_log import AuditLogger
from gateway.backup_manager import BackupManager


@pytest.fixture(autouse=True)
def _isolated_stress_db(tmp_path, monkeypatch):
    test_db = tmp_path / "state.db"
    monkeypatch.setattr("gateway.task_queue._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.circuit_breaker._db_path", lambda: test_db)
    monkeypatch.setattr("gateway.audit_log._db_path", lambda: test_db)
    yield test_db


def test_concurrent_multithreaded_stress(_isolated_stress_db):
    tq = PersistentTaskQueue()
    cb = ProviderCircuitBreaker()
    al = AuditLogger()

    num_threads = 8
    ops_per_thread = 25

    def worker(thread_idx: int):
        for i in range(ops_per_thread):
            # TaskQueue enqueue
            task = tq.enqueue(
                session_key=f"user_{thread_idx}",
                task_type="stress_test",
                payload={"thread": thread_idx, "i": i},
                idempotency_key=f"idem_{thread_idx}_{i}"
            )

            # Circuit breaker read & write
            cb.record_success(f"provider_{thread_idx % 3}")
            cb.can_execute(f"provider_{thread_idx % 3}")

            # AuditLogger write
            al.log(
                "stress.event",
                actor=f"user_{thread_idx}",
                details={"i": i},
                severity="INFO"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, idx) for idx in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            # If any database locking or concurrency error occurred, it will raise here!
            f.result()

    # Verify all records
    total_logs = len(al.query(limit=num_threads * ops_per_thread + 10))
    assert total_logs == num_threads * ops_per_thread

    # Verify DB integrity after intense multi-threaded write bombardment
    assert BackupManager.verify_integrity(_isolated_stress_db) is True
