import sqlite3
import time
from pathlib import Path
import pytest
from gateway.backup_manager import BackupManager


def test_backup_manager_list_and_prune_snapshots(tmp_path):
    db_file = tmp_path / "app.db"

    # Create dummy database
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE users (id INT, name TEXT)")
    conn.commit()
    conn.close()

    # Create 8 sequential snapshots with slight timestamp differences
    snapshots = []
    for i in range(8):
        snap = tmp_path / f"app.db.bak_{1000 + i}"
        snap.write_text(f"snapshot content {i}")
        snapshots.append(snap)

    all_snaps = BackupManager.list_snapshots(db_file)
    assert len(all_snaps) == 8

    # Prune keeping only the top 3
    pruned_count = BackupManager.prune_snapshots(db_file, keep_count=3)
    assert pruned_count == 5

    remaining = BackupManager.list_snapshots(db_file)
    assert len(remaining) == 3


def test_backup_manager_prune_when_fewer_than_limit(tmp_path):
    db_file = tmp_path / "small.db"
    db_file.write_text("data")

    # Only 2 snapshots
    (tmp_path / "small.db.bak_1").write_text("1")
    (tmp_path / "small.db.bak_2").write_text("2")

    pruned = BackupManager.prune_snapshots(db_file, keep_count=5)
    assert pruned == 0
    assert len(BackupManager.list_snapshots(db_file)) == 2
