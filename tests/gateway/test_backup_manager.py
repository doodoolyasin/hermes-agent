import sqlite3
import pytest
from gateway.backup_manager import BackupManager, MigrationManager


def test_backup_and_integrity_check(tmp_path):
    db_file = tmp_path / "test_state.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE sample (id INT, name TEXT)")
    conn.execute("INSERT INTO sample VALUES (1, 'alpha')")
    conn.commit()
    conn.close()

    assert BackupManager.verify_integrity(db_file) is True
    snapshot = BackupManager.create_snapshot(db_file)
    assert snapshot.exists()
    assert ".bak_" in snapshot.name


def test_migration_manager_idempotency_and_rollback(tmp_path):
    db_file = tmp_path / "test_migration.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE users (id INT, email TEXT)")
    conn.commit()
    conn.close()

    mm = MigrationManager(db_file)

    # 1. Apply successful migration
    success = mm.apply_migration("001_add_role", ["ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"])
    assert success is True
    assert mm.is_applied("001_add_role") is True

    # 2. Idempotency: re-running does not fail
    assert mm.apply_migration("001_add_role", ["ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"]) is True

    # 3. Faulty migration with syntax error: MUST rollback cleanly!
    with pytest.raises(RuntimeError, match="failed and was rolled back"):
        mm.apply_migration("002_broken", ["INVALID SQL SYNTAX HERE"])

    # Verify database is intact and broken migration is not marked applied
    assert mm.is_applied("002_broken") is False
    assert BackupManager.verify_integrity(db_file) is True
