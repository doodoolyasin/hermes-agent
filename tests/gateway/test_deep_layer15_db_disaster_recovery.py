import os
import sqlite3
import pytest
from gateway.backup_manager import BackupManager, MigrationManager


def test_corrupted_header_detection_and_disaster_rollback(tmp_path):
    db_file = tmp_path / "production.db"

    # 1. Create a healthy database with critical user data
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE orders (id INT, item TEXT, amount REAL)")
    conn.execute("INSERT INTO orders VALUES (101, 'server_license', 49.99)")
    conn.execute("INSERT INTO orders VALUES (102, 'vpn_subscription', 19.99)")
    conn.commit()
    conn.close()

    assert BackupManager.verify_integrity(db_file) is True

    # 2. Simulate disk corruption by zeroing out pages
    with open(db_file, "r+b") as f:
        f.seek(50)
        f.write(b"\x00" * 300)

    # verify_integrity MUST detect the corruption immediately!
    assert BackupManager.verify_integrity(db_file) is False

    # 3. Attempting migration on corrupted DB must abort safely without wiping or pretending success
    mm = MigrationManager(db_file)
    with pytest.raises(RuntimeError, match="Pre-migration integrity check failed"):
        mm.apply_migration("migration_x", ["CREATE TABLE dummy (x INT)"])


def test_migration_atomic_snapshot_recovery(tmp_path):
    db_file = tmp_path / "valid.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO config VALUES ('theme', 'dark')")
    conn.commit()
    conn.close()

    mm = MigrationManager(db_file)

    # Bad migration that fails
    with pytest.raises(RuntimeError):
        mm.apply_migration(
            "bad_migration",
            [
                "INSERT INTO config VALUES ('lang', 'fa')",
                "CRASHING SYNTAX ERROR HERE",
            ],
        )

    # After rollback, 'lang' was NOT persisted and DB is intact
    conn = sqlite3.connect(str(db_file))
    rows = conn.execute("SELECT key, val FROM config").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0] == ("theme", "dark")
    assert BackupManager.verify_integrity(db_file) is True
