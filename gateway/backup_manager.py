"""Automated Backup Manager and Migration Framework for Hermes SQLite databases.

Implements Directive Section 4:
- BALANCED + AUTOMATIC BACKUP policy
- Pre-migration integrity verification
- Automatic timestamped snapshot backup before any schema changes
- Atomic, idempotent migration application
- Post-migration integrity verification
- Automatic rollback to snapshot on any failure
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_process_hermes_home

logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    return get_process_hermes_home() / "state.db"


class BackupManager:
    """Manages automatic SQLite backups, integrity checks and rollbacks."""

    @staticmethod
    def verify_integrity(db_path: Path | str) -> bool:
        """Run PRAGMA integrity_check on the database."""
        path = Path(db_path)
        if not path.exists():
            return True
        try:
            conn = sqlite3.connect(str(path), timeout=5.0)
            cursor = conn.cursor()
            result = cursor.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            return result and result[0] == "ok"
        except Exception as exc:
            logger.error("Database integrity check failed for %s: %s", path, exc)
            return False

    @staticmethod
    def create_snapshot(db_path: Path | str) -> Path:
        """Create an automatic timestamped backup of the database."""
        path = Path(db_path)
        if not path.exists():
            return path

        timestamp = int(time.time())
        backup_path = path.parent / f"{path.name}.bak_{timestamp}"

        # Prefer SQLite online backup API for hot-safe atomic snapshot
        try:
            src = sqlite3.connect(str(path), timeout=10.0)
            dst = sqlite3.connect(str(backup_path), timeout=10.0)
            src.backup(dst)
            dst.close()
            src.close()
        except Exception:
            # Fallback to file copy if SQLite lock is contentious
            shutil.copy2(path, backup_path)

        logger.info("Automatic database snapshot created at %s", backup_path)
        return backup_path

    @staticmethod
    def restore_snapshot(snapshot_path: Path | str, target_path: Path | str) -> bool:
        """Restore database from a snapshot."""
        src_path = Path(snapshot_path)
        dst_path = Path(target_path)
        if not src_path.exists():
            logger.error("Snapshot file %s does not exist", src_path)
            return False

        try:
            shutil.copy2(src_path, dst_path)
            logger.info("Successfully restored %s from snapshot %s", dst_path, src_path)
            return True
        except Exception as exc:
            logger.error("Failed to restore snapshot: %s", exc)
            return False


class MigrationManager:
    """Applies idempotent migrations under the automated backup policy."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or get_default_db_path()

    def _ensure_migrations_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS _applied_migrations (
                migration_id TEXT PRIMARY KEY,
                applied_at REAL NOT NULL
            )"""
        )

    def is_applied(self, migration_id: str) -> bool:
        if not self.db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            self._ensure_migrations_table(conn)
            row = conn.execute(
                "SELECT 1 FROM _applied_migrations WHERE migration_id = ?",
                (migration_id,),
            ).fetchone()
            conn.close()
            return bool(row)
        except Exception:
            return False

    def apply_migration(self, migration_id: str, sql_commands: List[str]) -> bool:
        """Apply an idempotent migration with pre-check, auto-backup, and post-check."""
        # 1. Pre-migration integrity check MUST run first!
        if not BackupManager.verify_integrity(self.db_path):
            raise RuntimeError(f"Pre-migration integrity check failed for {self.db_path}")

        if self.is_applied(migration_id):
            logger.debug("Migration %s already applied, skipping", migration_id)
            return True

        logger.info("Starting migration %s...", migration_id)

        # 2. Automatic backup
        snapshot = BackupManager.create_snapshot(self.db_path)

        # 3. Apply migration inside transaction
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_migrations_table(conn)
            for sql in sql_commands:
                conn.execute(sql)
            conn.execute(
                "INSERT INTO _applied_migrations (migration_id, applied_at) VALUES (?, ?)",
                (migration_id, time.time()),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            conn.rollback()
            conn.close()
            logger.error("Migration %s failed: %s. Initiating automatic rollback...", migration_id, exc)
            BackupManager.restore_snapshot(snapshot, self.db_path)
            raise RuntimeError(f"Migration {migration_id} failed and was rolled back: {exc}") from exc

        # 4. Post-migration integrity check
        if not BackupManager.verify_integrity(self.db_path):
            logger.error("Post-migration integrity check failed! Rolling back to %s", snapshot)
            BackupManager.restore_snapshot(snapshot, self.db_path)
            raise RuntimeError(f"Post-migration integrity check failed for {migration_id}")

        logger.info("Migration %s applied successfully and verified.", migration_id)
        return True
