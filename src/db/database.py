"""SQLite connection wrapper with auto-migration.

Threading note: SQLite connections are per-thread by default. We open
connections with ``check_same_thread=False`` and rely on the SQLite-side
locking, which is sufficient for the dashboard's modest concurrency. For
heavier write loads, switch to a connection pool or move to Postgres.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from src.db.schema import MIGRATIONS, SCHEMA_VERSION_TABLE


class Database:
    """Thin wrapper around an sqlite3 connection."""

    def __init__(self, path: Optional[str | Path] = None):
        if path is None:
            path = os.getenv("DATABASE_PATH") or self._default_path()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use explicit BEGIN/COMMIT
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._migrate()

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "traffic.db"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block atomically. Rolls back on any exception."""
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run a single statement (autocommit)."""
        with self._lock:
            return self._conn.execute(sql, params)

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ─── Migration ──────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        self._conn.executescript(SCHEMA_VERSION_TABLE)
        applied = {
            row["version"]
            for row in self._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchall()
        }
        # NOTE: ``executescript`` implicitly commits any open transaction,
        # so we cannot wrap it in BEGIN/COMMIT. Each migration uses
        # ``IF NOT EXISTS`` and is safe to re-run; if the version-row insert
        # fails, the next startup simply retries the (idempotent) script.
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            logger.info("Applying migration {} ({})",
                        migration.version, migration.name)
            self._conn.executescript(migration.sql)
            self._conn.execute(
                "INSERT INTO schema_version (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )

    @property
    def applied_versions(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        return [r["version"] for r in rows]


_default_db: Optional[Database] = None
_default_db_lock = threading.Lock()


def get_default_db() -> Database:
    """Process-wide singleton — convenient for the dashboard / pipeline."""
    global _default_db
    with _default_db_lock:
        if _default_db is None:
            _default_db = Database()
        return _default_db
