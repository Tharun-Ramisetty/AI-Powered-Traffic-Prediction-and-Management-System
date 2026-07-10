"""Database schema and forward-only migrations.

Each migration is an idempotent ``CREATE TABLE IF NOT EXISTS`` (or
``ALTER TABLE`` for additive changes). The migration runner records applied
versions in the ``schema_version`` table so re-running is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATIONS: List[Migration] = [
    Migration(
        version=1,
        name="initial_schema",
        sql="""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id        TEXT    NOT NULL UNIQUE,
            alert_type      TEXT    NOT NULL,
            priority        TEXT    NOT NULL,
            title           TEXT    NOT NULL,
            message         TEXT    NOT NULL,
            location        TEXT    NOT NULL DEFAULT '',
            camera_id       TEXT    NOT NULL DEFAULT '',
            sent_via        TEXT    NOT NULL DEFAULT '',
            metadata_json   TEXT    NOT NULL DEFAULT '{}',
            timestamp       REAL    NOT NULL,
            created_at      REAL    NOT NULL DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_alerts_type      ON alerts(alert_type);
        CREATE INDEX IF NOT EXISTS idx_alerts_camera    ON alerts(camera_id);

        CREATE TABLE IF NOT EXISTS notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id     TEXT    NOT NULL,
            type         TEXT    NOT NULL,
            priority     TEXT    NOT NULL,
            title        TEXT    NOT NULL,
            message      TEXT    NOT NULL,
            location     TEXT    NOT NULL DEFAULT '',
            camera_id    TEXT    NOT NULL DEFAULT '',
            timestamp    REAL    NOT NULL,
            is_read      INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_notif_unread ON notifications(is_read, timestamp DESC);

        CREATE TABLE IF NOT EXISTS anpr_matches (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number  TEXT    NOT NULL,
            camera_id     TEXT    NOT NULL DEFAULT '',
            location      TEXT    NOT NULL DEFAULT '',
            is_flagged    INTEGER NOT NULL DEFAULT 0,
            reason        TEXT    NOT NULL DEFAULT '',
            confidence    REAL    NOT NULL DEFAULT 0.0,
            timestamp     REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_anpr_plate     ON anpr_matches(plate_number);
        CREATE INDEX IF NOT EXISTS idx_anpr_flagged   ON anpr_matches(is_flagged, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_anpr_timestamp ON anpr_matches(timestamp DESC);

        CREATE TABLE IF NOT EXISTS vehicle_counts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id     TEXT    NOT NULL,
            class_name    TEXT    NOT NULL,
            count         INTEGER NOT NULL,
            window_start  REAL    NOT NULL,
            window_end    REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_counts_camera_window
            ON vehicle_counts(camera_id, window_start);

        CREATE TABLE IF NOT EXISTS signal_changes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            direction   TEXT    NOT NULL,
            old_state   TEXT    NOT NULL,
            new_state   TEXT    NOT NULL,
            reason      TEXT    NOT NULL DEFAULT '',
            timestamp   REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_changes(timestamp DESC);

        CREATE TABLE IF NOT EXISTS flagged_vehicles (
            plate_number    TEXT PRIMARY KEY,
            reason          TEXT NOT NULL DEFAULT 'unknown',
            reported_date   TEXT NOT NULL DEFAULT '',
            owner_name      TEXT NOT NULL DEFAULT '',
            vehicle_type    TEXT NOT NULL DEFAULT '',
            priority        TEXT NOT NULL DEFAULT 'medium',
            created_at      REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
        """,
    ),
]


SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    applied_at REAL    NOT NULL DEFAULT (strftime('%s','now'))
);
"""
