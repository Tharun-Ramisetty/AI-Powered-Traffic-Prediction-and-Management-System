"""Typed repositories — one per table.

Repositories accept domain objects (``Alert``, ``MatchResult``, …) where
possible so callers do not need to know the SQL schema. Read methods return
plain dicts so they can be used directly by the Streamlit dashboard.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from src.db.database import Database


class _BaseRepo:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        return {k: row[k] for k in row.keys()}


class AlertRepository(_BaseRepo):
    """Persistent audit trail of every dispatched alert."""

    def insert(self, alert) -> int:
        """Persist an Alert. Returns the rowid. Idempotent on alert_id."""
        cur = self.db.execute(
            """
            INSERT INTO alerts
                (alert_id, alert_type, priority, title, message,
                 location, camera_id, sent_via, metadata_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                sent_via      = excluded.sent_via,
                metadata_json = excluded.metadata_json
            """,
            (
                alert.alert_id,
                alert.alert_type.value,
                alert.priority.name,
                alert.title,
                alert.message,
                alert.location,
                alert.camera_id,
                ",".join(alert.sent_via),
                json.dumps(alert.metadata or {}),
                alert.timestamp,
            ),
        )
        return cur.lastrowid

    def list_recent(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if alert_type:
            clauses.append("alert_type = ?")
            params.append(alert_type)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.fetchall(
            f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [self._row_to_dict(r) for r in rows]

    def count_by_type(self) -> Dict[str, int]:
        rows = self.db.fetchall(
            "SELECT alert_type, COUNT(*) AS n FROM alerts GROUP BY alert_type"
        )
        return {r["alert_type"]: r["n"] for r in rows}

    def delete_older_than(self, cutoff_timestamp: float) -> int:
        cur = self.db.execute(
            "DELETE FROM alerts WHERE timestamp < ?", (cutoff_timestamp,)
        )
        return cur.rowcount


class NotificationRepository(_BaseRepo):
    """In-app notification feed (replaces the JSON file store)."""

    def insert(self, alert) -> int:
        cur = self.db.execute(
            """
            INSERT INTO notifications
                (alert_id, type, priority, title, message,
                 location, camera_id, timestamp, is_read)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                alert.alert_id,
                alert.alert_type.value,
                alert.priority.name,
                alert.title,
                alert.message,
                alert.location,
                alert.camera_id,
                alert.timestamp,
            ),
        )
        return cur.lastrowid

    def get_unread(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM notifications WHERE is_read = 0 "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM notifications ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]

    def mark_read(self, notification_id: int) -> None:
        self.db.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ?",
            (notification_id,),
        )

    def mark_all_read(self) -> None:
        self.db.execute("UPDATE notifications SET is_read = 1")

    def unread_count(self) -> int:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM notifications WHERE is_read = 0"
        )
        return int(row["n"]) if row else 0


class ANPRMatchRepository(_BaseRepo):
    """History of every plate read, flagged or not."""

    def insert(
        self,
        plate_number: str,
        camera_id: str = "",
        location: str = "",
        is_flagged: bool = False,
        reason: str = "",
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO anpr_matches
                (plate_number, camera_id, location, is_flagged,
                 reason, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plate_number.upper().replace(" ", "").replace("-", ""),
                camera_id, location, int(bool(is_flagged)),
                reason, confidence,
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return cur.lastrowid

    def list_flagged(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM anpr_matches WHERE is_flagged = 1 "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]

    def history_for(self, plate_number: str, limit: int = 50) -> List[Dict[str, Any]]:
        cleaned = plate_number.upper().replace(" ", "").replace("-", "")
        rows = self.db.fetchall(
            "SELECT * FROM anpr_matches WHERE plate_number = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (cleaned, limit),
        )
        return [self._row_to_dict(r) for r in rows]


class VehicleCountRepository(_BaseRepo):
    """Time-series storage of per-camera, per-class vehicle counts."""

    def insert(
        self, camera_id: str, class_name: str,
        count: int, window_start: float, window_end: float,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO vehicle_counts
                (camera_id, class_name, count, window_start, window_end)
            VALUES (?, ?, ?, ?, ?)
            """,
            (camera_id, class_name, count, window_start, window_end),
        )
        return cur.lastrowid

    def insert_batch(self, rows: List[tuple]) -> None:
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO vehicle_counts
                    (camera_id, class_name, count, window_start, window_end)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def aggregate_by_class(
        self, camera_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> Dict[str, int]:
        clauses, params = [], []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if since is not None:
            clauses.append("window_start >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.fetchall(
            f"""
            SELECT class_name, SUM(count) AS total
            FROM vehicle_counts {where}
            GROUP BY class_name
            ORDER BY total DESC
            """,
            tuple(params),
        )
        return {r["class_name"]: int(r["total"]) for r in rows}


class SignalChangeRepository(_BaseRepo):
    """Audit log of every traffic-signal state change."""

    def insert(
        self, direction: str, old_state: str, new_state: str,
        reason: str = "", timestamp: Optional[float] = None,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO signal_changes
                (direction, old_state, new_state, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                direction, old_state, new_state, reason,
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return cur.lastrowid

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM signal_changes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]


class FlaggedVehicleRepository(_BaseRepo):
    """Persistent blacklist; replaces the JSON file."""

    def upsert(
        self, plate_number: str, reason: str = "unknown",
        reported_date: str = "", owner_name: str = "",
        vehicle_type: str = "", priority: str = "medium",
    ) -> None:
        cleaned = plate_number.upper().replace(" ", "").replace("-", "")
        self.db.execute(
            """
            INSERT INTO flagged_vehicles
                (plate_number, reason, reported_date,
                 owner_name, vehicle_type, priority)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(plate_number) DO UPDATE SET
                reason        = excluded.reason,
                reported_date = excluded.reported_date,
                owner_name    = excluded.owner_name,
                vehicle_type  = excluded.vehicle_type,
                priority      = excluded.priority
            """,
            (cleaned, reason, reported_date, owner_name, vehicle_type, priority),
        )

    def get(self, plate_number: str) -> Optional[Dict[str, Any]]:
        cleaned = plate_number.upper().replace(" ", "").replace("-", "")
        row = self.db.fetchone(
            "SELECT * FROM flagged_vehicles WHERE plate_number = ?",
            (cleaned,),
        )
        return self._row_to_dict(row) if row else None

    def remove(self, plate_number: str) -> bool:
        cleaned = plate_number.upper().replace(" ", "").replace("-", "")
        cur = self.db.execute(
            "DELETE FROM flagged_vehicles WHERE plate_number = ?", (cleaned,)
        )
        return cur.rowcount > 0

    def list_all(self) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM flagged_vehicles ORDER BY plate_number"
        )
        return [self._row_to_dict(r) for r in rows]

    def import_from_json(self, json_path: str) -> int:
        """Bulk-load from the legacy JSON file. Returns rows inserted."""
        from pathlib import Path
        path = Path(json_path)
        if not path.exists():
            logger.warning("Flagged-vehicles JSON not found: {}", json_path)
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("flagged_vehicles", []) if isinstance(data, dict) else []
        for entry in entries:
            self.upsert(
                entry["plate_number"],
                reason=entry.get("reason", "unknown"),
                reported_date=entry.get("reported_date", ""),
                owner_name=entry.get("owner_name", ""),
                vehicle_type=entry.get("vehicle_type", ""),
                priority=entry.get("priority", "medium"),
            )
        return len(entries)
