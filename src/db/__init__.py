"""SQLite persistence layer.

Lightweight, dependency-free (stdlib ``sqlite3``) audit / history store for
alerts, notifications, ANPR matches, vehicle counts, and signal changes.

Usage::

    from src.db import Database, AlertRepository

    db = Database()              # opens data/traffic.db, runs migrations
    repo = AlertRepository(db)
    repo.insert(alert)
"""

from src.db.database import Database, get_default_db
from src.db.repositories import (
    AlertRepository,
    NotificationRepository,
    ANPRMatchRepository,
    VehicleCountRepository,
    SignalChangeRepository,
    FlaggedVehicleRepository,
)

__all__ = [
    "Database",
    "get_default_db",
    "AlertRepository",
    "NotificationRepository",
    "ANPRMatchRepository",
    "VehicleCountRepository",
    "SignalChangeRepository",
    "FlaggedVehicleRepository",
]
