"""Check detected plates against a database of illegal/wanted vehicles."""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path


@dataclass
class VehicleRecord:
    """A record for a flagged vehicle."""
    plate_number: str
    reason: str  # "stolen", "wanted", "blacklisted", "expired_registration"
    reported_date: str = ""
    owner_name: str = ""
    vehicle_type: str = ""
    priority: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class MatchResult:
    """Result of checking a plate against the database."""
    plate_number: str
    is_flagged: bool
    record: Optional[VehicleRecord] = None
    timestamp: float = 0.0
    location: str = ""


class IllegalVehicleChecker:
    """Checks number plates against a local database of flagged vehicles.

    The database is a JSON file with plate numbers and their details.
    In production, this would connect to RTO / police databases.
    """

    def __init__(self, database_path: Optional[str] = None):
        self._database: Dict[str, VehicleRecord] = {}
        self._match_log: List[MatchResult] = []
        self._anpr_repo = None
        self._flagged_repo = None

        if database_path and Path(database_path).exists():
            self.load_database(database_path)

    def attach_db(self, anpr_match_repo, flagged_repo=None) -> None:
        """Persist every plate read (and optionally hydrate the in-memory
        blacklist from a FlaggedVehicleRepository)."""
        self._anpr_repo = anpr_match_repo
        self._flagged_repo = flagged_repo
        if flagged_repo is not None:
            for row in flagged_repo.list_all():
                self._database[row["plate_number"]] = VehicleRecord(
                    plate_number=row["plate_number"],
                    reason=row["reason"],
                    reported_date=row["reported_date"],
                    owner_name=row["owner_name"],
                    vehicle_type=row["vehicle_type"],
                    priority=row["priority"],
                )

    def load_database(self, path: str):
        """Load flagged vehicles from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        for entry in data.get("flagged_vehicles", []):
            plate = entry["plate_number"].upper().replace(" ", "").replace("-", "")
            self._database[plate] = VehicleRecord(
                plate_number=plate,
                reason=entry.get("reason", "unknown"),
                reported_date=entry.get("reported_date", ""),
                owner_name=entry.get("owner_name", ""),
                vehicle_type=entry.get("vehicle_type", ""),
                priority=entry.get("priority", "medium"),
            )

    def add_flagged_vehicle(self, record: VehicleRecord):
        """Add a vehicle to the flagged database."""
        plate = record.plate_number.upper().replace(" ", "").replace("-", "")
        record.plate_number = plate
        self._database[plate] = record

    def check_plate(self, plate_text: str, location: str = "") -> MatchResult:
        """Check a plate number against the database.

        Args:
            plate_text: Detected plate text.
            location: Camera/location identifier.

        Returns:
            MatchResult with flagged status and details.
        """
        cleaned = plate_text.upper().replace(" ", "").replace("-", "")

        record = self._database.get(cleaned)
        result = MatchResult(
            plate_number=cleaned,
            is_flagged=record is not None,
            record=record,
            timestamp=time.time(),
            location=location,
        )

        if result.is_flagged:
            self._match_log.append(result)

        if self._anpr_repo is not None:
            try:
                self._anpr_repo.insert(
                    plate_number=cleaned,
                    location=location,
                    is_flagged=result.is_flagged,
                    reason=record.reason if record else "",
                    timestamp=result.timestamp,
                )
            except Exception:
                from loguru import logger
                logger.exception("Failed to persist ANPR read for plate {}", cleaned)

        return result

    def get_flagged_plates(self) -> List[str]:
        """Return all plate numbers in the database."""
        return list(self._database.keys())

    def get_match_log(self, last_n: int = 50) -> List[MatchResult]:
        """Get recent matches against flagged vehicles."""
        return self._match_log[-last_n:]

    def save_database(self, path: str):
        """Save the current database to a JSON file."""
        data = {
            "flagged_vehicles": [
                {
                    "plate_number": r.plate_number,
                    "reason": r.reason,
                    "reported_date": r.reported_date,
                    "owner_name": r.owner_name,
                    "vehicle_type": r.vehicle_type,
                    "priority": r.priority,
                }
                for r in self._database.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def database_size(self) -> int:
        return len(self._database)
