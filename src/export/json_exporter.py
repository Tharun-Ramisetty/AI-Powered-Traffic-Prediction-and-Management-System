"""JSON export for vehicle count data."""

import json
from typing import Dict, List, Any
from datetime import datetime


class JSONExporter:
    """Exports vehicle count data to JSON files."""

    @staticmethod
    def export_counts(counts: Dict[str, int], filepath: str,
                      metadata: Dict[str, Any] = None):
        """Export counts with optional metadata to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "counts": counts,
        }
        if metadata:
            data["metadata"] = metadata

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def export_full_report(
        counts: Dict[str, int],
        density: str,
        crossing_log: List[Dict],
        filepath: str,
    ):
        """Export a comprehensive report as JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_vehicles": counts.get("total", 0),
                "per_class": {k: v for k, v in counts.items() if k != "total"},
                "traffic_density": density,
            },
            "crossing_events": crossing_log,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
