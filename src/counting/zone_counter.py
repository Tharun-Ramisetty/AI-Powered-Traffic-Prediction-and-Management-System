"""Zone-based vehicle counter.

Counts unique vehicles that enter a defined polygon region.
Useful for intersection monitoring.
"""

from typing import List, Dict, Set, Tuple

import numpy as np

from src.tracking.tracker import Track


class ZoneCounter:
    """Counts unique vehicles entering a polygon zone."""

    def __init__(self, polygon_points: List[Tuple[int, int]]):
        """
        Args:
            polygon_points: List of (x, y) vertices defining the zone polygon.
        """
        self.polygon = np.array(polygon_points, dtype=np.float32)
        self.counted_ids: Set[int] = set()
        self.inside_ids: Set[int] = set()
        self.counts: Dict[str, int] = {"total": 0}

    def update(self, tracks: List[Track]) -> Dict[str, int]:
        """Update counter with current frame's tracks.

        Args:
            tracks: List of Track objects from the tracker.

        Returns:
            Current cumulative counts dict.
        """
        current_inside = set()

        for track in tracks:
            cx, cy = track.centroid

            if self._point_in_polygon(cx, cy):
                current_inside.add(track.track_id)

                if track.track_id not in self.counted_ids:
                    self.counted_ids.add(track.track_id)
                    class_name = track.class_name

                    if class_name not in self.counts:
                        self.counts[class_name] = 0
                    self.counts[class_name] += 1
                    self.counts["total"] += 1

        self.inside_ids = current_inside
        return self.counts.copy()

    def get_current_inside_count(self) -> int:
        """Get the number of vehicles currently inside the zone."""
        return len(self.inside_ids)

    def get_counts(self) -> Dict[str, int]:
        return self.counts.copy()

    def _point_in_polygon(self, x: float, y: float) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(self.polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def reset(self):
        self.counted_ids.clear()
        self.inside_ids.clear()
        self.counts = {"total": 0}
