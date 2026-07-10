"""Line-crossing vehicle counter.

Counts unique vehicles that cross a horizontal counting line.
Each tracked vehicle is counted exactly once when its centroid
crosses from one side of the line to the other.
"""

from typing import List, Dict, Set, Tuple

from src.tracking.tracker import Track


class LineCrossingCounter:
    """Counts vehicles crossing a horizontal line in the frame.

    Also tracks unique vehicles seen across all frames (even if they
    don't cross the counting line), so the total is never zero when
    vehicles are present.
    """

    def __init__(self, line_y_fraction: float = 0.6, direction: str = "both"):
        """
        Args:
            line_y_fraction: Position of counting line as fraction of frame height (0-1).
            direction: Count direction - "down", "up", or "both".
        """
        self.line_y_fraction = line_y_fraction
        self.direction = direction

        self.counted_ids: Set[int] = set()
        self.prev_positions: Dict[int, float] = {}  # track_id -> previous centroid_y
        self.counts: Dict[str, int] = {"total": 0}
        self.crossing_log: List[Dict] = []

        # Track all unique vehicles seen (regardless of line crossing)
        self._all_seen_ids: Set[int] = set()
        self._seen_counts: Dict[str, int] = {"total": 0}

    def update(self, tracks: List[Track], frame_height: int) -> Dict[str, int]:
        """Update counter with current frame's tracks.

        Args:
            tracks: List of Track objects from the tracker.
            frame_height: Height of the frame in pixels.

        Returns:
            Current cumulative counts dict (class_name -> count + "total").
        """
        line_y = frame_height * self.line_y_fraction

        for track in tracks:
            centroid_y = track.centroid[1]
            track_id = track.track_id

            # Track unique confirmed vehicles (seen for 3+ frames = real vehicles)
            if track_id not in self._all_seen_ids and track.is_confirmed:
                self._all_seen_ids.add(track_id)
                class_name = track.class_name
                if class_name not in self._seen_counts:
                    self._seen_counts[class_name] = 0
                self._seen_counts[class_name] += 1
                self._seen_counts["total"] += 1

            # Line crossing detection
            if track_id in self.prev_positions and track_id not in self.counted_ids:
                prev_y = self.prev_positions[track_id]
                crossed = False

                if self.direction in ("down", "both") and prev_y < line_y <= centroid_y:
                    crossed = True
                if self.direction in ("up", "both") and prev_y > line_y >= centroid_y:
                    crossed = True

                if crossed:
                    self.counted_ids.add(track_id)
                    class_name = track.class_name

                    if class_name not in self.counts:
                        self.counts[class_name] = 0
                    self.counts[class_name] += 1
                    self.counts["total"] += 1

                    self.crossing_log.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "direction": "down" if centroid_y > prev_y else "up",
                    })

            self.prev_positions[track_id] = centroid_y

        return self.get_counts()

    def get_line_y(self, frame_height: int) -> int:
        """Get the pixel Y-coordinate of the counting line."""
        return int(frame_height * self.line_y_fraction)

    def get_counts(self) -> Dict[str, int]:
        """Get best available counts.

        Returns line-crossing counts if any crossings detected,
        otherwise returns unique confirmed vehicles seen.
        """
        if self.counts["total"] > 0:
            return self.counts.copy()
        return self._seen_counts.copy()

    def get_crossing_counts(self) -> Dict[str, int]:
        """Get only line-crossing counts."""
        return self.counts.copy()

    def get_unique_vehicle_counts(self) -> Dict[str, int]:
        """Get all unique vehicles seen regardless of line crossing."""
        return self._seen_counts.copy()

    def get_crossing_log(self) -> List[Dict]:
        """Get log of all crossing events."""
        return self.crossing_log.copy()

    def reset(self):
        """Reset all counts and state."""
        self.counted_ids.clear()
        self.prev_positions.clear()
        self.counts = {"total": 0}
        self.crossing_log.clear()
        self._all_seen_ids.clear()
        self._seen_counts = {"total": 0}
