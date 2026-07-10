"""Abstract base classes and data structures for object tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.detection.detector import Detection


@dataclass
class Track:
    """A tracked object with persistent identity across frames."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    age: int = 0                     # Total frames this track has existed
    hits: int = 0                    # Frames with a matched detection
    time_since_update: int = 0       # Frames since last matched detection
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/frame

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3


class BaseTracker(ABC):
    """Abstract base class for all object trackers."""

    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with new detections from the current frame.

        Args:
            detections: List of detections from the current frame.
            frame: The current BGR frame (used for appearance features).

        Returns:
            List of active Track objects with persistent IDs.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clear all tracks and reset the tracker state."""
        pass
