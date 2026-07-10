"""Abstract base classes and data structures for vehicle detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


@dataclass
class Detection:
    """A single detected object in a frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel coords
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height


class BaseDetector(ABC):
    """Abstract base class for all vehicle detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects found in the frame.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model metadata for benchmarking.

        Returns:
            Dict with keys: name, variant, params, input_size, etc.
        """
        pass
