"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import Detection
from src.tracking.tracker import Track


@pytest.fixture
def sample_frame():
    """640x480 synthetic BGR frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (100, 100, 100)  # Gray background
    return frame


@pytest.fixture
def sample_detections():
    """List of sample Detection objects."""
    return [
        Detection(bbox=(100, 200, 200, 280), confidence=0.92, class_id=1, class_name="car"),
        Detection(bbox=(300, 150, 500, 300), confidence=0.88, class_id=2, class_name="bus"),
        Detection(bbox=(50, 350, 120, 420), confidence=0.75, class_id=4, class_name="two_wheeler"),
    ]


@pytest.fixture
def sample_tracks():
    """List of sample Track objects."""
    return [
        Track(track_id=1, bbox=(100, 200, 200, 280), class_id=1,
              class_name="car", confidence=0.92, age=10, hits=8),
        Track(track_id=2, bbox=(300, 150, 500, 300), class_id=2,
              class_name="bus", confidence=0.88, age=15, hits=12),
        Track(track_id=3, bbox=(50, 350, 120, 420), class_id=4,
              class_name="two_wheeler", confidence=0.75, age=5, hits=4),
    ]
