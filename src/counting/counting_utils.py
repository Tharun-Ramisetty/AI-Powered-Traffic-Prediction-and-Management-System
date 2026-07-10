"""Utility functions for counting geometry."""

from typing import Tuple, List

import numpy as np
import cv2


def draw_counting_line(frame: np.ndarray, line_y: int,
                       color: Tuple[int, int, int] = (0, 0, 255),
                       thickness: int = 2) -> np.ndarray:
    """Draw the counting line on a frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, thickness)
    cv2.putText(frame, "Counting Line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_zone_polygon(frame: np.ndarray, polygon: List[Tuple[int, int]],
                      color: Tuple[int, int, int] = (0, 255, 255),
                      alpha: float = 0.3) -> np.ndarray:
    """Draw a semi-transparent polygon zone on a frame."""
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(frame, [pts], True, color, 2)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_track_info(frame: np.ndarray, bbox: Tuple[float, float, float, float],
                    track_id: int, class_name: str, confidence: float,
                    color: Tuple[int, int, int] = (50, 255, 50)) -> np.ndarray:
    """Draw bounding box with track ID and class info."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{track_id} {class_name} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def draw_counts_overlay(frame: np.ndarray, counts: dict,
                        position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw vehicle counts as text overlay on frame."""
    x, y = position
    for i, (cls, count) in enumerate(sorted(counts.items())):
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (x, y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
