"""Emergency vehicle detection - ambulance, fire truck, police vehicles."""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from src.detection.detector import Detection


@dataclass
class EmergencyEvent:
    """Represents a detected emergency vehicle."""
    event_id: str
    timestamp: float
    vehicle_type: str  # "ambulance", "fire_truck", "police"
    location: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    direction: str = "unknown"  # "approaching", "moving_away", "stationary"
    frame_number: int = 0
    siren_detected: bool = False


class EmergencyVehicleDetector:
    """Detects emergency vehicles using visual cues.

    Detection strategies:
    1. Color-based detection (red/white for ambulance, red for fire truck)
    2. Shape/size heuristics (large vehicles with specific aspect ratios)
    3. Text detection ("AMBULANCE", "FIRE" on vehicle body)
    4. Optional: siren sound detection via audio
    """

    # Emergency vehicle color ranges in HSV
    # Red (fire trucks, some ambulances)
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([160, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Blue (police lights)
    BLUE_LOWER = np.array([100, 150, 100])
    BLUE_UPPER = np.array([130, 255, 255])

    # White (ambulance body)
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 30, 255])

    EMERGENCY_KEYWORDS = ["AMBULANCE", "FIRE", "POLICE", "108", "101", "100", "RESCUE"]

    def __init__(
        self,
        color_threshold: float = 0.15,
        min_vehicle_area: int = 5000,
        use_ocr: bool = True,
        ocr_languages: list = None,
    ):
        self.color_threshold = color_threshold
        self.min_vehicle_area = min_vehicle_area
        self.use_ocr = use_ocr
        self._event_counter = 0
        self._recent_events: List[EmergencyEvent] = []
        self._ocr_reader = None

        if use_ocr:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(ocr_languages or ["en"], gpu=False)
            except ImportError:
                self.use_ocr = False

    def detect(
        self, frame: np.ndarray, detections: List[Detection], frame_number: int = 0
    ) -> List[EmergencyEvent]:
        """Detect emergency vehicles in the current frame.

        Args:
            frame: BGR image.
            detections: Vehicle detections from the main detector.
            frame_number: Current frame number.

        Returns:
            List of EmergencyEvent objects.
        """
        events: List[EmergencyEvent] = []

        for det in detections:
            # Only check larger vehicles (Bus, Truck, Tempo)
            if det.area < self.min_vehicle_area:
                continue

            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            # Check color patterns
            vehicle_type = self._classify_by_color(vehicle_roi)

            # Check text on vehicle
            if vehicle_type is None and self.use_ocr:
                vehicle_type = self._classify_by_text(vehicle_roi)

            if vehicle_type:
                self._event_counter += 1
                event = EmergencyEvent(
                    event_id=f"EMR_{self._event_counter:04d}",
                    timestamp=time.time(),
                    vehicle_type=vehicle_type,
                    location=det.centroid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    frame_number=frame_number,
                )
                events.append(event)

        self._recent_events.extend(events)
        if len(self._recent_events) > 100:
            self._recent_events = self._recent_events[-100:]

        return events

    def _classify_by_color(self, roi: np.ndarray) -> Optional[str]:
        """Classify emergency vehicle type by dominant colors."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]

        # Red detection (fire truck)
        red_mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        red_mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        red_mask = red_mask1 | red_mask2
        red_ratio = cv2.countNonZero(red_mask) / total_pixels

        # Blue detection (police)
        blue_mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

        # White detection (ambulance)
        white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
        white_ratio = cv2.countNonZero(white_mask) / total_pixels

        # Classification logic
        if red_ratio > self.color_threshold * 2:
            return "fire_truck"
        if blue_ratio > self.color_threshold:
            return "police"
        if white_ratio > 0.5 and red_ratio > self.color_threshold * 0.5:
            return "ambulance"

        return None

    def _classify_by_text(self, roi: np.ndarray) -> Optional[str]:
        """Check for emergency text on vehicle body."""
        if not self._ocr_reader:
            return None

        try:
            results = self._ocr_reader.readtext(roi)
            for _, text, conf in results:
                text_upper = text.upper().strip()
                for keyword in self.EMERGENCY_KEYWORDS:
                    if keyword in text_upper:
                        if keyword in ("AMBULANCE", "108"):
                            return "ambulance"
                        elif keyword in ("FIRE", "101", "RESCUE"):
                            return "fire_truck"
                        elif keyword in ("POLICE", "100"):
                            return "police"
        except Exception:
            pass

        return None

    def get_recent_events(self, last_n: int = 10) -> List[EmergencyEvent]:
        return self._recent_events[-last_n:]

    def reset(self):
        self._recent_events.clear()
        self._event_counter = 0
