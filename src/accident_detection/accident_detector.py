"""Accident detection using vehicle tracking data - sudden stops and collisions."""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from src.tracking.tracker import Track


@dataclass
class AccidentEvent:
    """Represents a detected accident event."""
    event_id: str
    timestamp: float
    location: Tuple[float, float]  # (x, y) pixel coordinates
    event_type: str  # "sudden_stop", "collision", "wrong_way"
    severity: str  # "low", "medium", "high", "critical"
    involved_track_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0
    frame_number: int = 0
    description: str = ""


class AccidentDetector:
    """Detects accidents by analyzing vehicle track behavior.

    Detection methods:
    1. Sudden deceleration - vehicle speed drops drastically
    2. Collision - two vehicle bounding boxes overlap suddenly
    3. Stationary detection - vehicle stops in traffic lane unexpectedly
    """

    def __init__(
        self,
        deceleration_threshold: float = 15.0,
        collision_iou_threshold: float = 0.3,
        stationary_frames: int = 90,
        min_speed_for_sudden_stop: float = 8.0,
        cooldown_seconds: float = 10.0,
    ):
        self.deceleration_threshold = deceleration_threshold
        self.collision_iou_threshold = collision_iou_threshold
        self.stationary_frames = stationary_frames
        self.min_speed_for_sudden_stop = min_speed_for_sudden_stop
        self.cooldown_seconds = cooldown_seconds

        # Track history: track_id -> list of (centroid, timestamp)
        self._track_history: Dict[int, List[Tuple[Tuple[float, float], float]]] = {}
        self._speed_history: Dict[int, List[float]] = {}
        self._stationary_counter: Dict[int, int] = {}
        self._event_counter = 0
        self._recent_events: List[AccidentEvent] = []
        self._last_event_time: float = 0.0

    def update(self, tracks: List[Track], frame_number: int) -> List[AccidentEvent]:
        """Analyze current tracks for accident indicators.

        Args:
            tracks: List of active Track objects from the tracker.
            frame_number: Current frame number.

        Returns:
            List of newly detected AccidentEvent objects.
        """
        current_time = time.time()
        events: List[AccidentEvent] = []

        # Update track histories
        for track in tracks:
            if not track.is_confirmed:
                continue

            tid = track.track_id
            centroid = track.centroid

            if tid not in self._track_history:
                self._track_history[tid] = []
                self._speed_history[tid] = []
                self._stationary_counter[tid] = 0

            self._track_history[tid].append((centroid, current_time))

            # Keep only last 60 positions
            if len(self._track_history[tid]) > 60:
                self._track_history[tid] = self._track_history[tid][-60:]

            # Calculate current speed (pixels/frame)
            speed = np.sqrt(track.velocity[0] ** 2 + track.velocity[1] ** 2)
            self._speed_history[tid].append(speed)
            if len(self._speed_history[tid]) > 30:
                self._speed_history[tid] = self._speed_history[tid][-30:]

        # Check for sudden stops
        sudden_stop_events = self._detect_sudden_stops(tracks, frame_number, current_time)
        events.extend(sudden_stop_events)

        # Check for collisions (bbox overlap)
        collision_events = self._detect_collisions(tracks, frame_number, current_time)
        events.extend(collision_events)

        # Check for stationary vehicles in lanes
        stationary_events = self._detect_stationary(tracks, frame_number, current_time)
        events.extend(stationary_events)

        # Clean up old track data
        active_ids = {t.track_id for t in tracks}
        stale_ids = [tid for tid in self._track_history if tid not in active_ids]
        for tid in stale_ids:
            self._track_history.pop(tid, None)
            self._speed_history.pop(tid, None)
            self._stationary_counter.pop(tid, None)

        self._recent_events.extend(events)
        # Keep only last 100 events
        if len(self._recent_events) > 100:
            self._recent_events = self._recent_events[-100:]

        return events

    def _detect_sudden_stops(
        self, tracks: List[Track], frame_number: int, current_time: float
    ) -> List[AccidentEvent]:
        events = []
        if current_time - self._last_event_time < self.cooldown_seconds:
            return events

        for track in tracks:
            tid = track.track_id
            if tid not in self._speed_history or len(self._speed_history[tid]) < 5:
                continue

            speeds = self._speed_history[tid]
            recent_avg = np.mean(speeds[-3:])
            prev_avg = np.mean(speeds[-8:-3]) if len(speeds) >= 8 else np.mean(speeds[:-3])

            if prev_avg >= self.min_speed_for_sudden_stop and recent_avg < 1.0:
                deceleration = prev_avg - recent_avg
                if deceleration >= self.deceleration_threshold:
                    severity = "critical" if deceleration > 25 else "high" if deceleration > 20 else "medium"
                    self._event_counter += 1
                    event = AccidentEvent(
                        event_id=f"ACC_{self._event_counter:04d}",
                        timestamp=current_time,
                        location=track.centroid,
                        event_type="sudden_stop",
                        severity=severity,
                        involved_track_ids=[tid],
                        confidence=min(deceleration / 30.0, 1.0),
                        frame_number=frame_number,
                        description=f"Vehicle #{tid} sudden stop: speed {prev_avg:.1f} -> {recent_avg:.1f} px/frame",
                    )
                    events.append(event)
                    self._last_event_time = current_time

        return events

    def _detect_collisions(
        self, tracks: List[Track], frame_number: int, current_time: float
    ) -> List[AccidentEvent]:
        events = []
        if current_time - self._last_event_time < self.cooldown_seconds:
            return events

        confirmed = [t for t in tracks if t.is_confirmed]
        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                iou = self._compute_iou(confirmed[i].bbox, confirmed[j].bbox)
                if iou >= self.collision_iou_threshold:
                    # Check if both vehicles were moving
                    speed_i = np.sqrt(confirmed[i].velocity[0] ** 2 + confirmed[i].velocity[1] ** 2)
                    speed_j = np.sqrt(confirmed[j].velocity[0] ** 2 + confirmed[j].velocity[1] ** 2)

                    if speed_i > 3.0 or speed_j > 3.0:
                        mid_x = (confirmed[i].centroid[0] + confirmed[j].centroid[0]) / 2
                        mid_y = (confirmed[i].centroid[1] + confirmed[j].centroid[1]) / 2
                        severity = "critical" if iou > 0.6 else "high" if iou > 0.4 else "medium"
                        self._event_counter += 1
                        event = AccidentEvent(
                            event_id=f"ACC_{self._event_counter:04d}",
                            timestamp=current_time,
                            location=(mid_x, mid_y),
                            event_type="collision",
                            severity=severity,
                            involved_track_ids=[confirmed[i].track_id, confirmed[j].track_id],
                            confidence=min(iou / 0.5, 1.0),
                            frame_number=frame_number,
                            description=f"Collision between #{confirmed[i].track_id} and #{confirmed[j].track_id} (IoU={iou:.2f})",
                        )
                        events.append(event)
                        self._last_event_time = current_time

        return events

    def _detect_stationary(
        self, tracks: List[Track], frame_number: int, current_time: float
    ) -> List[AccidentEvent]:
        events = []
        for track in tracks:
            if not track.is_confirmed:
                continue

            tid = track.track_id
            speed = np.sqrt(track.velocity[0] ** 2 + track.velocity[1] ** 2)

            if speed < 0.5:
                self._stationary_counter[tid] = self._stationary_counter.get(tid, 0) + 1
            else:
                self._stationary_counter[tid] = 0

            if self._stationary_counter[tid] == self.stationary_frames:
                self._event_counter += 1
                event = AccidentEvent(
                    event_id=f"ACC_{self._event_counter:04d}",
                    timestamp=current_time,
                    location=track.centroid,
                    event_type="stationary_vehicle",
                    severity="low",
                    involved_track_ids=[tid],
                    confidence=0.7,
                    frame_number=frame_number,
                    description=f"Vehicle #{tid} stationary for {self.stationary_frames} frames",
                )
                events.append(event)

        return events

    @staticmethod
    def _compute_iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_recent_events(self, last_n: int = 10) -> List[AccidentEvent]:
        return self._recent_events[-last_n:]

    def reset(self):
        self._track_history.clear()
        self._speed_history.clear()
        self._stationary_counter.clear()
        self._recent_events.clear()
        self._event_counter = 0
        self._last_event_time = 0.0
