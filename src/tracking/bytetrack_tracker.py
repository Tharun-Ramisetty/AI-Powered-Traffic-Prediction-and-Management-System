"""ByteTrack tracker implementation using the supervision library."""

from typing import List

import numpy as np
import supervision as sv

from config.settings import TrackingConfig
from src.detection.detector import Detection
from .tracker import BaseTracker, Track


class ByteTrackTracker(BaseTracker):
    """Vehicle tracker using ByteTrack algorithm.

    ByteTrack is a simple yet effective tracker that associates
    every detection box (including low-confidence ones) to reduce
    missed tracks. Faster than Deep SORT.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=config.max_age,
            minimum_matching_threshold=config.iou_threshold,
            frame_rate=30,
        )
        self._class_names = {}  # track_id -> class_name
        self._prev_centroids = {}  # track_id -> (cx, cy) for velocity
        self._track_hits = {}  # track_id -> number of frames seen

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        if not detections:
            return []

        # Convert to supervision Detections format
        bboxes = np.array([d.bbox for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids,
        )

        # Run ByteTrack
        tracked = self.tracker.update_with_detections(sv_detections)

        if tracked.tracker_id is None:
            return []

        # Build Track objects
        result = []
        active_ids = set()

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            class_id = int(tracked.class_id[i])
            active_ids.add(track_id)

            # Resolve class name from the detection itself (not from VEHICLE_CLASSES index)
            class_name = self._resolve_class_name(class_id, detections)
            self._class_names[track_id] = class_name

            bbox = tracked.xyxy[i]
            cx = (float(bbox[0]) + float(bbox[2])) / 2
            cy = (float(bbox[1]) + float(bbox[3])) / 2

            # Compute velocity from previous centroid
            vx, vy = 0.0, 0.0
            if track_id in self._prev_centroids:
                prev_cx, prev_cy = self._prev_centroids[track_id]
                vx = cx - prev_cx
                vy = cy - prev_cy
            self._prev_centroids[track_id] = (cx, cy)

            # Track hits (frames seen)
            self._track_hits[track_id] = self._track_hits.get(track_id, 0) + 1
            hits = self._track_hits[track_id]

            result.append(Track(
                track_id=track_id,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                class_id=class_id,
                class_name=class_name,
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                velocity=(vx, vy),
                hits=hits,
            ))

        # Clean up stale entries
        stale = [tid for tid in self._prev_centroids if tid not in active_ids]
        for tid in stale:
            if self._track_hits.get(tid, 0) > self.config.max_age:
                self._prev_centroids.pop(tid, None)
                self._track_hits.pop(tid, None)
                self._class_names.pop(tid, None)

        return result

    def reset(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self.config.max_age,
            minimum_matching_threshold=self.config.iou_threshold,
            frame_rate=30,
        )
        self._class_names.clear()
        self._prev_centroids.clear()
        self._track_hits.clear()

    @staticmethod
    def _resolve_class_name(class_id: int, detections: List[Detection]) -> str:
        # Always use the class name from the detection (works for both custom and COCO models)
        for d in detections:
            if d.class_id == class_id:
                return d.class_name
        return f"class_{class_id}"
