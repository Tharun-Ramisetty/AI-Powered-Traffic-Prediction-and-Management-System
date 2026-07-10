"""Deep SORT tracker implementation using deep-sort-realtime library."""

from typing import List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from config.settings import TrackingConfig
from src.detection.detector import Detection
from .tracker import BaseTracker, Track


class DeepSORTTracker(BaseTracker):
    """Vehicle tracker using Deep SORT with appearance features.

    Deep SORT combines Kalman filtering for motion prediction with
    a deep appearance descriptor (MobileNet) for re-identification.
    Better for crowded scenes but slower than ByteTrack.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.min_hits,
            max_iou_distance=1.0 - config.iou_threshold,
            embedder="mobilenet",
            embedder_gpu=True,
            half=True,
        )

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert to deep-sort-realtime format: ([x, y, w, h], confidence, class_name)
        raw_detections = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            raw_detections.append(
                ([x1, y1, x2 - x1, y2 - y1], d.confidence, d.class_name)
            )

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        result = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            ltrb = t.to_ltrb()  # [left, top, right, bottom]
            det_class = t.det_class if t.det_class else "unknown"

            # Find matching detection for class_id
            class_id = self._get_class_id(det_class)

            result.append(Track(
                track_id=t.track_id,
                bbox=(float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])),
                class_id=class_id,
                class_name=det_class,
                confidence=t.det_conf if t.det_conf else 0.0,
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
            ))

        return result

    def reset(self):
        self.tracker = DeepSort(
            max_age=self.config.max_age,
            n_init=self.config.min_hits,
            max_iou_distance=1.0 - self.config.iou_threshold,
            embedder="mobilenet",
            embedder_gpu=True,
            half=True,
        )

    @staticmethod
    def _get_class_id(class_name: str) -> int:
        from config.settings import VEHICLE_CLASSES
        try:
            return VEHICLE_CLASSES.index(class_name)
        except ValueError:
            return -1
