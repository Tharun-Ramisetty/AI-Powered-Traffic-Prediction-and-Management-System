"""Main video processing pipeline orchestrating detection, tracking, and counting."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import cv2
import numpy as np
import time

from config.settings import (
    PipelineConfig, DetectionConfig, TrackingConfig,
    CountingConfig, DensityConfig, CLASS_COLORS,
)
from src.detection.detector import Detection
from src.detection.model_loader import create_detector
from src.tracking.tracker import Track
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.deep_sort_tracker import DeepSORTTracker
from src.counting.line_counter import LineCrossingCounter
from src.counting.zone_counter import ZoneCounter
from src.counting.count_aggregator import CountAggregator
from src.counting.counting_utils import (
    draw_counting_line, draw_track_info, draw_counts_overlay,
)
from src.classification.density_classifier import DensityClassifier, DensityLevel


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    tracks: List[Track]
    counts: Dict[str, int]
    density: DensityLevel
    vehicles_in_frame: int
    fps: float
    timestamp: float


@dataclass
class VideoResult:
    """Result of processing an entire video."""
    total_counts: Dict[str, int]
    total_frames: int
    avg_fps: float
    duration_seconds: float
    density_history: List[DensityLevel] = field(default_factory=list)


class VideoPipeline:
    """Central orchestrator: Video -> Detection -> Tracking -> Counting.

    Processes video frames through the full pipeline and produces
    annotated frames with detection overlays, counts, and density.
    """

    def __init__(self, config: PipelineConfig = None):
        if config is None:
            config = PipelineConfig()
        self.config = config

        # Initialize detector
        self.detector = create_detector(
            config.detection.model_name, config.detection
        )

        # Initialize tracker
        if config.tracking.tracker_type == "deepsort":
            self.tracker = DeepSORTTracker(config.tracking)
        else:
            self.tracker = ByteTrackTracker(config.tracking)

        # Initialize counter
        if config.counting.mode == "zone" and config.counting.zone_points:
            self.counter = ZoneCounter(config.counting.zone_points)
        else:
            self.counter = LineCrossingCounter(
                line_y_fraction=config.counting.line_y_fraction,
                direction=config.counting.direction,
            )

        # Initialize density classifier
        self.density_clf = DensityClassifier(config.density)

        # Initialize count aggregator
        self.aggregator = CountAggregator(
            window_seconds=config.counting.aggregation_window_seconds
        )

    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> FrameResult:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image.
            timestamp: Unix timestamp. Uses current time if None.

        Returns:
            FrameResult with all detection, tracking, and counting data.
        """
        if timestamp is None:
            timestamp = time.time()

        start_time = time.perf_counter()

        # Resize large/vertical frames for better detection
        frame = self._preprocess_frame(frame)

        # 1. Detect
        detections = self.detector.detect(frame)

        # 2. Track
        tracks = self.tracker.update(detections, frame)

        # 3. Count
        h = frame.shape[0]
        if isinstance(self.counter, LineCrossingCounter):
            counts = self.counter.update(tracks, h)
        else:
            counts = self.counter.update(tracks)

        # 4. Classify density
        vehicles_in_frame = len(tracks)
        density = self.density_clf.classify(vehicles_in_frame)

        # 5. Aggregate
        self.aggregator.record(timestamp, counts)

        # 6. Annotate frame
        annotated = self._annotate_frame(frame, tracks, counts, density, vehicles_in_frame)

        elapsed = time.perf_counter() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return FrameResult(
            frame=frame,
            annotated_frame=annotated,
            detections=detections,
            tracks=tracks,
            counts=counts,
            density=density,
            vehicles_in_frame=vehicles_in_frame,
            fps=fps,
            timestamp=timestamp,
        )

    def process_video(self, video_path: str,
                      callback: Optional[Callable[[FrameResult], None]] = None,
                      max_frames: int = 0) -> VideoResult:
        """Process an entire video file.

        Args:
            video_path: Path to the video file.
            callback: Optional callback called with each FrameResult.
            max_frames: Max frames to process (0 = all).

        Returns:
            VideoResult with aggregate statistics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        fps_values = []
        density_history = []

        while cap.isOpened():
            if max_frames > 0 and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / video_fps
            result = self.process_frame(frame, timestamp)
            fps_values.append(result.fps)
            density_history.append(result.density)

            if callback:
                callback(result)

            frame_idx += 1

        cap.release()

        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

        return VideoResult(
            total_counts=self.counter.get_counts(),
            total_frames=frame_idx,
            avg_fps=avg_fps,
            duration_seconds=frame_idx / video_fps,
            density_history=density_history,
        )

    def _annotate_frame(self, frame: np.ndarray, tracks: List[Track],
                        counts: Dict[str, int], density: DensityLevel,
                        vehicles_in_frame: int) -> np.ndarray:
        """Draw all annotations on a copy of the frame."""
        annotated = frame.copy()

        # Draw counting line
        if isinstance(self.counter, LineCrossingCounter):
            line_y = self.counter.get_line_y(frame.shape[0])
            draw_counting_line(annotated, line_y)

        # Draw tracked vehicles
        for track in tracks:
            color = CLASS_COLORS.get(track.class_name, (50, 255, 50))
            draw_track_info(
                annotated, track.bbox, track.track_id,
                track.class_name, track.confidence, color
            )

        # Draw counts overlay
        draw_counts_overlay(annotated, counts, position=(10, 30))

        # Draw density badge
        density_color = density.color
        cv2.putText(
            annotated,
            f"Density: {density.value} ({vehicles_in_frame} vehicles)",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2,
        )

        return annotated

    @staticmethod
    def _preprocess_frame(frame: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        """Resize large or vertical frames for better detection.

        - Vertical videos (portrait) are resized to landscape-friendly dimensions
        - Very large frames (4K+) are downscaled to max_dim on the longest side
        """
        h, w = frame.shape[:2]

        # If frame is too large, resize proportionally
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def get_aggregator(self) -> CountAggregator:
        return self.aggregator

    def reset(self):
        """Reset all pipeline state."""
        self.tracker.reset()
        self.counter.reset()
        self.aggregator.reset()
