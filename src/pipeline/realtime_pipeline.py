"""Real-time pipeline for continuous live camera processing."""

import threading
import time
from typing import Optional

import cv2

from config.settings import PipelineConfig
from .video_pipeline import VideoPipeline, FrameResult


class RealtimePipeline:
    """Wraps VideoPipeline for continuous live processing from a camera source."""

    def __init__(self, config: PipelineConfig, source=0):
        """
        Args:
            config: Pipeline configuration.
            source: Video source - file path, RTSP URL, or device index.
        """
        self.pipeline = VideoPipeline(config)
        self.source = source
        self.running = False
        self.latest_result: Optional[FrameResult] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self):
        """Start processing in a background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def _run(self):
        """Main processing loop."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self.running = False
            return

        while self.running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # If video file, loop back to start
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            result = self.pipeline.process_frame(frame, time.time())

            with self._lock:
                self.latest_result = result

        self._cap.release()

    def get_latest(self) -> Optional[FrameResult]:
        """Get the most recent frame result (thread-safe)."""
        with self._lock:
            return self.latest_result

    def is_running(self) -> bool:
        return self.running

    def get_pipeline(self) -> VideoPipeline:
        return self.pipeline
