"""Annotated video writer for saving processed output."""

from typing import Optional

import cv2
import numpy as np


class VideoWriter:
    """Writes annotated frames to an output video file."""

    def __init__(self, output_path: str, fps: float = 30.0,
                 codec: str = "mp4v"):
        """
        Args:
            output_path: Output video file path.
            fps: Output video FPS.
            codec: FourCC codec string.
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray):
        """Write a single frame to the output video.

        Initializes the writer on first call using the frame dimensions.
        """
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (w, h)
            )

        self._writer.write(frame)

    def release(self):
        """Finalize and close the video file."""
        if self._writer:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
