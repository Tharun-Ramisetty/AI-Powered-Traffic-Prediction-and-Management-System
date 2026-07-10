"""Scene-level accident detection via optical-flow motion analysis.

Complements the tracking-based ``AccidentDetector`` for cases where
per-vehicle tracks break down (dashcam footage, severe occlusion, camera
shake during impact). Instead of analysing individual vehicle behaviour,
this detector watches the whole-frame motion energy and fires when it
spikes well above a rolling baseline — which is what an actual collision
looks like at the pixel level.
"""

import time
from collections import deque
from typing import List

import cv2
import numpy as np

from .accident_detector import AccidentEvent


class MotionAccidentDetector:
    """Detects accidents by spotting sudden whole-frame motion spikes.

    Uses dense optical flow (Farneback) to compute per-frame motion
    chaos (std of flow magnitude). Maintains a rolling baseline of
    chaos values; when the current frame exceeds baseline by a
    configurable factor, fires an :class:`AccidentEvent`.

    Severity mapping (ratio of current chaos to baseline):

    - ``ratio >= critical_factor``   → ``"critical"``
    - ``ratio >= high_factor``       → ``"high"``
    - ``ratio >= spike_factor``      → ``"medium"``
    """

    def __init__(
        self,
        baseline_window: int = 30,
        spike_factor: float = 3.0,
        high_factor: float = 5.0,
        critical_factor: float = 10.0,
        cooldown_frames: int = 50,
        min_baseline_samples: int = 10,
    ):
        self.baseline_window = baseline_window
        self.spike_factor = spike_factor
        self.high_factor = high_factor
        self.critical_factor = critical_factor
        self.cooldown_frames = cooldown_frames
        self.min_baseline_samples = min_baseline_samples

        self._prev_gray: np.ndarray | None = None
        self._chaos_history: deque[float] = deque(maxlen=baseline_window)
        self._last_event_frame: int = -10_000
        self._event_counter: int = 0
        self._all_events: List[AccidentEvent] = []

    def update(self, frame: np.ndarray, frame_idx: int) -> List[AccidentEvent]:
        """Process one frame; return any newly fired events.

        Args:
            frame: BGR image.
            frame_idx: Frame index in the source video.

        Returns:
            New events fired this frame (typically 0 or 1).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events: List[AccidentEvent] = []

        if self._prev_gray is None:
            self._prev_gray = gray
            return events

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        chaos = float(mag.std())

        baseline = (
            float(np.median(self._chaos_history))
            if len(self._chaos_history) >= self.min_baseline_samples
            else None
        )

        if (
            baseline is not None
            and baseline > 0
            and chaos > baseline * self.spike_factor
            and frame_idx - self._last_event_frame >= self.cooldown_frames
        ):
            ratio = chaos / baseline
            if ratio >= self.critical_factor:
                sev = "critical"
            elif ratio >= self.high_factor:
                sev = "high"
            else:
                sev = "medium"

            self._event_counter += 1
            h, w = gray.shape
            event = AccidentEvent(
                event_id=f"MOT_{self._event_counter:04d}",
                timestamp=time.time(),
                location=(w / 2.0, h / 2.0),
                event_type="motion_spike",
                severity=sev,
                involved_track_ids=[],
                confidence=min(ratio / 10.0, 1.0),
                frame_number=frame_idx,
                description=(
                    f"Scene motion spike: chaos {chaos:.2f} vs baseline "
                    f"{baseline:.2f} (ratio {ratio:.1f}x)"
                ),
            )
            events.append(event)
            self._all_events.append(event)
            self._last_event_frame = frame_idx

        # Cap each contribution to avoid one big spike polluting the baseline
        # for many frames after.
        if baseline is not None:
            self._chaos_history.append(min(chaos, baseline * 2))
        else:
            self._chaos_history.append(chaos)

        self._prev_gray = gray
        return events

    def get_events(self) -> List[AccidentEvent]:
        return list(self._all_events)

    def reset(self) -> None:
        self._prev_gray = None
        self._chaos_history.clear()
        self._last_event_frame = -10_000
        self._event_counter = 0
        self._all_events.clear()
