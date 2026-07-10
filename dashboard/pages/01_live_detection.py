"""Live Detection page - Upload video and run real-time detection + tracking."""

import tempfile
import time

import cv2
import numpy as np
import streamlit as st

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

from config.settings import PipelineConfig, DetectionConfig, TrackingConfig, VEHICLE_CLASSES
from src.pipeline.video_pipeline import VideoPipeline


@st.cache_resource(show_spinner="Loading detection model…")
def _build_pipeline(
    model_name: str,
    tracker_type: str,
    confidence: float,
    line_position: float,
    direction: str,
) -> VideoPipeline:
    """Build (and cache) a VideoPipeline.

    Cached on the tuple of user-controlled settings so re-clicking "Run" with
    the same settings reuses the loaded YOLO model instead of reloading it —
    which otherwise costs several seconds per click on CPU.
    """
    config = PipelineConfig()
    config.detection = DetectionConfig(
        model_name=model_name,
        confidence_threshold=confidence,
    )
    config.tracking = TrackingConfig(tracker_type=tracker_type)
    config.counting.line_y_fraction = line_position
    config.counting.direction = direction
    return VideoPipeline(config)


st.set_page_config(page_title="Live Detection", layout="wide")
st.title("Live Vehicle Detection & Tracking")

# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Detection Model",
        ["yolov8", "yolov9", "yolov10", "yolov8m_coco"],
        help="Custom models: yolov8/v9/v10 (9 Indian classes). "
             "yolov8m_coco: pre-trained on 80 COCO classes (works on any video).",
    )
    tracker_type = st.selectbox("Tracker", ["bytetrack", "deepsort"])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    line_position = st.slider("Counting Line Position", 0.1, 0.9, 0.6, 0.05)
    direction = st.selectbox("Count Direction", ["both", "down", "up"])
    max_frames = st.number_input(
        "Max Frames (0 = all)", 0, 100000, 300, 50,
        help="Cap processing for a fast preview. Set 0 to process the whole video.",
    )
    frame_stride = st.slider(
        "Frame Stride", 1, 10, 2, 1,
        help="Process every Nth frame. 2–3 is visually identical for traffic "
             "counting but 2–3× faster.",
    )

# ─── Video Source ─────────────────────────────────────────────────────────────
st.markdown("### Upload a Traffic Video")
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    # ``getbuffer`` avoids the extra copy that ``read()`` makes — worth it
    # on the large videos people upload to this page.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.getbuffer())
    tfile.flush()

    st.video(uploaded_file)

    if st.button("Run Detection & Tracking", type="primary"):
        # Cached pipeline: model weights load once per settings combination.
        pipeline = _build_pipeline(
            model_name, tracker_type, confidence, line_position, direction,
        )
        # Reset per-run state so counts don't leak between clicks.
        pipeline.reset()

        # UI placeholders
        col_video, col_stats = st.columns([3, 1])

        with col_video:
            frame_placeholder = st.empty()
        with col_stats:
            count_placeholder = st.empty()
            density_placeholder = st.empty()
            fps_placeholder = st.empty()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process video
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frame_limit = max_frames if max_frames > 0 else total_frames

        frame_idx = 0
        processed_count = 0
        density_history = []
        # Keep only a small rolling window of tracks/detections for downstream
        # pages. Accumulating every frame ballooned memory and slowed the UI
        # noticeably on videos longer than ~30s.
        recent_tracks: list = []
        recent_detections: list = []
        RECENT_MAX = 500

        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened() and frame_idx < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame-skip: only run inference on every Nth frame.
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / video_fps
            result = pipeline.process_frame(frame, timestamp)
            density_history.append(result.density)

            recent_tracks.extend(result.tracks)
            recent_detections.extend(result.detections)
            if len(recent_tracks) > RECENT_MAX:
                recent_tracks = recent_tracks[-RECENT_MAX:]
            if len(recent_detections) > RECENT_MAX:
                recent_detections = recent_detections[-RECENT_MAX:]

            # Update video display every processed frame (already strided).
            rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                rgb_frame, channels="RGB", use_container_width=True
            )

            count_placeholder.markdown("### Vehicle Counts")
            for cls, cnt in sorted(result.counts.items()):
                count_placeholder.markdown(f"**{cls}**: {cnt}")
            density_placeholder.markdown(f"### Density: {result.density.value}")
            fps_placeholder.markdown(f"**FPS**: {result.fps:.1f}")

            processed_count += 1
            progress_bar.progress(min((frame_idx + 1) / frame_limit, 1.0))
            status_text.text(
                f"Frame {frame_idx + 1}/{frame_limit} "
                f"(processed {processed_count}, stride={frame_stride})"
            )

            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        # Final results
        st.markdown("---")
        st.markdown("### Final Results")
        final_counts = pipeline.counter.get_counts()

        # Filter out non-vehicle classes (like 'person') for display
        vehicle_counts = {k: v for k, v in final_counts.items()
                         if k.lower() not in ("person", "total")}
        vehicle_total = sum(vehicle_counts.values())
        vehicle_counts["total"] = vehicle_total

        if vehicle_counts:
            num_cols = min(len(vehicle_counts), 9)
            cols = st.columns(num_cols)
            for i, (cls, cnt) in enumerate(sorted(vehicle_counts.items())):
                with cols[i % num_cols]:
                    st.metric(cls.replace("_", " ").title(), cnt)

        # Show crossing vs unique breakdown
        if hasattr(pipeline.counter, 'get_crossing_counts'):
            crossing = pipeline.counter.get_crossing_counts()
            unique = pipeline.counter.get_unique_vehicle_counts()
            if crossing["total"] > 0:
                st.caption(f"Line crossings: {crossing['total']} | Unique vehicles seen: {unique['total']}")
            else:
                st.caption(f"Unique vehicles detected: {unique['total']} (no line crossings in this video)")

        # Store data in session for other pages
        st.session_state["aggregator"] = pipeline.get_aggregator()
        st.session_state["final_counts"] = vehicle_counts
        st.session_state["density_history"] = density_history
        st.session_state["all_tracks"] = recent_tracks
        st.session_state["all_detections"] = recent_detections
        st.session_state["video_fps"] = video_fps
        st.session_state["total_frames"] = frame_idx

else:
    st.info("Upload a traffic video to start detection and tracking.")
    st.markdown("""
    **Supported formats**: MP4, AVI, MOV, MKV

    **Tips**:
    - Use 720p or higher resolution for best results
    - Adjust the counting line position to match the road in your video
    - ByteTrack is faster; Deep SORT is more accurate for crowded scenes
    """)
