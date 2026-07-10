"""Dashboard page for Accident Detection System - Real detection from video."""

import tempfile
import time

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Accident Detection", page_icon=":rotating_light:", layout="wide")
st.title(":rotating_light: Accident Detection System")
st.markdown("Detect sudden stops, collisions, and stationary vehicles in real-time.")

with st.expander("What this detector catches — and what it misses", expanded=False):
    st.markdown("""
    The detector is **heuristic / track-based**: it analyses how tracked
    vehicles move, not the visual appearance of a crash itself.

    **Catches well:**
    - Vehicles that brake to a near-complete stop from cruising speed
    - Vehicles whose bounding boxes overlap for multiple frames
      (sustained collision, not glancing impact)
    - Vehicles that remain stationary in a lane for ≥ N frames

    **Misses (known limitation):**
    - Accidents where the impact breaks the vehicle track (visual change,
      camera shake, occlusion). With no continuous track, there's no speed
      profile to analyse.
    - Glancing collisions with low bounding-box overlap (e.g. cameras at
      shallow angles to the road).
    - Low-resolution / dashcam footage where the tracker can only keep a
      vehicle locked for a few frames at a time.

    **For best results:** use higher-resolution traffic-camera footage with
    a top-down or slightly elevated view, and the `yolov8m_coco` model.
    Lower the sliders below if a clear accident isn't being flagged.
    """)
st.markdown("---")

# ─── Sidebar Configuration ──────────────────────────────────────────────────
st.sidebar.header("Accident Detection Settings")

detector_kind = st.sidebar.radio(
    "Detector Type",
    ["Motion-based (best for dashcam)", "Behavioural (best for traffic-cam)"],
    help=(
        "Motion-based watches whole-frame motion energy and catches "
        "crashes even when individual vehicle tracks break. "
        "Behavioural tracks each vehicle's speed and bbox overlaps; "
        "best on stable top-down traffic footage."
    ),
)
is_motion_mode = detector_kind.startswith("Motion")

if is_motion_mode:
    spike_factor = st.sidebar.slider(
        "Spike Factor (× baseline)", 2.0, 10.0, 3.0, 0.5,
        help="How many times above baseline motion must spike to fire an event.",
    )
    motion_cooldown_frames = st.sidebar.slider(
        "Motion Cooldown (frames)", 10, 200, 50, 10,
    )
    # Behavioural-only knobs are hidden in motion mode but kept defined
    # so downstream code can reference them safely.
    model_name = "yolov8m_coco"
    decel_threshold = 5.0
    collision_iou = 0.15
    stationary_frames = 60
    cooldown = 3.0
else:
    model_name = st.sidebar.selectbox("Detection Model", ["yolov8m_coco", "yolov8", "yolov9", "yolov10"])
    decel_threshold = st.sidebar.slider("Deceleration Threshold (px/frame)", 1.0, 30.0, 5.0, 1.0)
    collision_iou = st.sidebar.slider("Collision IoU Threshold", 0.05, 0.8, 0.15, 0.05)
    stationary_frames = st.sidebar.slider("Stationary Alert (frames)", 30, 300, 60, 10)
    cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 1.0, 60.0, 3.0, 1.0)
    spike_factor = 3.0
    motion_cooldown_frames = 50
max_frames_ui = st.sidebar.number_input(
    "Max Frames (0 = all)", 0, 100000, 2000, 100,
    help="Cap processing length for a fast preview. 0 = whole video. "
         "Bump higher if your accident is in the middle of a long clip.",
)
frame_stride = st.sidebar.slider(
    "Frame Stride", 1, 10, 2, 1,
    help="Process every Nth frame. Higher = faster, lower fidelity.",
)

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Live Detection", "Event Log", "Statistics"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Video Feed")
        uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

        if uploaded:
            st.video(uploaded)

    with col2:
        st.subheader("Detection Types")
        st.markdown("""
        - **Sudden Stop** - Vehicle decelerates rapidly
        - **Collision** - Two vehicles overlap unexpectedly
        - **Stationary** - Vehicle stops in traffic lane
        """)

        st.markdown("### Alert Severity")
        st.markdown("""
        - 🔴 **Critical** - Confirmed collision
        - 🟠 **High** - Severe sudden stop
        - 🟡 **Medium** - Moderate event
        - ⚪ **Low** - Stationary vehicle
        """)

    if uploaded and st.button("Run Accident Detection", type="primary"):
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.getvalue())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open the uploaded video. The file may be "
                     "corrupted or use an unsupported codec.")
            st.stop()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frame_placeholder = st.empty()
        event_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_events = []
        frame_idx = 0
        frame_limit = max_frames_ui if max_frames_ui > 0 else total_frames

        if is_motion_mode:
            from src.accident_detection.motion_detector import MotionAccidentDetector

            motion_det = MotionAccidentDetector(
                spike_factor=spike_factor,
                cooldown_frames=motion_cooldown_frames,
            )
            cap = cv2.VideoCapture(tfile.name)
            while cap.isOpened() and frame_idx < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                events = motion_det.update(frame, frame_idx)
                all_events.extend(events)

                annotated = frame.copy()
                for event in events:
                    loc = event.location
                    color = (0, 0, 255) if event.severity in ("critical", "high") else (0, 165, 255)
                    cv2.circle(annotated, (int(loc[0]), int(loc[1])), 60, color, 4)
                    cv2.putText(
                        annotated, f"{event.event_type} ({event.severity})",
                        (int(loc[0]) - 100, int(loc[1]) - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

                if frame_idx % 3 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                if events:
                    event_placeholder.warning(
                        f"Event detected: {events[0].event_type} - {events[0].description}"
                    )

                progress_bar.progress(min((frame_idx + 1) / frame_limit, 1.0))
                status_text.text(
                    f"Frame {frame_idx + 1}/{frame_limit} | "
                    f"Events: {len(all_events)} | mode=motion"
                )
                frame_idx += 1
            cap.release()
        else:
            from config.settings import PipelineConfig
            from src.pipeline.video_pipeline import VideoPipeline
            from src.accident_detection.accident_detector import AccidentDetector

            try:
                config = PipelineConfig()
                config.detection.model_name = model_name
                config.detection.confidence_threshold = 0.3
                pipeline = VideoPipeline(config)
            except FileNotFoundError as exc:
                st.error(
                    f"Model weights not found for `{model_name}`: {exc}\n\n"
                    "Train or download the model into `models/` first."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Failed to initialise detection pipeline: {exc}")
                st.stop()

            detector = AccidentDetector(
                deceleration_threshold=decel_threshold,
                collision_iou_threshold=collision_iou,
                stationary_frames=stationary_frames,
                cooldown_seconds=cooldown,
            )

            cap = cv2.VideoCapture(tfile.name)
            while cap.isOpened() and frame_idx < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / video_fps
                try:
                    result = pipeline.process_frame(frame, timestamp)
                    events = detector.update(result.tracks, frame_idx)
                except Exception as exc:
                    st.error(f"Frame {frame_idx} failed: {exc}")
                    break
                all_events.extend(events)

                annotated = result.annotated_frame.copy()
                for event in events:
                    loc = event.location
                    color = (0, 0, 255) if event.severity in ("critical", "high") else (0, 165, 255)
                    cv2.circle(annotated, (int(loc[0]), int(loc[1])), 30, color, 3)
                    cv2.putText(
                        annotated, f"{event.event_type} ({event.severity})",
                        (int(loc[0]) - 50, int(loc[1]) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

                if frame_idx % 3 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                if events:
                    event_placeholder.warning(f"Event detected: {events[0].event_type} - {events[0].description}")

                progress_bar.progress(min((frame_idx + 1) / frame_limit, 1.0))
                status_text.text(
                    f"Frame {frame_idx + 1}/{frame_limit} | "
                    f"Events: {len(all_events)} | stride={frame_stride}"
                )
                frame_idx += 1
            cap.release()
            st.session_state["accident_active_tracks"] = len(detector._track_history)

        progress_bar.progress(1.0)
        status_text.text(f"Complete! {len(all_events)} event(s) detected.")
        st.session_state["accident_events"] = all_events

with tab2:
    st.subheader("Accident Event Log")

    events = st.session_state.get("accident_events", [])

    if events:
        event_data = []
        for e in events:
            event_data.append({
                "Event ID": e.event_id,
                "Type": e.event_type,
                "Severity": e.severity,
                "Frame": e.frame_number,
                "Vehicles": ", ".join([f"#{tid}" for tid in e.involved_track_ids]),
                "Confidence": round(e.confidence, 2),
                "Description": e.description,
            })
        st.dataframe(pd.DataFrame(event_data), use_container_width=True)
    else:
        st.info("No events detected yet. Run accident detection on a video first.")

with tab3:
    st.subheader("Accident Statistics")

    events = st.session_state.get("accident_events", [])

    col1, col2, col3, col4 = st.columns(4)
    collisions = len([e for e in events if e.event_type == "collision"])
    sudden_stops = len([e for e in events if e.event_type == "sudden_stop"])
    stationary = len([e for e in events if e.event_type == "stationary_vehicle"])

    with col1:
        st.metric("Total Events", len(events))
    with col2:
        st.metric("Collisions", collisions)
    with col3:
        st.metric("Sudden Stops", sudden_stops)
    with col4:
        st.metric("Stationary", stationary)

    if events:
        st.markdown("---")
        # Severity distribution
        import plotly.express as px
        severity_counts = pd.Series([e.severity for e in events]).value_counts()
        fig = px.pie(names=severity_counts.index, values=severity_counts.values, title="Event Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Vehicle Tracking** - Each vehicle is tracked with a unique ID using Deep SORT / ByteTrack
    2. **Speed Analysis** - Vehicle speed is computed from centroid displacement across frames
    3. **Sudden Stop Detection** - If speed drops from > 8 px/frame to < 1 px/frame, it's flagged
    4. **Collision Detection** - If two moving vehicle bounding boxes overlap (IoU > threshold), it's flagged
    5. **Stationary Detection** - If a vehicle remains stationary for > N frames in a traffic lane
    6. **Alert Dispatch** - Events trigger email / app notifications to authorities
    """)
