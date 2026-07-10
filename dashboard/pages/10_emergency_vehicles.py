"""Dashboard page for Emergency Vehicle Detection & Signal Control - Real detection."""

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

st.set_page_config(page_title="Emergency Vehicle Detection", page_icon=":ambulance:", layout="wide")
st.title(":ambulance: Emergency Vehicle Detection")
st.markdown("Detect ambulances, fire trucks, and police vehicles. Auto-control traffic signals.")
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Emergency Detection Settings")
model_name = st.sidebar.selectbox("Detection Model", ["yolov8m_coco", "yolov8", "yolov9", "yolov10"],
                                   key="emg_model")
color_threshold = st.sidebar.slider("Color Detection Threshold", 0.05, 0.40, 0.15, 0.05)
min_area = st.sidebar.slider("Min Vehicle Area (px)", 1000, 15000, 5000, 500)
use_ocr = st.sidebar.checkbox("Enable Text Detection (OCR)", value=False)
green_duration = st.sidebar.slider("Emergency Green Duration (sec)", 10, 60, 30, 5)

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Live Detection", "Signal Control", "Event Log"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Video Feed")
        uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"], key="emg_upload")
        if uploaded:
            st.video(uploaded)

    with col2:
        st.subheader("Detection Methods")
        st.markdown("""
        #### 1. Color Analysis
        - 🔴 **Red dominant** -> Fire Truck
        - 🔵 **Blue dominant** -> Police
        - ⚪ **White + Red** -> Ambulance

        #### 2. Text Detection (OCR)
        - "AMBULANCE" / "108"
        - "FIRE" / "101" / "RESCUE"
        - "POLICE" / "100"
        """)

    if uploaded and st.button("Run Emergency Detection", type="primary"):
        from config.settings import PipelineConfig, DetectionConfig
        from src.pipeline.video_pipeline import VideoPipeline
        from src.emergency_detection.emergency_detector import EmergencyVehicleDetector
        from src.emergency_detection.signal_controller import SignalController

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.getvalue())
        tfile.flush()

        # Initialize pipeline
        config = PipelineConfig()
        config.detection.model_name = model_name
        config.detection.confidence_threshold = 0.3
        pipeline = VideoPipeline(config)

        # Initialize emergency detector
        emg_detector = EmergencyVehicleDetector(
            color_threshold=color_threshold,
            min_vehicle_area=min_area,
            use_ocr=use_ocr,
        )

        # Initialize signal controller
        signal_ctrl = SignalController(green_duration=green_duration)

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frame_placeholder = st.empty()
        signal_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_events = []
        frame_idx = 0
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process_frame(frame, frame_idx / video_fps)

            # Run emergency detection on detections
            events = emg_detector.detect(result.frame, result.detections, frame_idx)
            all_events.extend(events)

            # Handle signal control for detected emergency vehicles
            for event in events:
                directions = ["north", "south", "east", "west"]
                # Estimate direction from position in frame
                h, w = result.frame.shape[:2]
                cx = event.location[0]
                if cx < w * 0.25:
                    approach_dir = "west"
                elif cx > w * 0.75:
                    approach_dir = "east"
                elif event.location[1] < h * 0.5:
                    approach_dir = "north"
                else:
                    approach_dir = "south"
                signal_ctrl.handle_emergency(event, approach_dir)

            signal_ctrl.update()

            # Annotate emergency detections
            annotated = result.annotated_frame.copy()
            for event in events:
                x1, y1, x2, y2 = [int(c) for c in event.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"EMERGENCY: {event.vehicle_type}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if frame_idx % 3 == 0:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # Show signal states
                states = signal_ctrl.get_signal_states()
                signal_info = " | ".join([
                    f"{s['direction'].upper()}: {s['state'].upper()}"
                    + (" [EMERGENCY]" if s['is_emergency'] else "")
                    for s in states.values()
                ])
                signal_placeholder.info(f"Signals: {signal_info}")

            progress_bar.progress((frame_idx + 1) / total_frames)
            status_text.text(f"Frame {frame_idx + 1}/{total_frames} | Emergency Events: {len(all_events)}")
            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"Complete! {len(all_events)} emergency event(s) detected.")

        # Store results
        st.session_state["emergency_events"] = all_events
        st.session_state["signal_controller"] = signal_ctrl

with tab2:
    st.subheader("Traffic Signal Control Panel")

    signal_ctrl = st.session_state.get("signal_controller")

    if signal_ctrl:
        st.markdown("### Current Signal States")
        states = signal_ctrl.get_signal_states()

        sig_cols = st.columns(4)
        signal_colors = {"red": "🔴", "yellow": "🟡", "green": "🟢"}

        for i, (sig_id, state) in enumerate(states.items()):
            with sig_cols[i]:
                icon = signal_colors.get(state["state"], "⚪")
                emg_tag = " [OVERRIDE]" if state["is_emergency"] else ""
                st.markdown(f"### {state['direction'].upper()} {icon} **{state['state'].upper()}**{emg_tag}")
                if state["override_remaining"] > 0:
                    st.caption(f"Override expires in {state['override_remaining']:.0f}s")

        st.markdown("---")
        st.metric("Emergency Mode", "Active" if signal_ctrl.is_in_emergency_mode else "Normal")

        # Signal change log
        change_log = signal_ctrl.get_change_log()
        if change_log:
            st.markdown("### Signal Change Log")
            log_data = [{
                "Time": datetime.fromtimestamp(log.timestamp).strftime("%H:%M:%S"),
                "Signal": log.signal_id,
                "Old State": log.old_state,
                "New State": log.new_state,
                "Reason": log.reason,
            } for log in change_log]
            st.dataframe(pd.DataFrame(log_data), use_container_width=True)
    else:
        st.info("Run emergency detection first to see signal control states.")

    st.markdown("---")
    st.markdown("### Emergency Override Rules")
    st.markdown("""
    When an emergency vehicle is detected approaching:
    1. Signal in the vehicle's direction -> **GREEN**
    2. All other signals -> **RED**
    3. After vehicle passes or timeout -> **Revert to normal cycle**
    """)

with tab3:
    st.subheader("Emergency Vehicle Event Log")

    events = st.session_state.get("emergency_events", [])

    if events:
        event_data = [{
            "Event ID": e.event_id,
            "Type": e.vehicle_type,
            "Frame": e.frame_number,
            "Confidence": round(e.confidence, 2),
            "Location": f"({e.location[0]:.0f}, {e.location[1]:.0f})",
        } for e in events]
        st.dataframe(pd.DataFrame(event_data), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        ambulances = len([e for e in events if e.vehicle_type == "ambulance"])
        fire_trucks = len([e for e in events if e.vehicle_type == "fire_truck"])
        police = len([e for e in events if e.vehicle_type == "police"])
        with col1:
            st.metric("Ambulances", ambulances)
        with col2:
            st.metric("Fire Trucks", fire_trucks)
        with col3:
            st.metric("Police", police)
    else:
        st.info("Run emergency detection on a video to see events here.")
