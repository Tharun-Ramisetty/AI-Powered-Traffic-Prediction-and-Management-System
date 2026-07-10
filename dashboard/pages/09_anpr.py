"""Dashboard page for Automatic Number Plate Recognition (ANPR) - Real detection."""

import tempfile
import streamlit as st
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.anpr.illegal_vehicle_checker import IllegalVehicleChecker
from src.db import (
    ANPRMatchRepository, FlaggedVehicleRepository, get_default_db,
)

FLAGGED_JSON_PATH = Path("data/flagged_vehicles.json")


@st.cache_resource
def _get_anpr_checker() -> IllegalVehicleChecker:
    """One checker per session, hydrated from both the DB and the legacy JSON
    so that previously-added blacklisted plates are actually matched."""
    db = get_default_db()
    flagged_repo = FlaggedVehicleRepository(db)

    if FLAGGED_JSON_PATH.exists():
        flagged_repo.import_from_json(str(FLAGGED_JSON_PATH))

    checker = IllegalVehicleChecker(database_path=str(FLAGGED_JSON_PATH))
    checker.attach_db(ANPRMatchRepository(db), flagged_repo)
    return checker

st.set_page_config(page_title="ANPR System", page_icon=":oncoming_automobile:", layout="wide")
st.title(":oncoming_automobile: Number Plate Recognition (ANPR)")
st.markdown("Detect and read vehicle number plates. Track illegal/wanted vehicles.")
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("ANPR Settings")
detector_mode = st.sidebar.selectbox("Plate Detector", ["cascade", "yolo"])
ocr_confidence = st.sidebar.slider("OCR Min Confidence", 0.1, 1.0, 0.5, 0.05)
show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=False)

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Plate Detection", "Flagged Vehicles Database", "Detection Log"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Image / Video")
        upload_type = st.radio("Input Type", ["Image", "Video"], horizontal=True)

        if upload_type == "Image":
            uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded:
                # Read image
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

                if st.button("Detect Plates", type="primary"):
                    from src.anpr.plate_detector import PlateDetector
                    from src.anpr.plate_reader import PlateReader

                    with st.spinner("Detecting plates..."):
                        plate_detector = PlateDetector(mode=detector_mode)

                        # Single-pass: detect plates and read text together
                        try:
                            all_results = plate_detector.detect_and_read(img)
                        except Exception as e:
                            st.error(f"Detection error: {e}")
                            all_results = []

                        # Clean up text using PlateReader's formatting
                        try:
                            reader = PlateReader(gpu=False)
                            for r in all_results:
                                if r.get("text"):
                                    cleaned = reader._clean_plate_text(r["text"])
                                    if cleaned:
                                        r["text"] = cleaned
                        except Exception:
                            pass

                        # Filter by confidence
                        readable = [r for r in all_results if r.get("text") and r.get("confidence", 0) >= ocr_confidence]

                        # Fallback to best result
                        if not readable and all_results:
                            best = max(all_results, key=lambda r: r.get("confidence", 0))
                            if best.get("text"):
                                readable = [best]

                        # Debug output
                        if show_preprocessing:
                            for idx, r in enumerate(all_results):
                                st.write(f"Plate {idx+1}: text='{r.get('text')}', conf={r.get('confidence', 0):.2f}, bbox={r.get('bbox')}")
                                if r.get("plate_img") is not None:
                                    st.image(r["plate_img"], caption=f"Plate {idx+1} crop", width=300)

                        if readable:
                            st.success(f"Found {len(readable)} plate(s)!")
                        elif all_results:
                            st.warning("Plates detected but text could not be read. Try a clearer image.")

                        detection_results = []
                        annotated = img.copy()

                        for i, result in enumerate(readable):
                            x1, y1, x2, y2 = result["bbox"]
                            plate_text = result.get("text") or "Unreadable"
                            confidence = result.get("confidence", 0)

                            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (50, 255, 50), 2)
                            cv2.putText(annotated, f"{plate_text} ({confidence:.2f})",
                                      (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

                            detection_results.append({
                                "Plate #": i + 1,
                                "Text": plate_text,
                                "Confidence": round(confidence, 2),
                                "BBox": f"({x1},{y1},{x2},{y2})",
                            })

                            # Check against flagged database (hydrated checker)
                            if plate_text and plate_text not in ("N/A", "Unreadable"):
                                try:
                                    checker = _get_anpr_checker()
                                    match = checker.check_plate(plate_text)
                                    if match.is_flagged:
                                        reason = match.record.reason if match.record else "N/A"
                                        st.error(f"ALERT: Plate {plate_text} is FLAGGED! Reason: {reason}")
                                        detection_results[-1]["Flagged"] = True
                                        detection_results[-1]["Reason"] = reason
                                except Exception as exc:
                                    st.warning(f"Blacklist check failed: {exc}")

                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption="Detected Plates", use_container_width=True)
                        if detection_results:
                            st.dataframe(pd.DataFrame(detection_results), use_container_width=True)

                        # Store in session for log
                        if "anpr_log" not in st.session_state:
                            st.session_state["anpr_log"] = []
                        st.session_state["anpr_log"].extend(detection_results)

                        if not readable and not all_results:
                            st.warning("No plates detected in this image. Try adjusting the image or detector mode.")

        else:
            uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
            if uploaded:
                st.video(uploaded)

                vid_col_a, vid_col_b = st.columns(2)
                with vid_col_a:
                    frame_stride = st.number_input(
                        "Process every Nth frame", 1, 120, 30, 1,
                        help="Higher = faster but may miss short-lived plates.",
                    )
                with vid_col_b:
                    max_frames_to_scan = st.number_input(
                        "Max frames to scan", 30, 5000, 600, 30,
                        help="Caps total work to keep CPU runs interactive.",
                    )

                if st.button("Run ANPR on Video", type="primary"):
                    from src.anpr.plate_detector import PlateDetector
                    from src.anpr.plate_reader import PlateReader

                    # cv2.VideoCapture needs a real path, so persist the upload.
                    tfile = tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded.name).suffix or ".mp4",
                    )
                    tfile.write(uploaded.read())
                    tfile.flush()

                    detector = PlateDetector(mode=detector_mode)
                    try:
                        reader = PlateReader(gpu=False)
                    except Exception:
                        reader = None
                    try:
                        checker = _get_anpr_checker()
                    except Exception:
                        checker = None

                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("Could not open uploaded video. Try a different format.")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(max_frames_to_scan)
                        max_scan = min(total_frames, int(max_frames_to_scan))

                        # Dedup by cleaned plate text — same car across many
                        # frames should count once. Keep the highest-confidence
                        # detection per plate.
                        seen = {}
                        progress = st.progress(0)
                        status = st.empty()
                        frame_idx = 0
                        scanned = 0

                        while frame_idx < max_scan:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_idx % int(frame_stride) != 0:
                                frame_idx += 1
                                continue

                            try:
                                results = detector.detect_and_read(frame)
                            except Exception as exc:
                                st.warning(f"Frame {frame_idx} failed: {exc}")
                                frame_idx += 1
                                scanned += 1
                                continue

                            for r in results:
                                text = r.get("text", "")
                                conf = r.get("confidence", 0)
                                if not text or conf < ocr_confidence:
                                    continue

                                cleaned = reader._clean_plate_text(text) if reader else text
                                if not cleaned:
                                    continue

                                existing = seen.get(cleaned)
                                if existing and existing["confidence"] >= conf:
                                    continue

                                entry = {
                                    "text": cleaned,
                                    "confidence": float(conf),
                                    "first_seen_sec": frame_idx / fps,
                                    "frame_idx": frame_idx,
                                    "plate_img": r.get("plate_img"),
                                    "flagged": False,
                                    "reason": "",
                                }

                                if checker:
                                    try:
                                        match = checker.check_plate(cleaned)
                                        if match.is_flagged:
                                            entry["flagged"] = True
                                            entry["reason"] = (
                                                match.record.reason if match.record else "N/A"
                                            )
                                    except Exception:
                                        pass

                                seen[cleaned] = entry

                            scanned += 1
                            progress.progress(min(1.0, frame_idx / max(max_scan, 1)))
                            status.text(
                                f"Scanned frame {frame_idx}/{max_scan} — "
                                f"{len(seen)} unique plate(s) so far"
                            )
                            frame_idx += 1

                        cap.release()
                        progress.progress(1.0)
                        status.text(
                            f"Done. Scanned {scanned} frame(s) at stride "
                            f"{frame_stride}, found {len(seen)} unique plate(s)."
                        )

                        if not seen:
                            st.warning(
                                "No plates detected. Try lowering Frame Stride "
                                "or OCR Min Confidence."
                            )
                        else:
                            ordered = sorted(seen.values(), key=lambda v: v["first_seen_sec"])

                            for v in ordered:
                                if v["flagged"]:
                                    st.error(
                                        f"ALERT: Plate {v['text']} is FLAGGED! "
                                        f"Reason: {v['reason']}"
                                    )

                            st.success(f"Found {len(seen)} unique plate(s).")

                            df = pd.DataFrame([
                                {
                                    "Plate": v["text"],
                                    "Confidence": round(v["confidence"], 2),
                                    "First Seen (s)": round(v["first_seen_sec"], 2),
                                    "Frame": v["frame_idx"],
                                    "Flagged": "Yes" if v["flagged"] else "",
                                    "Reason": v["reason"],
                                }
                                for v in ordered
                            ])
                            st.dataframe(df, use_container_width=True)

                            st.markdown("### Plate thumbnails")
                            thumb_cols = st.columns(min(4, len(ordered)))
                            for i, v in enumerate(ordered):
                                with thumb_cols[i % len(thumb_cols)]:
                                    if v["plate_img"] is not None:
                                        st.image(
                                            cv2.cvtColor(v["plate_img"], cv2.COLOR_BGR2RGB),
                                            caption=f"{v['text']} @ {v['first_seen_sec']:.1f}s",
                                            use_container_width=True,
                                        )

                            # Mirror image-flow behaviour: feed the right-hand
                            # log + Detection Log tab.
                            st.session_state.setdefault("anpr_log", [])
                            st.session_state["anpr_log"].extend([
                                {
                                    "Plate #": i + 1,
                                    "Text": v["text"],
                                    "Confidence": round(v["confidence"], 2),
                                    "BBox": f"frame {v['frame_idx']}",
                                    "Flagged": v["flagged"],
                                    "Reason": v["reason"],
                                }
                                for i, v in enumerate(ordered)
                            ])

    with col2:
        st.subheader("Detection Results")
        if "anpr_log" in st.session_state and st.session_state["anpr_log"]:
            st.dataframe(pd.DataFrame(st.session_state["anpr_log"]), use_container_width=True)
        else:
            st.info("Upload an image and click **Detect Plates** to see results.")

        st.markdown("### Supported Formats")
        st.markdown("""
        **Indian Number Plate Format:**
        - `KA 01 AB 1234` (Karnataka)
        - `MH 02 CD 5678` (Maharashtra)
        - `TN 03 EF 9012` (Tamil Nadu)

        **Format:** `[State][District][Series][Number]`
        """)

with tab2:
    st.subheader("Flagged Vehicles Database")

    # Load database
    db_path = Path("data/flagged_vehicles.json")
    if db_path.exists():
        with open(db_path, "r") as f:
            data = json.load(f)

        vehicles = data.get("flagged_vehicles", [])
        if vehicles:
            df = pd.DataFrame(vehicles)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No flagged vehicles in database.")
    else:
        st.warning("Database file not found. Add a vehicle below to create it.")

    st.markdown("---")
    st.subheader("Add Flagged Vehicle")

    with st.form("add_vehicle"):
        col1, col2 = st.columns(2)
        with col1:
            plate = st.text_input("Plate Number", placeholder="KA01AB1234")
            reason = st.selectbox("Reason", ["stolen", "wanted", "blacklisted", "expired_registration"])
            owner = st.text_input("Owner Name (optional)")
        with col2:
            v_type = st.selectbox("Vehicle Type", ["Car", "Bike", "Bus", "Truck", "Auto", "Scooty"])
            priority = st.selectbox("Priority", ["critical", "high", "medium", "low"])

        submitted = st.form_submit_button("Add to Database")
        if submitted and plate:
            new_entry = {
                "plate_number": plate.upper().replace(" ", "").replace("-", ""),
                "reason": reason,
                "reported_date": str(pd.Timestamp.now().date()),
                "owner_name": owner,
                "vehicle_type": v_type,
                "priority": priority,
            }

            if db_path.exists():
                with open(db_path, "r") as f:
                    data = json.load(f)
            else:
                data = {"flagged_vehicles": []}

            data["flagged_vehicles"].append(new_entry)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(db_path, "w") as f:
                json.dump(data, f, indent=2)

            FlaggedVehicleRepository(get_default_db()).upsert(
                plate_number=new_entry["plate_number"],
                reason=new_entry["reason"],
                reported_date=new_entry["reported_date"],
                owner_name=new_entry["owner_name"],
                vehicle_type=new_entry["vehicle_type"],
                priority=new_entry["priority"],
            )
            _get_anpr_checker.clear()  # type: ignore[attr-defined]

            st.success(f"Added {plate} to flagged vehicles database!")
            st.rerun()

with tab3:
    st.subheader("Detection Log")

    anpr_log = st.session_state.get("anpr_log", [])
    if anpr_log:
        st.dataframe(pd.DataFrame(anpr_log), use_container_width=True)
        if st.button("Clear Log"):
            st.session_state["anpr_log"] = []
            st.rerun()
    else:
        st.info("Plate detections will be logged here after running detection.")
