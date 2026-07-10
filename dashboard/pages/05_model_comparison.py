"""Model Comparison page - Run real benchmarks on test video."""

import tempfile
import time

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison: YOLOv8 vs YOLOv9 vs YOLOv10")

st.markdown("""
Compare detection models by running them on a test video. This generates
**real benchmark data** instead of static numbers.
""")

# ─── Check available models ─────────────────────────────────────────────────
models_dir = Path("models")
trained_dir = Path("trained_models")

available_models = {}

# Check custom trained models
for name, subdir in [("yolov8", "yolov8"), ("yolov9", "yolov9"), ("yolov10", "yolov10")]:
    best_path = models_dir / subdir / "best.pt"
    trained_path = trained_dir / f"{name}_best.pt"
    if best_path.exists():
        available_models[name] = str(best_path)
    elif trained_path.exists():
        available_models[name] = str(trained_path)

# COCO pretrained
coco_path = Path("yolov8m.pt")
if coco_path.exists():
    available_models["yolov8m_coco"] = str(coco_path)

st.markdown(f"**Found {len(available_models)} model(s):** {', '.join(available_models.keys())}")

# ─── Load saved benchmark results if available ──────────────────────────────
benchmark_csv = Path("trained_models/model_comparison.csv")
saved_benchmarks = None
if benchmark_csv.exists():
    try:
        saved_benchmarks = pd.read_csv(benchmark_csv)
    except Exception:
        pass

# ─── Benchmark Controls ────────────────────────────────────────────────────
st.markdown("---")

# Test video selection
test_videos_dir = Path("testvideos")
test_videos = list(test_videos_dir.glob("*.mp4")) if test_videos_dir.exists() else []

col1, col2 = st.columns(2)
with col1:
    selected_models = st.multiselect(
        "Models to benchmark",
        list(available_models.keys()),
        default=list(available_models.keys()),
    )
with col2:
    if test_videos:
        video_file = st.selectbox("Test Video", [v.name for v in test_videos])
        video_path = str(test_videos_dir / video_file)
    else:
        video_path = None
        st.warning("No test videos found in `testvideos/` directory.")

max_frames = st.slider("Max Frames for Benchmark", 10, 200, 50, 10)

if st.button("Run Benchmark", type="primary") and selected_models and video_path:
    import cv2
    from ultralytics import YOLO

    results_list = []
    progress = st.progress(0)
    status = st.empty()

    for idx, model_name in enumerate(selected_models):
        status.text(f"Benchmarking {model_name}...")
        model_path = available_models[model_name]

        try:
            model = YOLO(model_path)

            cap = cv2.VideoCapture(video_path)
            frame_times = []
            total_detections = 0
            frame_count = 0

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize large frames
                h, w = frame.shape[:2]
                if max(h, w) > 1280:
                    scale = 1280 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                start = time.perf_counter()
                results = model(frame, conf=0.3, verbose=False)
                elapsed = time.perf_counter() - start
                frame_times.append(elapsed)
                total_detections += len(results[0].boxes)
                frame_count += 1

            cap.release()

            avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
            avg_detections = total_detections / max(frame_count, 1)

            # Get model info
            total_params = sum(p.numel() for p in model.model.parameters())
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

            results_list.append({
                "Model": model_name,
                "Avg FPS": round(avg_fps, 1),
                "Avg Detections/Frame": round(avg_detections, 1),
                "Total Detections": total_detections,
                "Frames Processed": frame_count,
                "Params (M)": round(total_params / 1e6, 2),
                "Size (MB)": round(model_size_mb, 1),
                "Avg Inference (ms)": round(np.mean(frame_times) * 1000, 1),
                "Classes": ", ".join(model.names.values()) if hasattr(model, 'names') else "N/A",
            })

        except Exception as e:
            st.error(f"Error benchmarking {model_name}: {e}")

        progress.progress((idx + 1) / len(selected_models))

    status.text("Benchmark complete!")

    if results_list:
        benchmark_data = pd.DataFrame(results_list)
        st.session_state["benchmark_data"] = benchmark_data

        # Save to CSV
        benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
        benchmark_data.to_csv(benchmark_csv, index=False)
        st.success(f"Results saved to `{benchmark_csv}`")

# ─── Display Results ────────────────────────────────────────────────────────
st.markdown("---")

benchmark_data = st.session_state.get("benchmark_data")
if benchmark_data is None and saved_benchmarks is not None:
    benchmark_data = saved_benchmarks

if benchmark_data is not None and len(benchmark_data) > 0:
    st.markdown("### Benchmark Results")
    st.dataframe(benchmark_data, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Speed Comparison")
        if "Avg FPS" in benchmark_data.columns:
            fig_fps = px.bar(
                benchmark_data, x="Model", y="Avg FPS",
                title="Average FPS by Model",
                color="Model",
            )
            fig_fps.update_layout(height=400)
            st.plotly_chart(fig_fps, use_container_width=True)

    with col2:
        st.markdown("### Detection Count Comparison")
        if "Avg Detections/Frame" in benchmark_data.columns:
            fig_det = px.bar(
                benchmark_data, x="Model", y="Avg Detections/Frame",
                title="Avg Detections Per Frame",
                color="Model",
            )
            fig_det.update_layout(height=400)
            st.plotly_chart(fig_det, use_container_width=True)

    # Speed vs Detections scatter
    if "Avg FPS" in benchmark_data.columns and "Avg Detections/Frame" in benchmark_data.columns:
        st.markdown("### Speed vs Detection Trade-off")
        fig_scatter = px.scatter(
            benchmark_data, x="Avg FPS", y="Avg Detections/Frame",
            size="Params (M)" if "Params (M)" in benchmark_data.columns else None,
            color="Model", size_max=40,
            title="FPS vs Detection Count (bubble size = parameters)",
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Model size comparison
    if "Size (MB)" in benchmark_data.columns:
        st.markdown("### Model Size")
        fig_size = px.bar(
            benchmark_data, x="Model", y="Size (MB)",
            title="Model File Size",
            color="Model",
        )
        fig_size.update_layout(height=300)
        st.plotly_chart(fig_size, use_container_width=True)

else:
    st.info("Click **Run Benchmark** above to generate real comparison data.")
