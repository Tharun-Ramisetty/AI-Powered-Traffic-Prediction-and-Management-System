"""Benchmark all detection models for comparison."""

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


def measure_fps(model, video_path: str, num_frames: int = 200) -> float:
    """Measure average inference FPS on a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    times = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        model(frame, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        count += 1

    cap.release()

    if not times:
        return 0.0
    return 1.0 / (sum(times) / len(times))


def benchmark(
    data_yaml: str = "data/datasets/vehicle_detection/data.yaml",
    test_video: str = None,
    output_csv: str = "outputs/benchmarks/model_comparison.csv",
):
    """Run benchmarks across all YOLO model variants.

    Args:
        data_yaml: Dataset config for accuracy evaluation.
        test_video: Video file for FPS measurement.
        output_csv: Output CSV path for results.
    """
    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
        "YOLOv9c": "yolov9c.pt",
        "YOLOv10n": "yolov10n.pt",
        "YOLOv10s": "yolov10s.pt",
    }

    # Also check for custom-trained models
    custom_models = {
        "YOLOv8m-custom": "models/yolov8/best.pt",
        "YOLOv9c-custom": "models/yolov9/best.pt",
        "YOLOv10n-custom": "models/yolov10/best.pt",
    }
    for name, path in custom_models.items():
        if Path(path).exists():
            models[name] = path

    results = []

    for model_name, weights in models.items():
        print(f"\nBenchmarking: {model_name}")
        try:
            model = YOLO(weights)
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue

        # Model info
        params = sum(p.numel() for p in model.model.parameters())
        try:
            import os
            size_mb = os.path.getsize(weights) / 1e6
        except OSError:
            size_mb = 0

        # Accuracy metrics
        try:
            metrics = model.val(data=data_yaml, split="test", verbose=False)
            map50 = round(metrics.box.map50 * 100, 1)
            map50_95 = round(metrics.box.map * 100, 1)
            precision = round(metrics.box.mp * 100, 1)
            recall = round(metrics.box.mr * 100, 1)
        except Exception as e:
            print(f"  Validation failed: {e}")
            map50 = map50_95 = precision = recall = 0.0

        # Speed metrics
        fps = 0.0
        if test_video and Path(test_video).exists():
            fps = round(measure_fps(model, test_video), 1)

        result = {
            "Model": model_name,
            "mAP@50": map50,
            "mAP@50-95": map50_95,
            "Precision": precision,
            "Recall": recall,
            "FPS": fps,
            "Params (M)": round(params / 1e6, 1),
            "Size (MB)": round(size_mb, 1),
        }
        results.append(result)
        print(f"  mAP@50={map50}, FPS={fps}, Params={result['Params (M)']}M")

    # Save results
    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO models")
    parser.add_argument("--data", default="data/datasets/vehicle_detection/data.yaml")
    parser.add_argument("--video", default=None, help="Video file for FPS test")
    parser.add_argument("--output", default="outputs/benchmarks/model_comparison.csv")
    args = parser.parse_args()

    benchmark(data_yaml=args.data, test_video=args.video, output_csv=args.output)
