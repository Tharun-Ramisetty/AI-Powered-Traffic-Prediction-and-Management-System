"""Train YOLOv8 on the custom vehicle detection dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str = "data/datasets/vehicle_detection/data.yaml",
    model_variant: str = "yolov8m.pt",
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    device: int = 0,
    patience: int = 30,
    project: str = "outputs/training",
    name: str = "yolov8_vehicle",
):
    """Train YOLOv8 model on vehicle detection dataset.

    Args:
        data_yaml: Path to dataset YAML config.
        model_variant: Pretrained model to fine-tune (yolov8n/s/m/l/x.pt).
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: GPU device index (use -1 for CPU).
        patience: Early stopping patience.
        project: Output project directory.
        name: Run name.
    """
    print(f"Loading model: {model_variant}")
    model = YOLO(model_variant)

    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10,
        project=project,
        name=name,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Training params
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
    )

    # Save best model to models directory
    best_path = Path(project) / name / "weights" / "best.pt"
    target_path = Path("models/yolov8/best.pt")
    if best_path.exists():
        import shutil
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"Best model saved to: {target_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.val(data=data_yaml, split="test")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 vehicle detector")
    parser.add_argument("--data", default="data/datasets/vehicle_detection/data.yaml")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
    )
