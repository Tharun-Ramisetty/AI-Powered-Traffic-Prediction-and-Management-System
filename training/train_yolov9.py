"""Train YOLOv9 on the custom vehicle detection dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str = "data/datasets/vehicle_detection/data.yaml",
    model_variant: str = "yolov9c.pt",
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    device: int = 0,
    patience: int = 30,
):
    print(f"Loading YOLOv9 model: {model_variant}")
    model = YOLO(model_variant)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10,
        project="outputs/training",
        name="yolov9_vehicle",
        optimizer="Adam",
        lr0=0.001,
    )

    best_path = Path("outputs/training/yolov9_vehicle/weights/best.pt")
    target_path = Path("models/yolov9/best.pt")
    if best_path.exists():
        import shutil
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"Best model saved to: {target_path}")

    print("\nEvaluating on test set...")
    metrics = model.val(data=data_yaml, split="test")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv9 vehicle detector")
    parser.add_argument("--data", default="data/datasets/vehicle_detection/data.yaml")
    parser.add_argument("--model", default="yolov9c.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    train(data_yaml=args.data, model_variant=args.model,
          epochs=args.epochs, batch=args.batch, device=args.device)
