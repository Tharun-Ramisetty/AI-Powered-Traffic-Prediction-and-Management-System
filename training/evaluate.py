"""Evaluate a trained model on the test set."""

import argparse

from ultralytics import YOLO


def evaluate(
    model_path: str = "models/yolov8/best.pt",
    data_yaml: str = "data/datasets/vehicle_detection/data.yaml",
    split: str = "test",
):
    """Evaluate a trained YOLO model.

    Args:
        model_path: Path to trained model weights.
        data_yaml: Dataset config.
        split: Dataset split to evaluate on.
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Evaluating on {split} set...")
    metrics = model.val(data=data_yaml, split=split)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@50:    {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.1f}%)")
    print(f"mAP@50-95: {metrics.box.map:.4f} ({metrics.box.map * 100:.1f}%)")
    print(f"Precision: {metrics.box.mp:.4f} ({metrics.box.mp * 100:.1f}%)")
    print(f"Recall:    {metrics.box.mr:.4f} ({metrics.box.mr * 100:.1f}%)")

    # Per-class results
    if hasattr(metrics.box, 'ap_class_index'):
        print("\nPer-Class AP@50:")
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            ap = metrics.box.ap50[i]
            print(f"  {names[cls_idx]:15s}: {ap:.4f} ({ap * 100:.1f}%)")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", default="models/yolov8/best.pt")
    parser.add_argument("--data", default="data/datasets/vehicle_detection/data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(model_path=args.model, data_yaml=args.data, split=args.split)
