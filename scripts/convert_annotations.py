"""Convert annotations between formats and create train/val/test splits."""

import argparse
import json
import shutil
import random
from pathlib import Path

from tqdm import tqdm


# Class mapping for our 5 classes
CLASS_MAP = {
    "auto": 0,
    "car": 1,
    "bus": 2,
    "truck": 3,
    "two_wheeler": 4,
    # Common aliases
    "auto_rickshaw": 0,
    "auto-rickshaw": 0,
    "rickshaw": 0,
    "motorcycle": 4,
    "motorbike": 4,
    "bike": 4,
    "two-wheeler": 4,
    "van": 3,
    "lorry": 3,
}


def coco_to_yolo(
    coco_json: str,
    images_dir: str,
    output_dir: str,
):
    """Convert COCO format annotations to YOLO format.

    Args:
        coco_json: Path to COCO annotations JSON.
        images_dir: Directory containing the images.
        output_dir: Output directory for YOLO labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    # Build category mapping
    coco_cats = {cat["id"]: cat["name"].lower() for cat in coco["categories"]}

    # Build image id -> filename mapping
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    converted = 0
    for img_id, anns in tqdm(img_annotations.items(), desc="Converting"):
        img_info = images.get(img_id)
        if not img_info:
            continue

        img_w, img_h = img_info["width"], img_info["height"]
        label_file = output_dir / (Path(img_info["file_name"]).stem + ".txt")

        lines = []
        for ann in anns:
            cat_name = coco_cats.get(ann["category_id"], "")
            if cat_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[cat_name]
            x, y, w, h = ann["bbox"]  # COCO: x, y, w, h (absolute)

            # Convert to YOLO: cx, cy, w, h (normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            with open(label_file, "w") as f:
                f.write("\n".join(lines))
            converted += 1

    print(f"Converted {converted} annotation files to: {output_dir}")


def create_train_val_test_split(
    images_dir: str,
    labels_dir: str,
    output_dir: str = "data/datasets/vehicle_detection",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split annotated images into train/val/test sets.

    Args:
        images_dir: Directory with images.
        labels_dir: Directory with YOLO label files.
        output_dir: Output dataset directory.
        train_ratio: Training set fraction.
        val_ratio: Validation set fraction.
        test_ratio: Test set fraction.
        seed: Random seed.
    """
    random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Find images with corresponding labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paired = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                paired.append((img_path, label_path))

    print(f"Found {len(paired)} image-label pairs")

    # Shuffle and split
    random.shuffle(paired)
    n = len(paired)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": paired[:n_train],
        "val": paired[n_train:n_train + n_val],
        "test": paired[n_train + n_val:],
    }

    for split_name, split_data in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in tqdm(split_data, desc=f"{split_name}"):
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

        print(f"  {split_name}: {len(split_data)} samples")

    print(f"\nDataset created at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert annotations and create splits")
    subparsers = parser.add_subparsers(dest="command")

    # COCO to YOLO
    coco_parser = subparsers.add_parser("coco2yolo", help="Convert COCO to YOLO format")
    coco_parser.add_argument("--json", required=True, help="COCO JSON file")
    coco_parser.add_argument("--images", required=True, help="Images directory")
    coco_parser.add_argument("--output", default="data/processed/annotations")

    # Create split
    split_parser = subparsers.add_parser("split", help="Create train/val/test split")
    split_parser.add_argument("--images", required=True)
    split_parser.add_argument("--labels", required=True)
    split_parser.add_argument("--output", default="data/datasets/vehicle_detection")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    if args.command == "coco2yolo":
        coco_to_yolo(args.json, args.images, args.output)
    elif args.command == "split":
        create_train_val_test_split(
            args.images, args.labels, args.output,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )
    else:
        parser.print_help()
