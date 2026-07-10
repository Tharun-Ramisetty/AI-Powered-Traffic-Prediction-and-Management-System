"""Download real vehicle detection datasets from Roboflow Universe (public, free).

Available datasets (all pre-annotated with bounding boxes):
1. 'vehicles-openimages' - Large dataset with cars, buses, trucks from Open Images
2. 'vehicle-detection-epmce' - Indian traffic vehicles (auto, car, bus, truck, two-wheeler)
3. 'vehicle-count' - Vehicle counting dataset with multiple classes

Usage:
    python scripts/download_real_dataset.py --dataset vehicles
    python scripts/download_real_dataset.py --dataset indian-traffic
"""

import argparse
import os
import sys
from pathlib import Path


def download_from_roboflow(
    workspace: str,
    project: str,
    version: int,
    output_dir: str = "data/datasets/vehicle_detection",
    api_key: str = None,
):
    """Download a dataset from Roboflow Universe.

    Args:
        workspace: Roboflow workspace name.
        project: Project name.
        version: Dataset version number.
        output_dir: Output directory.
        api_key: Roboflow API key (uses env var if not provided).
    """
    from roboflow import Roboflow

    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("=" * 60)
        print("ROBOFLOW API KEY REQUIRED")
        print("=" * 60)
        print()
        print("To download real datasets, you need a free Roboflow account:")
        print("1. Go to https://app.roboflow.com/")
        print("2. Sign up (free)")
        print("3. Go to Settings -> API Keys")
        print("4. Copy your API key")
        print("5. Set it in your .env file:")
        print("   ROBOFLOW_API_KEY=your_key_here")
        print()
        print("Or pass it directly:")
        print(f"   python {sys.argv[0]} --api-key YOUR_KEY")
        print()
        return False

    print(f"Connecting to Roboflow: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(
        "yolov8",
        location=output_dir,
    )
    print(f"Dataset downloaded to: {output_dir}")
    return True


def download_ultralytics_pretrained():
    """Download pretrained YOLO models from Ultralytics."""
    from ultralytics import YOLO

    models_dir = Path("models")

    models = {
        "yolov8m.pt": "yolov8",
        "yolov9c.pt": "yolov9",
        "yolov10n.pt": "yolov10",
    }

    for model_name, folder in models.items():
        target = models_dir / folder
        target.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {model_name}...")
        try:
            model = YOLO(model_name)
            print(f"  Downloaded: {model_name}")
        except Exception as e:
            print(f"  Failed: {e}")


# ─── Curated public dataset options ─────────────────────────────────────────
DATASETS = {
    "indian-traffic": {
        "description": "Indian road traffic with auto, car, bus, truck, two-wheeler classes",
        "workspace": "roboflow-universe-projects",
        "project": "vehicle-detection-epmce",
        "version": 1,
    },
    "vehicles": {
        "description": "General vehicle detection (cars, buses, trucks, motorcycles)",
        "workspace": "roboflow-universe-projects",
        "project": "vehicles-openimages",
        "version": 1,
    },
    "traffic-counting": {
        "description": "Traffic counting dataset optimized for counting vehicles",
        "workspace": "roboflow-universe-projects",
        "project": "vehicle-count",
        "version": 1,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download real vehicle detection datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all", "pretrained"],
        default="indian-traffic",
        help="Which dataset to download",
    )
    parser.add_argument("--api-key", default=None, help="Roboflow API key")
    parser.add_argument("--output", default="data/datasets/vehicle_detection")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:20s} - {info['description']}")
        print(f"  {'pretrained':20s} - Download pretrained YOLO models only")
        return

    if args.dataset == "pretrained":
        download_ultralytics_pretrained()
        return

    if args.dataset == "all":
        for name, info in DATASETS.items():
            output = f"{args.output}_{name}"
            download_from_roboflow(
                info["workspace"], info["project"], info["version"],
                output, args.api_key,
            )
    else:
        info = DATASETS[args.dataset]
        download_from_roboflow(
            info["workspace"], info["project"], info["version"],
            args.output, args.api_key,
        )


if __name__ == "__main__":
    main()
