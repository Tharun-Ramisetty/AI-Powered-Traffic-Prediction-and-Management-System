"""Factory for creating detector instances by model name."""

from pathlib import Path

from config.settings import DetectionConfig, MODELS_DIR, PROJECT_ROOT
from .detector import BaseDetector


def create_detector(model_name: str, config: DetectionConfig = None) -> BaseDetector:
    """Create a detector instance by model name.

    Args:
        model_name: One of "yolov8", "yolov9", "yolov10", "yolov8m_coco".
        config: Detection configuration. If None, uses defaults.

    Returns:
        A BaseDetector instance.
    """
    if config is None:
        config = DetectionConfig()

    # Override model path based on model name if using default path
    default_paths = {
        "yolov8": str(MODELS_DIR / "yolov8" / "best.pt"),
        "yolov9": str(MODELS_DIR / "yolov9" / "best.pt"),
        "yolov10": str(MODELS_DIR / "yolov10" / "best.pt"),
        "yolov8m_coco": str(PROJECT_ROOT / "yolov8m.pt"),
    }

    if model_name in default_paths and config.model_path == str(MODELS_DIR / "yolov8" / "best.pt"):
        config.model_path = default_paths[model_name]

    if model_name == "yolov8m_coco":
        from .yolov8_detector import YOLOv8Detector
        return YOLOv8Detector(config)
    elif model_name == "yolov8":
        from .yolov8_detector import YOLOv8Detector
        return YOLOv8Detector(config)
    elif model_name == "yolov9":
        from .yolov9_detector import YOLOv9Detector
        return YOLOv9Detector(config)
    elif model_name == "yolov10":
        from .yolov10_detector import YOLOv10Detector
        return YOLOv10Detector(config)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Choose from: yolov8, yolov9, yolov10, yolov8m_coco"
        )
