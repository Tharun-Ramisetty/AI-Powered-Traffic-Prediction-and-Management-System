"""YOLOv8 vehicle detector using Ultralytics."""

from typing import List, Dict

import numpy as np
from ultralytics import YOLO

from config.settings import DetectionConfig, VEHICLE_CLASSES
from .detector import BaseDetector, Detection


class YOLOv8Detector(BaseDetector):
    """Vehicle detector using YOLOv8."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.conf = config.confidence_threshold
        self.iou = config.iou_threshold
        self.img_size = config.img_size

        # Resolve device
        if config.device == "auto":
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names.get(class_id, f"class_{class_id}")

            detections.append(Detection(
                bbox=tuple(box.xyxy[0].cpu().numpy().tolist()),
                confidence=float(box.conf[0]),
                class_id=class_id,
                class_name=class_name,
            ))

        return detections

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.model.parameters())
        return {
            "name": "YOLOv8",
            "model_path": self.config.model_path,
            "params": total_params,
            "params_M": round(total_params / 1e6, 2),
            "input_size": self.img_size,
            "device": self.device,
        }
