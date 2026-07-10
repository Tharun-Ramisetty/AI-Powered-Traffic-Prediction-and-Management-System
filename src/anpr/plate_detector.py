"""Number plate detection using YOLOv8, OpenCV cascade, or OCR-based detection."""

import re
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


class PlateDetector:
    """Detects number plates in vehicle images/frames."""

    def __init__(
        self,
        mode: str = "cascade",
        model_path: Optional[str] = None,
        confidence: float = 0.5,
        cascade_path: Optional[str] = None,
    ):
        self.mode = mode
        self.confidence = confidence
        self._ocr_reader = None

        if mode == "yolo" and model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        else:
            self.mode = "cascade"
            if cascade_path:
                self.cascade = cv2.CascadeClassifier(cascade_path)
            else:
                cascade_file = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
                self.cascade = cv2.CascadeClassifier(cascade_file)

    def detect_and_read(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect plates and read text in a single pass.

        Returns list of dicts with keys: bbox, text, confidence, plate_img
        """
        h, w = frame.shape[:2]

        # Always run OCR on full image — most reliable for Indian plates
        ocr_results = self._ocr_full_image(frame)

        if ocr_results:
            # Add cropped plate images
            for r in ocr_results:
                r["plate_img"] = self._extract_region(frame, r["bbox"])
            return ocr_results

        # Fallback: try cascade detection + OCR on crops
        if self.mode == "cascade":
            cascade_plates = self._detect_cascade(frame)
            if cascade_plates:
                results = []
                for bbox in cascade_plates:
                    crop = self._extract_region(frame, bbox)
                    text, conf = self._ocr_crop(crop)
                    results.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": conf,
                        "plate_img": crop,
                    })
                return results

        return []

    def detect_plates(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Legacy method — returns only bboxes."""
        results = self.detect_and_read(frame)
        return [r["bbox"] for r in results]

    def _ocr_full_image(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run EasyOCR on full image, find and merge plate-like text."""
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=False)

        try:
            results = self._ocr_reader.readtext(frame)
        except Exception:
            return []

        if not results:
            return []

        # Collect all text detections with their positions
        detections = []
        for bbox_pts, text, conf in results:
            if conf < 0.1:
                continue
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text)
            if len(cleaned) < 2:
                continue

            pts = np.array(bbox_pts, dtype=np.int32)
            x1, y1 = pts.min(axis=0).tolist()
            x2, y2 = pts.max(axis=0).tolist()

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "text": text.strip(),
                "conf": conf,
            })

        if not detections:
            return []

        # Merge vertically close text lines into plate groups
        merged = self._merge_into_plates(detections)
        return merged

    def _merge_into_plates(self, detections: list) -> List[Dict[str, Any]]:
        """Merge text detections that form multi-line plates."""
        detections.sort(key=lambda d: d["bbox"][1])

        merged = []
        used = set()

        for i, d1 in enumerate(detections):
            if i in used:
                continue

            x1, y1, x2, y2 = d1["bbox"]
            texts = [d1["text"]]
            confs = [d1["conf"]]

            for j, d2 in enumerate(detections):
                if j <= i or j in used:
                    continue

                bx1, by1, bx2, by2 = d2["bbox"]

                # Check vertical proximity
                vertical_gap = by1 - y2
                line_height = y2 - y1
                if vertical_gap < -5 or vertical_gap > line_height * 2.0:
                    continue

                # Check horizontal overlap
                overlap_left = max(x1, bx1)
                overlap_right = min(x2, bx2)
                min_width = min(x2 - x1, bx2 - bx1)
                if min_width > 0 and (overlap_right - overlap_left) < min_width * 0.2:
                    continue

                # Merge this line into the plate
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)
                texts.append(d2["text"])
                confs.append(d2["conf"])
                used.add(j)

            used.add(i)

            combined_text = " ".join(texts)
            avg_conf = sum(confs) / len(confs)

            # Only keep if it looks like a plate (has both letters and digits)
            alpha_num = re.sub(r"[^A-Za-z0-9]", "", combined_text)
            has_letters = bool(re.search(r"[A-Za-z]", alpha_num))
            has_digits = bool(re.search(r"[0-9]", alpha_num))

            if has_letters and has_digits and len(alpha_num) >= 4:
                merged.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": combined_text,
                    "confidence": avg_conf,
                })

        return merged

    def _detect_cascade(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        h, w = gray.shape[:2]
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(30, w // 20), max(10, h // 30)),
            maxSize=(min(w // 2, 600), min(h // 3, 200)),
        )

        plates = []
        for (x, y, bw, bh) in detections:
            aspect = bw / bh if bh > 0 else 0
            if 1.5 <= aspect <= 6.0:
                plates.append((x, y, x + bw, y + bh))
        return plates

    def _ocr_crop(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Run OCR on a cropped plate region."""
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=False)

        try:
            results = self._ocr_reader.readtext(crop)
            if results:
                results.sort(key=lambda r: r[0][0][1])
                text = " ".join(r[1] for r in results)
                conf = sum(r[2] for r in results) / len(results)
                return text.strip(), conf
        except Exception:
            pass
        return None, 0.0

    def _extract_region(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        """Extract and scale up a plate region."""
        h, w = frame.shape[:2]
        x1 = int(max(0, bbox[0] - padding))
        y1 = int(max(0, bbox[1] - padding))
        x2 = int(min(w, bbox[2] + padding))
        y2 = int(min(h, bbox[3] + padding))

        plate_img = frame[y1:y2, x1:x2].copy()

        if plate_img.size == 0:
            return np.zeros((100, 300, 3), dtype=np.uint8)

        ph, pw = plate_img.shape[:2]
        scale = max(1.0, 300.0 / pw)
        new_w = int(pw * scale)
        new_h = int(ph * scale)
        plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return plate_img

    # Keep for backward compatibility
    def extract_plate_region(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        return self._extract_region(frame, bbox, padding)
