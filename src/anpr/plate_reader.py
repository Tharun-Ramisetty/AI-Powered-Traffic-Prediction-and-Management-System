"""OCR-based number plate text reader using EasyOCR."""

import re
from typing import Optional, Tuple

import cv2
import numpy as np


class PlateReader:
    """Reads text from number plate images using EasyOCR.

    Supports Indian vehicle number plate format: XX-00-XX-0000
    """

    INDIAN_PLATE_PATTERN = re.compile(
        r"^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}$"
    )

    def __init__(self, languages: list = None, gpu: bool = False):
        import easyocr
        self.reader = easyocr.Reader(
            languages or ["en"],
            gpu=gpu,
        )

    def read_plate(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Read text from a plate image. Handles single and two-line plates."""
        # Try 1: read as-is
        results = self._run_ocr(plate_image)

        # Try 2: convert to grayscale
        if not results:
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            results = self._run_ocr(gray)

        # Try 3: grayscale + CLAHE
        if not results:
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            results = self._run_ocr(enhanced)

        if not results:
            return None, 0.0

        # Sort top-to-bottom for multi-line plates
        results.sort(key=lambda r: r[0][0][1])

        full_text = " ".join([r[1] for r in results])
        avg_confidence = sum(r[2] for r in results) / len(results)

        cleaned = self._clean_plate_text(full_text)
        if cleaned:
            return cleaned, avg_confidence
        return full_text.upper().strip(), avg_confidence

    def _run_ocr(self, image: np.ndarray) -> list:
        """Run EasyOCR with allowlist for plate characters."""
        try:
            results = self.reader.readtext(
                image,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
            )
            return [r for r in results if r[2] > 0.05]
        except Exception:
            return []

    def _clean_plate_text(self, text: str) -> Optional[str]:
        """Clean and validate plate text to match Indian format."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())

        if len(cleaned) < 4:
            return None

        corrections = {
            "O": "0", "I": "1", "S": "5", "B": "8",
            "Z": "2", "G": "6",
        }

        if len(cleaned) >= 6:
            state = cleaned[:2]
            rest = cleaned[2:]

            state = state.replace("0", "O").replace("1", "I").replace("5", "S")

            district = rest[:2]
            for old, new in corrections.items():
                district = district.replace(old, new)

            series_and_num = rest[2:]
            return f"{state}{district}{series_and_num}"

        return cleaned

    def validate_indian_plate(self, plate_text: str) -> bool:
        """Check if text matches Indian vehicle number plate format."""
        if not plate_text:
            return False
        cleaned = re.sub(r"[\s-]", "", plate_text.upper())
        return bool(self.INDIAN_PLATE_PATTERN.match(cleaned))
