"""Tests for detection module data structures."""

import pytest
from src.detection.detector import Detection


class TestDetection:
    def test_centroid(self):
        d = Detection(bbox=(100, 200, 200, 300), confidence=0.9,
                      class_id=1, class_name="car")
        assert d.centroid == (150.0, 250.0)

    def test_width_height(self):
        d = Detection(bbox=(100, 200, 300, 400), confidence=0.9,
                      class_id=1, class_name="car")
        assert d.width == 200.0
        assert d.height == 200.0

    def test_area(self):
        d = Detection(bbox=(0, 0, 100, 50), confidence=0.9,
                      class_id=1, class_name="car")
        assert d.area == 5000.0

    def test_confidence_range(self, sample_detections):
        for d in sample_detections:
            assert 0.0 <= d.confidence <= 1.0

    def test_bbox_format(self, sample_detections):
        for d in sample_detections:
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1
            assert y2 > y1
