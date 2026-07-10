"""Tests for density classification module."""

import pytest
from config.settings import DensityConfig
from src.classification.density_classifier import (
    DensityClassifier, DensityLevel, AdaptiveThresholdCalibrator,
)


class TestDensityClassifier:
    def setup_method(self):
        self.config = DensityConfig(
            low_threshold=5, medium_threshold=15, high_threshold=30,
        )
        self.clf = DensityClassifier(self.config)

    def test_low_density(self):
        assert self.clf.classify(0) == DensityLevel.LOW
        assert self.clf.classify(3) == DensityLevel.LOW
        assert self.clf.classify(4) == DensityLevel.LOW

    def test_medium_density(self):
        assert self.clf.classify(5) == DensityLevel.MEDIUM
        assert self.clf.classify(10) == DensityLevel.MEDIUM
        assert self.clf.classify(14) == DensityLevel.MEDIUM

    def test_high_density(self):
        assert self.clf.classify(15) == DensityLevel.HIGH
        assert self.clf.classify(20) == DensityLevel.HIGH
        assert self.clf.classify(29) == DensityLevel.HIGH

    def test_congested(self):
        assert self.clf.classify(30) == DensityLevel.CONGESTED
        assert self.clf.classify(50) == DensityLevel.CONGESTED
        assert self.clf.classify(100) == DensityLevel.CONGESTED

    def test_speed_override_to_congested(self):
        """Low speed with medium count should be classified as congested."""
        result = self.clf.classify_with_speed(vehicle_count=10, avg_speed=3.0)
        assert result == DensityLevel.CONGESTED

    def test_speed_normal_keeps_classification(self):
        """Normal speed should not override count-based classification."""
        result = self.clf.classify_with_speed(vehicle_count=10, avg_speed=20.0)
        assert result == DensityLevel.MEDIUM

    def test_density_level_has_color(self):
        for level in DensityLevel:
            assert level.color is not None
            assert level.hex_color is not None


class TestAdaptiveCalibrator:
    def test_calibrate_from_data(self):
        import pandas as pd
        data = pd.Series([1, 2, 5, 8, 10, 15, 20, 25, 30, 40])
        config = AdaptiveThresholdCalibrator.calibrate(data)

        assert config.low_threshold > 0
        assert config.medium_threshold > config.low_threshold
        assert config.high_threshold > config.medium_threshold
        assert config.use_adaptive is True
