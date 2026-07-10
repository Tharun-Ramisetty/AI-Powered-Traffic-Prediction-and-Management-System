"""Traffic density classification module."""

from enum import Enum
from typing import Optional

import pandas as pd

from config.settings import DensityConfig


class DensityLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CONGESTED = "Congested"

    @property
    def color(self):
        """Color associated with each density level (BGR)."""
        return {
            DensityLevel.LOW: (50, 255, 50),      # Bright Green
            DensityLevel.MEDIUM: (0, 255, 255),    # Yellow
            DensityLevel.HIGH: (0, 165, 255),      # Orange
            DensityLevel.CONGESTED: (0, 0, 255),   # Red
        }[self]

    @property
    def hex_color(self):
        """Hex color for dashboard display."""
        return {
            DensityLevel.LOW: "#00FF00",
            DensityLevel.MEDIUM: "#FFD700",
            DensityLevel.HIGH: "#FF8C00",
            DensityLevel.CONGESTED: "#FF0000",
        }[self]


class DensityClassifier:
    """Classifies current traffic density based on vehicle count."""

    def __init__(self, config: DensityConfig = None):
        if config is None:
            config = DensityConfig()
        self.config = config

    def classify(self, vehicle_count: int) -> DensityLevel:
        """Classify traffic density from vehicle count.

        Args:
            vehicle_count: Number of vehicles currently in frame.

        Returns:
            DensityLevel enum value.
        """
        if vehicle_count < self.config.low_threshold:
            return DensityLevel.LOW
        elif vehicle_count < self.config.medium_threshold:
            return DensityLevel.MEDIUM
        elif vehicle_count < self.config.high_threshold:
            return DensityLevel.HIGH
        else:
            return DensityLevel.CONGESTED

    def classify_with_speed(self, vehicle_count: int,
                            avg_speed: float) -> DensityLevel:
        """Enhanced classification using both count and average speed.

        Low speed + moderate count = likely congested.

        Args:
            vehicle_count: Vehicles in frame.
            avg_speed: Average vehicle speed in pixels/frame.

        Returns:
            DensityLevel enum value.
        """
        count_level = self.classify(vehicle_count)

        # Override to congested if vehicles are barely moving
        if avg_speed < 5.0 and vehicle_count >= self.config.low_threshold:
            return DensityLevel.CONGESTED

        return count_level


class AdaptiveThresholdCalibrator:
    """Auto-calibrate density thresholds from historical count data."""

    @staticmethod
    def calibrate(historical_counts: pd.Series) -> DensityConfig:
        """Compute thresholds from percentiles of historical data.

        Args:
            historical_counts: Series of vehicle counts over time.

        Returns:
            DensityConfig with calibrated thresholds.
        """
        return DensityConfig(
            low_threshold=int(historical_counts.quantile(0.25)),
            medium_threshold=int(historical_counts.quantile(0.50)),
            high_threshold=int(historical_counts.quantile(0.75)),
            use_adaptive=True,
        )
