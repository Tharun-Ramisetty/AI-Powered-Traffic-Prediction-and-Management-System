"""Prepare time-series data for LSTM training from video processing results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_timeseries(
    output_path: str = "data/datasets/traffic_timeseries/counts_camera_01.csv",
    days: int = 60,
):
    """Generate sample time-series traffic data for LSTM training.

    Simulates realistic Indian urban traffic patterns with:
    - Morning peak (8-10 AM)
    - Lunch peak (12-2 PM)
    - Evening peak (5-8 PM)
    - Weekend reduction
    - Random noise

    Args:
        output_path: Output CSV path.
        days: Number of days to simulate.
    """
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2024-06-01 00:00",
        periods=days * 24,
        freq="h",
    )

    records = []
    for ts in timestamps:
        hour = ts.hour
        is_weekend = ts.dayofweek >= 5

        # Base traffic pattern
        if 0 <= hour < 5:
            base = 3
        elif 5 <= hour < 7:
            base = 8
        elif 7 <= hour < 10:
            base = 30  # Morning peak
        elif 10 <= hour < 12:
            base = 18
        elif 12 <= hour < 14:
            base = 22  # Lunch peak
        elif 14 <= hour < 17:
            base = 16
        elif 17 <= hour < 20:
            base = 35  # Evening peak
        elif 20 <= hour < 22:
            base = 15
        else:
            base = 8

        # Weekend adjustment
        if is_weekend:
            base = int(base * 0.6)

        # Per-class distribution
        total = max(1, base + np.random.randint(-3, 4))
        car = int(total * 0.35 + np.random.randint(-1, 2))
        two_wheeler = int(total * 0.28 + np.random.randint(-1, 2))
        auto = int(total * 0.18 + np.random.randint(-1, 2))
        bus = int(total * 0.08 + np.random.randint(0, 2))
        truck = int(total * 0.11 + np.random.randint(-1, 2))

        # Ensure non-negative
        car = max(0, car)
        two_wheeler = max(0, two_wheeler)
        auto = max(0, auto)
        bus = max(0, bus)
        truck = max(0, truck)
        total = car + two_wheeler + auto + bus + truck

        records.append({
            "timestamp": ts,
            "total": total,
            "car": car,
            "bus": bus,
            "truck": truck,
            "auto": auto,
            "two_wheeler": two_wheeler,
        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} hourly records ({days} days) -> {output_path}")


def generate_weather_data(
    output_path: str = "data/datasets/traffic_timeseries/weather_merged.csv",
    days: int = 60,
):
    """Generate sample weather data aligned with traffic data.

    Args:
        output_path: Output CSV path.
        days: Number of days.
    """
    np.random.seed(123)

    timestamps = pd.date_range(
        start="2024-06-01 00:00",
        periods=days * 24,
        freq="h",
    )

    records = []
    for ts in timestamps:
        hour = ts.hour
        month = ts.month

        # Temperature pattern (India, summer)
        base_temp = 28 + 5 * np.sin(np.pi * (hour - 6) / 12)
        temp = base_temp + np.random.normal(0, 2)

        # Humidity (higher at night)
        humidity = 60 + 15 * np.cos(np.pi * (hour - 14) / 12) + np.random.normal(0, 5)

        # Rain (monsoon season: June-September)
        rain_prob = 0.3 if 6 <= month <= 9 else 0.05
        rain = np.random.exponential(2) if np.random.random() < rain_prob else 0.0

        # Visibility
        visibility = 10.0 - (rain * 0.5) + np.random.normal(0, 0.5)

        # Wind
        wind = 5 + np.random.exponential(3)

        records.append({
            "timestamp": ts,
            "temperature": round(max(15, min(45, temp)), 1),
            "humidity": round(max(20, min(100, humidity)), 1),
            "rain_1h": round(max(0, rain), 2),
            "visibility": round(max(0.5, min(15, visibility)), 1),
            "wind_speed": round(max(0, wind), 1),
        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} weather records -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LSTM training data")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--counts-output",
                        default="data/datasets/traffic_timeseries/counts_camera_01.csv")
    parser.add_argument("--weather-output",
                        default="data/datasets/traffic_timeseries/weather_merged.csv")
    args = parser.parse_args()

    generate_sample_timeseries(args.counts_output, args.days)
    generate_weather_data(args.weather_output, args.days)
