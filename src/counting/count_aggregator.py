"""Time-windowed count aggregation for analytics and LSTM training data."""

from typing import Dict, List, Optional

import pandas as pd


class CountAggregator:
    """Aggregates raw per-frame counts into time-windowed summaries.

    Collects vehicle count snapshots over time and provides
    resampled DataFrames suitable for charting and LSTM training.
    """

    def __init__(self, window_seconds: int = 300):
        """
        Args:
            window_seconds: Aggregation window size (default 300 = 5 minutes).
        """
        self.window_seconds = window_seconds
        self.records: List[Dict] = []

    def record(self, timestamp: float, counts: Dict[str, int]):
        """Record a count snapshot at a given timestamp.

        Args:
            timestamp: Unix timestamp (seconds).
            counts: Dict of class_name -> count including "total".
        """
        self.records.append({"timestamp": timestamp, **counts})

    def get_dataframe(self) -> pd.DataFrame:
        """Get raw records as a DataFrame."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df.set_index("timestamp").fillna(0).astype(int)

    def get_windowed_counts(self) -> pd.DataFrame:
        """Get counts resampled into time windows.

        Returns:
            DataFrame with one row per window, columns are vehicle classes + total.
        """
        df = self.get_dataframe()
        if df.empty:
            return df
        return df.resample(f"{self.window_seconds}s").last().ffill()

    def get_current_rate(self, window_seconds: int = 60) -> Optional[Dict[str, float]]:
        """Get vehicle count rate (vehicles/minute) over recent window.

        Args:
            window_seconds: How far back to look.

        Returns:
            Dict of class_name -> vehicles per minute, or None if no data.
        """
        if not self.records:
            return None

        latest_ts = self.records[-1]["timestamp"]
        cutoff = latest_ts - window_seconds

        recent = [r for r in self.records if r["timestamp"] >= cutoff]
        if len(recent) < 2:
            return None

        first = recent[0]
        last = recent[-1]
        elapsed_min = (last["timestamp"] - first["timestamp"]) / 60.0

        if elapsed_min <= 0:
            return None

        rates = {}
        for key in last:
            if key == "timestamp":
                continue
            diff = last.get(key, 0) - first.get(key, 0)
            rates[key] = round(diff / elapsed_min, 1)

        return rates

    def export_csv(self, path: str, windowed: bool = True):
        """Export counts to CSV file.

        Args:
            path: Output file path.
            windowed: If True, export windowed counts; else raw records.
        """
        df = self.get_windowed_counts() if windowed else self.get_dataframe()
        df.to_csv(path)

    def reset(self):
        self.records.clear()
