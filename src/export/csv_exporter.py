"""CSV export for vehicle count data."""

from typing import Dict, List
import pandas as pd


class CSVExporter:
    """Exports vehicle count data to CSV files."""

    @staticmethod
    def export_counts(counts: Dict[str, int], filepath: str):
        """Export a single count snapshot to CSV."""
        df = pd.DataFrame([counts])
        df.to_csv(filepath, index=False)

    @staticmethod
    def export_timeseries(df: pd.DataFrame, filepath: str):
        """Export time-series count data to CSV."""
        df.to_csv(filepath)

    @staticmethod
    def export_crossing_log(log: List[Dict], filepath: str):
        """Export line-crossing event log to CSV."""
        df = pd.DataFrame(log)
        df.to_csv(filepath, index=False)
