"""Tests for export module."""

import json
import pytest

from src.export.csv_exporter import CSVExporter
from src.export.json_exporter import JSONExporter
from src.export.pdf_report import PDFReportGenerator


class TestCSVExporter:
    def test_export_counts(self, tmp_path):
        counts = {"total": 10, "car": 5, "bus": 3, "truck": 2}
        filepath = str(tmp_path / "counts.csv")
        CSVExporter.export_counts(counts, filepath)

        import pandas as pd
        df = pd.read_csv(filepath)
        assert df["total"].iloc[0] == 10


class TestJSONExporter:
    def test_export_counts(self, tmp_path):
        counts = {"total": 10, "car": 5, "bus": 3}
        filepath = str(tmp_path / "counts.json")
        JSONExporter.export_counts(counts, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert data["counts"]["total"] == 10

    def test_export_full_report(self, tmp_path):
        filepath = str(tmp_path / "report.json")
        JSONExporter.export_full_report(
            counts={"total": 10, "car": 5},
            density="High",
            crossing_log=[{"track_id": 1, "class_name": "car", "direction": "down"}],
            filepath=filepath,
        )

        with open(filepath) as f:
            data = json.load(f)
        assert data["summary"]["total_vehicles"] == 10
        assert data["summary"]["traffic_density"] == "High"


class TestPDFReport:
    def test_generate_report(self, tmp_path):
        filepath = str(tmp_path / "report.pdf")
        gen = PDFReportGenerator()
        gen.generate_report(
            counts={"total": 25, "car": 10, "bus": 5, "truck": 4, "auto": 3, "two_wheeler": 3},
            density="High",
            duration_seconds=120.5,
            avg_fps=35.2,
            model_name="YOLOv8m",
            output_path=filepath,
        )

        import os
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
