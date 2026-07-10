"""Tests for vehicle counting module."""

import pytest
from src.counting.line_counter import LineCrossingCounter
from src.counting.zone_counter import ZoneCounter
from src.counting.count_aggregator import CountAggregator
from src.tracking.tracker import Track


class TestLineCrossingCounter:
    def test_counts_vehicle_exactly_once(self):
        """A vehicle crossing the line should be counted exactly once."""
        counter = LineCrossingCounter(line_y_fraction=0.5, direction="both")
        frame_height = 480
        line_y = 240  # 0.5 * 480

        # Simulate track moving from y=100 to y=300 (crossing at y=240)
        for y in range(100, 300, 10):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counts = counter.update(tracks, frame_height)

        assert counts["total"] == 1
        assert counts["car"] == 1

    def test_no_double_count(self):
        """Same track ID should never be counted twice."""
        counter = LineCrossingCounter(line_y_fraction=0.5, direction="both")
        frame_height = 480

        # Move track across line and then back
        for y in list(range(100, 300, 10)) + list(range(300, 100, -10)):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counter.update(tracks, frame_height)

        counts = counter.get_counts()
        assert counts["total"] == 1

    def test_counts_multiple_vehicles(self):
        """Multiple different vehicles should each be counted."""
        counter = LineCrossingCounter(line_y_fraction=0.5, direction="both")
        frame_height = 480

        # Vehicle 1 crosses
        for y in range(100, 300, 20):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counter.update(tracks, frame_height)

        # Vehicle 2 crosses
        for y in range(100, 300, 20):
            tracks = [Track(
                track_id=2, bbox=(200, y - 20, 300, y + 20),
                class_id=2, class_name="bus", confidence=0.85,
            )]
            counter.update(tracks, frame_height)

        counts = counter.get_counts()
        assert counts["total"] == 2
        assert counts["car"] == 1
        assert counts["bus"] == 1

    def test_direction_down_only(self):
        """With direction='down', only downward crossings are counted."""
        counter = LineCrossingCounter(line_y_fraction=0.5, direction="down")
        frame_height = 480

        # Upward crossing (should NOT be counted)
        for y in range(300, 100, -10):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counter.update(tracks, frame_height)

        assert counter.get_counts()["total"] == 0

    def test_vehicle_not_crossing_is_not_counted(self):
        """A vehicle that stays on one side should not be counted."""
        counter = LineCrossingCounter(line_y_fraction=0.5, direction="both")
        frame_height = 480

        # Vehicle stays above line
        for y in range(50, 200, 10):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counter.update(tracks, frame_height)

        assert counter.get_counts()["total"] == 0

    def test_reset(self):
        """Reset should clear all state."""
        counter = LineCrossingCounter(line_y_fraction=0.5)
        frame_height = 480

        for y in range(100, 300, 20):
            tracks = [Track(
                track_id=1, bbox=(50, y - 20, 150, y + 20),
                class_id=1, class_name="car", confidence=0.9,
            )]
            counter.update(tracks, frame_height)

        counter.reset()
        assert counter.get_counts()["total"] == 0
        assert len(counter.counted_ids) == 0


class TestZoneCounter:
    def test_counts_vehicle_entering_zone(self):
        """Vehicle entering the zone should be counted."""
        zone = [(100, 100), (400, 100), (400, 400), (100, 400)]
        counter = ZoneCounter(zone)

        tracks = [Track(
            track_id=1, bbox=(150, 200, 250, 300),
            class_id=1, class_name="car", confidence=0.9,
        )]
        counts = counter.update(tracks)

        assert counts["total"] == 1
        assert counts["car"] == 1

    def test_vehicle_outside_zone_not_counted(self):
        """Vehicle outside the zone should not be counted."""
        zone = [(100, 100), (200, 100), (200, 200), (100, 200)]
        counter = ZoneCounter(zone)

        tracks = [Track(
            track_id=1, bbox=(300, 300, 400, 400),
            class_id=1, class_name="car", confidence=0.9,
        )]
        counts = counter.update(tracks)

        assert counts["total"] == 0

    def test_current_inside_count(self):
        """get_current_inside_count returns vehicles currently in zone."""
        zone = [(0, 0), (500, 0), (500, 500), (0, 500)]
        counter = ZoneCounter(zone)

        tracks = [
            Track(track_id=1, bbox=(50, 50, 100, 100),
                  class_id=1, class_name="car", confidence=0.9),
            Track(track_id=2, bbox=(200, 200, 300, 300),
                  class_id=2, class_name="bus", confidence=0.85),
        ]
        counter.update(tracks)
        assert counter.get_current_inside_count() == 2


class TestCountAggregator:
    def test_record_and_retrieve(self):
        """Records should be retrievable as DataFrame."""
        agg = CountAggregator(window_seconds=60)

        agg.record(1000.0, {"total": 5, "car": 3, "bus": 2})
        agg.record(1030.0, {"total": 8, "car": 5, "bus": 3})

        df = agg.get_dataframe()
        assert len(df) == 2
        assert "total" in df.columns

    def test_export_csv(self, tmp_path):
        """CSV export should create a valid file."""
        agg = CountAggregator()
        agg.record(1000.0, {"total": 5, "car": 3})
        agg.record(1060.0, {"total": 8, "car": 5})

        csv_path = str(tmp_path / "counts.csv")
        agg.export_csv(csv_path, windowed=False)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_reset(self):
        agg = CountAggregator()
        agg.record(1000.0, {"total": 5})
        agg.reset()
        assert len(agg.records) == 0
