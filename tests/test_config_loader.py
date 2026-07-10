"""Tests for YAML config validation."""

import pytest

from src.utils.config_loader import (
    ConfigValidationError,
    load_camera_config,
    load_density_thresholds,
    validate_all_configs,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_camera_config_requires_cameras_key(tmp_path):
    path = _write(tmp_path / "camera_config.yaml", "other: 1\n")
    with pytest.raises(ConfigValidationError, match="cameras"):
        load_camera_config(path)


def test_camera_config_rejects_empty_list(tmp_path):
    path = _write(tmp_path / "camera_config.yaml", "cameras: []\n")
    with pytest.raises(ConfigValidationError):
        load_camera_config(path)


def test_camera_config_requires_id_and_source(tmp_path):
    path = _write(
        tmp_path / "camera_config.yaml",
        "cameras:\n  - id: cam-01\n",
    )
    with pytest.raises(ConfigValidationError, match="source"):
        load_camera_config(path)


def test_valid_camera_config(tmp_path):
    path = _write(
        tmp_path / "camera_config.yaml",
        "cameras:\n  - id: cam-01\n    source: rtsp://0.0.0.0\n",
    )
    data = load_camera_config(path)
    assert data["cameras"][0]["id"] == "cam-01"


def test_density_thresholds_require_ordering(tmp_path):
    path = _write(
        tmp_path / "density_thresholds.yaml",
        "low_threshold: 10\nmedium_threshold: 5\nhigh_threshold: 20\n",
    )
    with pytest.raises(ConfigValidationError, match="low"):
        load_density_thresholds(path)


def test_density_thresholds_type_check(tmp_path):
    path = _write(
        tmp_path / "density_thresholds.yaml",
        "low_threshold: low\nmedium_threshold: 5\nhigh_threshold: 20\n",
    )
    with pytest.raises(ConfigValidationError):
        load_density_thresholds(path)


def test_validate_all_reports_missing_files(tmp_path):
    ok, errors = validate_all_configs(tmp_path)
    assert ok is False
    assert any("camera_config.yaml" in e for e in errors)
