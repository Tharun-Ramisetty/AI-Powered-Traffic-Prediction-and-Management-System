"""YAML config loader with schema validation.

Falls back to a best-effort validator when ``pydantic`` is not installed so
the module can be imported in minimal environments (for example, the test
matrix) without pulling in heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from loguru import logger


class ConfigValidationError(ValueError):
    """Raised when a YAML config does not match its expected schema."""


def _require_keys(data: Dict[str, Any], required: List[str], file: Path) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigValidationError(
            f"{file.name} is missing required keys: {', '.join(missing)}"
        )


def _typecheck(data: Dict[str, Any], spec: Dict[str, type], file: Path) -> None:
    errors: List[str] = []
    for key, expected_type in spec.items():
        if key not in data:
            continue
        if not isinstance(data[key], expected_type):
            errors.append(
                f"{key}: expected {expected_type.__name__}, "
                f"got {type(data[key]).__name__}"
            )
    if errors:
        raise ConfigValidationError(
            f"{file.name} has invalid types:\n  - " + "\n  - ".join(errors)
        )


def load_camera_config(path: Path) -> Dict[str, Any]:
    """Load and validate ``camera_config.yaml``."""
    data = _read_yaml(path)
    _require_keys(data, ["cameras"], path)
    if not isinstance(data["cameras"], list) or not data["cameras"]:
        raise ConfigValidationError(
            f"{path.name}: 'cameras' must be a non-empty list."
        )
    for i, cam in enumerate(data["cameras"]):
        if not isinstance(cam, dict):
            raise ConfigValidationError(
                f"{path.name}: cameras[{i}] must be a mapping."
            )
        _require_keys(cam, ["id", "source"], path)
    return data


def load_density_thresholds(path: Path) -> Dict[str, Any]:
    """Load and validate ``density_thresholds.yaml``."""
    data = _read_yaml(path)
    _typecheck(data, {
        "low_threshold": int,
        "medium_threshold": int,
        "high_threshold": int,
    }, path)
    low = data.get("low_threshold", 0)
    med = data.get("medium_threshold", 0)
    high = data.get("high_threshold", 0)
    if not (low < med < high):
        raise ConfigValidationError(
            f"{path.name}: thresholds must satisfy "
            f"low ({low}) < medium ({med}) < high ({high})."
        )
    return data


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"{path.name}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    return data


def validate_all_configs(config_dir: Path) -> Tuple[bool, List[str]]:
    """Validate every known YAML config in ``config_dir``.

    Returns ``(ok, errors)`` so callers can decide whether to log and
    continue or abort.
    """
    errors: List[str] = []
    loaders = [
        ("camera_config.yaml", load_camera_config),
        ("density_thresholds.yaml", load_density_thresholds),
    ]
    for filename, loader in loaders:
        path = config_dir / filename
        if not path.exists():
            errors.append(f"{filename}: missing")
            continue
        try:
            loader(path)
            logger.debug("{} validated OK", filename)
        except (ConfigValidationError, yaml.YAMLError) as exc:
            errors.append(f"{filename}: {exc}")

    return (not errors, errors)
