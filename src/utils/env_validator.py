"""Startup environment-variable validation.

Fails fast with a clear, actionable error message if any required secret or
configuration value is missing. Optional variables log a warning but do not
abort startup.

Usage::

    from src.utils.env_validator import validate_env
    validate_env()  # at application entry point
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from loguru import logger


@dataclass(frozen=True)
class EnvVarSpec:
    """Declaration of an environment variable the app consumes."""

    name: str
    required: bool = False
    feature: str = "general"
    description: str = ""
    default: Optional[str] = None

    def is_set(self) -> bool:
        value = os.getenv(self.name, "").strip()
        if not value:
            return False
        if value.lower().startswith(("your_", "changeme", "placeholder")):
            return False
        return True


DEFAULT_SPECS: tuple[EnvVarSpec, ...] = (
    EnvVarSpec("ROBOFLOW_API_KEY", False, "detection",
               "Roboflow dataset API key. Only needed for dataset downloads."),
    EnvVarSpec("OPENWEATHER_API_KEY", False, "prediction",
               "OpenWeatherMap key for weather-aware LSTM forecasts."),
    EnvVarSpec("WEATHER_CITY", False, "prediction",
               "City name for weather lookups.", default="Tumkur,IN"),
    EnvVarSpec("EMAIL_USER", False, "email_alerts",
               "From-address / SMTP login for email alerts."),
    EnvVarSpec("EMAIL_APP_PASSWORD", False, "email_alerts",
               "Gmail App Password (or SMTP password) for email alerts."),
    EnvVarSpec("ORS_API_KEY", False, "maps",
               "OpenRouteService key for route suggestions."),
    EnvVarSpec("FIREBASE_CREDENTIALS_PATH", False, "firebase",
               "Path to Firebase Admin SDK JSON credentials."),
    EnvVarSpec("FIREBASE_PROJECT_ID", False, "firebase",
               "Firebase project ID."),
    EnvVarSpec("DASHBOARD_PASSWORD", False, "auth",
               "Shared password protecting the Streamlit dashboard. "
               "If unset, the dashboard runs unauthenticated (NOT recommended "
               "for production)."),
    EnvVarSpec("DATABASE_PATH", False, "persistence",
               "SQLite file path. Defaults to data/traffic.db.",
               default="data/traffic.db"),
)


class EnvValidator:
    """Validates environment variables against a list of specs."""

    def __init__(self, specs: Iterable[EnvVarSpec] = DEFAULT_SPECS,
                 env_file: Optional[Path] = None):
        self.specs: List[EnvVarSpec] = list(specs)
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        else:
            load_dotenv(override=False)

    def validate(self, fail_on_missing_required: bool = True) -> dict[str, list[str]]:
        """Validate all specs.

        Returns a dict with keys ``missing_required``, ``missing_optional``,
        and ``disabled_features``. Raises ``SystemExit`` when required vars are
        missing and ``fail_on_missing_required`` is True.
        """
        missing_required: List[str] = []
        missing_optional: List[str] = []
        disabled_features: set[str] = set()

        for spec in self.specs:
            if spec.is_set():
                continue
            if spec.required:
                missing_required.append(spec.name)
                disabled_features.add(spec.feature)
            else:
                missing_optional.append(spec.name)
                disabled_features.add(spec.feature)

        if missing_required:
            msg = self._format_missing_required_message(missing_required)
            logger.error(msg)
            if fail_on_missing_required:
                sys.exit(1)

        if missing_optional:
            logger.info(
                "Optional env vars not set ({} features degraded): {}",
                len({s.feature for s in self.specs
                     if s.name in missing_optional}),
                ", ".join(missing_optional),
            )

        return {
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "disabled_features": sorted(disabled_features),
        }

    def feature_enabled(self, feature: str) -> bool:
        """True when every spec belonging to ``feature`` has a real value."""
        return all(spec.is_set() for spec in self.specs if spec.feature == feature)

    def _format_missing_required_message(self, missing: List[str]) -> str:
        lines = ["Required environment variables are not configured:"]
        for name in missing:
            spec = next((s for s in self.specs if s.name == name), None)
            if spec:
                lines.append(f"  - {name}  [{spec.feature}] — {spec.description}")
            else:
                lines.append(f"  - {name}")
        lines.append("")
        lines.append("Copy .env.example to .env and fill in the missing values.")
        return "\n".join(lines)


def validate_env(
    specs: Optional[Iterable[EnvVarSpec]] = None,
    fail_on_missing_required: bool = True,
) -> dict[str, list[str]]:
    """Convenience entry point: run validation with the default spec list."""
    validator = EnvValidator(specs=specs or DEFAULT_SPECS)
    return validator.validate(fail_on_missing_required=fail_on_missing_required)
