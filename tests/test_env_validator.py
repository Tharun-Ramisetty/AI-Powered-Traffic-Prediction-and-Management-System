"""Tests for the environment-variable validator."""

import pytest

from src.utils.env_validator import EnvValidator, EnvVarSpec


def test_required_var_set(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "real-value-123")
    spec = EnvVarSpec("TEST_API_KEY", required=True, feature="x")
    result = EnvValidator(specs=[spec]).validate(fail_on_missing_required=False)
    assert result["missing_required"] == []


def test_required_var_missing_aborts(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    spec = EnvVarSpec("TEST_API_KEY", required=True, feature="x")
    with pytest.raises(SystemExit):
        EnvValidator(specs=[spec]).validate(fail_on_missing_required=True)


def test_placeholder_value_treated_as_unset(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "your_api_key_here")
    spec = EnvVarSpec("TEST_API_KEY", required=True, feature="x")
    with pytest.raises(SystemExit):
        EnvValidator(specs=[spec]).validate(fail_on_missing_required=True)


def test_optional_var_missing_does_not_abort(monkeypatch):
    monkeypatch.delenv("OPTIONAL_KEY", raising=False)
    spec = EnvVarSpec("OPTIONAL_KEY", required=False, feature="analytics")
    result = EnvValidator(specs=[spec]).validate(fail_on_missing_required=True)
    assert "OPTIONAL_KEY" in result["missing_optional"]
    assert "analytics" in result["disabled_features"]


def test_feature_enabled_tracks_all_specs(monkeypatch):
    monkeypatch.setenv("A", "1")
    monkeypatch.delenv("B", raising=False)
    specs = [
        EnvVarSpec("A", required=False, feature="f"),
        EnvVarSpec("B", required=False, feature="f"),
    ]
    validator = EnvValidator(specs=specs)
    validator.validate(fail_on_missing_required=False)
    assert validator.feature_enabled("f") is False

    monkeypatch.setenv("B", "2")
    assert validator.feature_enabled("f") is True
