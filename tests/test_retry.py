"""Tests for the retry decorator."""

import pytest

from src.utils.retry import with_retry


def test_retry_succeeds_after_transient_failure():
    calls = {"n": 0}

    @with_retry(exceptions=(ConnectionError,), max_attempts=3, initial_wait=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    @with_retry(exceptions=(ConnectionError,), max_attempts=2, initial_wait=0.01)
    def always_fails():
        calls["n"] += 1
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        always_fails()
    assert calls["n"] == 2


def test_retry_does_not_swallow_non_retryable_errors():
    @with_retry(exceptions=(ConnectionError,), max_attempts=3, initial_wait=0.01)
    def bad_usage():
        raise ValueError("bad arg")

    with pytest.raises(ValueError):
        bad_usage()
