"""Tests for the Firebase FCM notification sender."""

import json
from pathlib import Path

import pytest

from src.firebase_notifications.fcm_sender import FCMNotificationSender


@pytest.fixture
def unconfigured_sender(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREBASE_CREDENTIALS_PATH", raising=False)
    sender = FCMNotificationSender(credentials_path="")
    sender._tokens_file = tmp_path / "tokens.json"
    return sender


def test_unconfigured_sender_reports_not_configured(unconfigured_sender):
    assert unconfigured_sender.is_configured is False


def test_register_device_deduplicates(unconfigured_sender):
    unconfigured_sender.register_device("tok-1")
    unconfigured_sender.register_device("tok-1")
    unconfigured_sender.register_device("tok-2")
    assert unconfigured_sender._device_tokens == ["tok-1", "tok-2"]


def test_register_device_persists(unconfigured_sender):
    unconfigured_sender.register_device("tok-1")
    raw = json.loads(unconfigured_sender._tokens_file.read_text())
    assert raw == ["tok-1"]


def test_unregister_device(unconfigured_sender):
    unconfigured_sender.register_device("tok-1")
    unconfigured_sender.register_device("tok-2")
    unconfigured_sender.unregister_device("tok-1")
    assert unconfigured_sender._device_tokens == ["tok-2"]


def test_send_to_device_skips_when_not_initialised(unconfigured_sender):
    assert (
        unconfigured_sender.send_to_device("t", "title", "body")
        is False
    )


def test_missing_credentials_leaves_uninitialized(tmp_path):
    sender = FCMNotificationSender(credentials_path=str(tmp_path / "nope.json"))
    assert sender.is_configured is False
