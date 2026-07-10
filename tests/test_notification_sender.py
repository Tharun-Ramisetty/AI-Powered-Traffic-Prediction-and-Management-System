"""Tests for the in-app notification store."""

import json

import pytest

from src.alerts.alert_manager import Alert, AlertPriority, AlertType
from src.alerts.notification_sender import NotificationSender


def _alert(alert_id="ALT_00001"):
    return Alert(
        alert_id=alert_id,
        alert_type=AlertType.HEAVY_TRAFFIC,
        priority=AlertPriority.MEDIUM,
        title="Heavy traffic",
        message="Density spiked on Lane 2",
        timestamp=1_700_000_000.0,
        location="NH44",
        camera_id="cam-01",
    )


@pytest.fixture
def sender(tmp_path):
    return NotificationSender(storage_path=str(tmp_path / "notif.json"))


def test_send_stores_and_persists(sender):
    assert sender.send(_alert()) is True
    assert sender.get_unread_count() == 1

    saved = json.loads(sender._storage_path.read_text())
    assert saved[0]["id"] == "ALT_00001"


def test_mark_read_flips_unread_flag(sender):
    sender.send(_alert("A"))
    sender.send(_alert("B"))
    sender.mark_read("A")
    unread_ids = [n["id"] for n in sender.get_unread()]
    assert unread_ids == ["B"]


def test_store_trims_to_max(tmp_path):
    sender = NotificationSender(
        storage_path=str(tmp_path / "n.json"), max_stored=5,
    )
    for i in range(8):
        sender.send(_alert(f"A{i}"))
    assert len(sender.get_all(last_n=100)) == 5


def test_load_existing_notifications(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text(json.dumps([{
        "id": "X", "type": "heavy_traffic", "priority": "LOW",
        "title": "t", "message": "m", "timestamp": 0,
        "location": "", "camera_id": "", "read": False,
    }]))
    sender = NotificationSender(storage_path=str(path))
    assert sender.get_unread_count() == 1


def test_corrupt_store_recovers_to_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    sender = NotificationSender(storage_path=str(path))
    assert sender.get_unread_count() == 0
