"""Tests for the AlertManager dispatch and cooldown logic."""

import time

import pytest

from src.alerts.alert_manager import (
    AlertManager,
    AlertPriority,
    AlertType,
)


class _StubSender:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.sent = []

    def send(self, alert):
        if self.should_fail:
            raise RuntimeError("simulated transport failure")
        self.sent.append(alert)
        return True


@pytest.fixture
def manager():
    return AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=0)


def test_alert_below_priority_is_dropped(manager):
    manager.min_priority = AlertPriority.HIGH
    alert = manager.create_alert(
        AlertType.HEAVY_TRAFFIC, AlertPriority.LOW, "t", "m"
    )
    assert alert is None


def test_cooldown_suppresses_duplicate_alerts():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=60)
    first = mgr.create_alert(AlertType.ACCIDENT, AlertPriority.HIGH, "t", "m")
    second = mgr.create_alert(AlertType.ACCIDENT, AlertPriority.HIGH, "t", "m")
    assert first is not None
    assert second is None


def test_cooldown_allows_different_alert_types():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=60)
    a = mgr.create_alert(AlertType.ACCIDENT, AlertPriority.HIGH, "t", "m")
    b = mgr.create_alert(AlertType.HEAVY_TRAFFIC, AlertPriority.HIGH, "t", "m")
    assert a is not None and b is not None


def test_dispatch_sends_to_all_channels(manager):
    email, push = _StubSender(), _StubSender()
    manager.set_email_sender(email)
    manager.set_notification_sender(push)

    alert = manager.create_alert(
        AlertType.EMERGENCY_VEHICLE, AlertPriority.CRITICAL, "t", "m"
    )
    assert alert is not None
    assert "email" in alert.sent_via
    assert "app" in alert.sent_via
    assert len(email.sent) == 1 and len(push.sent) == 1


def test_channel_failure_does_not_block_other_channels(manager):
    email = _StubSender(should_fail=True)
    push = _StubSender()
    manager.set_email_sender(email)
    manager.set_notification_sender(push)

    alert = manager.create_alert(
        AlertType.ACCIDENT, AlertPriority.CRITICAL, "t", "m"
    )
    assert alert is not None
    assert "email" not in alert.sent_via
    assert "app" in alert.sent_via
    assert len(push.sent) == 1


def test_subscriber_callback_invoked():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=0)
    received = []
    mgr.subscribe(received.append)

    mgr.create_alert(AlertType.HEAVY_TRAFFIC, AlertPriority.LOW, "t", "m")
    assert len(received) == 1


def test_subscriber_exception_isolated_from_other_subscribers():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=0)
    received = []

    def bad(alert):
        raise RuntimeError("subscriber bug")

    mgr.subscribe(bad)
    mgr.subscribe(received.append)
    mgr.create_alert(AlertType.HEAVY_TRAFFIC, AlertPriority.LOW, "t", "m")
    assert len(received) == 1


def test_alert_stats_counts_by_type():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=0)
    mgr.create_alert(AlertType.ACCIDENT, AlertPriority.HIGH, "a", "m")
    mgr.create_alert(AlertType.ACCIDENT, AlertPriority.HIGH, "a", "m")
    mgr.create_alert(AlertType.HEAVY_TRAFFIC, AlertPriority.HIGH, "t", "m")
    stats = mgr.get_alert_stats()
    assert stats["total"] == 3
    assert stats["accident"] == 2
    assert stats["heavy_traffic"] == 1


def test_alert_buffer_capped_at_500():
    mgr = AlertManager(min_priority=AlertPriority.LOW, cooldown_seconds=0)
    for _ in range(520):
        mgr.create_alert(AlertType.HEAVY_TRAFFIC, AlertPriority.LOW, "t", "m")
    assert len(mgr.get_alerts(last_n=1000)) == 500
