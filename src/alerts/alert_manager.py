"""Central alert management system for traffic events."""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

from loguru import logger


class AlertType(Enum):
    ACCIDENT = "accident"
    HEAVY_TRAFFIC = "heavy_traffic"
    ILLEGAL_VEHICLE = "illegal_vehicle"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    SIGNAL_OVERRIDE = "signal_override"


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """A system alert to be dispatched."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: float = 0.0
    location: str = ""
    camera_id: str = ""
    is_sent: bool = False
    sent_via: List[str] = field(default_factory=list)  # "app", "email"
    metadata: Dict = field(default_factory=dict)


class AlertManager:
    """Manages and dispatches alerts through configured channels.

    Supports:
    - Email via SMTP
    - App push notifications
    - In-app dashboard alerts
    - Configurable alert rules and thresholds
    """

    def __init__(
        self,
        notification_enabled: bool = True,
        email_enabled: bool = False,
        min_priority: AlertPriority = AlertPriority.MEDIUM,
        cooldown_seconds: float = 30.0,
    ):
        self.notification_enabled = notification_enabled
        self.email_enabled = email_enabled
        self.min_priority = min_priority
        self.cooldown_seconds = cooldown_seconds

        self._alert_counter = 0
        self._alerts: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}  # type -> last_sent_time
        self._notification_sender = None
        self._email_sender = None
        self._subscribers: List[Callable[[Alert], None]] = []
        self._alert_repo = None  # set via set_alert_repository

    def set_notification_sender(self, sender):
        """Attach a NotificationSender instance."""
        self._notification_sender = sender
        self.notification_enabled = True

    def set_email_sender(self, sender):
        """Attach an EmailSender instance."""
        self._email_sender = sender
        self.email_enabled = True

    def subscribe(self, callback: Callable[[Alert], None]):
        """Register a callback for real-time alert notifications."""
        self._subscribers.append(callback)

    def set_alert_repository(self, repo) -> None:
        """Persist every dispatched alert to the given AlertRepository.

        Optional — when unset, alerts live only in the in-memory ring buffer.
        """
        self._alert_repo = repo

    def create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        location: str = "",
        camera_id: str = "",
        metadata: Dict = None,
    ) -> Optional[Alert]:
        """Create and dispatch an alert if it meets threshold and cooldown.

        Returns:
            Alert object if created, None if suppressed by cooldown/priority.
        """
        # Priority check
        if priority.value < self.min_priority.value:
            return None

        # Cooldown check per alert type
        current_time = time.time()
        type_key = alert_type.value
        if type_key in self._last_alert_time:
            if current_time - self._last_alert_time[type_key] < self.cooldown_seconds:
                return None

        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALT_{self._alert_counter:05d}",
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            timestamp=current_time,
            location=location,
            camera_id=camera_id,
            metadata=metadata or {},
        )

        # Dispatch through channels
        self._dispatch(alert)

        self._alerts.append(alert)
        self._last_alert_time[type_key] = current_time

        # Keep only last 500 alerts
        if len(self._alerts) > 500:
            self._alerts = self._alerts[-500:]

        return alert

    def _dispatch(self, alert: Alert):
        """Send alert through all configured channels.

        Each channel is isolated: a failure in one channel must never prevent
        the others from receiving the alert. Failures are logged with full
        context so delivery issues surface in observability instead of being
        silently swallowed.
        """
        if self.notification_enabled and self._notification_sender:
            try:
                self._notification_sender.send(alert)
                alert.sent_via.append("app")
            except Exception:
                logger.exception(
                    "Push notification dispatch failed for alert {} ({})",
                    alert.alert_id, alert.alert_type.value,
                )

        if self.email_enabled and self._email_sender:
            try:
                if self._email_sender.send(alert):
                    alert.sent_via.append("email")
            except Exception:
                logger.exception(
                    "Email dispatch failed for alert {} ({})",
                    alert.alert_id, alert.alert_type.value,
                )

        for callback in self._subscribers:
            try:
                callback(alert)
            except Exception:
                logger.exception(
                    "Alert subscriber callback {} failed for alert {}",
                    getattr(callback, "__name__", repr(callback)),
                    alert.alert_id,
                )

        alert.is_sent = True
        logger.info(
            "Alert dispatched: {} [{}] via {}",
            alert.alert_id, alert.priority.name,
            ", ".join(alert.sent_via) or "no channels",
        )

        if self._alert_repo is not None:
            try:
                self._alert_repo.insert(alert)
            except Exception:
                logger.exception(
                    "Failed to persist alert {} to database", alert.alert_id,
                )

    def get_alerts(
        self,
        last_n: int = 50,
        alert_type: Optional[AlertType] = None,
        min_priority: Optional[AlertPriority] = None,
    ) -> List[Alert]:
        """Get recent alerts with optional filters."""
        filtered = self._alerts
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        if min_priority:
            filtered = [a for a in filtered if a.priority.value >= min_priority.value]
        return filtered[-last_n:]

    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        stats = {"total": len(self._alerts)}
        for at in AlertType:
            type_alerts = [a for a in self._alerts if a.alert_type == at]
            stats[at.value] = len(type_alerts)
        return stats

    def clear(self):
        self._alerts.clear()
        self._alert_counter = 0
        self._last_alert_time.clear()
