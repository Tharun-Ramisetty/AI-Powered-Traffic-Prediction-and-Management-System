"""App push notification sender."""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class NotificationSender:
    """Sends push notifications and stores them for the dashboard.

    For production, integrate with:
    - Firebase Cloud Messaging (FCM)
    - OneSignal
    - Custom WebSocket server

    Currently stores notifications locally for the Streamlit dashboard to display.
    """

    def __init__(self, storage_path: Optional[str] = None, max_stored: int = 200):
        self.max_stored = max_stored
        self._notifications: List[Dict] = []

        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            self._storage_path = Path("outputs") / "notifications.json"

        # Load existing notifications
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    self._notifications = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._notifications = []

    def send(self, alert) -> bool:
        """Store notification for dashboard display and optional push delivery.

        Args:
            alert: An Alert object.

        Returns:
            True if notification was stored successfully.
        """
        notification = {
            "id": alert.alert_id,
            "type": alert.alert_type.value,
            "priority": alert.priority.name,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "location": alert.location,
            "camera_id": alert.camera_id,
            "read": False,
        }

        self._notifications.append(notification)

        # Trim old notifications
        if len(self._notifications) > self.max_stored:
            self._notifications = self._notifications[-self.max_stored:]

        # Persist to file
        self._save()

        logger.info(f"Notification stored: {alert.alert_id} - {alert.title}")
        return True

    def get_unread(self) -> List[Dict]:
        """Get all unread notifications."""
        return [n for n in self._notifications if not n.get("read")]

    def get_all(self, last_n: int = 50) -> List[Dict]:
        """Get recent notifications."""
        return self._notifications[-last_n:]

    def mark_read(self, notification_id: str):
        """Mark a notification as read."""
        for n in self._notifications:
            if n["id"] == notification_id:
                n["read"] = True
                break
        self._save()

    def mark_all_read(self):
        """Mark all notifications as read."""
        for n in self._notifications:
            n["read"] = True
        self._save()

    def get_unread_count(self) -> int:
        return len(self.get_unread())

    def _save(self):
        """Persist notifications to disk."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, "w") as f:
                json.dump(self._notifications, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save notifications: {e}")

    def clear(self):
        self._notifications.clear()
        self._save()
