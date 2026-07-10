"""Firebase Cloud Messaging (FCM) push notification sender.

Setup:
1. Create a Firebase project at https://console.firebase.google.com
2. Download service account key JSON
3. Set FIREBASE_CREDENTIALS_PATH in .env
4. Users install your mobile app and register for notifications
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class FCMNotificationSender:
    """Sends push notifications via Firebase Cloud Messaging.

    Supports:
    - Individual device notifications
    - Topic-based broadcasts (e.g., all users subscribed to "traffic_alerts")
    - Data messages for app processing
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self.credentials_path = credentials_path or os.getenv(
            "FIREBASE_CREDENTIALS_PATH", ""
        )
        self.project_id = project_id or os.getenv("FIREBASE_PROJECT_ID", "")
        self._initialized = False
        self._device_tokens: List[str] = []
        self._tokens_file = Path("data/fcm_device_tokens.json")

        self._load_tokens()
        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase Admin SDK."""
        if not self.credentials_path or not Path(self.credentials_path).exists():
            logger.warning(
                "Firebase credentials not found. FCM notifications disabled."
            )
            return

        try:
            import firebase_admin
            from firebase_admin import credentials

            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_path)
                firebase_admin.initialize_app(cred)

            self._initialized = True
            logger.info("Firebase Admin SDK initialized.")
        except ImportError:
            logger.warning("firebase-admin package not installed.")
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

    @property
    def is_configured(self) -> bool:
        return self._initialized

    def send_to_device(
        self,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict] = None,
    ) -> bool:
        """Send notification to a specific device.

        Args:
            token: FCM device registration token.
            title: Notification title.
            body: Notification body text.
            data: Optional data payload for app processing.

        Returns:
            True if sent successfully.
        """
        if not self._initialized:
            logger.warning("Firebase not initialized. Skipping notification.")
            return False

        try:
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                token=token,
            )
            response = messaging.send(message)
            logger.info(f"FCM sent to device: {response}")
            return True
        except Exception as e:
            logger.error(f"FCM send failed: {e}")
            return False

    def send_to_topic(
        self,
        topic: str,
        title: str,
        body: str,
        data: Optional[Dict] = None,
    ) -> bool:
        """Send notification to all devices subscribed to a topic.

        Topics: "traffic_alerts", "accident_alerts", "emergency_alerts"

        Args:
            topic: Topic name.
            title: Notification title.
            body: Notification body.
            data: Optional data payload.

        Returns:
            True if sent successfully.
        """
        if not self._initialized:
            return False

        try:
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                topic=topic,
            )
            response = messaging.send(message)
            logger.info(f"FCM sent to topic '{topic}': {response}")
            return True
        except Exception as e:
            logger.error(f"FCM topic send failed: {e}")
            return False

    def broadcast_alert(self, alert) -> int:
        """Send alert to all registered devices and relevant topic.

        Args:
            alert: An Alert object from the alert system.

        Returns:
            Number of successful sends.
        """
        topic_map = {
            "accident": "accident_alerts",
            "heavy_traffic": "traffic_alerts",
            "illegal_vehicle": "security_alerts",
            "emergency_vehicle": "emergency_alerts",
            "signal_override": "traffic_alerts",
        }

        title = alert.title
        body = f"{alert.message}\nLocation: {alert.location or 'N/A'}"
        data = {
            "alert_id": alert.alert_id,
            "type": alert.alert_type.value,
            "priority": alert.priority.name,
            "camera_id": alert.camera_id or "",
            "timestamp": str(alert.timestamp),
        }

        sent_count = 0

        # Send to topic
        topic = topic_map.get(alert.alert_type.value, "traffic_alerts")
        if self.send_to_topic(topic, title, body, data):
            sent_count += 1

        # Send to individual registered devices
        for token in self._device_tokens:
            if self.send_to_device(token, title, body, data):
                sent_count += 1

        return sent_count

    def register_device(self, token: str):
        """Register a device token for direct notifications."""
        if token not in self._device_tokens:
            self._device_tokens.append(token)
            self._save_tokens()

    def unregister_device(self, token: str):
        """Remove a device token."""
        self._device_tokens = [t for t in self._device_tokens if t != token]
        self._save_tokens()

    def _save_tokens(self):
        try:
            self._tokens_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tokens_file, "w") as f:
                json.dump(self._device_tokens, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save FCM tokens: {e}")

    def _load_tokens(self):
        if self._tokens_file.exists():
            try:
                with open(self._tokens_file, "r") as f:
                    self._device_tokens = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._device_tokens = []

    @property
    def device_count(self) -> int:
        return len(self._device_tokens)
