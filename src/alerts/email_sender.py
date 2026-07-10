"""Email alert sender via SMTP (works with Gmail app passwords, etc.)."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from loguru import logger

from src.utils.retry import with_retry


_SMTP_TRANSIENT = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    smtplib.SMTPHeloError,
    ConnectionError,
    TimeoutError,
    OSError,
)


class EmailSender:
    """Sends alert emails via SMTP.

    Defaults are tuned for Gmail (smtp.gmail.com:587 + STARTTLS); override
    via env vars or constructor args for other providers. For Gmail you must
    use an App Password (https://myaccount.google.com/apppasswords), not your
    account password.

    Env vars (used as fallback when constructor args are empty):
    - EMAIL_HOST           (default: smtp.gmail.com)
    - EMAIL_PORT           (default: 587)
    - EMAIL_USER           (the From address — also the SMTP login)
    - EMAIL_APP_PASSWORD   (Gmail App Password or SMTP password)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        app_password: Optional[str] = None,
        to_addresses: Optional[List[str]] = None,
    ):
        self.host = host or os.getenv("EMAIL_HOST", "smtp.gmail.com")
        self.port = int(port or os.getenv("EMAIL_PORT", "587"))
        self.user = user or os.getenv("EMAIL_USER", "")
        self.app_password = app_password or os.getenv("EMAIL_APP_PASSWORD", "")
        self.to_addresses: List[str] = to_addresses or []

    @property
    def is_configured(self) -> bool:
        return bool(self.user and self.app_password and self.to_addresses)

    def send(self, alert) -> bool:
        """Send an alert email to all configured recipients.

        Returns True if at least one recipient was delivered to.
        """
        if not self.is_configured:
            logger.warning("Email sender not configured. Skipping.")
            return False

        subject = f"[{alert.priority.name}] {alert.title}"
        body_lines = [
            alert.message,
            "",
            f"Alert ID: {alert.alert_id}",
            f"Type:     {alert.alert_type.value}",
            f"Priority: {alert.priority.name}",
        ]
        if alert.location:
            body_lines.append(f"Location: {alert.location}")
        if alert.camera_id:
            body_lines.append(f"Camera:   {alert.camera_id}")
        body = "\n".join(body_lines)

        msg = MIMEMultipart()
        msg["From"] = self.user
        msg["To"] = ", ".join(self.to_addresses)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            self._send_message(msg)
            logger.info(
                "Email sent to {} recipient(s) for alert {}",
                len(self.to_addresses), alert.alert_id,
            )
            return True
        except _SMTP_TRANSIENT as exc:
            logger.error(
                "Email send failed after retries ({}): {}",
                type(exc).__name__, exc,
            )
        except Exception:
            logger.exception("Unexpected email failure for alert {}", alert.alert_id)
        return False

    @with_retry(exceptions=_SMTP_TRANSIENT, max_attempts=3, initial_wait=1.0)
    def _send_message(self, msg: MIMEMultipart) -> None:
        """Open an SMTP connection, STARTTLS, login, send. Retried on transient errors."""
        with smtplib.SMTP(self.host, self.port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(self.user, self.app_password)
            smtp.send_message(msg)

    def add_recipient(self, email: str) -> None:
        if email not in self.to_addresses:
            self.to_addresses.append(email)

    def remove_recipient(self, email: str) -> None:
        self.to_addresses = [a for a in self.to_addresses if a != email]
