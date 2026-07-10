"""Tests for the SMTP email sender — no real SMTP traffic."""

import smtplib
from unittest.mock import MagicMock, patch

from src.alerts.alert_manager import AlertPriority, AlertType, Alert
from src.alerts.email_sender import EmailSender


def _alert():
    return Alert(
        alert_id="ALT_00001",
        alert_type=AlertType.ACCIDENT,
        priority=AlertPriority.HIGH,
        title="Collision",
        message="Two-vehicle collision detected",
        timestamp=0.0,
        location="NH44 / Cam-03",
        camera_id="cam_03",
    )


def test_is_configured_false_without_recipients():
    sender = EmailSender(user="a@b.com", app_password="pw", to_addresses=[])
    assert sender.is_configured is False


def test_is_configured_false_without_credentials(monkeypatch):
    # Constructor falls back to EMAIL_USER / EMAIL_APP_PASSWORD env vars when
    # args are empty — clear them so this test isolates the not-configured path.
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.delenv("EMAIL_APP_PASSWORD", raising=False)
    sender = EmailSender(user="", app_password="", to_addresses=["x@y.com"])
    assert sender.is_configured is False


def test_send_skips_when_not_configured():
    sender = EmailSender(user="", app_password="", to_addresses=[])
    assert sender.send(_alert()) is False


def test_send_success_sends_message_with_alert_fields():
    sender = EmailSender(
        user="alerts@example.com", app_password="pw",
        to_addresses=["ops@example.com", "oncall@example.com"],
    )

    with patch("src.alerts.email_sender.smtplib.SMTP") as smtp_cls:
        smtp_inst = MagicMock()
        smtp_cls.return_value.__enter__.return_value = smtp_inst

        assert sender.send(_alert()) is True

    smtp_inst.starttls.assert_called_once()
    smtp_inst.login.assert_called_once_with("alerts@example.com", "pw")
    smtp_inst.send_message.assert_called_once()

    msg = smtp_inst.send_message.call_args.args[0]
    assert "Collision" in msg["Subject"]
    assert "[HIGH]" in msg["Subject"]
    assert msg["From"] == "alerts@example.com"
    assert "ops@example.com" in msg["To"]
    body = msg.get_payload()[0].get_payload()
    assert "Two-vehicle collision detected" in body
    assert "NH44" in body
    assert "cam_03" in body


def test_send_reports_failure_after_retries():
    sender = EmailSender(
        user="a@b.com", app_password="pw", to_addresses=["x@y.com"],
    )
    with patch("src.alerts.email_sender.smtplib.SMTP") as smtp_cls:
        smtp_cls.side_effect = ConnectionError("network down")
        assert sender.send(_alert()) is False
        # Retry decorator should attempt at least twice before giving up.
        assert smtp_cls.call_count >= 2


def test_add_and_remove_recipient():
    sender = EmailSender()
    sender.add_recipient("a@b.com")
    sender.add_recipient("a@b.com")  # dedup
    sender.add_recipient("c@d.com")
    assert sender.to_addresses == ["a@b.com", "c@d.com"]
    sender.remove_recipient("a@b.com")
    assert sender.to_addresses == ["c@d.com"]


def test_env_vars_are_used_when_constructor_args_empty(monkeypatch):
    monkeypatch.setenv("EMAIL_HOST", "smtp.example.com")
    monkeypatch.setenv("EMAIL_PORT", "2525")
    monkeypatch.setenv("EMAIL_USER", "env_user@example.com")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "env_pw")
    sender = EmailSender(to_addresses=["x@y.com"])
    assert sender.host == "smtp.example.com"
    assert sender.port == 2525
    assert sender.user == "env_user@example.com"
    assert sender.app_password == "env_pw"
    assert sender.is_configured is True
