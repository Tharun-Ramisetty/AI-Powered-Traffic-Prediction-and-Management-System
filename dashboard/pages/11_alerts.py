"""Dashboard page for Alert System - Real AlertManager + NotificationSender."""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Alert System", page_icon=":bell:", layout="wide")
st.title(":bell: Alert System")
st.markdown("Email and app notifications for traffic events, accidents, and emergencies.")
st.markdown("---")

# ─── Initialize Alert Manager ───────────────────────────────────────────────
from src.alerts.alert_manager import AlertManager, AlertType, AlertPriority
from src.alerts.notification_sender import NotificationSender
from src.db import AlertRepository, NotificationRepository, get_default_db

# Initialize or get from session
if "alert_manager" not in st.session_state:
    notif_sender = NotificationSender()
    alert_mgr = AlertManager(notification_enabled=True)
    alert_mgr.set_notification_sender(notif_sender)

    # Persist every dispatched alert to SQLite for the audit trail.
    try:
        alert_mgr.set_alert_repository(AlertRepository(get_default_db()))
    except Exception as exc:
        st.warning(f"DB persistence disabled: {exc}")

    # Auto-attach email sender if env vars are present so alerts go out
    # without the user needing to fill in the sidebar each session.
    try:
        from src.alerts.email_sender import EmailSender
        env_email = EmailSender()
        if env_email.user and env_email.app_password:
            alert_mgr.set_email_sender(env_email)
            st.session_state["email_sender"] = env_email
    except Exception as exc:
        st.warning(f"Email auto-config skipped: {exc}")

    st.session_state["alert_manager"] = alert_mgr
    st.session_state["notification_sender"] = notif_sender
    # Tracks event IDs we've already dispatched so reruns don't duplicate.
    st.session_state["_dispatched_event_ids"] = set()
else:
    alert_mgr = st.session_state["alert_manager"]
    notif_sender = st.session_state["notification_sender"]
    st.session_state.setdefault("_dispatched_event_ids", set())

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Alert Settings")
notification_enabled = st.sidebar.checkbox("Enable App Notifications", value=True)
min_priority = st.sidebar.selectbox("Minimum Priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1)
cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 5, 120, 30, 5)

# Update alert manager settings
alert_mgr.min_priority = AlertPriority[min_priority]
alert_mgr.cooldown_seconds = cooldown

# ─── Email Alerts ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
email_enabled = st.sidebar.checkbox(
    "Enable Email Alerts",
    value=alert_mgr.email_enabled,
    help="Sends an email for every dispatched alert. Free with Gmail App Passwords.",
)
alert_mgr.email_enabled = email_enabled

if email_enabled:
    st.sidebar.markdown("### SMTP / Email Config")
    from src.alerts.email_sender import EmailSender

    # Pre-fill from env-attached sender (set in init block) when present.
    seeded = st.session_state.get("email_sender")
    default_user = seeded.user if seeded else ""
    default_to = ", ".join(seeded.to_addresses) if seeded else ""

    email_user = st.sidebar.text_input(
        "From Email", value=default_user,
        placeholder="alerts@gmail.com", key="email_user",
    )
    email_pw = st.sidebar.text_input(
        "App Password", type="password", key="email_pw",
        help="Gmail: generate at https://myaccount.google.com/apppasswords",
    )
    email_to = st.sidebar.text_input(
        "Recipients (comma-separated)", value=default_to,
        placeholder="ops@org.com, oncall@org.com", key="email_to",
    )

    # Sidebar wins when filled; otherwise fall back to env-loaded sender.
    if email_user and (email_pw or (seeded and seeded.app_password)):
        try:
            sender = EmailSender(
                user=email_user,
                app_password=email_pw or (seeded.app_password if seeded else ""),
                to_addresses=[a.strip() for a in email_to.split(",") if a.strip()],
            )
            if sender.is_configured:
                alert_mgr.set_email_sender(sender)
                st.session_state["email_sender"] = sender
                st.sidebar.success(f"Email connected ({len(sender.to_addresses)} recipient(s))")
            else:
                st.sidebar.warning("Add at least one recipient to enable email.")
        except Exception as e:
            st.sidebar.error(f"Email config error: {e}")

# ─── Auto-generate alerts from detection results ───────────────────────────
# Streamlit re-runs this script on every interaction. Without dedup we'd
# spam the alert manager with the same event repeatedly; the cooldown only
# helps within its window. Track dispatched event IDs in session state.
accident_events = st.session_state.get("accident_events", [])
emergency_events = st.session_state.get("emergency_events", [])
final_counts = st.session_state.get("final_counts", {})
dispatched: set = st.session_state["_dispatched_event_ids"]

severity_map = {
    "critical": AlertPriority.CRITICAL,
    "high": AlertPriority.HIGH,
    "medium": AlertPriority.MEDIUM,
    "low": AlertPriority.LOW,
}

for event in accident_events:
    key = ("accident", getattr(event, "event_id", id(event)))
    if key in dispatched:
        continue
    alert_mgr.create_alert(
        alert_type=AlertType.ACCIDENT,
        priority=severity_map.get(event.severity, AlertPriority.MEDIUM),
        title=f"{event.event_type.replace('_', ' ').title()} detected",
        message=event.description,
        camera_id="cam_01",
    )
    dispatched.add(key)

for event in emergency_events:
    key = ("emergency",
           getattr(event, "event_id", None) or getattr(event, "frame_number", id(event)))
    if key in dispatched:
        continue
    alert_mgr.create_alert(
        alert_type=AlertType.EMERGENCY_VEHICLE,
        priority=AlertPriority.HIGH,
        title=f"Emergency vehicle: {event.vehicle_type}",
        message=f"{event.vehicle_type} detected at frame {event.frame_number}",
        camera_id="cam_01",
    )
    dispatched.add(key)

# Heavy traffic alert — keyed by (count_bucket) so it fires once per spike
# rather than every rerun.
if final_counts.get("total", 0) >= 30:
    bucket = final_counts["total"] // 10
    key = ("heavy_traffic", bucket)
    if key not in dispatched:
        alert_mgr.create_alert(
            alert_type=AlertType.HEAVY_TRAFFIC,
            priority=AlertPriority.MEDIUM,
            title="Heavy traffic detected",
            message=f"Total vehicles: {final_counts['total']}",
            camera_id="cam_01",
        )
        dispatched.add(key)

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Notifications", "Alert Configuration", "Statistics"])

with tab1:
    st.subheader("Recent Notifications")

    # Load from NotificationSender
    notifications = notif_sender.get_all(last_n=50)

    if notifications:
        unread = notif_sender.get_unread()
        st.info(f"{len(unread)} unread notification(s) out of {len(notifications)} total.")

        if st.button("Mark All as Read"):
            notif_sender.mark_all_read()
            st.rerun()

        for notif in reversed(notifications[-20:]):
            priority_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "⚪",
            }.get(notif.get("priority", ""), "⚪")

            read_status = "" if notif.get("read") else " :new:"
            ts = datetime.fromtimestamp(notif["timestamp"]).strftime("%H:%M:%S") if notif.get("timestamp") else ""

            with st.expander(f"{priority_icon} {notif['title']}{read_status} -- {ts}"):
                st.markdown(f"**Type:** {notif.get('type', 'N/A')}")
                st.markdown(f"**Message:** {notif.get('message', '')}")
                st.markdown(f"**Location:** {notif.get('location', 'N/A')}")
                st.markdown(f"**Camera:** {notif.get('camera_id', 'N/A')}")
    else:
        st.info("No notifications yet. Run detection on other pages to generate real alerts.")

    # Manual alert creation
    st.markdown("---")
    st.subheader("Create Manual Alert")
    with st.form("manual_alert"):
        m_type = st.selectbox("Alert Type", [t.value for t in AlertType])
        m_priority = st.selectbox("Priority", [p.name for p in AlertPriority])
        m_title = st.text_input("Title", placeholder="Heavy traffic at junction")
        m_message = st.text_area("Message", placeholder="Traffic congestion detected...")
        m_location = st.text_input("Location", placeholder="MG Road Junction")

        if st.form_submit_button("Send Alert"):
            alert = alert_mgr.create_alert(
                alert_type=AlertType(m_type),
                priority=AlertPriority[m_priority],
                title=m_title,
                message=m_message,
                location=m_location,
            )
            if alert:
                st.success(f"Alert {alert.alert_id} created and dispatched!")
                st.rerun()
            else:
                st.warning("Alert suppressed by cooldown or priority filter.")

with tab2:
    st.subheader("Alert Rules Configuration")

    st.markdown("### Alert Types & Channels")

    alert_types = {
        "Accident Detected": {"email": True, "app": True, "priority": "CRITICAL"},
        "Illegal Vehicle Spotted": {"email": True, "app": True, "priority": "HIGH"},
        "Heavy Traffic": {"email": False, "app": True, "priority": "MEDIUM"},
        "Emergency Vehicle": {"email": False, "app": True, "priority": "MEDIUM"},
        "Signal Override": {"email": False, "app": True, "priority": "LOW"},
    }

    for alert_name, config in alert_types.items():
        with st.expander(f"{alert_name} -- {config['priority']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.checkbox("Email", value=config["email"], key=f"email_{alert_name}")
            with col2:
                st.checkbox("App Notification", value=config["app"], key=f"app_{alert_name}")
            with col3:
                st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                           index=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(config["priority"]),
                           key=f"pri_{alert_name}")

with tab3:
    st.subheader("Alert Statistics")

    stats = alert_mgr.get_alert_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Alerts", stats.get("total", 0))
    with col2:
        st.metric("Accidents", stats.get("accident", 0))
    with col3:
        st.metric("Illegal Vehicles", stats.get("illegal_vehicle", 0))
    with col4:
        st.metric("Heavy Traffic", stats.get("heavy_traffic", 0))
    with col5:
        st.metric("Emergency", stats.get("emergency_vehicle", 0))

    if st.button("Clear All Alerts"):
        alert_mgr.clear()
        notif_sender.clear()
        st.session_state["_dispatched_event_ids"] = set()
        st.rerun()

    st.markdown("---")
    st.markdown("### Alert Flow")
    st.markdown("""
    ```
    Event Detected -> Priority Check -> Cooldown Check -> Dispatch
                                                         |-- Email (SMTP) -> Authorities
                                                         +-- App Notification -> Dashboard
    ```
    """)
