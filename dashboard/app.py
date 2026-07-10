"""Main Streamlit dashboard entry point."""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.auth import require_login  # noqa: E402
from src.utils.env_validator import validate_env  # noqa: E402

APP_TITLE = "AI-Powered Traffic Prediction and Management System"
APP_TAGLINE = "Smart City Traffic Intelligence · Detection · Prediction · Control"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":vertical_traffic_light:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "_env_validated" not in st.session_state:
    validate_env(fail_on_missing_required=False)
    st.session_state["_env_validated"] = True

if not require_login():
    st.stop()

# Logo in the top-left of the horizontal navbar
LOGO_PATH = Path(__file__).resolve().parent.parent / "logo.jpeg"
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")


# ---------------------------------------------------------------------------
# Global theme / CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* Tighten top padding so the hero sits closer to the navbar */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    color: #ffffff;
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -60px; top: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,229,255,0.25) 0%, rgba(0,229,255,0) 70%);
    pointer-events: none;
}
.hero h1 {
    margin: 0 0 .35rem 0;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: .3px;
    line-height: 1.15;
}
.hero p.tag { margin: 0; opacity: .88; font-size: 1.02rem; }
.hero .pill-row { margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: .55rem; }
.hero .pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: #fff;
    padding: .35rem .75rem;
    border-radius: 999px;
    font-size: .82rem;
    backdrop-filter: blur(4px);
}
.hero .pill .dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00e676;
    margin-right: .45rem;
    box-shadow: 0 0 0 0 rgba(0,230,118,.7);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0   rgba(0,230,118,.7); }
    70%  { box-shadow: 0 0 0 10px rgba(0,230,118,0); }
    100% { box-shadow: 0 0 0 0   rgba(0,230,118,0); }
}

/* ---------- Section headers ---------- */
.section-head {
    display: flex; align-items: center; gap: .6rem;
    margin: 1.6rem 0 .8rem 0;
}
.section-head .bar {
    width: 5px; height: 26px; border-radius: 4px;
    background: linear-gradient(180deg, #00e5ff, #2c5364);
}
.section-head h3 { margin: 0; font-weight: 700; }

/* ---------- Feature cards ---------- */
.feat-card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 1.1rem 1.15rem 1rem 1.15rem;
    height: 100%;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    position: relative;
    min-height: 168px;
}
.feat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.10);
    border-color: rgba(0,229,255,.55);
}
.feat-card .icon {
    font-size: 1.7rem;
    display: inline-flex;
    width: 42px; height: 42px;
    align-items: center; justify-content: center;
    border-radius: 10px;
    margin-bottom: .6rem;
}
.feat-card h4 { margin: .1rem 0 .35rem 0; font-size: 1.02rem; font-weight: 700; }
.feat-card p  { margin: 0; font-size: .86rem; opacity: .82; line-height: 1.35; }

/* Accent colors per category */
.accent-core     .icon { background: rgba(0,188,212,.14); color: #00bcd4; }
.accent-safety   .icon { background: rgba(244,67,54,.14);  color: #f44336; }
.accent-iot      .icon { background: rgba(124,77,255,.14); color: #7c4dff; }
.accent-analytics .icon{ background: rgba(255,152,0,.14);  color: #ff9800; }

/* ---------- Vehicle class chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: .4rem; }
.chip {
    background: rgba(0,229,255,.08);
    border: 1px solid rgba(0,229,255,.25);
    padding: .4rem .75rem;
    border-radius: 999px;
    font-size: .84rem;
    font-weight: 500;
}

/* ---------- Footer ---------- */
.footer {
    margin-top: 2rem;
    padding: 1rem 0 .2rem 0;
    border-top: 1px solid rgba(128,128,128,0.18);
    text-align: center;
    font-size: .82rem;
    opacity: .7;
}

/* Hide default Streamlit chrome we don't need on the landing page */
footer[data-testid="stFooter"] { visibility: hidden; }
</style>
"""


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------
def section_header(title: str) -> None:
    st.markdown(
        f'<div class="section-head"><div class="bar"></div><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def feature_card(
    icon: str,
    title: str,
    description: str,
    page_path: str,
    accent: str = "accent-core",
) -> None:
    """Render a feature card with an icon, text and a page-link button."""
    st.markdown(
        f"""
        <div class="feat-card {accent}">
            <div class="icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(page_path, label="Open →", icon=None)


# ---------------------------------------------------------------------------
# Home / Landing page
# ---------------------------------------------------------------------------
def home() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # -------- Hero --------
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p class="tag">{APP_TAGLINE}</p>
            <div class="pill-row">
                <span class="pill"><span class="dot"></span>System Online</span>
                <span class="pill">YOLOv8 / v9 / v10</span>
                <span class="pill">LSTM + Attention</span>
                <span class="pill">Deep SORT · ByteTrack</span>
                <span class="pill">{datetime.now().strftime("%a, %d %b %Y")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- Quick stats strip --------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Vehicle Classes", "9", "Indian traffic")
    m2.metric("Detection Models", "3", "v8 · v9 · v10")
    m3.metric("Dashboard Modules", "12", "+Home")
    m4.metric("Trackers", "2", "DeepSORT + ByteTrack")
    m5.metric("Forecast Horizon", "60 min", "LSTM + Attn")

    # -------- Core capabilities --------
    section_header("Core Capabilities")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        feature_card(
            "🎥", "Live Detection",
            "Real-time vehicle detection from video files or RTSP camera streams.",
            "pages/01_live_detection.py", "accent-core",
        )
    with c2:
        feature_card(
            "📊", "Vehicle Counts",
            "Time-series charts and per-class counts with downloadable CSV.",
            "pages/02_vehicle_counts.py", "accent-analytics",
        )
    with c3:
        feature_card(
            "🌡️", "Traffic Density",
            "Density classification, heatmaps and lane-level congestion view.",
            "pages/03_traffic_density.py", "accent-analytics",
        )
    with c4:
        feature_card(
            "🔮", "Predictions",
            "LSTM + Attention forecasts with weather-aware adjustments.",
            "pages/04_predictions.py", "accent-core",
        )

    # -------- Safety & enforcement --------
    section_header("Smart Safety & Enforcement")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        feature_card(
            "🚨", "Accident Detection",
            "Sudden stop & collision detection with automatic authority alerts.",
            "pages/08_accident_detection.py", "accent-safety",
        )
    with c2:
        feature_card(
            "🔎", "ANPR",
            "Number plate recognition to track illegal or blacklisted vehicles.",
            "pages/09_anpr.py", "accent-safety",
        )
    with c3:
        feature_card(
            "🚑", "Emergency Vehicles",
            "Detect ambulance / fire truck and auto-switch the signal to green.",
            "pages/10_emergency_vehicles.py", "accent-safety",
        )
    with c4:
        feature_card(
            "🔔", "Alert System",
            "Email & app notifications for heavy traffic, accidents and emergencies.",
            "pages/11_alerts.py", "accent-safety",
        )

    # -------- Connected systems --------
    section_header("Connected Systems")
    feature_card(
        "🗺️", "Traffic Map",
        "Live density on OpenStreetMap with route suggestions for drivers.",
        "pages/12_traffic_map.py", "accent-iot",
    )

    # -------- Research / ops --------
    section_header("Research & Operations")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        feature_card(
            "⚖️", "Model Comparison",
            "Benchmark YOLOv8 vs YOLOv9 vs YOLOv10 on accuracy and speed.",
            "pages/05_model_comparison.py", "accent-core",
        )
    with c2:
        feature_card(
            "📤", "Export",
            "Download reports in CSV, JSON or PDF for stakeholders.",
            "pages/07_export.py", "accent-analytics",
        )
    with c3:
        feature_card(
            "🏠", "All Modules",
            "Use the top navigation bar to jump to any of the 12 modules.",
            "pages/01_live_detection.py", "accent-core",
        )

    # -------- Supported vehicle classes --------
    section_header("Supported Vehicle Classes")
    classes = [
        ("🛺", "Auto"), ("🏍️", "Bike"), ("🚌", "Bus"),
        ("🚗", "Car"), ("🛵", "Scooty"), ("🚖", "Taxi"),
        ("🚐", "Tempo"), ("🛺", "Toto"), ("🚚", "Truck"),
    ]
    chips_html = "".join(
        f'<div class="chip">{icon} {name}</div>' for icon, name in classes
    )
    st.markdown(f'<div class="chip-row">{chips_html}</div>', unsafe_allow_html=True)

    # -------- Footer --------
    st.markdown(
        f'<div class="footer">{APP_TITLE} · Built with YOLOv8 · LSTM · Streamlit</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES_DIR = Path(__file__).resolve().parent / "pages"

NAV_PAGES = [
    st.Page(home, title="Home", icon=":material/home:", default=True),
    st.Page(str(PAGES_DIR / "01_live_detection.py"), title="Live Detection", icon=":material/videocam:"),
    st.Page(str(PAGES_DIR / "02_vehicle_counts.py"), title="Vehicle Counts", icon=":material/bar_chart:"),
    st.Page(str(PAGES_DIR / "03_traffic_density.py"), title="Traffic Density", icon=":material/density_medium:"),
    st.Page(str(PAGES_DIR / "04_predictions.py"), title="Predictions", icon=":material/insights:"),
    st.Page(str(PAGES_DIR / "05_model_comparison.py"), title="Model Comparison", icon=":material/compare_arrows:"),
    st.Page(str(PAGES_DIR / "07_export.py"), title="Export", icon=":material/download:"),
    st.Page(str(PAGES_DIR / "08_accident_detection.py"), title="Accident Detection", icon=":material/crisis_alert:"),
    st.Page(str(PAGES_DIR / "09_anpr.py"), title="ANPR", icon=":material/directions_car:"),
    st.Page(str(PAGES_DIR / "10_emergency_vehicles.py"), title="Emergency Vehicles", icon=":material/emergency:"),
    st.Page(str(PAGES_DIR / "11_alerts.py"), title="Alerts", icon=":material/notifications:"),
    st.Page(str(PAGES_DIR / "12_traffic_map.py"), title="Traffic Map", icon=":material/map:"),
]

pg = st.navigation(NAV_PAGES, position="top")
pg.run()
