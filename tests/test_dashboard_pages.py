"""Headless Streamlit tests for the dashboard pages.

Uses ``streamlit.testing.v1.AppTest`` to render each page without a browser.
These tests verify that every page imports cleanly and renders its initial
widgets — i.e. no ``ImportError``, ``AttributeError``, or unguarded
``NameError`` between a fresh session and the first user interaction.

Heavy per-click workflows (YOLO inference, EasyOCR, real Twilio) are NOT
exercised here — they belong in an integration suite that requires model
weights and network access. This suite catches the import / wiring bugs
that silently shipped before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so ``from src...`` and
# ``from config...`` resolve when AppTest loads the page script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    pytest.skip(
        "streamlit.testing unavailable in this environment",
        allow_module_level=True,
    )


PAGES_DIR = PROJECT_ROOT / "dashboard" / "pages"


def _page(name: str) -> AppTest:
    """Load a dashboard page by filename."""
    path = PAGES_DIR / name
    return AppTest.from_file(str(path), default_timeout=30)


@pytest.fixture(autouse=True)
def _isolated_data_dirs(tmp_path, monkeypatch):
    """Redirect all file-backed state under the project root to a tmp dir.

    Without this, running the tests would write into (and read from) real
    ``data/*.json`` and ``outputs/*.json`` files, which is both slow and
    pollutes the developer's workspace.
    """
    # Each test gets its own DB + notification stores.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    yield


# ─── Home / app.py auth gate ────────────────────────────────────────────────


def test_home_renders_without_password(monkeypatch):
    """Without DASHBOARD_PASSWORD the landing page renders its hero section
    and displays a warning banner (auth bypassed)."""
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception

    # The warning banner should have been emitted by require_login().
    warnings = [w.body for w in at.warning]
    assert any("without authentication" in w.lower() for w in warnings), (
        f"Expected auth warning banner. Got warnings: {warnings}"
    )


def test_home_blocks_when_password_set(monkeypatch):
    """When DASHBOARD_PASSWORD is set the login screen appears and the
    rest of the home page is NOT rendered."""
    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret")
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception

    # Login screen present
    titles = [t.value for t in at.title]
    assert any("login" in t.lower() for t in titles), (
        f"Expected login title, got: {titles}"
    )

    # Landing-page hero / metrics should NOT have been rendered.
    assert len(at.metric) == 0, (
        "Home landing page leaked through the auth gate — metrics visible "
        "before login."
    )


# ─── Alerts page ────────────────────────────────────────────────────────────


def test_alerts_page_initial_render():
    """Alerts page wires up AlertManager + DB persistence without crashing."""
    at = _page("11_alerts.py")
    at.run()
    assert not at.exception

    titles = [t.value for t in at.title]
    assert any("alert" in t.lower() for t in titles)

    # Stats tab shows 5 metric tiles (total + 4 categories).
    assert len(at.metric) >= 5


def test_alerts_page_dedup_across_reruns():
    """The fix for duplicate alerts: rerunning the page does NOT re-dispatch
    the same events. This guards against regressions of the bug where every
    Streamlit rerun created duplicate alerts."""
    at = _page("11_alerts.py")
    at.run()
    assert not at.exception

    # Seed fake accident events into session state, then rerun twice.
    class _Ev:
        event_id = "ACC_0001"
        event_type = "collision"
        severity = "high"
        description = "two-vehicle collision"

    at.session_state["accident_events"] = [_Ev()]
    at.run()
    first_total = next(
        int(m.value) for m in at.metric if m.label == "Total Alerts"
    )

    at.run()  # second rerun with the same event in state
    second_total = next(
        int(m.value) for m in at.metric if m.label == "Total Alerts"
    )

    assert first_total == second_total == 1, (
        f"Alert dedup regressed: total went {first_total} -> {second_total} "
        f"on rerun with the same event."
    )


def test_alerts_page_manual_alert_form_submits():
    """Submitting the manual-alert form creates an alert."""
    at = _page("11_alerts.py")
    at.run()
    assert not at.exception

    # The sidebar "Minimum Priority" selectbox defaults to MEDIUM (index=1).
    # The form's priority field defaults to LOW (first enum value) which
    # would be filtered out — push it up to HIGH so the alert survives.
    for sb in at.selectbox:
        if sb.label == "Priority":
            sb.set_value("HIGH")
            break

    # Fill the form fields (there's only one title/message/location trio).
    for ti in at.text_input:
        if ti.label == "Title":
            ti.set_value("Test alert")
        elif ti.label == "Location":
            ti.set_value("Cam-99")
    for ta in at.text_area:
        if ta.label == "Message":
            ta.set_value("Body")

    # Submit the form by clicking its submit button
    for btn in at.button:
        if btn.label == "Send Alert":
            btn.click().run()
            break
    else:
        pytest.fail("Send Alert button not found")

    total_tile = next(m for m in at.metric if m.label == "Total Alerts")
    assert int(total_tile.value) >= 1


# ─── Upload-form pages (no model invocation) ────────────────────────────────


def test_accident_detection_page_initial_render():
    """Accident page renders all three tabs without triggering model load."""
    at = _page("08_accident_detection.py")
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3


def test_anpr_page_initial_render():
    """ANPR page renders without calling OCR/plate detector on load."""
    at = _page("09_anpr.py")
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3


# ─── Remaining pages: smoke test that every page at least loads ────────────


@pytest.mark.parametrize("page_file", sorted(p.name for p in PAGES_DIR.glob("*.py")))
def test_every_page_imports_cleanly(page_file):
    """Parametrised smoke test over all dashboard pages.

    Pages that legitimately require resources (model weights, live cameras)
    only trigger those on user interaction, so initial render must always
    be exception-free. This catches import errors, missing modules, and
    wrong class names *before* the user clicks anything.
    """
    at = _page(page_file)
    at.run()
    if at.exception:
        pytest.fail(
            f"{page_file} raised on initial render:\n"
            + "\n".join(str(e) for e in at.exception)
        )
