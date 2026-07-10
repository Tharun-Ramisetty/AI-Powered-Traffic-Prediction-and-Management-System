"""Shared-password gate for the Streamlit dashboard.

Simple, single-secret protection suitable for internal/demo deployments. For
production, swap this for an IdP-backed auth layer (OAuth, SAML, SSO).

Set ``DASHBOARD_PASSWORD`` in the environment (or Streamlit secrets) to turn
the gate on. When unset, ``require_login`` is a no-op — the dashboard stays
open for local development — but a banner warns the operator.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from loguru import logger


_SESSION_KEY = "_dashboard_authenticated"
_ATTEMPT_KEY = "_dashboard_auth_attempts"
_MAX_ATTEMPTS = 5


def _get_expected_password() -> Optional[str]:
    pw = os.getenv("DASHBOARD_PASSWORD", "").strip()
    if pw:
        return pw
    try:
        import streamlit as st
        return st.secrets.get("DASHBOARD_PASSWORD")  # type: ignore[attr-defined]
    except Exception:
        return None


def require_login() -> bool:
    """Block dashboard rendering until the user enters the shared password.

    Returns True when the session is authenticated (or auth is disabled),
    False when the caller should stop rendering the current page.
    """
    import streamlit as st

    expected = _get_expected_password()
    if not expected:
        st.warning(
            "⚠️ Dashboard is running **without authentication**. "
            "Set `DASHBOARD_PASSWORD` in your environment to enable the "
            "password gate before exposing this UI to others.",
            icon="⚠️",
        )
        return True

    if st.session_state.get(_SESSION_KEY):
        return True

    attempts = st.session_state.get(_ATTEMPT_KEY, 0)
    if attempts >= _MAX_ATTEMPTS:
        st.error("Too many failed attempts. Restart the app to try again.")
        st.stop()
        return False

    st.title("🔒 Dashboard Login")
    st.caption("Enter the shared dashboard password to continue.")
    pw = st.text_input("Password", type="password", key="_auth_pw_input")

    if st.button("Sign in", type="primary"):
        if hmac.compare_digest(pw, expected):
            st.session_state[_SESSION_KEY] = True
            st.session_state[_ATTEMPT_KEY] = 0
            logger.info("Dashboard login succeeded.")
            st.rerun()
        else:
            st.session_state[_ATTEMPT_KEY] = attempts + 1
            logger.warning(
                "Dashboard login failed (attempt {}/{}).",
                attempts + 1, _MAX_ATTEMPTS,
            )
            st.error("Incorrect password.")

    st.stop()
    return False


def logout() -> None:
    """Clear the authenticated session (hook up to a sidebar button)."""
    import streamlit as st
    st.session_state.pop(_SESSION_KEY, None)
    st.session_state.pop(_ATTEMPT_KEY, None)
