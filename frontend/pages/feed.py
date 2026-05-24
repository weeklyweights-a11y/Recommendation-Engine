"""Personalized job feed page (expanded in Step 5.2)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render feed placeholder until Step 5.2."""
    if not st.session_state.get("onboarding_complete"):
        st.warning("Complete onboarding to see your job feed.")
        return
    st.info("Loading feed…")
