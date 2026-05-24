"""Reusable candidate preferences form."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from config.settings import get_settings
from frontend.ui_settings import load_frontend_options
from frontend.utils.preferences import preferences_to_api_payload, split_locations, split_roles


def render_preference_form(
    key_prefix: str = "pref",
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Render preference fields and return raw form values."""
    options = load_frontend_options()
    defaults = defaults or {}
    ingestion = get_settings().ingestion

    if st.button("Skip all — use defaults", key=f"{key_prefix}_skip_defaults"):
        st.session_state[f"{key_prefix}_skipped"] = True

    job_types = st.multiselect(
        "Job type",
        options.get("job_types", []),
        default=defaults.get("job_types", []),
        key=f"{key_prefix}_job_types",
    )
    work_models = st.multiselect(
        "Work model",
        options.get("work_models", []),
        default=defaults.get("work_models", []),
        key=f"{key_prefix}_work_models",
    )
    locations_text = st.text_input(
        "Locations (comma-separated)",
        value=", ".join(defaults.get("locations", [])),
        placeholder="New York, San Francisco, Austin",
        key=f"{key_prefix}_locations",
    )
    visa_choice = st.radio(
        "Visa sponsorship needed",
        options=["Yes", "No", "Not applicable"],
        index=2,
        horizontal=True,
        key=f"{key_prefix}_visa",
    )
    visa_map = {"Yes": True, "No": False, "Not applicable": None}
    col1, col2 = st.columns(2)
    with col1:
        salary_min = st.number_input(
            "Minimum salary (USD)",
            min_value=0,
            value=int(defaults.get("salary_min") or 0),
            key=f"{key_prefix}_salary_min",
        )
    with col2:
        salary_max_val = defaults.get("salary_max")
        salary_max = st.number_input(
            "Maximum salary (USD)",
            min_value=0,
            value=int(salary_max_val) if salary_max_val else 0,
            key=f"{key_prefix}_salary_max",
        )
    st.caption("Jobs without listed salary won't be excluded.")

    stage_options = list(options.get("company_stages", []))
    company_stages = st.multiselect(
        "Company stage",
        stage_options,
        default=defaults.get("company_stages", []),
        key=f"{key_prefix}_stages",
    )
    if st.checkbox("No preference (company stage)", key=f"{key_prefix}_no_stage"):
        company_stages = []

    company_sizes = st.multiselect(
        "Company size",
        options.get("company_sizes", []),
        default=defaults.get("company_sizes", []),
        key=f"{key_prefix}_sizes",
    )
    roles_text = st.text_input(
        "Target roles (comma-separated)",
        value=", ".join(defaults.get("target_roles", [])),
        placeholder="ML Engineer, AI Engineer",
        key=f"{key_prefix}_roles",
    )
    industries_target = st.multiselect(
        "Industries to target",
        options.get("industries", []),
        default=defaults.get("industries_target", []),
        key=f"{key_prefix}_ind_target",
    )
    industries_avoid = st.multiselect(
        "Industries to avoid",
        options.get("industries", []),
        default=defaults.get("industries_avoid", []),
        key=f"{key_prefix}_ind_avoid",
    )
    priorities = st.multiselect(
        "What matters most (pick up to 5, in priority order)",
        options.get("priorities", []),
        default=defaults.get("priority_ranking", [])[:5],
        max_selections=5,
        key=f"{key_prefix}_priorities",
    )

    max_bytes = ingestion.resume_max_file_bytes
    st.caption(f"Resume size limit: {max_bytes // (1024 * 1024)} MB")

    raw = {
        "job_types": job_types,
        "work_models": work_models,
        "locations": split_locations(locations_text),
        "visa_sponsorship_needed": visa_map[visa_choice],
        "salary_min": int(salary_min) if salary_min > 0 else None,
        "salary_max": int(salary_max) if salary_max > 0 else None,
        "company_stages": company_stages,
        "company_sizes": company_sizes,
        "target_roles": split_roles(roles_text),
        "industries_target": industries_target,
        "industries_avoid": industries_avoid,
        "priority_ranking": priorities,
    }
    return raw


def render_preference_form_api(
    key_prefix: str = "pref",
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Render form and return API-ready preferences dict."""
    raw = render_preference_form(key_prefix=key_prefix, defaults=defaults)
    return preferences_to_api_payload(raw)
