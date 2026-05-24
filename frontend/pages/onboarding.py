"""Onboarding wizard page."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import streamlit as st

from config.settings import get_settings
from frontend.components.preference_form import render_preference_form_api
from frontend.utils.api_client import (
    ApiError,
    create_candidate,
    get_recommendations,
    github_preview,
    patch_preferences,
)


def _depth_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "orange"
    return "gray"


def _step_resume() -> None:
    st.markdown("## Let's get to know you")
    st.write(
        "Upload your resume and we'll build your profile. We analyze your skills, "
        "experience, and career trajectory to find jobs that actually fit you."
    )
    max_bytes = get_settings().ingestion.resume_max_file_bytes
    uploaded = st.file_uploader(
        "Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        key="onboarding_resume_upload",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) > max_bytes:
            st.error(f"File is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.")
            return
        st.session_state.resume_bytes = data
        st.session_state.resume_filename = uploaded.name
        st.success(f"Uploaded: {uploaded.name}")
    if st.session_state.get("resume_bytes"):
        if st.button("Next", key="resume_next"):
            st.session_state.onboarding_step = 2
            st.rerun()


def _step_github() -> None:
    st.markdown("## Connect your GitHub")
    st.write(
        "Optional but powerful. We analyze your repos to understand what you "
        "actually build, not just what you claim."
    )
    username = st.text_input("GitHub username", key="onboarding_github_user")
    preview = st.session_state.get("github_preview")
    if st.button("Preview", key="github_preview_btn") and username.strip():
        try:
            st.session_state.github_preview = github_preview(username)
            st.session_state.github_username = username.strip().lstrip("@")
        except ApiError as exc:
            st.error(exc.message)
    if preview:
        cols = st.columns([1, 3])
        with cols[0]:
            if preview.get("avatar_url"):
                st.image(preview["avatar_url"], width=80)
        with cols[1]:
            st.write(f"**{preview.get('name', '')}** (@{preview.get('username', '')})")
            st.write(f"Public repos: {preview.get('public_repos', 0)}")
            langs = preview.get("top_languages") or []
            if langs:
                st.write("Top languages: " + ", ".join(langs))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Skip", key="github_skip"):
            st.session_state.github_username = None
            st.session_state.onboarding_step = 3
            st.rerun()
    with col_b:
        if st.button("Next", key="github_next"):
            if username.strip():
                st.session_state.github_username = username.strip().lstrip("@")
            st.session_state.onboarding_step = 3
            st.rerun()


def _step_preferences() -> None:
    st.markdown("## What are you looking for?")
    st.write(
        "Tell us your non-negotiables. We'll use these as hard filters — "
        "you'll never see jobs that don't match."
    )
    prefs = render_preference_form_api(key_prefix="onboard")
    st.session_state.preferences = prefs
    if st.button("Next", key="prefs_next"):
        st.session_state.onboarding_step = 4
        st.session_state.profile_build_started = False
        st.rerun()


def _step_build_profile() -> None:
    st.markdown("## Building your profile...")
    if st.session_state.get("candidate_id"):
        st.session_state.onboarding_step = 5
        st.rerun()
        return
    if st.session_state.get("profile_build_started"):
        return
    st.session_state.profile_build_started = True
    stages = [
        "Parsing resume...",
        "Analyzing GitHub...",
        "Building your profile...",
        "Linking skills...",
        "Generating embeddings...",
    ]
    progress = st.progress(0)
    status = st.empty()
    try:
        for idx, label in enumerate(stages):
            status.info(label)
            progress.progress((idx + 1) / len(stages))
            time.sleep(0.4)
        result = create_candidate(
            resume_bytes=st.session_state.resume_bytes,
            filename=st.session_state.resume_filename,
            github_username=st.session_state.get("github_username"),
            preferences=st.session_state.get("preferences"),
        )
        st.session_state.candidate_id = result["id"]
        st.session_state.profile = result.get("profile") or {}
        st.session_state.candidate_name = result.get("name")
        st.session_state.onboarding_step = 5
        st.rerun()
    except ApiError as exc:
        st.session_state.profile_build_started = False
        st.error(exc.message)
        if st.button("Retry", key="build_retry"):
            st.rerun()


def _render_inferred_preferences(profile: dict[str, Any]) -> None:
    """Show inferred preferences with confirm/edit toggles."""
    prefs = profile.get("preferences") or {}
    if not isinstance(prefs, dict):
        return
    inferred_fields = [
        ("preferred_company_stage", "company stage"),
        ("preferred_team_size", "team size"),
        ("preferred_work_style", "work style"),
        ("likely_looking_for", "role focus"),
    ]
    edits = st.session_state.setdefault("step5_preference_edits", {})
    for field_key, label in inferred_fields:
        field = prefs.get(field_key) or {}
        if not isinstance(field, dict):
            continue
        if field.get("source") != "inferred" or not field.get("value"):
            continue
        value = field["value"]
        st.write(f"Based on your experience, we think you prefer **{label}**: {value}")
        choice = st.radio(
            f"Confirm {label}",
            options=["Yes, that's right", "No, change this"],
            key=f"infer_{field_key}",
            horizontal=True,
        )
        if choice == "No, change this":
            edits[field_key] = st.text_input(
                f"Your preferred {label}",
                value=str(value),
                key=f"edit_{field_key}",
            )
            st.session_state.step5_preferences_dirty = True


def _step_review() -> None:
    st.markdown("## Here's what we found")
    profile = st.session_state.get("profile") or {}
    if isinstance(profile, str):
        profile = json.loads(profile)
    name = profile.get("name") or st.session_state.get("candidate_name") or "Candidate"
    st.subheader(name)
    if profile.get("location"):
        st.write(profile["location"])
    skills = profile.get("skills") or []
    if skills:
        st.markdown("**Skills**")
        chips = []
        for skill in skills[:20]:
            if isinstance(skill, dict):
                label = skill.get("name", "")
                color = _depth_color(float(skill.get("depth_score") or 0))
                chips.append(f":{color}[{label}]")
        st.markdown(" ".join(chips))
    experience = profile.get("experience") or []
    if experience:
        st.markdown("**Experience**")
        for exp in experience[:3]:
            if isinstance(exp, dict):
                st.write(
                    f"- **{exp.get('title', '')}** at {exp.get('company', '')} "
                    f"({exp.get('start_date', '')} – {exp.get('end_date') or 'present'})"
                )
    col1, col2 = st.columns(2)
    with col1:
        if profile.get("role_archetype"):
            st.info(f"Role: {profile['role_archetype']}")
    with col2:
        if profile.get("career_trajectory"):
            st.info(f"Trajectory: {profile['career_trajectory']}")
    if profile.get("domains"):
        st.write("Domains: " + ", ".join(profile["domains"][:8]))
    gh = profile.get("github_summary")
    if gh and isinstance(gh, dict):
        st.markdown("**GitHub**")
        st.write(
            f"Repos: {gh.get('total_repos', 0)} · "
            f"Languages: {', '.join(gh.get('top_languages') or [])} · "
            f"Activity: {gh.get('activity_level', 'n/a')}"
        )
    _render_inferred_preferences(profile)
    if st.button("Edit preferences", key="review_edit_prefs"):
        st.session_state.onboarding_step = 3
        st.rerun()
    if st.button("Looks good — show me my jobs", type="primary", key="review_finish"):
        st.session_state.onboarding_step = 6
        st.rerun()


def _build_step5_patch_payload() -> Optional[dict[str, Any]]:
    """Build preferences patch from step 5 edits if any."""
    if not st.session_state.get("step5_preferences_dirty"):
        return None
    edits = st.session_state.get("step5_preference_edits") or {}
    base = dict(st.session_state.get("preferences") or {})
    if edits.get("preferred_company_stage"):
        base.setdefault("company_stages", [])
        if edits["preferred_company_stage"] not in base["company_stages"]:
            base["company_stages"] = [edits["preferred_company_stage"]]
    return base if base else None


def _step_generate_feed() -> None:
    st.markdown("## Finding your best matches...")
    candidate_id = st.session_state.get("candidate_id")
    if not candidate_id:
        st.error("Profile not found. Please go back and rebuild your profile.")
        return
    try:
        patch_payload = _build_step5_patch_payload()
        refresh = bool(patch_payload) or st.session_state.get("step5_preferences_dirty")
        if patch_payload:
            patch_preferences(candidate_id, patch_payload)
        get_recommendations(candidate_id, refresh=refresh)
        st.session_state.onboarding_complete = True
        st.session_state.current_page = "Feed"
        st.success("Your personalized feed is ready!")
        st.rerun()
    except ApiError as exc:
        st.error(exc.message)


def render() -> None:
    """Render the onboarding wizard."""
    step = int(st.session_state.get("onboarding_step") or 1)
    steps = {
        1: _step_resume,
        2: _step_github,
        3: _step_preferences,
        4: _step_build_profile,
        5: _step_review,
        6: _step_generate_feed,
    }
    steps.get(step, _step_resume)()
