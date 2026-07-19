"""Job recommendation card component."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import streamlit as st

from frontend.ui_settings import get_frontend_settings
from frontend.utils.formatting import (
    format_factor_scores,
    format_posted_freshness,
    match_pct_color,
    parse_explanation,
)


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _salary_label(job: dict[str, Any]) -> str:
    smin, smax = job.get("salary_min"), job.get("salary_max")
    currency = job.get("currency") or "USD"
    if smin and smax:
        return f"{currency} {smin:,} – {smax:,}"
    if smax:
        return f"Up to {currency} {smax:,}"
    if smin:
        return f"From {currency} {smin:,}"
    return ""


def _normalize_skill_name(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


def _profile_skill_keys(skill: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("name", "esco_label"):
        value = skill.get(field)
        if value:
            norm = _normalize_skill_name(str(value))
            if norm:
                keys.add(norm)
    return keys


def _skills_equivalent(candidate_keys: set[str], job_name: str) -> bool:
    job_keys = {_normalize_skill_name(job_name)}
    if not job_keys or not candidate_keys:
        return False
    if candidate_keys & job_keys:
        return True
    for ck in candidate_keys:
        for jk in job_keys:
            if ck == jk or ck in jk or jk in ck:
                return True
    return False


def _skill_tags(rec: dict[str, Any], profile_skills: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """Return matched labels, gap labels, and optional via captions."""
    matched: list[str] = []
    gaps: list[str] = []
    via_notes: list[str] = []

    display = rec.get("skill_match_display")
    if not display and rec.get("retrieval_scores"):
        raw = rec["retrieval_scores"].get("skill_match_display")
        if isinstance(raw, dict):
            display = raw

    if isinstance(display, dict):
        for item in display.get("matched") or []:
            if isinstance(item, dict):
                name = str(item.get("skill") or "")
                if name:
                    matched.append(name)
                    via = item.get("via")
                    if via and via not in ("direct", "profile"):
                        via_notes.append(f"{name} (via {via})")
        for item in display.get("gaps") or []:
            if isinstance(item, dict):
                name = str(item.get("skill") or "")
                if name:
                    gaps.append(name)
        return matched[:16], gaps[:12], via_notes[:6]

    candidate_keys: set[str] = set()
    for skill in profile_skills:
        if isinstance(skill, dict):
            candidate_keys |= _profile_skill_keys(skill)

    job = rec.get("job") or {}
    raw_skills = job.get("skills_extracted")
    job_skill_rows: list[dict[str, Any]] = []
    if isinstance(raw_skills, dict) and isinstance(raw_skills.get("skills"), list):
        job_skill_rows = [s for s in raw_skills["skills"] if isinstance(s, dict)]
    elif isinstance(raw_skills, list):
        for raw in raw_skills:
            if isinstance(raw, dict):
                job_skill_rows.append(raw)
            elif isinstance(raw, str) and raw.strip():
                job_skill_rows.append({"name": raw.strip()})

    for row in job_skill_rows:
        name = str(row.get("name") or row.get("esco_label") or "")
        if not name:
            continue
        if _skills_equivalent(candidate_keys, name):
            matched.append(name)
        else:
            gaps.append(name)

    if not job_skill_rows:
        graph = rec.get("graph_matched_skills") or []
        for item in graph:
            if isinstance(item, dict):
                name = (
                    item.get("skill")
                    or item.get("candidate_skill")
                    or item.get("name")
                    or ""
                )
                if name:
                    matched.append(str(name))
    return matched[:16], gaps[:12], via_notes


def render_job_card(
    rec: dict[str, Any],
    profile_skills: list[dict[str, Any]],
    *,
    candidate_id: Optional[str] = None,
    saved_highlight: bool = False,
) -> None:
    """Render a single recommendation card."""
    job = rec.get("job") or {}
    explanation = parse_explanation(rec.get("explanation"))
    pct = int(rec.get("match_percentage") or round(float(rec.get("match_score", 0)) * 100))
    color = match_pct_color(pct)

    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:4px solid {color}; padding-left:8px'>",
            unsafe_allow_html=True,
        )
        st.markdown(f"### {job.get('title', 'Role')}")
        st.markdown(f"**{job.get('company', '')}**")
        location = job.get("location") or "Location flexible"
        remote = job.get("remote_type") or ""
        st.caption(f"{location} · {remote}" if remote else location)
        salary = _salary_label(job)
        if salary:
            st.write(salary)
        posted = _parse_date(job.get("posted_date"))
        label, freshness = format_posted_freshness(posted)
        st.caption(label)

        st.metric("Match", f"{pct}%")
        reasons = explanation.get("reasons") or []
        for reason in reasons[:3]:
            st.markdown(f"- {reason}")

        matched, gap_skills, via_notes = _skill_tags(rec, profile_skills)
        if matched:
            st.markdown("**Skills matched**")
            st.markdown(" ".join(f"`{s}`" for s in matched))
        gap_text = explanation.get("gaps")
        if gap_skills:
            st.markdown("**Skills to develop**")
            st.markdown(" ".join(f"`{s}`" for s in gap_skills))
        elif gap_text and gap_text != "None significant":
            st.markdown(f"**Gaps:** {gap_text}")
        for note in via_notes:
            st.caption(note)

        with st.expander("See full details"):
            if explanation.get("summary"):
                st.write(explanation["summary"])
            factors = format_factor_scores(rec.get("factor_scores") or {})
            if factors:
                st.bar_chart({label: score for label, score in factors})
            if salary:
                st.write(f"Salary: {salary}")
            url = job.get("source_url")
            if url:
                st.link_button("View original posting →", url)
            desc = job.get("description") or ""
            limit = get_frontend_settings().job_description_preview_chars
            st.write(desc[:limit] + ("…" if len(desc) > limit else ""))

        if candidate_id:
            from frontend.components.feedback_buttons import render_feedback_buttons

            job = rec.get("job") or {}
            render_feedback_buttons(
                candidate_id,
                str(rec.get("job_id")),
                job.get("source_url"),
                key_suffix=str(rec.get("job_id")),
            )
