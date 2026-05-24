"""Job recommendation card component."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import streamlit as st

from frontend.config import get_frontend_settings
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


def _skill_tags(rec: dict[str, Any], profile_skills: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return matched and gap skill labels."""
    matched: list[str] = []
    gaps: list[str] = []
    graph = rec.get("graph_matched_skills") or []
    if graph:
        for item in graph:
            if isinstance(item, dict):
                name = item.get("skill") or item.get("name") or ""
                if name:
                    matched.append(str(name))
        return matched[:12], gaps[:8]

    profile_names = {
        str(s.get("name", "")).lower()
        for s in profile_skills
        if isinstance(s, dict) and s.get("name")
    }
    job_skills = rec.get("job", {}).get("skills_extracted") or []
    for raw in job_skills:
        if isinstance(raw, dict):
            name = raw.get("name") or raw.get("skill")
        else:
            name = str(raw)
        if not name:
            continue
        if str(name).lower() in profile_names:
            matched.append(str(name))
        else:
            gaps.append(str(name))
    return matched[:12], gaps[:8]


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

    border = "2px solid #f0c040" if saved_highlight else "1px solid #e0e0e0"
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

        matched, gap_skills = _skill_tags(rec, profile_skills)
        if matched:
            st.markdown("**Skills matched**")
            st.markdown(" ".join(f"`{s}`" for s in matched))
        gap_text = explanation.get("gaps")
        if gap_skills:
            st.markdown("**Skills to develop**")
            st.markdown(" ".join(f"`{s}`" for s in gap_skills))
        elif gap_text and gap_text != "None significant":
            st.markdown(f"**Gaps:** {gap_text}")

        graph = rec.get("graph_matched_skills") or []
        for item in graph[:3]:
            if isinstance(item, dict) and item.get("expansion"):
                skill = item.get("skill") or item.get("name") or "skill"
                st.caption(f"{skill} → matched via: {item['expansion']}")

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
