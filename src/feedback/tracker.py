"""Aggregate candidate feedback into patterns."""

from __future__ import annotations

from collections import Counter
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.api.schemas.candidate import CandidateProfile
from src.db.models import Candidate, Feedback, Job, Recommendation
from src.embeddings.skills_extracted_parser import parse_skills_extracted
from src.feedback.schemas import FeedbackSummary
from src.matching.reranker import FACTOR_KEYS


class FeedbackTracker:
    """Load and summarize feedback for weight adjustment."""

    def __init__(self, session: Session, settings: Optional[Settings] = None) -> None:
        """Initialize with a database session."""
        self._session = session
        self._settings = settings or get_settings()

    def get_feedback_summary(self, candidate_id: UUID) -> FeedbackSummary:
        """Build a feedback summary for the candidate."""
        cfg = self._settings.feedback
        rows = self._session.scalars(
            select(Feedback).where(Feedback.candidate_id == candidate_id),
        ).all()
        if not rows:
            return FeedbackSummary()

        positive_industries: Counter[str] = Counter()
        positive_stages: Counter[str] = Counter()
        positive_sizes: Counter[str] = Counter()
        positive_remote: Counter[str] = Counter()
        positive_skills: Counter[str] = Counter()
        negative_industries: Counter[str] = Counter()
        negative_stages: Counter[str] = Counter()
        negative_remote: Counter[str] = Counter()
        skill_fits: list[float] = []
        semantics: list[float] = []
        other_factors: list[float] = []

        saved_count = dismissed_count = applied_count = 0
        mult = cfg.applied_signal_multiplier

        candidate = self._session.get(Candidate, candidate_id)
        profile_skills: set[str] = set()
        if candidate and candidate.profile:
            try:
                profile = CandidateProfile.model_validate(candidate.profile)
                profile_skills = {s.name.lower() for s in profile.skills if s.name}
            except Exception:
                profile_skills = set()

        for row in rows:
            job = self._session.get(Job, row.job_id)
            if job is None:
                continue
            weight = 1
            if row.action == "saved":
                saved_count += 1
            elif row.action == "dismissed":
                dismissed_count += 1
                weight = 0
            elif row.action == "applied":
                applied_count += 1
                weight = mult

            if weight == 0:
                if job.industry:
                    negative_industries[job.industry.lower()] += 1
                if job.company_stage:
                    negative_stages[job.company_stage.lower()] += 1
                if job.remote_type:
                    negative_remote[job.remote_type.lower()] += 1
                continue

            if job.industry:
                positive_industries[job.industry.lower()] += weight
            if job.company_stage:
                positive_stages[job.company_stage.lower()] += weight
            if job.company_size:
                positive_sizes[job.company_size.lower()] += weight
            if job.remote_type:
                positive_remote[job.remote_type.lower()] += weight
            for skill in parse_skills_extracted(job.skills_extracted):
                if skill.name:
                    positive_skills[skill.name.lower()] += weight

            rec = self._session.scalar(
                select(Recommendation)
                .where(
                    Recommendation.candidate_id == candidate_id,
                    Recommendation.job_id == row.job_id,
                )
                .order_by(Recommendation.created_at.desc())
                .limit(1),
            )
            if rec and rec.factor_scores:
                fs = rec.factor_scores
                if "skill_fit" in fs:
                    skill_fits.append(float(fs["skill_fit"]) * weight)
                if "semantic_similarity" in fs:
                    semantics.append(float(fs["semantic_similarity"]) * weight)
                others = [
                    float(fs[k])
                    for k in FACTOR_KEYS
                    if k not in ("skill_fit", "semantic_similarity") and k in fs
                ]
                if others:
                    other_factors.append(sum(others) / len(others) * weight)

        total = len(rows)
        summary = FeedbackSummary(
            total_actions=total,
            saved_count=saved_count,
            dismissed_count=dismissed_count,
            applied_count=applied_count,
            preferred_industries=_top_freq(positive_industries),
            preferred_stages=_top_freq(positive_stages),
            preferred_sizes=_top_freq(positive_sizes),
            preferred_remote_types=_top_freq(positive_remote),
            preferred_skills=_top_freq(positive_skills),
            avoided_industries=_top_freq(negative_industries),
            avoided_stages=_top_freq(negative_stages),
            avoided_remote_types=_top_freq(negative_remote),
            has_enough_data=total >= cfg.min_actions_for_adjustment,
        )
        if skill_fits:
            summary.avg_skill_fit_saved = sum(skill_fits) / len(skill_fits)
        if semantics:
            summary.avg_semantic_saved = sum(semantics) / max(len(semantics), 1)
        if other_factors:
            summary.avg_other_factors_saved = sum(other_factors) / max(len(other_factors), 1)

        summary.strong_positive_signals = _contrast_signals(
            positive_remote,
            negative_remote,
            "remote",
            "on-site",
        )
        summary.strong_negative_signals = _contrast_signals(
            negative_remote,
            positive_remote,
            "on-site",
            "remote",
        )

        pivot = [
            skill
            for skill, count in positive_skills.items()
            if skill not in profile_skills and count >= 2
        ]
        if pivot:
            summary.strong_positive_signals.append(
                f"Interested in developing: {', '.join(pivot[:5])}",
            )

        return summary


def _top_freq(counter: Counter[str], limit: int = 10) -> list[tuple[str, float]]:
    total = sum(counter.values()) or 1
    return [(k, v / total) for k, v in counter.most_common(limit)]


def _contrast_signals(
    positive: Counter[str],
    negative: Counter[str],
    pos_label: str,
    neg_label: str,
) -> list[str]:
    pos_total = sum(positive.values())
    neg_total = sum(negative.values())
    signals: list[str] = []
    if pos_total and neg_total:
        remote_pos = positive.get("remote", 0) / pos_total
        onsite_neg = negative.get("onsite", 0) / neg_total
        if remote_pos > 0.7 and onsite_neg > 0.7:
            signals.append(f"strongly prefers {pos_label}")
    return signals
