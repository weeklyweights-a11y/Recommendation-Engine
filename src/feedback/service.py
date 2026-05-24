"""Orchestrate feedback weight persistence."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from src.db.models import Candidate
from src.feedback.schemas import AdjustedWeights
from src.feedback.tracker import FeedbackTracker
from src.feedback.weight_adjuster import WeightAdjuster
from src.matching.recommendation_pipeline import _normalize_utility_weights


def apply_feedback_weights(
    candidate_id: UUID,
    session: Session,
    settings: Optional[Settings] = None,
) -> Optional[AdjustedWeights]:
    """Analyze feedback and persist adjusted utility weights on the candidate."""
    cfg = settings or get_settings()
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        return None

    summary = FeedbackTracker(session, cfg).get_feedback_summary(candidate_id)
    if not summary.has_enough_data:
        return None

    adjuster = WeightAdjuster(cfg)
    current = _normalize_utility_weights(candidate.utility_weights) or adjuster._default_weights()
    result = WeightAdjuster(cfg).adjust_weights(current, summary)
    if not result.adjustments_made:
        return result

    candidate.utility_weights = {
        "weights": result.weights,
        "adjustments_made": result.adjustments_made,
        "previous_weights": result.previous_weights,
        "adjustment_magnitude": result.adjustment_magnitude,
    }
    session.add(candidate)
    session.commit()
    return result
