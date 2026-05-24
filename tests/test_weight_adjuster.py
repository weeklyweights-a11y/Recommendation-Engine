"""Tests for feedback-driven weight adjustment."""

from __future__ import annotations

from src.feedback.schemas import FeedbackSummary
from src.feedback.weight_adjuster import WeightAdjuster


def test_no_adjustment_below_threshold() -> None:
    """Weights unchanged when not enough feedback."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(total_actions=2, has_enough_data=False)
    result = adjuster.adjust_weights(current, summary)
    assert result.weights == current
    assert result.adjustments_made == []


def test_company_stage_weight_increases() -> None:
    """Concentrated stage preference increases company_stage_alignment."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(
        total_actions=6,
        has_enough_data=True,
        preferred_stages=[("seed", 0.8)],
        avoided_stages=[("enterprise", 0.9)],
    )
    result = adjuster.adjust_weights(current, summary)
    assert result.weights["company_stage_alignment"] >= current["company_stage_alignment"]
    assert any("company stage" in msg.lower() for msg in result.adjustments_made)


def test_domain_concentration_increases_domain_weight() -> None:
    """Fintech concentration increases domain_relevance."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(
        total_actions=5,
        has_enough_data=True,
        preferred_industries=[("fintech", 0.75)],
    )
    result = adjuster.adjust_weights(current, summary)
    assert result.weights["domain_relevance"] >= current["domain_relevance"]


def test_domain_diversity_decreases_domain_weight() -> None:
    """Many industries decreases domain_relevance."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(
        total_actions=8,
        has_enough_data=True,
        preferred_industries=[
            ("fintech", 0.2),
            ("healthcare", 0.2),
            ("saas", 0.2),
            ("gaming", 0.2),
            ("other", 0.2),
        ],
    )
    result = adjuster.adjust_weights(current, summary)
    assert result.weights["domain_relevance"] <= current["domain_relevance"]


def test_skill_stretch_decreases_skill_fit() -> None:
    """Low skill overlap on saves decreases skill_fit weight."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(
        total_actions=5,
        has_enough_data=True,
        avg_skill_fit_saved=0.3,
    )
    result = adjuster.adjust_weights(current, summary)
    assert result.weights["skill_fit"] <= current["skill_fit"]


def test_weights_clamped_and_normalized() -> None:
    """Output weights sum to 1 and respect bounds."""
    adjuster = WeightAdjuster()
    current = {k: 0.5 for k in adjuster._default_weights()}
    summary = FeedbackSummary(
        total_actions=10,
        has_enough_data=True,
        preferred_stages=[("seed", 0.9)],
        preferred_industries=[("tech", 0.8)],
        applied_count=5,
        avg_semantic_saved=0.9,
        avg_other_factors_saved=0.5,
        strong_positive_signals=["strongly prefers remote"],
    )
    result = adjuster.adjust_weights(current, summary)
    total = sum(result.weights.values())
    assert abs(total - 1.0) < 0.01
    min_w = adjuster._cfg.weight_min
    max_w = adjuster._cfg.weight_max
    for value in result.weights.values():
        assert min_w - 0.001 <= value <= max_w + 0.001


def test_idempotent_same_summary() -> None:
    """Same input produces same output."""
    adjuster = WeightAdjuster()
    current = adjuster._default_weights()
    summary = FeedbackSummary(
        total_actions=6,
        has_enough_data=True,
        preferred_stages=[("series-a", 0.75)],
    )
    first = adjuster.adjust_weights(current, summary)
    second = adjuster.adjust_weights(current, summary)
    assert first.weights == second.weights
