"""Heuristic utility weight adjustment from feedback patterns."""

from __future__ import annotations

from typing import Optional

from config.settings import Settings, get_settings
from src.feedback.schemas import AdjustedWeights, FeedbackSummary
from src.matching.reranker import FACTOR_KEYS


class WeightAdjuster:
    """Adjust reranker utility weights based on feedback summary."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Load bounds and thresholds from settings."""
        self._settings = settings or get_settings()
        self._cfg = self._settings.feedback
        self._rerank_cfg = self._settings.reranker

    def _default_weights(self) -> dict[str, float]:
        c = self._rerank_cfg
        return {
            "skill_fit": c.skill_fit_weight,
            "experience_alignment": c.experience_alignment_weight,
            "domain_relevance": c.domain_relevance_weight,
            "role_shape_match": c.role_shape_weight,
            "location_fit": c.location_fit_weight,
            "company_stage_alignment": c.company_stage_weight,
            "semantic_similarity": c.semantic_similarity_weight,
        }

    def adjust_weights(
        self,
        current_weights: dict[str, float],
        feedback_summary: FeedbackSummary,
    ) -> AdjustedWeights:
        """Apply heuristic rules and return normalized adjusted weights."""
        if not feedback_summary.has_enough_data:
            return AdjustedWeights(
                weights=dict(current_weights),
                previous_weights=dict(current_weights),
            )

        weights = dict(current_weights)
        previous = dict(current_weights)
        adjustments: list[str] = []
        rate = self._cfg.adjustment_rate
        cfg = self._cfg

        if feedback_summary.preferred_stages:
            top_stage, freq = feedback_summary.preferred_stages[0]
            dismissed_stages = {s for s, _ in feedback_summary.avoided_stages}
            if freq >= cfg.stage_concentration_threshold and (
                not dismissed_stages or top_stage not in dismissed_stages
            ):
                weights["company_stage_alignment"] = weights.get(
                    "company_stage_alignment",
                    0.0,
                ) + rate
                adjustments.append(
                    f"Increased company stage weight (prefers {top_stage})",
                )

        if feedback_summary.preferred_industries:
            top_ind, freq = feedback_summary.preferred_industries[0]
            unique_inds = len(feedback_summary.preferred_industries)
            if freq >= cfg.domain_concentration_threshold and unique_inds <= 2:
                weights["domain_relevance"] = weights.get("domain_relevance", 0.0) + rate
                adjustments.append(f"Increased domain relevance (focus on {top_ind})")
            elif unique_inds >= cfg.domain_diversity_min_industries:
                weights["domain_relevance"] = max(
                    0.0,
                    weights.get("domain_relevance", 0.0) - rate,
                )
                adjustments.append("Decreased domain relevance (generalist pattern)")

        if feedback_summary.applied_count >= 2 and feedback_summary.preferred_stages:
            weights["role_shape_match"] = weights.get("role_shape_match", 0.0) + rate
            adjustments.append("Increased role shape match (consistent applied roles)")

        if any("prefers remote" in s for s in feedback_summary.strong_positive_signals):
            weights["location_fit"] = weights.get("location_fit", 0.0) + rate
            adjustments.append("Increased location fit (remote preference)")

        avg_skill = feedback_summary.avg_skill_fit_saved
        if avg_skill is not None:
            if avg_skill < cfg.skill_stretch_max_avg:
                weights["skill_fit"] = max(0.0, weights.get("skill_fit", 0.0) - rate)
                adjustments.append("Decreased skill fit (willing to stretch)")
            elif avg_skill >= cfg.skill_overlap_min_avg:
                weights["skill_fit"] = weights.get("skill_fit", 0.0) + rate
                adjustments.append("Increased skill fit (high overlap saves)")

        avg_sem = feedback_summary.avg_semantic_saved
        avg_other = feedback_summary.avg_other_factors_saved
        if (
            avg_sem is not None
            and avg_sem >= cfg.semantic_high_min
            and avg_other is not None
            and avg_other <= cfg.semantic_factor_moderate_max
        ):
            weights["semantic_similarity"] = weights.get("semantic_similarity", 0.0) + rate
            adjustments.append("Increased semantic similarity (vibe-driven matches)")

        weights = self._clamp_normalize(weights)
        magnitude = sum(abs(weights[k] - previous.get(k, 0.0)) for k in weights)
        return AdjustedWeights(
            weights=weights,
            adjustments_made=adjustments,
            previous_weights=previous,
            adjustment_magnitude=magnitude,
        )

    def _clamp_normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp each weight and renormalize to sum 1."""
        min_w = self._cfg.weight_min
        max_w = self._cfg.weight_max
        cleaned = {k: weights.get(k, 0.0) for k in FACTOR_KEYS}
        for key in cleaned:
            cleaned[key] = max(min_w, min(max_w, cleaned[key]))
        total = sum(cleaned.values())
        if total <= 0:
            return self._normalize_dict(self._default_weights())
        return {k: v / total for k, v in cleaned.items()}

    @staticmethod
    def _normalize_dict(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}
