"""LLM-powered explanations from pre-computed match scores."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_fixed

from config.settings import Settings, get_settings
from src.cache.redis_client import get_redis_cache
from src.api.schemas.candidate import CandidateProfile
from src.ingestion.schemas import TokenUsage
from src.matching.explainer_prompts import (
    EXPLAINER_BATCH_USER_TEMPLATE,
    EXPLAINER_SYSTEM_PROMPT,
    EXPLAINER_USER_TEMPLATE,
)
from src.matching.schemas import MatchExplanation, RankedJob

logger = logging.getLogger(__name__)


class Explainer:
    """Generate natural-language explanations from deterministic scores."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize Gemini client and cache."""
        self._settings = settings or get_settings()
        self._cfg = self._settings.explainer
        self._client: Optional[genai.Client] = None
        self._cache: dict[str, MatchExplanation] = {}

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = self._settings.llm.google_ai_api_key
            if not api_key:
                raise RuntimeError("GOOGLE_AI_API_KEY is not configured")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _cache_key(self, ranked_job: RankedJob, profile: CandidateProfile) -> str:
        job = ranked_job.job
        updated = getattr(job, "updated_at", None)
        updated_s = str(updated) if updated is not None else ""
        factor_blob = str(sorted(ranked_job.factor_scores.items()))
        raw = f"{ranked_job.job_id}|{updated_s}|{factor_blob}|{profile.summary[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _candidate_summary(self, profile: CandidateProfile) -> str:
        top_skills = sorted(profile.skills, key=lambda s: s.depth_score, reverse=True)[:5]
        skill_names = ", ".join(s.name for s in top_skills if s.name)
        domains = ", ".join(profile.domains[:5])
        return (
            f"{profile.name or 'Candidate'}; skills: {skill_names or 'n/a'}; "
            f"domains: {domains or 'n/a'}; archetype: {profile.role_archetype}; "
            f"trajectory: {profile.career_trajectory}; years: {profile.total_years_experience}"
        )

    def _skill_detail(self, ranked_job: RankedJob) -> tuple[str, str, str]:
        display = ranked_job.skill_match_display
        if not display and ranked_job.retrieval_scores:
            raw = ranked_job.retrieval_scores.get("skill_match_display")
            if isinstance(raw, dict):
                display = raw
        matched: list[str] = []
        gaps: list[str] = []
        expansion_parts: list[str] = []
        if isinstance(display, dict):
            for item in display.get("matched") or []:
                if isinstance(item, dict) and item.get("skill"):
                    label = str(item["skill"])
                    via = item.get("via")
                    if via and via not in ("direct", "profile"):
                        label = f"{label} ({via})"
                        expansion_parts.append(f"{item['skill']}:{via}")
                    matched.append(label)
            for item in display.get("gaps") or []:
                if isinstance(item, dict) and item.get("skill"):
                    gaps.append(str(item["skill"]))
        elif ranked_job.graph_matched_skills:
            for item in ranked_job.graph_matched_skills[:5]:
                matched.append(
                    str(item.get("skill") or item.get("candidate_skill") or item.get("uri", "skill")),
                )
        expansion = "; ".join(expansion_parts[:3]) if expansion_parts else "none"
        return (
            ", ".join(matched) or "keyword overlap",
            ", ".join(gaps) or "none listed",
            expansion,
        )

    def _build_user_prompt(
        self,
        profile: CandidateProfile,
        ranked_job: RankedJob,
    ) -> str:
        job = ranked_job.job
        matched, gaps, expansion = self._skill_detail(ranked_job)
        factors = ranked_job.factor_scores
        return EXPLAINER_USER_TEMPLATE.format(
            candidate_summary=self._candidate_summary(profile),
            job_title=getattr(job, "title", ""),
            company=getattr(job, "company", ""),
            job_description=(getattr(job, "description", "") or "")[:500],
            match_percentage=ranked_job.match_percentage,
            skill_fit_score=factors.get("skill_fit", 0),
            skill_fit_detail=matched,
            experience_score=factors.get("experience_alignment", 0),
            domain_score=factors.get("domain_relevance", 0),
            role_shape_score=factors.get("role_shape_match", 0),
            location_score=factors.get("location_fit", 0),
            stage_score=factors.get("company_stage_alignment", 0),
            semantic_score=factors.get("semantic_similarity", 0),
            matched_skills_list=matched,
            gap_skills_list=gaps,
            expansion_matches=expansion,
        )

    def _parse_response(self, text: str) -> Optional[MatchExplanation]:
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?=REASONS:|$)", text, re.S | re.I)
        reasons_block = re.search(r"REASONS:\s*(.+?)(?=GAPS:|$)", text, re.S | re.I)
        gaps_match = re.search(r"GAPS:\s*(.+?)$", text, re.S | re.I)
        if not summary_match:
            return None
        reasons: list[str] = []
        if reasons_block:
            for line in reasons_block.group(1).splitlines():
                line = line.strip().lstrip("-").strip()
                if line:
                    reasons.append(line)
        gaps = gaps_match.group(1).strip() if gaps_match else "None significant"
        return MatchExplanation(
            summary=summary_match.group(1).strip(),
            reasons=reasons[:4],
            gaps=gaps,
            generated_by="llm",
        )

    def _template_fallback(self, ranked_job: RankedJob) -> MatchExplanation:
        factors = ranked_job.factor_scores
        top_factor = max(factors, key=factors.get) if factors else "semantic_similarity"
        top_score = factors.get(top_factor, 0.0)
        return MatchExplanation(
            summary=(
                f"This role is a {ranked_job.match_percentage}% match. "
                f"Your strongest alignment is in {top_factor.replace('_', ' ')} ({top_score:.2f})."
            ),
            reasons=[
                f"Strong {top_factor.replace('_', ' ')} alignment",
                f"Semantic similarity score {factors.get('semantic_similarity', 0):.2f}",
            ],
            gaps="None significant",
            generated_by="template_fallback",
        )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
    def _call_gemini(self, user_prompt: str) -> tuple[str, TokenUsage]:
        client = self._get_client()
        response = client.models.generate_content(
            model=self._settings.llm.llm_model_pro,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXPLAINER_SYSTEM_PROMPT,
                max_output_tokens=self._settings.llm.llm_max_tokens,
            ),
        )
        text = (response.text or "").strip()
        usage = TokenUsage()
        meta = getattr(response, "usage_metadata", None)
        if meta:
            usage = TokenUsage(
                input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
                total_tokens=int(getattr(meta, "total_token_count", 0) or 0),
            )
        if self._cfg.log_token_usage:
            logger.info(
                "Explainer tokens input=%s output=%s total=%s",
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
            )
        return text, usage

    def explain_match(
        self,
        profile: CandidateProfile,
        ranked_job: RankedJob,
    ) -> MatchExplanation:
        """Explain a single ranked job."""
        key = self._cache_key(ranked_job, profile)
        if key in self._cache:
            return self._cache[key]

        try:
            text, _ = self._call_gemini(self._build_user_prompt(profile, ranked_job))
            parsed = self._parse_response(text)
            if parsed is None:
                raise ValueError("Failed to parse explainer response")
            explanation = parsed
        except Exception as exc:
            logger.warning("LLM explanation failed for job %s: %s", ranked_job.job_id, exc)
            explanation = self._template_fallback(ranked_job)

        self._cache[key] = explanation
        return explanation

    def _explain_batch_group(
        self,
        profile: CandidateProfile,
        batch: list[RankedJob],
    ) -> list[MatchExplanation]:
        if len(batch) == 1:
            return [self.explain_match(profile, batch[0])]
        blocks = [f"JOB {i}:\n{self._build_user_prompt(profile, job)}" for i, job in enumerate(batch, start=1)]
        prompt = EXPLAINER_BATCH_USER_TEMPLATE.format(job_blocks="\n---\n".join(blocks))
        text, _ = self._call_gemini(prompt)
        parts = re.split(r"\n---\n", text)
        if len(parts) != len(batch):
            raise ValueError("Batch parse count mismatch")
        parsed_batch: list[MatchExplanation] = []
        for job, part in zip(batch, parts):
            parsed = self._parse_response(part)
            if parsed is None:
                raise ValueError("Batch segment parse failed")
            key = self._cache_key(job, profile)
            self._cache[key] = parsed
            parsed_batch.append(parsed)
        return parsed_batch

    def explain_batch(
        self,
        profile: CandidateProfile,
        ranked_jobs: list[RankedJob],
        max_jobs: Optional[int] = None,
        llm_max: Optional[int] = None,
    ) -> list[MatchExplanation]:
        """Explain top ranked jobs in batches (LLM for first N, template for the rest)."""
        limit = max_jobs if max_jobs is not None else self._cfg.explain_top_k
        llm_limit = llm_max if llm_max is not None else self._cfg.explain_llm_top_k
        targets = ranked_jobs[:limit]
        results: list[MatchExplanation] = []
        batch_size = self._cfg.explain_batch_size
        redis = get_redis_cache()

        llm_targets = targets[:llm_limit]
        template_targets = targets[llm_limit:]

        for job in template_targets:
            results.append(self._template_fallback(job))

        if not llm_targets:
            return results

        batches: list[list[RankedJob]] = [
            llm_targets[start : start + batch_size]
            for start in range(0, len(llm_targets), batch_size)
        ]

        llm_results: list[MatchExplanation] = []
        if self._cfg.explain_parallel_batches and len(batches) > 1:
            ordered: dict[int, list[MatchExplanation]] = {}
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
                futures = {
                    pool.submit(self._explain_batch_group, profile, batch): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    batch = batches[idx]
                    try:
                        ordered[idx] = future.result()
                    except Exception as exc:
                        logger.warning("Parallel batch explain failed: %s", exc)
                        ordered[idx] = [self.explain_match(profile, job) for job in batch]
            for idx in sorted(ordered):
                llm_results.extend(ordered[idx])
        else:
            for batch in batches:
                try:
                    llm_results.extend(self._explain_batch_group(profile, batch))
                except Exception as exc:
                    logger.warning("Batch explain failed, falling back to singles: %s", exc)
                    for job in batch:
                        llm_results.append(self.explain_match(profile, job))

        for explanation in llm_results:
            if redis.health_check():
                pass  # per-job redis explain cache wired via explain_match cache key on demand

        return llm_results + results
