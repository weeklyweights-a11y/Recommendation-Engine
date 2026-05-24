"""LLM-powered structured resume profile extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from config.settings import Settings, get_settings
from src.ingestion.exceptions import ExtractionFailedError
from src.ingestion.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    GITHUB_SECTION_TEMPLATE,
    RETRY_USER_PROMPT,
)
from src.ingestion.schemas import ExtractedProfile, ExtractionResult, TokenUsage

logger = logging.getLogger(__name__)


def _strip_json_response(text: str) -> str:
    """Remove markdown fences and leading prose from LLM output."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_usage(response: Any) -> TokenUsage:
    """Extract token counts from Gemini usage metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return TokenUsage()
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens))
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _log_token_usage(usage: TokenUsage, label: str) -> None:
    """Log token usage at INFO for cost tracking."""
    logger.info(
        "Gemini %s tokens — input=%s output=%s total=%s",
        label,
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
    )


def _build_user_prompt(resume_text: str, github_summary: str | None) -> str:
    """Build the extraction user prompt."""
    github_section = ""
    if github_summary:
        github_section = GITHUB_SECTION_TEMPLATE.format(github_summary=github_summary)
    return EXTRACTION_USER_TEMPLATE.format(
        resume_text=resume_text,
        github_section=github_section,
    )


def _call_gemini(
    client: genai.Client,
    model: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, TokenUsage]:
    """Invoke Gemini and return text plus usage."""
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    usage = _parse_usage(response)
    return text, usage


def extract_profile(
    resume_text: str,
    github_summary: str | None = None,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> ExtractedProfile:
    """Extract a structured profile from resume text via Gemini Pro."""
    return extract_profile_with_usage(
        resume_text,
        github_summary=github_summary,
        settings=settings,
        client=client,
    ).profile


def extract_profile_with_usage(
    resume_text: str,
    github_summary: str | None = None,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> ExtractionResult:
    """Extract profile and return token usage metadata."""
    cfg = settings or get_settings()
    api_key = cfg.llm.google_ai_api_key
    if not api_key:
        raise ExtractionFailedError("GOOGLE_AI_API_KEY is not configured")

    gemini_client = client or genai.Client(api_key=api_key)
    model = cfg.llm.llm_model_pro
    max_tokens = cfg.llm.llm_max_tokens
    user_prompt = _build_user_prompt(resume_text, github_summary)

    last_error: Optional[Exception] = None
    total_usage = TokenUsage()

    for attempt, prompt in enumerate([user_prompt, RETRY_USER_PROMPT]):
        try:
            raw_text, usage = _call_gemini(gemini_client, model, prompt, max_tokens)
            total_usage = TokenUsage(
                input_tokens=total_usage.input_tokens + usage.input_tokens,
                output_tokens=total_usage.output_tokens + usage.output_tokens,
                total_tokens=total_usage.total_tokens + usage.total_tokens,
            )
            _log_token_usage(usage, f"extract attempt={attempt + 1}")

            cleaned = _strip_json_response(raw_text)
            data = json.loads(cleaned)
            profile = ExtractedProfile.model_validate(data)
            _log_token_usage(total_usage, "extract total")
            return ExtractionResult(profile=profile, token_usage=total_usage)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("LLM extraction parse failed (attempt %s): %s", attempt + 1, exc)
            if attempt == 0:
                continue

    msg = f"LLM extraction failed after retry: {last_error}"
    raise ExtractionFailedError(msg) from last_error
