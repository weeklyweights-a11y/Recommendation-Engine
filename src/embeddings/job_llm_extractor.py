"""LLM job field extraction via Gemini Flash."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from config.settings import Settings, get_settings
from src.db.models import Job
from src.embeddings.job_prompts import (
    JOB_EXTRACTION_RETRY_PROMPT,
    JOB_EXTRACTION_SYSTEM_PROMPT,
    JOB_EXTRACTION_USER_TEMPLATE,
)
from src.embeddings.schemas import JobFields

logger = logging.getLogger(__name__)


def _strip_json_response(text: str) -> str:
    """Remove markdown fences from LLM output."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    start = cleaned.find("{")
    if start >= 0:
        return cleaned[start:].strip()
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output with repair fallback."""
    cleaned = _strip_json_response(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            import json_repair

            data = json.loads(json_repair.repair_json(cleaned))
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    return data


def _log_token_usage(usage_metadata: Any, label: str) -> None:
    """Log Gemini token usage."""
    if usage_metadata is None:
        return
    input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
    total = int(getattr(usage_metadata, "total_token_count", 0) or (input_tokens + output_tokens))
    logger.info(
        "Gemini job extract %s — input=%s output=%s total=%s",
        label,
        input_tokens,
        output_tokens,
        total,
    )


def extract_job_fields_llm(
    job: Job,
    settings: Optional[Settings] = None,
    client: Optional[genai.Client] = None,
) -> JobFields:
    """Extract JobFields using Gemini Flash."""
    cfg = settings or get_settings()
    api_key = cfg.llm.google_ai_api_key
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY is not configured")

    gemini_client = client or genai.Client(api_key=api_key)
    model = cfg.llm.llm_model_flash
    user_prompt = JOB_EXTRACTION_USER_TEMPLATE.format(
        title=job.title or "",
        company=job.company or "",
        description=(job.description or "")[:12000],
    )

    last_error: Optional[Exception] = None
    for attempt, prompt in enumerate([user_prompt, JOB_EXTRACTION_RETRY_PROMPT + "\n\n" + user_prompt]):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JOB_EXTRACTION_SYSTEM_PROMPT,
                    max_output_tokens=cfg.llm.llm_max_tokens,
                    response_mime_type="application/json",
                ),
            )
            _log_token_usage(getattr(response, "usage_metadata", None), f"attempt={attempt + 1}")
            data = _parse_json_object(response.text or "")
            fields = JobFields.model_validate(data)
            fields.extraction_method = "llm"
            fields.job_title = job.title or fields.job_title
            fields.industry = job.industry or fields.industry
            fields.company_stage = job.company_stage or fields.company_stage
            fields.company_size = job.company_size or fields.company_size
            fields.remote_type = job.remote_type or fields.remote_type
            return fields
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("Job LLM parse failed attempt %s: %s", attempt + 1, exc)

    msg = f"Job LLM extraction failed after retry: {last_error}"
    raise ValueError(msg) from last_error
