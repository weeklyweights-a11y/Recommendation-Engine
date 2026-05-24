"""Input sanitization for API payloads."""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(value: str, max_length: int) -> str:
    """Strip control characters and enforce max length."""
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    return cleaned[:max_length]
