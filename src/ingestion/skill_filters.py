"""Heuristics to drop resume noise misclassified as skills."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[@\s][\w.+-]+@[\w.-]+\.\w+", re.I)
_EMAIL_ONLY_RE = re.compile(r"^[\w.+-]+@[\w.-]+\.\w+$", re.I)
_PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d|^\d{3}[-.\s]?\d{3}[-.\s]?\d{4})")
_URL_RE = re.compile(r"https?://|www\.|linkedin\.com|github\.com", re.I)
_DATE_RANGE_RE = re.compile(r"\b\d{4}\s*[-–]\s*(\d{4}|present)\b", re.I)
_CURRENCY_RE = re.compile(r"\$[\d,]+|[\d,]+\s*(?:k|m|million|billion)\b", re.I)
_SECTION_HEADER_RE = re.compile(
    r"^(summary|professional experience|work experience|education|projects|"
    r"technical skills|skills|certifications|references|objective|profile)\s*$",
    re.I,
)

# Resume prose / contact / metrics — not skills.
_JUNK_SUBSTRINGS: tuple[str, ...] = (
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "united states",
    "what they did",
    "key achievements",
    "professional experience",
    "technical skills",
    "years of experience",
    "proven track record",
    "scaled to",
    "saved clients",
    "revenue",
    "remote)",
    "hyderabad",
    "los angeles",
    "anderson school",
    "national institute",
    "driving an",
    "designed and",
    "engineered features",
    "enabling analysts",
    "reducing misclassification",
    "maintaining model",
    "categorization accuracy",
    "decision-makers",
    "build-in-public",
    "patent)",
    "platforms (",
    "programming &",
    "genai &",
    "data engineering &",
    "model deployment &",
    "visualization &",
)

# Known tech tokens (lowercase) — single-word supplements must match or be close.
_KNOWN_SHORT: frozenset[str] = frozenset(
    {
        "r",
        "go",
        "sql",
        "aws",
        "gcp",
        "nlp",
        "llm",
        "ml",
        "ai",
        "ci",
        "cd",
        "etl",
        "api",
        "gpu",
        "tpu",
        "mas",
        "rag",
        "xai",
        "nas",
        "cv",
    },
)


def is_plausible_skill(name: str) -> bool:
    """Return False for contact info, section headers, and resume sentence fragments."""
    raw = (name or "").strip()
    if not raw:
        return False
    if len(raw) < 2 or len(raw) > 56:
        return False

    lower = raw.lower()
    words = lower.split()
    if len(words) > 5:
        return False

    if _EMAIL_RE.search(raw) or _EMAIL_ONLY_RE.match(raw):
        return False
    if _PHONE_RE.search(raw):
        return False
    if _URL_RE.search(raw):
        return False
    if _DATE_RANGE_RE.search(raw):
        return False
    if _CURRENCY_RE.search(raw):
        return False
    if _SECTION_HEADER_RE.match(raw):
        return False

    for junk in _JUNK_SUBSTRINGS:
        if junk in lower:
            return False

    # Mostly punctuation or digits
    alpha = sum(1 for c in raw if c.isalpha())
    if alpha < max(2, len(raw) // 4):
        return False

    # Single generic words that appear in prose
    if len(words) == 1:
        if lower in {
            "remote",
            "summary",
            "education",
            "projects",
            "experience",
            "dates",
            "india",
            "usa",
            "ca",
            "and",
            "with",
            "using",
            "ensuring",
            "including",
            "enabling",
            "reducing",
            "improving",
            "processing",
            "driving",
            "designed",
            "developed",
            "implemented",
            "maintaining",
            "collaboration",
            "monitoring",
            "deployment",
            "latency",
            "accuracy",
            "efficiency",
            "retention",
            "stock",
            "levels",
            "cloud",
            "agentic",
            "sentence",
            "transformers",
        }:
            return False
        if lower not in _KNOWN_SHORT and len(lower) <= 2 and lower not in {"c", "r"}:
            return False

    # Sentence-like fragments
    if raw.endswith(".") and len(words) > 2:
        return False
    if lower.startswith(("and ", "with ", "to ", "for ", "the ", "a ", "an ")):
        return False
    if "(" in raw and ")" not in raw:
        return False

    return True


def filter_skill_names(names: list[str]) -> list[str]:
    """Keep only plausible skill names, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not is_plausible_skill(name):
            continue
        key = re.sub(r"\s+", " ", name.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(name.strip())
    return out
