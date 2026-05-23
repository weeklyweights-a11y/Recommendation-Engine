"""Logging configuration for PersonalMatch."""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with a structured text format.

    Args:
        level: Log level name (e.g. INFO, DEBUG). Defaults to INFO.
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
