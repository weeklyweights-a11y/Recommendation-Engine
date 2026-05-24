"""Streamlit frontend configuration from environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config.settings import FrontendSettings, get_settings


@lru_cache
def get_frontend_settings() -> FrontendSettings:
    """Return cached frontend settings."""
    return get_settings().frontend


@lru_cache
def load_frontend_options() -> dict[str, Any]:
    """Load YAML option lists for forms."""
    path = Path(get_frontend_settings().frontend_options_path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
