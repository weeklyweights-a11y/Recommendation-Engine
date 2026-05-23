"""Pytest fixtures and markers."""

import os

import pytest

from config.settings import get_settings


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring Docker services",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip integration tests unless RUN_INTEGRATION=1."""
    if os.getenv("RUN_INTEGRATION", "").strip() == "1":
        return
    skip_integration = pytest.mark.skip(
        reason="Set RUN_INTEGRATION=1 to run integration tests",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def settings():
    """Application settings fixture."""
    get_settings.cache_clear()
    return get_settings()
