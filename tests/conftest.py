"""Pytest fixtures and markers."""

import os
import socket
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from config.settings import get_settings
from src.api.main import create_app


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring Docker services",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that may call live LLM or full pipeline",
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


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def integration_services_available() -> bool:
    """Return True when core Docker services appear reachable."""
    return all(
        (
            _port_open("127.0.0.1", 5432),
            _port_open("127.0.0.1", 6379),
            _port_open("127.0.0.1", 9200),
            _port_open("127.0.0.1", 7687),
        ),
    )


@pytest.fixture(scope="session")
def integration_services_ok() -> bool:
    return integration_services_available()


@pytest.fixture
async def api_client():
    """Async HTTP client against the FastAPI app (no uvicorn)."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def resume_pdf_path(tmp_path: Path) -> Path:
    """Minimal PDF bytes for upload tests."""
    path = tmp_path / "resume.pdf"
    path.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj "
        b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF\n",
    )
    return path


@pytest.fixture
def fixtures_resume_dir() -> Path:
    path = Path(__file__).parent / "fixtures" / "resumes"
    path.mkdir(parents=True, exist_ok=True)
    return path
