"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, TimeoutError as SQLTimeoutError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config.settings import get_settings
from src.api.dependencies import get_embedding_encoder, get_faiss_manager, get_neo4j
from src.api.routes import candidates, feedback, github, health, recommendations

logger = logging.getLogger(__name__)


def _parse_cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm infrastructure clients on startup."""
    settings = get_settings()
    logging.basicConfig(level=settings.app.log_level, format=settings.cache.log_format)
    try:
        get_faiss_manager(settings)
        get_embedding_encoder(settings)._load_model()
        logger.info("FAISS indexes and embedding model warmed")
    except Exception as exc:
        logger.warning("Startup warm-up partial failure: %s", exc)
    yield
    try:
        get_neo4j(settings).close()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.api.rate_limit])

    app = FastAPI(
        title="Job Recommendation API",
        description="Hybrid retrieval with explainable multi-factor matching",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(settings.api.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(OperationalError)
    async def postgres_unavailable_handler(_request: Request, _exc: OperationalError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Database is temporarily unavailable. Please try again shortly."},
        )

    @app.exception_handler(SQLTimeoutError)
    async def postgres_timeout_handler(_request: Request, _exc: SQLTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"detail": "The system is under heavy load, please try again."},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    api_router = APIRouter(prefix="/api/v1")
    api_router.include_router(health.router)
    api_router.include_router(candidates.router)
    api_router.include_router(github.router)
    api_router.include_router(recommendations.router)
    api_router.include_router(feedback.router)
    app.include_router(api_router)
    return app


app = create_app()
