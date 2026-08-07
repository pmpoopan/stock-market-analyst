"""Application factory and dependency wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.middleware import RequestLoggingMiddleware
from app.api.routes import router as api_router
from app.config.logging_config import setup_logging
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    logger.info("Starting %s v%s [%s]", settings.app_name, __version__, settings.app_env)

    if settings.cache_enabled:
        from app.data.cache import create_cache

        cache = create_cache(settings.cache_db_path)
        app.state.cache = cache
        logger.info("Cache initialized at %s", settings.cache_db_path)

    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="AI-powered Stock Market Analyst for Indian equities",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(api_router, prefix=settings.api_prefix)

    return app
