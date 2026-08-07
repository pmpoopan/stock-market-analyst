"""Uvicorn entry point for the Buddy FastAPI backend."""

import uvicorn

from app.config.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
