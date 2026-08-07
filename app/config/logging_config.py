"""Central logging configuration for Buddy."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.config.settings import Settings

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_NOISY_LOGGERS = (
    "urllib3",
    "httpx",
    "httpcore",
    "watchfiles",
    "uvicorn.access",
)


def setup_logging(settings: Settings) -> None:
    """Configure root logger with console and optional file output."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=settings.log_format or DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger("app").setLevel(level)
