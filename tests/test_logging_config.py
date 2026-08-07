"""Tests for centralized logging configuration."""

import logging

from app.config.logging_config import setup_logging
from app.config.settings import Settings


def test_setup_logging_configures_root_logger(tmp_path):
    log_file = tmp_path / "buddy.log"
    settings = Settings(
        log_level="DEBUG",
        log_file=str(log_file),
        cache_db_path=str(tmp_path / "cache.db"),
    )

    setup_logging(settings)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(handler, logging.FileHandler) for handler in root.handlers)

    test_logger = logging.getLogger("app.test")
    test_logger.info("logging smoke test")
    assert log_file.read_text(encoding="utf-8").strip()


def test_setup_logging_quiets_noisy_loggers():
    settings = Settings(log_level="INFO", cache_db_path="data/test_cache.db")
    setup_logging(settings)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING
