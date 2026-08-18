"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Buddy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Buddy"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_file: str | None = None
    log_format: str | None = None

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"

    # LLM (Groq)
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    # Data providers
    yahoo_finance_timeout: int = 30
    web_search_provider: Literal["duckduckgo"] = "duckduckgo"
    web_search_max_results: int = 10

    # Cache (SQLite MVP)
    cache_enabled: bool = True
    cache_db_path: str = "data/cache.db"
    cache_ttl_quotes_seconds: int = 300
    cache_ttl_historical_seconds: int = 3600
    cache_ttl_financials_seconds: int = 86400
    cache_ttl_search_seconds: int = 1800

    # Scoring weights (must sum to 1.0)
    weight_fundamental: float = 0.40
    weight_technical: float = 0.35
    weight_sentiment: float = 0.15
    weight_risk: float = 0.10

    # Rating thresholds (0–100 scale)
    rating_strong_buy_min: float = 80.0
    rating_buy_min: float = 65.0
    rating_hold_min: float = 45.0
    # Below rating_hold_min → Avoid


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
