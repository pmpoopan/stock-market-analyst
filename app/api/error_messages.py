"""Map internal error details to user-facing API messages."""

from __future__ import annotations

import re

from app.models.schemas import ErrorDetail

_GROQ_RATE_LIMIT = re.compile(
    r"tokens per minute|\btpm\b|rate limit reached for model",
    re.IGNORECASE,
)
_MARKET_DATA_RATE_LIMIT = re.compile(
    r"too many requests|rate limit|rate limited",
    re.IGNORECASE,
)
_NEWS_RATE_LIMIT = re.compile(r"403|ratelimit|rate limit", re.IGNORECASE)
_INCOMPLETE_COMPARISON = re.compile(
    r"at least two stocks with complete analysis",
    re.IGNORECASE,
)

_ERROR_COMPONENT_PRIORITY = (
    "comparison",
    "final_response",
    "portfolio",
    "master_analyst",
    "decision_engine",
    "fundamental_analyst",
    "technical_analyst",
    "sentiment_analyst",
    "market_data",
    "query_parser",
)


def user_facing_api_error(message: str) -> str:
    """Return a clean API error message without raw provider details."""
    if _INCOMPLETE_COMPARISON.search(message):
        return (
            "Unable to compare stocks: fewer than two completed analyses. "
            "Some market data may be temporarily unavailable — please try again shortly."
        )
    if message.startswith("Quote unavailable for "):
        return "Market data is temporarily unavailable. Please try again shortly."
    if _GROQ_RATE_LIMIT.search(message):
        return "Analysis service is temporarily busy. Please try again in a minute."
    if _MARKET_DATA_RATE_LIMIT.search(message):
        return "Market data is temporarily unavailable. Please try again shortly."
    if _NEWS_RATE_LIMIT.search(message):
        return "News data is temporarily unavailable. Analysis may be limited."

    return message


def select_primary_error(errors: list[ErrorDetail]) -> str | None:
    """Pick the most relevant error for API responses."""
    if not errors:
        return None

    for component in _ERROR_COMPONENT_PRIORITY:
        for error in errors:
            if error.component == component:
                return user_facing_api_error(error.message)

    return user_facing_api_error(errors[0].message)
