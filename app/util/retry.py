"""Bounded retry helpers with exponential backoff for rate-limited providers."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "429",
    "403 ratelimit",
)

RETRY_AFTER_SECONDS_RE = re.compile(
    r"retry[- ]?after[:\s]+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
RETRY_AFTER_MS_RE = re.compile(
    r"retry[- ]?after[:\s]+(\d+(?:\.\d+)?)\s*ms",
    re.IGNORECASE,
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when an exception looks like an upstream rate-limit response."""
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 403}:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status in {429, 403}:
            return True

    message = str(exc).lower()
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract retry-after delay from exception message or response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError):
                pass

    message = str(exc)
    ms_match = RETRY_AFTER_MS_RE.search(message)
    if ms_match:
        return max(float(ms_match.group(1)) / 1000.0, 0.0)

    sec_match = RETRY_AFTER_SECONDS_RE.search(message)
    if sec_match:
        return max(float(sec_match.group(1)), 0.0)

    return None


def _compute_backoff_delay(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    retry_after: float | None,
) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), max_delay)

    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0.0, delay * 0.25)
    return min(delay + jitter, max_delay)


def sync_retry_with_backoff(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    operation_name: str,
    retry_on: Callable[[BaseException], bool] | None = None,
) -> T:
    """Run a sync callable with bounded exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    should_retry = retry_on or is_rate_limit_error
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if not should_retry(exc) or attempt >= max_attempts - 1:
                raise

            delay = _compute_backoff_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_after=parse_retry_after_seconds(exc),
            )
            logger.warning(
                "%s rate limited (attempt %d/%d); retrying in %.2fs: %s",
                operation_name,
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


async def async_retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    operation_name: str,
    retry_on: Callable[[BaseException], bool] | None = None,
) -> T:
    """Run an async callable with bounded exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    should_retry = retry_on or is_rate_limit_error
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as exc:
            last_exc = exc
            if not should_retry(exc) or attempt >= max_attempts - 1:
                raise

            delay = _compute_backoff_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_after=parse_retry_after_seconds(exc),
            )
            logger.warning(
                "%s rate limited (attempt %d/%d); retrying in %.2fs: %s",
                operation_name,
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
