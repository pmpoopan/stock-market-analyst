"""Web search provider for news and sentiment research.

Uses DuckDuckGo — exposes NewsSearchProvider interface only.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from duckduckgo_search import DDGS

from app.config.settings import Settings, get_settings
from app.data.exceptions import DataProviderError
from app.data.interfaces import CacheProvider
from app.models.schemas import NewsArticle
from app.util.retry import is_rate_limit_error, sync_retry_with_backoff

logger = logging.getLogger(__name__)

CACHE_NS_SEARCH = "search"
CACHE_NS_SEARCH_RATELIMIT = "search_ratelimit"
DEFAULT_REGION = "in-en"


class DuckDuckGoSearchProvider:
    """DuckDuckGo news search implementation with optional caching."""

    def __init__(
        self,
        cache: CacheProvider | None = None,
        max_results: int = 10,
        settings: Settings | None = None,
    ) -> None:
        self._cache = cache
        self._max_results = max_results
        self._settings = settings or get_settings()

    @property
    def _ttl_search(self) -> int:
        return self._settings.cache_ttl_search_seconds

    def search_news(self, query: str, max_results: int | None = None) -> list[NewsArticle]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty")

        limit = max_results if max_results is not None else self._max_results
        cache_key = f"{normalized_query}:{limit}"

        if self._cache is not None:
            if self._cache.get(CACHE_NS_SEARCH_RATELIMIT, cache_key) is not None:
                logger.warning(
                    "Skipping DuckDuckGo search due to recent rate limit: %s",
                    cache_key,
                )
                return []

            cached = self._cache.get(CACHE_NS_SEARCH, cache_key)
            if cached is not None:
                logger.debug("Cache hit for news search: %s", cache_key)
                return self._deserialize_articles(cached)

        logger.debug("Searching DuckDuckGo news: %s (limit=%d)", normalized_query, limit)
        try:
            articles = sync_retry_with_backoff(
                lambda: self._fetch_news(normalized_query, limit),
                max_attempts=self._settings.web_search_retry_max_attempts,
                base_delay=self._settings.web_search_retry_base_delay_seconds,
                max_delay=self._settings.web_search_retry_max_delay_seconds,
                operation_name=f"DuckDuckGo news '{normalized_query}'",
                retry_on=is_rate_limit_error,
            )
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.error(
                    "DuckDuckGo rate limit for '%s' after retries: %s",
                    normalized_query,
                    exc,
                )
                self._cache_rate_limited(cache_key)
                return []
            raise DataProviderError(
                f"News search failed for '{normalized_query}': {exc}"
            ) from exc

        if self._cache is not None:
            self._cache.set(
                CACHE_NS_SEARCH,
                cache_key,
                self._serialize_articles(articles),
                ttl_seconds=self._ttl_search,
            )

        return articles

    def _cache_rate_limited(self, cache_key: str) -> None:
        if self._cache is None:
            return
        self._cache.set(
            CACHE_NS_SEARCH_RATELIMIT,
            cache_key,
            "1",
            ttl_seconds=self._settings.web_search_rate_limit_cache_seconds,
        )

    def _fetch_news(self, query: str, limit: int) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=query,
                region=DEFAULT_REGION,
                timelimit="m",
                max_results=limit,
            )

        for row in results:
            article = self._parse_result(row)
            if article is not None:
                articles.append(article)

        return articles

    def _parse_result(self, row: dict) -> NewsArticle | None:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if not title or not url:
            return None

        source = (row.get("source") or "unknown").strip()
        snippet = row.get("body") or row.get("excerpt")
        published_at = self._parse_date(row.get("date"))

        return NewsArticle(
            title=title,
            source=source,
            url=url,
            published_at=published_at,
            snippet=snippet,
        )

    @staticmethod
    def _parse_date(value: str | date | datetime | None) -> date | None:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value).strip()
        if not text:
            return None
        try:
            if "T" in text:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
            return date.fromisoformat(text[:10])
        except ValueError:
            return None

    @staticmethod
    def _serialize_articles(articles: list[NewsArticle]) -> str:
        return json.dumps([article.model_dump(mode="json") for article in articles])

    @staticmethod
    def _deserialize_articles(payload: str) -> list[NewsArticle]:
        data = json.loads(payload)
        return [NewsArticle.model_validate(item) for item in data]
