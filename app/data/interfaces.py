"""Data provider interfaces and implementations.

Agents depend on these abstractions — not on Yahoo Finance or search internals.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from app.models.schemas import FinancialMetrics, HistoricalData, NewsArticle, Quote


@runtime_checkable
class MarketDataProvider(Protocol):
    """Interface for market and financial data retrieval."""

    def get_quote(self, symbol: str) -> Quote:
        """Return current quote for a symbol (e.g. RELIANCE.NS)."""
        ...

    def get_historical_data(self, symbol: str, period: str = "1y") -> HistoricalData:
        """Return OHLCV historical data for the given period."""
        ...

    def get_financials(self, symbol: str) -> FinancialMetrics:
        """Return computed fundamental metrics for a symbol."""
        ...


@runtime_checkable
class NewsSearchProvider(Protocol):
    """Interface for web search / news retrieval."""

    def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
        """Search for recent news articles relevant to the query."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Interface for caching external data (SQLite MVP, swappable later)."""

    def get(self, namespace: str, key: str) -> str | None:
        """Retrieve cached value if present and not expired."""
        ...

    def set(
        self,
        namespace: str,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a value with optional TTL."""
        ...

    def delete(self, namespace: str, key: str) -> None:
        """Remove a cached entry."""
        ...

    def clear_namespace(self, namespace: str) -> None:
        """Clear all entries in a namespace."""
        ...


class BaseDataProvider(ABC):
    """Optional base class for concrete data providers."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    @abstractmethod
    def get_historical_data(self, symbol: str, period: str = "1y") -> HistoricalData:
        raise NotImplementedError

    @abstractmethod
    def get_financials(self, symbol: str) -> FinancialMetrics:
        raise NotImplementedError
