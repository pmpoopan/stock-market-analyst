"""Yahoo Finance data provider for Indian equities.

Wraps yfinance; exposes MarketDataProvider interface only.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from app.config.settings import Settings, get_settings
from app.data.exceptions import DataNotFoundError, DataProviderError
from app.data.interfaces import BaseDataProvider, CacheProvider
from app.models.schemas import FinancialMetrics, HistoricalData, OHLCVBar, Quote
from app.util.retry import is_rate_limit_error, sync_retry_with_backoff

logger = logging.getLogger(__name__)

CACHE_NS_QUOTES = "quotes"
CACHE_NS_HISTORICAL = "historical"
CACHE_NS_FINANCIALS = "financials"


class TransientYahooQuoteError(Exception):
    """Raised when Yahoo returns an empty quote that may succeed on retry."""


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance implementation with optional caching."""

    def __init__(
        self,
        cache: CacheProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._cache = cache
        self._settings = settings or get_settings()
        self._ticker_lock = threading.Lock()
        self._tickers: dict[str, yf.Ticker] = {}

    @property
    def _ttl_quotes(self) -> int:
        return self._settings.cache_ttl_quotes_seconds

    @property
    def _ttl_historical(self) -> int:
        return self._settings.cache_ttl_historical_seconds

    @property
    def _ttl_financials(self) -> int:
        return self._settings.cache_ttl_financials_seconds

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        normalized = self._normalize_symbol(symbol)
        with self._ticker_lock:
            ticker = self._tickers.get(normalized)
            if ticker is None:
                ticker = yf.Ticker(normalized)
                self._tickers[normalized] = ticker
            return ticker

    def _with_retry(self, operation_name: str, operation, *, retry_on=None):
        return sync_retry_with_backoff(
            operation,
            max_attempts=self._settings.yahoo_retry_max_attempts,
            base_delay=self._settings.yahoo_retry_base_delay_seconds,
            max_delay=self._settings.yahoo_retry_max_delay_seconds,
            operation_name=operation_name,
            retry_on=retry_on or is_rate_limit_error,
        )

    def _cache_get_stale(
        self,
        namespace: str,
        key: str,
        model: type[Quote | HistoricalData | FinancialMetrics],
    ):
        if self._cache is None or not hasattr(self._cache, "get_allow_stale"):
            return None
        raw = self._cache.get_allow_stale(namespace, key)
        if raw is None:
            return None
        return model.model_validate_json(raw)

    def _cache_get(self, namespace: str, key: str, model: type[Quote | HistoricalData | FinancialMetrics]):
        if self._cache is None:
            return None
        raw = self._cache.get(namespace, key)
        if raw is None:
            return None
        return model.model_validate_json(raw)

    def _cache_set(
        self,
        namespace: str,
        key: str,
        value: Quote | HistoricalData | FinancialMetrics,
        ttl_seconds: int,
    ) -> None:
        if self._cache is None:
            return
        self._cache.set(namespace, key, value.model_dump_json(), ttl_seconds=ttl_seconds)

    def get_quote(self, symbol: str) -> Quote:
        normalized = self._normalize_symbol(symbol)
        cache_key = normalized

        cached = self._cache_get(CACHE_NS_QUOTES, cache_key, Quote)
        if cached is not None:
            logger.debug("Cache hit for quote: %s", normalized)
            return cached

        logger.debug("Fetching quote from Yahoo Finance: %s", normalized)

        def _fetch_quote() -> Quote:
            try:
                ticker = self._get_ticker(normalized)
                info = ticker.info or {}
            except Exception as exc:
                if is_rate_limit_error(exc):
                    raise
                raise DataProviderError(
                    f"Failed to fetch quote for {normalized}: {exc}"
                ) from exc

            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            if price is None:
                raise TransientYahooQuoteError(f"No price data available for {normalized}")

            previous_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
            change = None
            change_percent = None
            if previous_close is not None:
                change = round(price - previous_close, 4)
                if previous_close != 0:
                    change_percent = round((change / previous_close) * 100, 4)

            return Quote(
                symbol=normalized,
                name=info.get("shortName") or info.get("longName"),
                price=float(price),
                currency=info.get("currency", "INR"),
                change=change,
                change_percent=change_percent,
                market_cap=info.get("marketCap"),
                timestamp=datetime.now(timezone.utc),
            )

        def _is_retryable_quote_error(exc: BaseException) -> bool:
            return is_rate_limit_error(exc) or isinstance(exc, TransientYahooQuoteError)

        try:
            quote = self._with_retry(
                f"Yahoo quote {normalized}",
                _fetch_quote,
                retry_on=_is_retryable_quote_error,
            )
        except DataNotFoundError:
            raise
        except TransientYahooQuoteError as exc:
            stale = self._cache_get_stale(CACHE_NS_QUOTES, cache_key, Quote)
            if stale is not None:
                logger.warning(
                    "Using stale cached quote for %s after empty Yahoo response",
                    normalized,
                )
                return stale
            raise DataNotFoundError(str(exc)) from exc
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.error("Yahoo Finance rate limit fetching quote for %s: %s", normalized, exc)
                stale = self._cache_get_stale(CACHE_NS_QUOTES, cache_key, Quote)
                if stale is not None:
                    logger.warning("Using stale cached quote for %s after rate limit", normalized)
                    return stale
            raise DataProviderError(
                f"Failed to fetch quote for {normalized}: {exc}"
            ) from exc

        self._cache_set(CACHE_NS_QUOTES, cache_key, quote, self._ttl_quotes)
        return quote

    def get_historical_data(self, symbol: str, period: str = "1y") -> HistoricalData:
        normalized = self._normalize_symbol(symbol)
        cache_key = f"{normalized}:{period}"

        cached = self._cache_get(CACHE_NS_HISTORICAL, cache_key, HistoricalData)
        if cached is not None:
            logger.debug("Cache hit for historical data: %s", cache_key)
            return cached

        logger.debug("Fetching historical data from Yahoo Finance: %s period=%s", normalized, period)

        def _fetch_historical() -> HistoricalData:
            try:
                ticker = self._get_ticker(normalized)
                df = ticker.history(period=period, auto_adjust=True)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    raise
                raise DataProviderError(
                    f"Failed to fetch historical data for {normalized}: {exc}"
                ) from exc

            if df is None or df.empty:
                raise DataNotFoundError(
                    f"No historical data available for {normalized} (period={period})"
                )

            bars: list[OHLCVBar] = []
            for ts, row in df.iterrows():
                bar_date = ts.date() if hasattr(ts, "date") else ts
                bars.append(
                    OHLCVBar(
                        date=bar_date,
                        open=round(float(row["Open"]), 4),
                        high=round(float(row["High"]), 4),
                        low=round(float(row["Low"]), 4),
                        close=round(float(row["Close"]), 4),
                        volume=int(row["Volume"]),
                    )
                )

            return HistoricalData(symbol=normalized, period=period, bars=bars)

        try:
            historical = self._with_retry(
                f"Yahoo historical {normalized}",
                _fetch_historical,
            )
        except DataNotFoundError:
            raise
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.error(
                    "Yahoo Finance rate limit fetching historical data for %s: %s",
                    normalized,
                    exc,
                )
            raise DataProviderError(
                f"Failed to fetch historical data for {normalized}: {exc}"
            ) from exc

        self._cache_set(CACHE_NS_HISTORICAL, cache_key, historical, self._ttl_historical)
        return historical

    def get_financials(self, symbol: str) -> FinancialMetrics:
        from app.analysis.fundamental_metrics import FundamentalMetricsCalculator

        normalized = self._normalize_symbol(symbol)
        cache_key = normalized

        cached = self._cache_get(CACHE_NS_FINANCIALS, cache_key, FinancialMetrics)
        if cached is not None:
            logger.debug("Cache hit for financials: %s", normalized)
            return cached

        logger.debug("Fetching financials from Yahoo Finance: %s", normalized)

        def _fetch_financials() -> FinancialMetrics:
            try:
                ticker = self._get_ticker(normalized)
                info = ticker.info or {}
                income_stmt = self._safe_statement(ticker, ["financials", "income_stmt"])
                balance_sheet = self._safe_statement(ticker, ["balance_sheet"])
                cashflow = self._safe_statement(ticker, ["cashflow", "cash_flow"])
            except Exception as exc:
                if is_rate_limit_error(exc):
                    raise
                raise DataProviderError(
                    f"Failed to fetch financials for {normalized}: {exc}"
                ) from exc

            if not info and income_stmt is None and balance_sheet is None:
                raise DataNotFoundError(f"No financial data available for {normalized}")

            raw_data = {
                "symbol": normalized,
                "info": info,
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cashflow": cashflow,
                "data_sources": self._collect_data_sources(
                    info, income_stmt, balance_sheet, cashflow
                ),
            }

            metrics = FundamentalMetricsCalculator.compute(raw_data)

            if self._metrics_are_empty(metrics):
                raise DataNotFoundError(f"Insufficient fundamental metrics for {normalized}")

            return metrics

        try:
            metrics = self._with_retry(
                f"Yahoo financials {normalized}",
                _fetch_financials,
            )
        except DataNotFoundError:
            raise
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.error(
                    "Yahoo Finance rate limit fetching financials for %s: %s",
                    normalized,
                    exc,
                )
            raise DataProviderError(
                f"Failed to fetch financials for {normalized}: {exc}"
            ) from exc

        self._cache_set(CACHE_NS_FINANCIALS, cache_key, metrics, self._ttl_financials)
        return metrics

    @staticmethod
    def _safe_statement(ticker: Any, attr_names: list[str]) -> Any | None:
        for name in attr_names:
            try:
                value = getattr(ticker, name, None)
                if value is not None and hasattr(value, "empty") and not value.empty:
                    return value
            except Exception:
                continue
        return None

    @staticmethod
    def _collect_data_sources(
        info: dict,
        income_stmt: Any | None,
        balance_sheet: Any | None,
        cashflow: Any | None,
    ) -> list[str]:
        sources: list[str] = []
        if info:
            sources.append("yahoo_finance_info")
        if income_stmt is not None:
            sources.append("yahoo_income_statement")
        if balance_sheet is not None:
            sources.append("yahoo_balance_sheet")
        if cashflow is not None:
            sources.append("yahoo_cashflow")
        return sources

    @staticmethod
    def _metrics_are_empty(metrics: FinancialMetrics) -> bool:
        core_fields = [
            metrics.revenue,
            metrics.net_profit,
            metrics.eps,
            metrics.pe_ratio,
            metrics.market_cap,
            metrics.roe,
        ]
        return all(v is None for v in core_fields)
