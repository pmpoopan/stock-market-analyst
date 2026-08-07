"""Mock market data for tests — no live API or LLM calls."""

from datetime import date, datetime, timezone

import pandas as pd

from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.ohlcv_bars import make_trending_bars

from app.data.interfaces import MarketDataProvider
from app.models.schemas import FinancialMetrics, HistoricalData, OHLCVBar, Quote

MOCK_SYMBOL = "RELIANCE.NS"
MOCK_SYMBOL_2 = "TATAMOTORS.NS"


def make_mock_quote(symbol: str = MOCK_SYMBOL) -> Quote:
    return Quote(
        symbol=symbol.upper(),
        name="Reliance Industries Limited" if "RELIANCE" in symbol.upper() else "Tata Motors Limited",
        price=1450.25,
        currency="INR",
        change=12.50,
        change_percent=0.87,
        market_cap=9_800_000_000_000,
        timestamp=datetime(2026, 1, 15, 9, 15, tzinfo=timezone.utc),
    )


def make_mock_bars(count: int = 5, start_price: float = 1400.0) -> list[OHLCVBar]:
    bars = []
    for i in range(count):
        close = start_price + i * 10
        bars.append(
            OHLCVBar(
                date=date(2026, 1, 1 + i),
                open=close - 5,
                high=close + 8,
                low=close - 10,
                close=close,
                volume=1_000_000 + i * 50_000,
            )
        )
    return bars


def make_mock_historical(symbol: str = MOCK_SYMBOL, period: str = "1y") -> HistoricalData:
    return HistoricalData(
        symbol=symbol.upper(),
        period=period,
        bars=make_mock_bars(),
    )


def make_mock_historical_long(
    symbol: str = MOCK_SYMBOL,
    period: str = "1y",
    bar_count: int = 250,
) -> HistoricalData:
    """Enough bars for SMA 200 and full technical analysis."""
    return HistoricalData(
        symbol=symbol.upper(),
        period=period,
        bars=make_trending_bars(count=bar_count),
    )


def make_mock_history_dataframe(count: int = 5, start_price: float = 1400.0) -> pd.DataFrame:
    """Pandas DataFrame mimicking yfinance ticker.history() output."""
    rows = []
    index = []
    for i in range(count):
        close = start_price + i * 10
        index.append(pd.Timestamp(f"2026-01-{1 + i:02d}"))
        rows.append(
            {
                "Open": close - 5,
                "High": close + 8,
                "Low": close - 10,
                "Close": close,
                "Volume": 1_000_000 + i * 50_000,
            }
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


MOCK_YAHOO_INFO: dict = {
    "shortName": "Reliance Industries Limited",
    "longName": "Reliance Industries Limited",
    "currentPrice": 1450.25,
    "regularMarketPrice": 1450.25,
    "regularMarketPreviousClose": 1437.75,
    "previousClose": 1437.75,
    "currency": "INR",
    "marketCap": 9_800_000_000_000,
}


class MockMarketDataProvider:
    """In-memory market data provider for unit/integration tests."""

    def __init__(
        self,
        quotes: dict[str, Quote] | None = None,
        historical: dict[str, HistoricalData] | None = None,
        financials: dict[str, FinancialMetrics] | None = None,
    ) -> None:
        self._quotes = quotes or {MOCK_SYMBOL: make_mock_quote()}
        self._historical = historical or {f"{MOCK_SYMBOL}:1y": make_mock_historical()}
        self._financials = financials or {MOCK_SYMBOL: make_mock_financial_metrics()}

    def get_quote(self, symbol: str) -> Quote:
        key = symbol.upper()
        if key not in self._quotes:
            raise KeyError(f"No mock quote for {key}")
        return self._quotes[key]

    def get_historical_data(self, symbol: str, period: str = "1y") -> HistoricalData:
        key = f"{symbol.upper()}:{period}"
        if key not in self._historical:
            raise KeyError(f"No mock historical data for {key}")
        return self._historical[key]

    def get_financials(self, symbol: str) -> FinancialMetrics:
        key = symbol.upper()
        if key not in self._financials:
            raise KeyError(f"No mock financials for {key}")
        return self._financials[key]
