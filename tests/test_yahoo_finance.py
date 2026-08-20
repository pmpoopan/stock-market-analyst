"""Tests for Yahoo Finance provider — uses mocked yfinance, no live API calls."""

from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from app.data.exceptions import DataNotFoundError, DataProviderError
from app.data.yahoo_finance import YahooFinanceProvider
from app.models.schemas import FinancialMetrics, HistoricalData, Quote
from tests.fixtures.fundamental_data import MOCK_FUNDAMENTAL_INFO, make_mock_financial_metrics
from tests.fixtures.market_data import (
    MOCK_SYMBOL,
    MOCK_YAHOO_INFO,
    make_mock_history_dataframe,
    make_mock_quote,
)


@pytest.fixture
def provider(cache, test_settings):
    return YahooFinanceProvider(cache=cache, settings=test_settings)


def _mock_ticker(
    info: dict | None = None,
    history: pd.DataFrame | None = None,
    income_stmt: pd.DataFrame | None = None,
    balance_sheet: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
):
    ticker = MagicMock()
    ticker.info = info if info is not None else MOCK_YAHOO_INFO
    ticker.history.return_value = (
        history if history is not None else make_mock_history_dataframe()
    )
    ticker.financials = income_stmt
    ticker.income_stmt = income_stmt
    ticker.balance_sheet = balance_sheet
    ticker.cashflow = cashflow
    return ticker


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_fetches_from_yahoo(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker()

    quote = provider.get_quote(MOCK_SYMBOL)

    assert isinstance(quote, Quote)
    assert quote.symbol == MOCK_SYMBOL
    assert quote.price == 1450.25
    assert quote.name == "Reliance Industries Limited"
    assert quote.change == pytest.approx(12.50)
    mock_ticker_cls.assert_called_once_with(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_uses_cache_on_second_call(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker()

    provider.get_quote(MOCK_SYMBOL)
    provider.get_quote(MOCK_SYMBOL)

    mock_ticker_cls.assert_called_once()


def test_get_quote_without_cache(test_settings, tmp_cache_path):
    uncached = YahooFinanceProvider(cache=None, settings=test_settings)
    with patch("app.data.yahoo_finance.yf.Ticker") as mock_ticker_cls:
        mock_ticker = _mock_ticker()
        mock_ticker_cls.return_value = mock_ticker
        quote1 = uncached.get_quote(MOCK_SYMBOL)
        quote2 = uncached.get_quote(MOCK_SYMBOL)
    assert quote1.price == quote2.price
    assert mock_ticker_cls.call_count == 1
    assert mock_ticker.info is not None


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_normalizes_symbol(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker()
    quote = provider.get_quote("  reliance.ns  ")
    assert quote.symbol == "RELIANCE.NS"
    mock_ticker_cls.assert_called_once_with("RELIANCE.NS")


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_raises_when_no_price(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker(info={})
    with patch("app.util.retry.time.sleep"):
        with pytest.raises(DataNotFoundError, match="No price data"):
            provider.get_quote(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_wraps_fetch_errors(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.side_effect = RuntimeError("network down")
    with pytest.raises(DataProviderError, match="Failed to fetch quote"):
        provider.get_quote(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_historical_data_fetches_from_yahoo(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker()

    historical = provider.get_historical_data(MOCK_SYMBOL, period="1y")

    assert isinstance(historical, HistoricalData)
    assert historical.symbol == MOCK_SYMBOL
    assert historical.period == "1y"
    assert len(historical.bars) == 5
    assert historical.bars[0].close == 1400.0
    assert historical.bars[-1].close == 1440.0
    mock_ticker_cls.return_value.history.assert_called_once_with(period="1y", auto_adjust=True)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_historical_data_uses_cache(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker()

    provider.get_historical_data(MOCK_SYMBOL, period="6mo")
    provider.get_historical_data(MOCK_SYMBOL, period="6mo")

    mock_ticker_cls.assert_called_once()


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_historical_data_empty_raises(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker(history=pd.DataFrame())
    with pytest.raises(DataNotFoundError, match="No historical data"):
        provider.get_historical_data(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_historical_data_wraps_fetch_errors(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = MagicMock()
    mock_ticker_cls.return_value.history.side_effect = RuntimeError("timeout")
    with pytest.raises(DataProviderError, match="Failed to fetch historical"):
        provider.get_historical_data(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_financials_fetches_from_yahoo(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker(info=MOCK_FUNDAMENTAL_INFO)

    metrics = provider.get_financials(MOCK_SYMBOL)

    assert isinstance(metrics, FinancialMetrics)
    assert metrics.symbol == MOCK_SYMBOL
    assert metrics.revenue == 2_500_000_000_000
    assert metrics.pe_ratio == pytest.approx(22.5)
    mock_ticker_cls.assert_called_once_with(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_financials_uses_cache(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker(info=MOCK_FUNDAMENTAL_INFO)

    provider.get_financials(MOCK_SYMBOL)
    provider.get_financials(MOCK_SYMBOL)

    mock_ticker_cls.assert_called_once()


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_financials_empty_raises(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.return_value = _mock_ticker(info={})
    with pytest.raises(DataNotFoundError, match="No financial data"):
        provider.get_financials(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_financials_wraps_fetch_errors(mock_ticker_cls, provider: YahooFinanceProvider):
    mock_ticker_cls.side_effect = RuntimeError("network down")
    with pytest.raises(DataProviderError, match="Failed to fetch financials"):
        provider.get_financials(MOCK_SYMBOL)


def test_mock_market_data_provider():
    from tests.fixtures.market_data import MockMarketDataProvider

    mock = MockMarketDataProvider()
    quote = mock.get_quote(MOCK_SYMBOL)
    historical = mock.get_historical_data(MOCK_SYMBOL)
    financials = mock.get_financials(MOCK_SYMBOL)
    assert quote.price == make_mock_quote().price
    assert len(historical.bars) == 5
    assert financials.symbol == MOCK_SYMBOL


@patch("app.data.yahoo_finance.yf.Ticker")
def test_quote_and_financials_reuse_yahoo_info(mock_ticker_cls, provider: YahooFinanceProvider):
    info = {**MOCK_YAHOO_INFO, **MOCK_FUNDAMENTAL_INFO}
    ticker = _mock_ticker(info=info)
    info_property = PropertyMock(return_value=info)
    type(ticker).info = info_property
    mock_ticker_cls.return_value = ticker

    quote = provider.get_quote(MOCK_SYMBOL)
    metrics = provider.get_financials(MOCK_SYMBOL)

    assert quote.price == 1450.25
    assert metrics.pe_ratio == pytest.approx(22.5)
    assert metrics.pb_ratio == pytest.approx(2.4)
    assert metrics.revenue_growth == pytest.approx(12.0)
    assert metrics.earnings_growth == pytest.approx(15.0)
    assert info_property.call_count == 1


@patch("app.data.yahoo_finance.yf.Ticker")
@patch.object(YahooFinanceProvider, "_cache_get", return_value=None)
def test_get_financials_uses_stale_cache_after_rate_limit(
    mock_cache_get, mock_ticker_cls, provider: YahooFinanceProvider
):
    stale = make_mock_financial_metrics(MOCK_SYMBOL)
    provider._cache_set("financials", MOCK_SYMBOL, stale, ttl_seconds=86400)
    mock_ticker_cls.side_effect = RuntimeError("Too Many Requests. Rate limited.")

    with patch("app.util.retry.time.sleep"):
        metrics = provider.get_financials(MOCK_SYMBOL)

    assert metrics.pe_ratio == stale.pe_ratio
    assert metrics.pb_ratio == stale.pb_ratio
