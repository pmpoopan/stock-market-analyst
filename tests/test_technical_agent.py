"""Tests for TechnicalAnalyst — mock market data and mock LLM, no live APIs."""

import pytest

from app.agents.llm_client import MockLLMClient
from app.agents.technical_agent import TechnicalAnalyst, MIN_BARS_REQUIRED
from app.data.exceptions import DataNotFoundError
from app.models.schemas import TechnicalAnalysisResult, TechnicalInterpretation, TrendDirection
from tests.fixtures.market_data import MOCK_SYMBOL, make_mock_historical_long, MockMarketDataProvider
from tests.fixtures.ohlcv_bars import make_trending_bars


@pytest.fixture
def technical_analyst(mock_market_long, mock_llm):
    return TechnicalAnalyst(market_data=mock_market_long, llm=mock_llm)


@pytest.mark.asyncio
async def test_analyze_returns_structured_result(technical_analyst):
    result = await technical_analyst.analyze(MOCK_SYMBOL)

    assert isinstance(result, TechnicalAnalysisResult)
    assert result.stock == MOCK_SYMBOL
    assert 0 <= result.score <= 100
    assert isinstance(result.trend, TrendDirection)
    assert len(result.signals) > 0
    assert result.support is not None
    assert result.resistance is not None
    assert result.momentum
    assert result.volatility
    assert result.summary
    assert "bar_count" in result.indicators


@pytest.mark.asyncio
async def test_analyze_uptrend_high_score(technical_analyst):
    result = await technical_analyst.analyze(MOCK_SYMBOL)
    assert result.trend in (TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND)
    assert result.score >= 55


@pytest.mark.asyncio
async def test_analyze_calls_mock_llm(technical_analyst, mock_llm):
    await technical_analyst.analyze(MOCK_SYMBOL)
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["structured_output"] is TechnicalInterpretation


@pytest.mark.asyncio
async def test_analyze_uses_custom_llm_response(mock_market_long):
    custom_llm = MockLLMClient(
        structured_responses={
            TechnicalInterpretation: TechnicalInterpretation(
                momentum="Custom momentum narrative.",
                volatility="Custom volatility narrative.",
                summary="Custom summary for testing.",
            )
        }
    )
    analyst = TechnicalAnalyst(market_data=mock_market_long, llm=custom_llm)
    result = await analyst.analyze(MOCK_SYMBOL)

    assert result.momentum == "Custom momentum narrative."
    assert result.volatility == "Custom volatility narrative."
    assert result.summary == "Custom summary for testing."


@pytest.mark.asyncio
async def test_analyze_insufficient_data_raises():
    short_historical = {
        f"{MOCK_SYMBOL}:1y": make_mock_historical_long(bar_count=MIN_BARS_REQUIRED - 1)
    }
    market = MockMarketDataProvider(historical=short_historical)
    analyst = TechnicalAnalyst(market_data=market, llm=MockLLMClient())

    with pytest.raises(DataNotFoundError, match="Insufficient historical data"):
        await analyst.analyze(MOCK_SYMBOL)


@pytest.mark.asyncio
async def test_analyze_fallback_when_llm_fails(mock_market_long):
    class FailingLLM(MockLLMClient):
        async def generate(self, prompt, system=None, structured_output=None):
            raise RuntimeError("LLM unavailable")

    analyst = TechnicalAnalyst(market_data=mock_market_long, llm=FailingLLM())
    result = await analyst.analyze(MOCK_SYMBOL)

    assert MOCK_SYMBOL in result.summary
    assert "ATR" in result.volatility


@pytest.mark.asyncio
async def test_analyze_normalizes_symbol(mock_market_long, mock_llm):
    analyst = TechnicalAnalyst(market_data=mock_market_long, llm=mock_llm)
    result = await analyst.analyze("  reliance.ns  ")
    assert result.stock == MOCK_SYMBOL


@pytest.mark.asyncio
async def test_indicators_slim_response_not_full_series(technical_analyst):
    result = await technical_analyst.analyze(MOCK_SYMBOL)
    latest = result.indicators["latest"]
    assert isinstance(latest, dict)
    assert "close" in latest
    # Full SMA series should not be in API response
    assert "sma_20" not in result.indicators or isinstance(result.indicators.get("sma_20"), (int, float))
