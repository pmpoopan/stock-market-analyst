"""Tests for FundamentalAnalyst — mock market data and mock LLM."""

import pytest

from app.agents.fundamental_agent import FundamentalAnalyst
from app.agents.llm_client import MockLLMClient
from app.models.schemas import FundamentalAnalysisResult, FundamentalInterpretation, Rating
from tests.fixtures.market_data import MOCK_SYMBOL, MockMarketDataProvider
from tests.fixtures.fundamental_data import make_mock_financial_metrics


@pytest.fixture
def fundamental_analyst(mock_llm):
    return FundamentalAnalyst(
        market_data=MockMarketDataProvider(),
        llm=mock_llm,
    )


@pytest.mark.asyncio
async def test_analyze_returns_structured_result(fundamental_analyst):
    result = await fundamental_analyst.analyze(MOCK_SYMBOL)

    assert isinstance(result, FundamentalAnalysisResult)
    assert result.stock == MOCK_SYMBOL
    assert 0 <= result.score <= 100
    assert isinstance(result.rating, Rating)
    assert result.metrics.symbol == MOCK_SYMBOL
    assert len(result.strengths) > 0
    assert len(result.risks) > 0
    assert result.summary


@pytest.mark.asyncio
async def test_analyze_calls_mock_llm(fundamental_analyst, mock_llm):
    await fundamental_analyst.analyze(MOCK_SYMBOL)
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["structured_output"] is FundamentalInterpretation


@pytest.mark.asyncio
async def test_analyze_custom_llm_response():
    custom_llm = MockLLMClient(
        structured_responses={
            FundamentalInterpretation: FundamentalInterpretation(
                strengths=["Custom strength"],
                weaknesses=["Custom weakness"],
                risks=["Custom risk"],
                summary="Custom fundamental summary.",
            )
        }
    )
    analyst = FundamentalAnalyst(market_data=MockMarketDataProvider(), llm=custom_llm)
    result = await analyst.analyze(MOCK_SYMBOL)

    assert result.strengths == ["Custom strength"]
    assert result.weaknesses == ["Custom weakness"]
    assert result.risks == ["Custom risk"]
    assert result.summary == "Custom fundamental summary."


@pytest.mark.asyncio
async def test_analyze_fallback_when_llm_fails():
    class FailingLLM(MockLLMClient):
        async def generate(self, prompt, system=None, structured_output=None):
            raise RuntimeError("LLM unavailable")

    analyst = FundamentalAnalyst(market_data=MockMarketDataProvider(), llm=FailingLLM())
    result = await analyst.analyze(MOCK_SYMBOL)

    assert MOCK_SYMBOL in result.summary
    assert len(result.strengths) > 0


@pytest.mark.asyncio
async def test_analyze_normalizes_symbol(mock_llm):
    analyst = FundamentalAnalyst(market_data=MockMarketDataProvider(), llm=mock_llm)
    result = await analyst.analyze("  reliance.ns  ")
    assert result.stock == MOCK_SYMBOL


@pytest.mark.asyncio
async def test_analyze_missing_financials_raises(mock_llm):
    market = MockMarketDataProvider(financials={})
    analyst = FundamentalAnalyst(market_data=market, llm=mock_llm)

    with pytest.raises(KeyError, match="No mock financials"):
        await analyst.analyze("INFY.NS")
