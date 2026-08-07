"""Tests for PortfolioAnalyst — mock data and mock LLM."""

import pytest

from app.agents.llm_client import MockLLMClient
from app.models.schemas import (
    DecisionResult,
    PortfolioHolding,
    PortfolioInterpretation,
    Rating,
)
from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.market_data import MOCK_SYMBOL


@pytest.fixture
def portfolio_analyst(graph_deps):
    return graph_deps.portfolio_analyst


@pytest.mark.asyncio
async def test_analyze_full_pipeline(portfolio_analyst):
    holdings = [PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)]
    result = await portfolio_analyst.analyze(holdings)

    assert len(result.holdings) == 1
    assert result.holdings[0].holding.symbol == MOCK_SYMBOL
    assert result.portfolio_score >= 0
    assert result.summary
    assert result.portfolio_risk


@pytest.mark.asyncio
async def test_analyze_from_state(portfolio_analyst):
    from app.models.schemas import FundamentalAnalysisResult

    holdings = [PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)]
    decision = DecisionResult(
        stock=MOCK_SYMBOL,
        overall_score=72.0,
        rating=Rating.BUY,
        fundamental_score=75.0,
        technical_score=70.0,
        sentiment_score=60.0,
        risk_adjustment=0.0,
        key_reasons=["k"],
        major_risks=["r"],
    )
    metrics = make_mock_financial_metrics()
    fundamental = FundamentalAnalysisResult(
        stock=MOCK_SYMBOL,
        score=75.0,
        rating=Rating.BUY,
        metrics=metrics,
        strengths=["s"],
        weaknesses=["w"],
        risks=["r"],
        summary="summary",
    )

    result = await portfolio_analyst.analyze_from_state(
        holdings=holdings,
        decisions={MOCK_SYMBOL: decision},
        market_data={
            MOCK_SYMBOL: {
                "symbol": MOCK_SYMBOL,
                "price": 1450.25,
                "currency": "INR",
                "name": "Reliance",
            }
        },
        fundamental_analysis={MOCK_SYMBOL: fundamental},
    )

    assert len(result.holdings) == 1
    assert result.holdings[0].decision.overall_score == 72.0
    assert "Energy" in result.sector_concentration or result.sector_concentration


@pytest.mark.asyncio
async def test_analyze_uses_custom_llm_interpretation(graph_deps):
    custom = PortfolioInterpretation(
        portfolio_risk="Custom portfolio risk.",
        summary="Custom portfolio summary.",
    )
    llm = MockLLMClient(structured_responses={PortfolioInterpretation: custom})
    analyst = graph_deps.portfolio_analyst
    analyst._llm = llm

    holdings = [PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)]
    result = await analyst.analyze(holdings)

    assert result.portfolio_risk == "Custom portfolio risk."
    assert result.summary == "Custom portfolio summary."
