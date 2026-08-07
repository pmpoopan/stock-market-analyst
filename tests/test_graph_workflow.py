"""Tests for LangGraph workflow — mock data providers and mock LLM."""

import pytest

from app.models.schemas import QueryIntent
from tests.fixtures.market_data import MOCK_SYMBOL


@pytest.fixture
def orchestrator(graph_deps):
    from app.graph.workflow import AnalysisOrchestrator

    return AnalysisOrchestrator(deps=graph_deps)


@pytest.mark.asyncio
async def test_analyze_single_stock_workflow(orchestrator, mock_llm):
    state = await orchestrator.analyze("How is Reliance doing?")

    assert state["parsed_query"].intent == QueryIntent.ANALYZE_STOCK
    assert state["stocks"] == [MOCK_SYMBOL]
    assert MOCK_SYMBOL in state["fundamental_analysis"]
    assert MOCK_SYMBOL in state["technical_analysis"]
    assert MOCK_SYMBOL in state["sentiment_analysis"]
    assert MOCK_SYMBOL in state["master_analysis"]
    assert MOCK_SYMBOL in state["decision"]

    master = state["master_analysis"][MOCK_SYMBOL]
    assert master.narrative
    assert len(master.agreement_points) > 0

    # Master + fundamental + technical + sentiment LLM calls
    assert len(mock_llm.calls) >= 4

    response = state.get("stock_response")
    assert response is not None
    assert response.symbol == MOCK_SYMBOL
    assert response.decision.overall_score >= 0
    assert any("Agreement" in r or "Fundamental" in r for r in response.decision.key_reasons)


@pytest.mark.asyncio
async def test_parallel_analysts_all_produce_output(orchestrator):
    state = await orchestrator.analyze("How is Reliance doing?")

    fund = state["fundamental_analysis"][MOCK_SYMBOL]
    tech = state["technical_analysis"][MOCK_SYMBOL]
    sent = state["sentiment_analysis"][MOCK_SYMBOL]

    assert fund.score >= 0
    assert tech.score >= 0
    assert sent.sentiment_score >= 0


@pytest.mark.asyncio
async def test_compare_workflow_completes(orchestrator, mock_llm):
    state = await orchestrator.compare(["TATAMOTORS.NS", "M&M.NS"])

    assert state["parsed_query"].intent == QueryIntent.COMPARE_STOCKS
    assert len(state["stocks"]) == 2
    assert "TATAMOTORS.NS" in state["fundamental_analysis"]
    assert "M&M.NS" in state["fundamental_analysis"]
    assert state.get("stock_response") is None

    comparison = state.get("comparison_analysis")
    assert comparison is not None
    assert len(comparison.stocks) == 2
    assert "TATAMOTORS.NS" in comparison.overall_scores
    assert "M&M.NS" in comparison.overall_scores
    assert comparison.valuation_comparison
    assert comparison.relative_assessment
    assert "Mock" in comparison.relative_assessment or "leads" in comparison.relative_assessment

    errors = [e.message for e in state.get("errors", [])]
    assert not any("Phase 10" in msg for msg in errors)

    assert len(mock_llm.calls) >= 7


@pytest.mark.asyncio
async def test_portfolio_workflow_completes(orchestrator, mock_llm):
    from app.models.schemas import PortfolioHolding

    holdings = [PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)]
    state = await orchestrator.portfolio(holdings)

    assert state["parsed_query"].intent == QueryIntent.ANALYZE_PORTFOLIO
    assert MOCK_SYMBOL in state["fundamental_analysis"]
    assert MOCK_SYMBOL in state["decision"]

    portfolio = state.get("portfolio_analysis")
    assert portfolio is not None
    assert len(portfolio.holdings) == 1
    assert portfolio.holdings[0].holding.symbol == MOCK_SYMBOL
    assert portfolio.portfolio_score >= 0
    assert portfolio.summary
    assert "Mock portfolio" in portfolio.summary or "Portfolio of" in portfolio.summary

    errors = [e.message for e in state.get("errors", [])]
    assert not any("Phase 9" in msg for msg in errors)

    assert len(mock_llm.calls) >= 5


@pytest.mark.asyncio
async def test_invalid_query_records_error(orchestrator):
    state = await orchestrator.analyze("What is the weather today?")

    assert state.get("stock_response") is None
    errors = state.get("errors", [])
    assert len(errors) > 0
    assert errors[0].component == "query_parser"
