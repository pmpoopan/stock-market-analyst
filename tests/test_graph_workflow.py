"""Tests for LangGraph workflow — mock data providers and mock LLM."""

from unittest.mock import patch

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
@patch("app.data.yahoo_finance.yf.Ticker")
async def test_analyze_stock_uses_stale_quote_and_real_yahoo_ratios(
    mock_ticker_cls, cache, test_settings, mock_llm, mock_news_search
):
    import time
    from unittest.mock import MagicMock, PropertyMock, patch

    import pandas as pd

    from app.agents.comparison_agent import ComparisonAnalyst
    from app.agents.fundamental_agent import FundamentalAnalyst
    from app.agents.master_agent import MasterAnalyst
    from app.agents.portfolio_agent import PortfolioAnalyst
    from app.agents.query_parser import QueryParser
    from app.agents.sentiment_agent import SentimentAnalyst
    from app.agents.technical_agent import TechnicalAnalyst
    from app.analysis.scoring import ScoringEngine
    from app.config.settings import Settings
    from app.data.yahoo_finance import YahooFinanceProvider
    from app.graph.deps import GraphDependencies
    from app.graph.workflow import AnalysisOrchestrator
    from tests.fixtures.fundamental_data import MOCK_FUNDAMENTAL_INFO
    from tests.fixtures.market_data import MOCK_YAHOO_INFO, make_mock_quote

    stale_quote = make_mock_quote(MOCK_SYMBOL)
    cache.set("quotes", MOCK_SYMBOL, stale_quote.model_dump_json(), ttl_seconds=1)
    time.sleep(1.1)
    assert cache.get("quotes", MOCK_SYMBOL) is None

    combined_info = {**MOCK_YAHOO_INFO, **MOCK_FUNDAMENTAL_INFO}
    ticker = MagicMock()
    type(ticker).info = PropertyMock(
        side_effect=[
            RuntimeError("Too Many Requests. Rate limited."),
            RuntimeError("Too Many Requests. Rate limited."),
            combined_info,
            combined_info,
        ]
    )
    index = pd.bdate_range("2025-01-02", periods=60)
    close = 1400.0 + pd.Series(range(60), index=index) * 2
    ticker.history.return_value = pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 8,
            "Low": close - 10,
            "Close": close,
            "Volume": [1_000_000] * 60,
        },
        index=index,
    )
    ticker.financials = None
    ticker.income_stmt = None
    ticker.balance_sheet = None
    ticker.cashflow = None
    mock_ticker_cls.return_value = ticker

    settings = Settings(
        cache_enabled=True,
        cache_db_path=test_settings.cache_db_path,
        cache_ttl_quotes_seconds=300,
        yahoo_retry_max_attempts=2,
        yahoo_retry_base_delay_seconds=0.01,
        yahoo_retry_max_delay_seconds=0.05,
        groq_api_key="",
    )
    market = YahooFinanceProvider(cache=cache, settings=settings)
    fundamental = FundamentalAnalyst(market, mock_llm)
    technical = TechnicalAnalyst(market, mock_llm)
    sentiment = SentimentAnalyst(mock_news_search, mock_llm)
    master = MasterAnalyst(mock_llm)
    deps = GraphDependencies(
        query_parser=QueryParser(),
        fundamental_analyst=fundamental,
        technical_analyst=technical,
        sentiment_analyst=sentiment,
        master_analyst=master,
        comparison_analyst=ComparisonAnalyst(mock_llm),
        portfolio_analyst=PortfolioAnalyst(
            market_data=market,
            fundamental_analyst=fundamental,
            technical_analyst=technical,
            sentiment_analyst=sentiment,
            scoring_engine=ScoringEngine(),
            llm=mock_llm,
            master_analyst=master,
        ),
        scoring_engine=ScoringEngine(),
        market_data=market,
    )
    orchestrator = AnalysisOrchestrator(deps=deps)

    with patch("app.util.retry.time.sleep"):
        state = await orchestrator.analyze("How is Reliance doing?")

    assert state["parsed_query"].intent == QueryIntent.ANALYZE_STOCK
    response = state.get("stock_response")
    assert response is not None
    assert response.current_price == stale_quote.price
    assert response.fundamental.metrics.pe_ratio == pytest.approx(22.5)
    assert response.fundamental.metrics.pb_ratio == pytest.approx(2.4)
    assert response.fundamental.metrics.revenue_growth == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_invalid_query_records_error(orchestrator):
    state = await orchestrator.analyze("What is the weather today?")

    assert state.get("stock_response") is None
    errors = state.get("errors", [])
    assert len(errors) > 0
    assert errors[0].component == "query_parser"
