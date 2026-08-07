"""Shared pytest fixtures."""

import pytest

from app.config.settings import Settings
from app.data.cache import SQLiteCache


@pytest.fixture
def tmp_cache_path(tmp_path):
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def cache(tmp_cache_path):
    return SQLiteCache(db_path=tmp_cache_path)


@pytest.fixture
def test_settings(tmp_cache_path):
    return Settings(
        cache_enabled=True,
        cache_db_path=tmp_cache_path,
        cache_ttl_quotes_seconds=300,
        cache_ttl_historical_seconds=3600,
        groq_api_key="",
    )


@pytest.fixture
def mock_llm():
    from app.agents.llm_client import MockLLMClient

    return MockLLMClient()


@pytest.fixture
def mock_market(mock_llm):
    from tests.fixtures.market_data import MockMarketDataProvider

    return MockMarketDataProvider()


@pytest.fixture
def mock_news_search():
    from tests.fixtures.news_data import MockNewsSearchProvider

    return MockNewsSearchProvider()


@pytest.fixture
def mock_market_long():
    from tests.fixtures.market_data import MOCK_SYMBOL, MockMarketDataProvider, make_mock_historical_long

    return MockMarketDataProvider(
        historical={f"{MOCK_SYMBOL}:1y": make_mock_historical_long()}
    )


@pytest.fixture
def graph_deps(mock_llm, mock_news_search):
    from app.agents.comparison_agent import ComparisonAnalyst
    from app.agents.fundamental_agent import FundamentalAnalyst
    from app.agents.master_agent import MasterAnalyst
    from app.agents.query_parser import QueryParser
    from app.agents.portfolio_agent import PortfolioAnalyst
    from app.agents.sentiment_agent import SentimentAnalyst
    from app.agents.technical_agent import TechnicalAnalyst
    from app.analysis.scoring import ScoringEngine
    from app.graph.deps import GraphDependencies
    from tests.fixtures.fundamental_data import make_mock_financial_metrics
    from tests.fixtures.market_data import (
        MOCK_SYMBOL,
        MockMarketDataProvider,
        make_mock_historical_long,
        make_mock_quote,
    )
    from tests.fixtures.news_data import MockNewsSearchProvider

    market = MockMarketDataProvider(
        historical={
            f"{MOCK_SYMBOL}:1y": make_mock_historical_long(),
            "TATAMOTORS.NS:1y": make_mock_historical_long("TATAMOTORS.NS"),
            "M&M.NS:1y": make_mock_historical_long("M&M.NS"),
            "INFY.NS:1y": make_mock_historical_long("INFY.NS"),
        },
        financials={
            MOCK_SYMBOL: make_mock_financial_metrics(),
            "TATAMOTORS.NS": make_mock_financial_metrics("TATAMOTORS.NS"),
            "M&M.NS": make_mock_financial_metrics("M&M.NS"),
            "INFY.NS": make_mock_financial_metrics("INFY.NS"),
        },
        quotes={
            MOCK_SYMBOL: make_mock_quote(MOCK_SYMBOL),
            "TATAMOTORS.NS": make_mock_quote("TATAMOTORS.NS"),
            "M&M.NS": make_mock_quote("M&M.NS"),
            "INFY.NS": make_mock_quote("INFY.NS"),
        },
    )
    fundamental = FundamentalAnalyst(market, mock_llm)
    technical = TechnicalAnalyst(market, mock_llm)
    sentiment = SentimentAnalyst(mock_news_search, mock_llm)
    master = MasterAnalyst(mock_llm)
    return GraphDependencies(
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
