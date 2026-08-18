"""Tests for SentimentAnalyst — mock search and mock LLM."""

import pytest

from app.agents.llm_client import MockLLMClient
from app.agents.sentiment_agent import SentimentAnalyst
from app.models.schemas import NewsArticle, SentimentAnalysisResult, SentimentInterpretation
from tests.fixtures.market_data import MOCK_SYMBOL
from tests.fixtures.news_data import (
    MOCK_NEWS_DUPLICATE,
    MOCK_NEWS_IRRELEVANT,
    MOCK_NEWS_POSITIVE,
    MockNewsSearchProvider,
    make_mock_articles,
)


@pytest.fixture
def sentiment_analyst(mock_llm):
    return SentimentAnalyst(news_search=MockNewsSearchProvider(), llm=mock_llm)


@pytest.mark.asyncio
async def test_analyze_returns_structured_result(sentiment_analyst):
    result = await sentiment_analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")

    assert isinstance(result, SentimentAnalysisResult)
    assert result.stock == MOCK_SYMBOL
    assert 0 <= result.sentiment_score <= 100
    assert len(result.articles) > 0
    assert len(result.sources) == len(result.articles)
    assert len(result.publication_dates) == len(result.articles)
    assert result.summary


@pytest.mark.asyncio
async def test_analyze_calls_mock_llm(sentiment_analyst, mock_llm):
    await sentiment_analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["structured_output"] is SentimentInterpretation


@pytest.mark.asyncio
async def test_analyze_custom_llm_response():
    custom_llm = MockLLMClient(
        structured_responses={
            SentimentInterpretation: SentimentInterpretation(
                positive_catalysts=["Custom positive"],
                negative_catalysts=["Custom negative"],
                key_events=["Custom event"],
                summary="Custom sentiment summary.",
            )
        }
    )
    analyst = SentimentAnalyst(news_search=MockNewsSearchProvider(), llm=custom_llm)
    result = await analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")

    assert result.positive_catalysts == ["Custom positive"]
    assert result.summary == "Custom sentiment summary."


@pytest.mark.asyncio
async def test_deduplicate_articles():
    analyst = SentimentAnalyst(news_search=MockNewsSearchProvider(), llm=MockLLMClient())
    articles = make_mock_articles() + [MOCK_NEWS_DUPLICATE]
    deduped = analyst._deduplicate_articles(articles)
    assert len(deduped) == len(make_mock_articles())


@pytest.mark.asyncio
async def test_filter_removes_irrelevant():
    analyst = SentimentAnalyst(news_search=MockNewsSearchProvider(), llm=MockLLMClient())
    articles = make_mock_articles() + [MOCK_NEWS_IRRELEVANT]
    queries = analyst._build_search_queries(MOCK_SYMBOL, "Reliance Industries")
    filtered = analyst._filter_relevant(articles, MOCK_SYMBOL, "Reliance Industries", queries)
    assert MOCK_NEWS_IRRELEVANT.title not in [a.title for a in filtered]


@pytest.mark.asyncio
async def test_analyze_no_articles_neutral():
    empty_search = MockNewsSearchProvider(articles_by_query={})
    analyst = SentimentAnalyst(news_search=empty_search, llm=MockLLMClient())
    result = await analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")

    assert result.sentiment_score == 50.0
    assert len(result.articles) == 0
    assert "No recent news" in result.summary


@pytest.mark.asyncio
async def test_analyze_fallback_when_llm_fails():
    class FailingLLM(MockLLMClient):
        async def generate(self, prompt, system=None, structured_output=None, max_tokens=None):
            raise RuntimeError("LLM unavailable")

    analyst = SentimentAnalyst(
        news_search=MockNewsSearchProvider(),
        llm=FailingLLM(),
    )
    result = await analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")
    assert MOCK_SYMBOL in result.summary


@pytest.mark.asyncio
async def test_analyze_survives_search_timeout():
    class TimeoutSearchProvider(MockNewsSearchProvider):
        def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
            raise TimeoutError("search timed out")

    analyst = SentimentAnalyst(news_search=TimeoutSearchProvider(), llm=MockLLMClient())
    result = await analyst.analyze(MOCK_SYMBOL, company_name="Reliance Industries")

    assert result.stock == MOCK_SYMBOL
    assert len(result.articles) == 0
    assert "No recent news" in result.summary
