"""Tests for DuckDuckGo search provider — mocked DDGS, no live API."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.data.exceptions import DataProviderError
from app.data.web_search import DuckDuckGoSearchProvider
from app.models.schemas import NewsArticle


@pytest.fixture
def search_provider(cache, test_settings):
    return DuckDuckGoSearchProvider(cache=cache, settings=test_settings, max_results=5)


MOCK_DDG_RESULTS = [
    {
        "title": "Reliance profit beats estimates",
        "url": "https://example.com/reliance-profit",
        "source": "Economic Times",
        "body": "Reliance reported strong profit growth in the latest quarter.",
        "date": "2026-01-10T00:00:00+00:00",
    },
    {
        "title": "Reliance expansion plans in retail",
        "url": "https://example.com/reliance-retail",
        "source": "Business Standard",
        "body": "Company announced retail expansion across India.",
        "date": "2026-01-09T00:00:00+00:00",
    },
]


@patch("app.data.web_search.DDGS")
def test_search_news_fetches_from_ddg(mock_ddgs_cls, search_provider):
    mock_ddgs = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.news.return_value = MOCK_DDG_RESULTS

    articles = search_provider.search_news("Reliance stock India news")

    assert len(articles) == 2
    assert all(isinstance(a, NewsArticle) for a in articles)
    assert articles[0].title == "Reliance profit beats estimates"
    mock_ddgs.news.assert_called_once()


@patch("app.data.web_search.DDGS")
def test_search_news_uses_cache(mock_ddgs_cls, search_provider):
    mock_ddgs = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.news.return_value = MOCK_DDG_RESULTS

    search_provider.search_news("Reliance stock India news")
    search_provider.search_news("Reliance stock India news")

    mock_ddgs_cls.assert_called_once()


@patch("app.data.web_search.DDGS")
def test_search_news_skips_invalid_rows(mock_ddgs_cls, search_provider):
    mock_ddgs = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs.news.return_value = [
        {"title": "", "url": "https://example.com/empty", "source": "X"},
        MOCK_DDG_RESULTS[0],
    ]

    articles = search_provider.search_news("Reliance news")
    assert len(articles) == 1


def test_search_news_empty_query_raises(search_provider):
    with pytest.raises(ValueError, match="empty"):
        search_provider.search_news("  ")


@patch("app.data.web_search.DDGS")
def test_search_news_wraps_errors(mock_ddgs_cls, search_provider):
    mock_ddgs_cls.return_value.__enter__.side_effect = RuntimeError("network")
    with pytest.raises(DataProviderError, match="News search failed"):
        search_provider.search_news("Reliance news")


def test_serialize_deserialize_roundtrip(search_provider):
    articles = [
        NewsArticle(
            title="Test",
            source="Source",
            url="https://example.com/a",
            published_at=None,
            snippet="snippet",
        )
    ]
    payload = search_provider._serialize_articles(articles)
    restored = search_provider._deserialize_articles(payload)
    assert len(restored) == 1
    assert restored[0].title == "Test"
