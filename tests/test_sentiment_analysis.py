"""Tests for SentimentAnalysisEngine — deterministic, no LLM."""

from app.analysis.sentiment_analysis import SentimentAnalysisEngine
from app.models.schemas import SentimentClassification
from tests.fixtures.news_data import (
    MOCK_NEWS_NEGATIVE,
    MOCK_NEWS_POSITIVE,
    make_mock_articles,
)


def test_positive_articles_score_above_neutral():
    engine = SentimentAnalysisEngine([MOCK_NEWS_POSITIVE])
    assert engine.compute_score() > 50


def test_negative_articles_score_below_neutral():
    engine = SentimentAnalysisEngine([MOCK_NEWS_NEGATIVE])
    assert engine.compute_score() < 50


def test_empty_articles_neutral_score():
    engine = SentimentAnalysisEngine([])
    score = engine.compute_score()
    assert score == 50.0
    assert engine.classify(score) == SentimentClassification.NEUTRAL


def test_classification_mapping():
    engine = SentimentAnalysisEngine(make_mock_articles())
    score = engine.compute_score()
    classification = engine.classify(score)
    assert classification in SentimentClassification


def test_build_catalysts():
    engine = SentimentAnalysisEngine(make_mock_articles())
    assert len(engine.build_positive_catalysts()) > 0
    assert isinstance(engine.build_negative_catalysts(), list)
    assert len(engine.build_key_events()) > 0
