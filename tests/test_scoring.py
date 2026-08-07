"""Tests for ScoringEngine.compute_decision."""

from app.analysis.scoring import ScoringEngine
from app.analysis.master_synthesis import build_master_analysis_fallback
from app.models.schemas import (
    FundamentalAnalysisResult,
    Rating,
    SentimentAnalysisResult,
    SentimentClassification,
    TechnicalAnalysisResult,
    TrendDirection,
)
from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.market_data import MOCK_SYMBOL


def _sample_inputs():
    metrics = make_mock_financial_metrics()
    fundamental = FundamentalAnalysisResult(
        stock=MOCK_SYMBOL,
        score=75.0,
        rating=Rating.BUY,
        metrics=metrics,
        strengths=["s"],
        weaknesses=["w"],
        risks=["Fundamental risk"],
        summary="summary",
    )
    technical = TechnicalAnalysisResult(
        stock=MOCK_SYMBOL,
        score=70.0,
        trend=TrendDirection.UPTREND,
        signals=[],
        momentum="m",
        volatility="v",
        summary="t",
    )
    sentiment = SentimentAnalysisResult(
        stock=MOCK_SYMBOL,
        sentiment_score=60.0,
        sentiment_classification=SentimentClassification.POSITIVE,
        positive_catalysts=["p"],
        negative_catalysts=["Sentiment risk"],
        key_events=["e"],
        sources=[],
        publication_dates=[],
        articles=[],
        summary="s",
    )
    return fundamental, technical, sentiment


def test_compute_decision_weighted_score():
    engine = ScoringEngine()
    fundamental, technical, sentiment = _sample_inputs()

    decision = engine.compute_decision(MOCK_SYMBOL, fundamental, technical, sentiment)
    assert 0 <= decision.overall_score <= 100
    assert decision.fundamental_score == 75.0
    assert decision.rating in Rating


def test_compute_decision_uses_master_for_reasons():
    engine = ScoringEngine()
    fundamental, technical, sentiment = _sample_inputs()
    master = build_master_analysis_fallback(MOCK_SYMBOL, fundamental, technical, sentiment)

    decision = engine.compute_decision(
        MOCK_SYMBOL, fundamental, technical, sentiment, master=master
    )

    assert any("Agreement" in reason for reason in decision.key_reasons)
    assert decision.risk_adjustment >= 0

