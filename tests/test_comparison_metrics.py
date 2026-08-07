"""Tests for ComparisonMetricsCalculator — deterministic comparison math."""

from app.analysis.comparison_metrics import ComparisonMetricsCalculator
from app.analysis.scoring import ScoringEngine
from app.models.schemas import (
    FundamentalAnalysisResult,
    Rating,
    TechnicalAnalysisResult,
    TrendDirection,
)
from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.market_data import MOCK_SYMBOL, MOCK_SYMBOL_2


def _decision(symbol: str, overall: float, fundamental: float, technical: float, sentiment: float):
    from app.models.schemas import DecisionResult

    return DecisionResult(
        stock=symbol,
        overall_score=overall,
        rating=Rating.BUY,
        fundamental_score=fundamental,
        technical_score=technical,
        sentiment_score=sentiment,
        risk_adjustment=5.0,
        key_reasons=["reason"],
        major_risks=["risk"],
    )


def _fundamental(symbol: str) -> FundamentalAnalysisResult:
    return FundamentalAnalysisResult(
        stock=symbol,
        score=75.0,
        rating=Rating.BUY,
        metrics=make_mock_financial_metrics(symbol),
        strengths=["strength"],
        weaknesses=["weakness"],
        risks=["risk"],
        summary="summary",
    )


def _technical(symbol: str, score: float = 70.0, trend: TrendDirection = TrendDirection.UPTREND):
    return TechnicalAnalysisResult(
        stock=symbol,
        score=score,
        trend=trend,
        signals=[],
        momentum="Positive",
        volatility="Moderate",
        summary="summary",
    )


def test_score_maps_and_winner():
    decisions = {
        MOCK_SYMBOL: _decision(MOCK_SYMBOL, 80, 78, 72, 65),
        MOCK_SYMBOL_2: _decision(MOCK_SYMBOL_2, 65, 60, 62, 58),
    }
    fundamental, technical, sentiment, overall = ComparisonMetricsCalculator.score_maps(decisions)

    assert fundamental[MOCK_SYMBOL] == 78
    assert technical[MOCK_SYMBOL_2] == 62
    assert sentiment[MOCK_SYMBOL] == 65
    assert overall[MOCK_SYMBOL] == 80

    winner = ComparisonMetricsCalculator.determine_winner(overall)
    assert winner == MOCK_SYMBOL


def test_build_comparison_returns_narratives():
    stocks = [MOCK_SYMBOL, MOCK_SYMBOL_2]
    decisions = {
        MOCK_SYMBOL: _decision(MOCK_SYMBOL, 80, 78, 72, 65),
        MOCK_SYMBOL_2: _decision(MOCK_SYMBOL_2, 65, 60, 62, 58),
    }
    fundamentals = {symbol: _fundamental(symbol) for symbol in stocks}
    technical = {
        MOCK_SYMBOL: _technical(MOCK_SYMBOL, 72),
        MOCK_SYMBOL_2: _technical(MOCK_SYMBOL_2, 62, TrendDirection.SIDEWAYS),
    }

    result = ComparisonMetricsCalculator.build_comparison(
        stocks, decisions, fundamentals, technical
    )

    assert result.stocks == stocks
    assert result.winner == MOCK_SYMBOL
    assert "PE" in result.valuation_comparison
    assert "growth" in result.growth_comparison.lower()
    assert "risk" in result.risk_comparison.lower()
    assert "technical" in result.technical_trend_comparison.lower()
    assert MOCK_SYMBOL in result.relative_assessment


def test_scoring_engine_compare_scores():
    engine = ScoringEngine()
    decisions = {
        MOCK_SYMBOL: _decision(MOCK_SYMBOL, 80, 78, 72, 65),
        MOCK_SYMBOL_2: _decision(MOCK_SYMBOL_2, 65, 60, 62, 58),
    }

    assessment = engine.compare_scores(decisions)
    assert MOCK_SYMBOL in assessment
    assert "80" in assessment
