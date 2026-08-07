"""Tests for FundamentalAnalysisEngine — deterministic scoring, no LLM."""

import pytest

from app.analysis.fundamental_analysis import FundamentalAnalysisEngine
from app.models.schemas import Rating
from tests.fixtures.fundamental_data import make_mock_financial_metrics


def test_score_range():
    engine = FundamentalAnalysisEngine(make_mock_financial_metrics())
    score = engine.compute_score()
    assert 0 <= score <= 100


def test_strong_metrics_score_well():
    engine = FundamentalAnalysisEngine(make_mock_financial_metrics())
    score = engine.compute_score()
    assert score >= 65


def test_rating_mapping():
    engine = FundamentalAnalysisEngine(make_mock_financial_metrics())
    score = engine.compute_score()
    rating = engine.score_to_rating(score)
    assert rating in Rating


def test_strengths_and_weaknesses():
    engine = FundamentalAnalysisEngine(make_mock_financial_metrics())
    strengths = engine.build_strengths()
    weaknesses = engine.build_weaknesses()
    assert len(strengths) > 0
    assert isinstance(weaknesses, list)


def test_weak_metrics_lower_score():
    weak = make_mock_financial_metrics().model_copy(
        update={
            "revenue_growth": -8.0,
            "earnings_growth": -12.0,
            "roe": 4.0,
            "pe_ratio": 55.0,
            "debt_to_equity": 2.0,
            "free_cash_flow": -50_000_000_000,
        }
    )
    strong_engine = FundamentalAnalysisEngine(make_mock_financial_metrics())
    weak_engine = FundamentalAnalysisEngine(weak)
    assert weak_engine.compute_score() < strong_engine.compute_score()
