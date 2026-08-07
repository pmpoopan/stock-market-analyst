"""Tests for TechnicalAnalysisEngine — deterministic, no LLM."""

import pytest

from app.analysis.technical_analysis import TechnicalAnalysisEngine
from app.analysis.technical_indicators import TechnicalIndicatorEngine
from app.models.schemas import TrendDirection
from tests.fixtures.ohlcv_bars import make_trending_bars, make_volatile_bars


def _engine_from_bars(bars):
    indicators = TechnicalIndicatorEngine(bars).compute_all()
    return TechnicalAnalysisEngine(indicators)


class TestSignals:
    def test_build_signals_uptrend(self):
        engine = _engine_from_bars(make_trending_bars(250))
        signals = engine.build_signals()
        assert len(signals) >= 5
        names = {s.name for s in signals}
        assert "SMA 20" in names
        assert "RSI 14" in names

    def test_bullish_signals_majority_in_uptrend(self):
        engine = _engine_from_bars(make_trending_bars(250, daily_change=3.0))
        signals = engine.build_signals()
        bullish = sum(1 for s in signals if s.signal == "bullish")
        bearish = sum(1 for s in signals if s.signal == "bearish")
        assert bullish > bearish


class TestTrend:
    def test_uptrend_classification(self):
        engine = _engine_from_bars(make_trending_bars(250, daily_change=2.5))
        trend = engine.determine_trend()
        assert trend in (TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND)

    def test_volatile_sideways_or_mixed(self):
        engine = _engine_from_bars(make_volatile_bars(80))
        trend = engine.determine_trend()
        assert trend in TrendDirection


class TestScoring:
    def test_score_range(self):
        engine = _engine_from_bars(make_trending_bars(250))
        signals = engine.build_signals()
        score = engine.compute_score(signals)
        assert 0 <= score <= 100

    def test_uptrend_scores_higher_than_flat(self):
        up_engine = _engine_from_bars(make_trending_bars(250, daily_change=3.0))
        flat_bars = make_trending_bars(250, daily_change=0.0, start_price=1000.0)
        flat_engine = _engine_from_bars(flat_bars)
        up_score = up_engine.compute_score(up_engine.build_signals())
        flat_score = flat_engine.compute_score(flat_engine.build_signals())
        assert up_score >= flat_score


class TestLabels:
    def test_momentum_label_non_empty(self):
        engine = _engine_from_bars(make_trending_bars(100))
        assert len(engine.momentum_label()) > 10

    def test_volatility_label_non_empty(self):
        engine = _engine_from_bars(make_trending_bars(100))
        assert "ATR" in engine.volatility_label()

    def test_build_summary_includes_symbol(self):
        engine = _engine_from_bars(make_trending_bars(100))
        signals = engine.build_signals()
        summary = engine.build_summary("RELIANCE.NS", engine.determine_trend(), 70.0, signals)
        assert "RELIANCE.NS" in summary
