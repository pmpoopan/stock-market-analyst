"""Tests for TechnicalIndicatorEngine — uses mock OHLCV data, no LLM calls."""

import pytest

from app.analysis.technical_indicators import TechnicalIndicatorEngine
from app.models.schemas import OHLCVBar
from tests.fixtures.market_data import make_mock_bars
from tests.fixtures.ohlcv_bars import make_flat_bars, make_trending_bars, make_volatile_bars


class TestSMA:
    def test_sma_known_values(self):
        bars = make_mock_bars(count=5, start_price=100.0)
        engine = TechnicalIndicatorEngine(bars)
        sma3 = engine.sma(3)
        assert sma3[0] is None
        assert sma3[1] is None
        # closes: 100, 110, 120, 130, 140 → last SMA(3) = 130
        assert sma3[2] == pytest.approx(110.0)
        assert sma3[4] == pytest.approx(130.0)

    def test_sma_flat_series(self):
        engine = TechnicalIndicatorEngine(make_flat_bars(25, price=50.0))
        sma20 = engine.sma(20)
        assert sma20[-1] == pytest.approx(50.0)

    def test_sma_invalid_period(self):
        engine = TechnicalIndicatorEngine(make_mock_bars(5))
        with pytest.raises(ValueError, match="SMA period"):
            engine.sma(0)


class TestEMA:
    def test_ema_produces_values_after_seed(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        ema20 = engine.ema(20)
        assert ema20[18] is None
        assert ema20[19] is not None
        assert ema20[-1] is not None
        assert ema20[-1] > ema20[19]

    def test_ema_insufficient_data(self):
        engine = TechnicalIndicatorEngine(make_mock_bars(5))
        ema20 = engine.ema(20)
        assert all(v is None for v in ema20)


class TestRSI:
    def test_rsi_range(self):
        engine = TechnicalIndicatorEngine(make_volatile_bars(50))
        rsi = engine.rsi(14)
        valid = [v for v in rsi if v is not None]
        assert len(valid) > 0
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_uptrend_high(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50, daily_change=5.0))
        rsi = engine.rsi(14)
        assert rsi[-1] is not None
        assert rsi[-1] > 60


class TestMACD:
    def test_macd_structure(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(100))
        macd = engine.macd()
        assert set(macd.keys()) == {"macd", "signal", "histogram"}
        assert len(macd["macd"]) == 100
        assert macd["macd"][-1] is not None


class TestStochastic:
    def test_stochastic_range(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        stoch = engine.stochastic()
        k_valid = [v for v in stoch["k"] if v is not None]
        assert all(0 <= v <= 100 for v in k_valid)


class TestVolatility:
    def test_atr_positive(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        atr = engine.atr(14)
        valid = [v for v in atr if v is not None]
        assert all(v > 0 for v in valid)

    def test_bollinger_bands_ordering(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        bb = engine.bollinger_bands(20)
        idx = -1
        assert bb["upper"][idx] is not None
        assert bb["middle"][idx] is not None
        assert bb["lower"][idx] is not None
        assert bb["upper"][idx] > bb["middle"][idx] > bb["lower"][idx]


class TestVolume:
    def test_volume_change(self):
        bars = make_mock_bars(5)
        engine = TechnicalIndicatorEngine(bars)
        vc = engine.volume_change()
        assert vc[0] is None
        assert vc[1] is not None

    def test_obv_trending_up_increases(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(20))
        obv = engine.obv()
        assert obv[-1] is not None
        assert obv[0] is not None
        assert obv[-1] > obv[0]

    def test_volume_ma(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(30))
        vma = engine.volume_moving_average(10)
        assert vma[8] is None
        assert vma[9] is not None


class TestPriceLevels:
    def test_52_week_high_low(self):
        bars = make_trending_bars(300)
        engine = TechnicalIndicatorEngine(bars)
        w52 = engine.week_52_high_low()
        assert w52["lookback_bars"] == 252
        assert w52["high_52w"] > w52["low_52w"]

    def test_52_week_uses_all_bars_when_short(self):
        bars = make_trending_bars(50)
        engine = TechnicalIndicatorEngine(bars)
        w52 = engine.week_52_high_low()
        assert w52["lookback_bars"] == 50

    def test_support_resistance(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        sr = engine.support_resistance(lookback=20)
        assert sr["support"] < sr["resistance"]

    def test_drawdown_uptrend_near_zero(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        dd = engine.drawdown()
        assert dd["current_drawdown_pct"] == pytest.approx(0.0, abs=0.01)
        assert dd["max_drawdown_pct"] <= 0

    def test_breakout_conditions_keys(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(50))
        bc = engine.breakout_conditions()
        assert "assessment" in bc
        assert isinstance(bc["above_resistance"], bool)
        assert isinstance(bc["volume_surge"], bool)


class TestComputeAll:
    def test_compute_all_structure(self):
        engine = TechnicalIndicatorEngine(make_trending_bars(250))
        result = engine.compute_all()

        assert result["bar_count"] == 250
        assert "trend" in result
        assert "momentum" in result
        assert "volatility" in result
        assert "volume" in result
        assert "price_levels" in result
        assert "latest" in result

        assert "sma_20" in result["trend"]
        assert "sma_200" in result["trend"]
        assert "ema_50" in result["trend"]
        assert "rsi_14" in result["momentum"]
        assert "macd" in result["momentum"]
        assert result["latest"]["close"] is not None
        assert result["latest"]["sma_200"] is not None

    def test_empty_bars_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            TechnicalIndicatorEngine([])

    def test_single_bar_minimal(self):
        bar = OHLCVBar(
            date=make_trending_bars(1)[0].date,
            open=100,
            high=105,
            low=95,
            close=102,
            volume=1_000_000,
        )
        engine = TechnicalIndicatorEngine([bar])
        w52 = engine.week_52_high_low()
        assert w52["high_52w"] == 105
        assert w52["low_52w"] == 95
