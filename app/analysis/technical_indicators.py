"""Deterministic technical indicator calculations.

All indicator math lives here — agents/LLMs must NOT compute indicators.
"""

from __future__ import annotations

from statistics import mean, pstdev

from app.models.schemas import OHLCVBar

TRADING_DAYS_52W = 252
DEFAULT_SR_LOOKBACK = 20


class TechnicalIndicatorEngine:
    """Compute trend, momentum, volatility, and volume indicators from OHLCV bars."""

    def __init__(self, bars: list[OHLCVBar]) -> None:
        if not bars:
            raise ValueError("At least one OHLCV bar is required")
        self._bars = bars

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    def _closes(self) -> list[float]:
        return [b.close for b in self._bars]

    def _opens(self) -> list[float]:
        return [b.open for b in self._bars]

    def _highs(self) -> list[float]:
        return [b.high for b in self._bars]

    def _lows(self) -> list[float]:
        return [b.low for b in self._bars]

    def _volumes(self) -> list[int]:
        return [b.volume for b in self._bars]

    @staticmethod
    def _last_valid(values: list[float | None]) -> float | None:
        for value in reversed(values):
            if value is not None:
                return value
        return None

    @staticmethod
    def _round_series(values: list[float | None], places: int = 4) -> list[float | None]:
        return [round(v, places) if v is not None else None for v in values]

    # --- Trend ---
    def sma(self, period: int) -> list[float | None]:
        if period < 1:
            raise ValueError("SMA period must be >= 1")

        closes = self._closes()
        result: list[float | None] = [None] * len(closes)

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1 : i + 1]
            result[i] = mean(window)

        return self._round_series(result)

    def ema(self, period: int) -> list[float | None]:
        if period < 1:
            raise ValueError("EMA period must be >= 1")

        closes = self._closes()
        result: list[float | None] = [None] * len(closes)
        if len(closes) < period:
            return result

        multiplier = 2 / (period + 1)
        seed = mean(closes[:period])
        result[period - 1] = seed

        for i in range(period, len(closes)):
            prev = result[i - 1]
            assert prev is not None
            result[i] = (closes[i] - prev) * multiplier + prev

        return self._round_series(result)

    def _ema_from_series(
        self, series: list[float | None], period: int
    ) -> list[float | None]:
        """EMA over a series that may contain None values (used for MACD signal line)."""
        result: list[float | None] = [None] * len(series)
        values = [(i, v) for i, v in enumerate(series) if v is not None]
        if len(values) < period:
            return result

        start_idx = values[period - 1][0]
        seed = mean(v for _, v in values[:period])
        result[start_idx] = seed
        multiplier = 2 / (period + 1)

        prev = seed
        for idx in range(start_idx + 1, len(series)):
            val = series[idx]
            if val is None:
                result[idx] = None
                continue
            prev = (val - prev) * multiplier + prev
            result[idx] = prev

        return self._round_series(result)

    # --- Momentum ---
    def rsi(self, period: int = 14) -> list[float | None]:
        if period < 1:
            raise ValueError("RSI period must be >= 1")

        closes = self._closes()
        result: list[float | None] = [None] * len(closes)
        if len(closes) <= period:
            return result

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]

        avg_gain = mean(gains[:period])
        avg_loss = mean(losses[:period])

        def _calc_rsi(g: float, l: float) -> float:
            if l == 0:
                return 100.0
            rs = g / l
            return 100 - (100 / (1 + rs))

        result[period] = _calc_rsi(avg_gain, avg_loss)

        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            result[i + 1] = _calc_rsi(avg_gain, avg_loss)

        return self._round_series(result)

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, list[float | None]]:
        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)

        macd_line: list[float | None] = [None] * self.bar_count
        for i in range(self.bar_count):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line[i] = ema_fast[i] - ema_slow[i]

        signal_line = self._ema_from_series(macd_line, signal)

        histogram: list[float | None] = [None] * self.bar_count
        for i in range(self.bar_count):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram[i] = macd_line[i] - signal_line[i]

        return {
            "macd": self._round_series(macd_line),
            "signal": signal_line,
            "histogram": self._round_series(histogram),
        }

    def stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> dict[str, list[float | None]]:
        if k_period < 1 or d_period < 1:
            raise ValueError("Stochastic periods must be >= 1")

        highs = self._highs()
        lows = self._lows()
        closes = self._closes()
        k_values: list[float | None] = [None] * len(closes)

        for i in range(k_period - 1, len(closes)):
            window_high = max(highs[i - k_period + 1 : i + 1])
            window_low = min(lows[i - k_period + 1 : i + 1])
            if window_high == window_low:
                k_values[i] = 50.0
            else:
                k_values[i] = ((closes[i] - window_low) / (window_high - window_low)) * 100

        k_rounded = self._round_series(k_values)
        d_from_k: list[float | None] = [None] * len(closes)
        for i in range(d_period - 1, len(closes)):
            window = k_rounded[i - d_period + 1 : i + 1]
            if any(v is None for v in window):
                continue
            d_from_k[i] = mean(window)  # type: ignore[arg-type]

        return {
            "k": k_rounded,
            "d": self._round_series(d_from_k),
        }

    # --- Volatility ---
    def _true_ranges(self) -> list[float]:
        highs = self._highs()
        lows = self._lows()
        closes = self._closes()
        trs = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return trs

    def atr(self, period: int = 14) -> list[float | None]:
        if period < 1:
            raise ValueError("ATR period must be >= 1")

        trs = self._true_ranges()
        result: list[float | None] = [None] * len(trs)
        if len(trs) < period:
            return result

        atr_val = mean(trs[:period])
        result[period - 1] = atr_val

        for i in range(period, len(trs)):
            atr_val = ((atr_val * (period - 1)) + trs[i]) / period
            result[i] = atr_val

        return self._round_series(result)

    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict[str, list[float | None]]:
        if period < 1:
            raise ValueError("Bollinger period must be >= 1")

        closes = self._closes()
        middle = self.sma(period)
        upper: list[float | None] = [None] * len(closes)
        lower: list[float | None] = [None] * len(closes)

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1 : i + 1]
            mid = middle[i]
            if mid is None:
                continue
            deviation = pstdev(window)
            upper[i] = mid + std_dev * deviation
            lower[i] = mid - std_dev * deviation

        return {
            "upper": self._round_series(upper),
            "middle": middle,
            "lower": self._round_series(lower),
        }

    # --- Volume ---
    def volume_moving_average(self, period: int = 20) -> list[float | None]:
        if period < 1:
            raise ValueError("Volume MA period must be >= 1")

        volumes = [float(v) for v in self._volumes()]
        result: list[float | None] = [None] * len(volumes)

        for i in range(period - 1, len(volumes)):
            result[i] = mean(volumes[i - period + 1 : i + 1])

        return self._round_series(result)

    def volume_change(self) -> list[float | None]:
        volumes = self._volumes()
        result: list[float | None] = [None] * len(volumes)

        for i in range(1, len(volumes)):
            prev = volumes[i - 1]
            if prev == 0:
                result[i] = None
            else:
                result[i] = ((volumes[i] - prev) / prev) * 100

        return self._round_series(result)

    def obv(self) -> list[float | None]:
        closes = self._closes()
        volumes = self._volumes()
        result: list[float | None] = [None] * len(closes)
        obv_val = 0.0
        result[0] = float(volumes[0])

        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv_val = (result[i - 1] or 0.0) + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv_val = (result[i - 1] or 0.0) - volumes[i]
            else:
                obv_val = result[i - 1] or 0.0
            result[i] = obv_val

        return self._round_series(result)

    # --- Price levels ---
    def week_52_high_low(self) -> dict[str, float | None]:
        lookback = min(TRADING_DAYS_52W, self.bar_count)
        window = self._bars[-lookback:]
        highs = [b.high for b in window]
        lows = [b.low for b in window]
        return {
            "high_52w": round(max(highs), 4),
            "low_52w": round(min(lows), 4),
            "lookback_bars": lookback,
        }

    def support_resistance(self, lookback: int = DEFAULT_SR_LOOKBACK) -> dict[str, float | None]:
        if lookback < 1:
            raise ValueError("Support/resistance lookback must be >= 1")

        window = self._bars[-min(lookback, self.bar_count) :]
        return {
            "support": round(min(b.low for b in window), 4),
            "resistance": round(max(b.high for b in window), 4),
            "lookback": min(lookback, self.bar_count),
        }

    def drawdown(self) -> dict[str, float | None]:
        closes = self._closes()
        peak = closes[0]
        max_drawdown = 0.0
        current_drawdown = 0.0

        for close in closes:
            if close > peak:
                peak = close
            if peak == 0:
                dd = 0.0
            else:
                dd = ((close - peak) / peak) * 100
            if dd < max_drawdown:
                max_drawdown = dd
            current_drawdown = dd

        return {
            "current_drawdown_pct": round(current_drawdown, 4),
            "max_drawdown_pct": round(max_drawdown, 4),
        }

    def breakout_conditions(
        self,
        volume_surge_multiplier: float = 1.5,
        near_extreme_pct: float = 2.0,
    ) -> dict[str, bool | str]:
        closes = self._closes()
        volumes = self._volumes()
        latest_close = closes[-1]
        latest_volume = volumes[-1]

        sr = self.support_resistance()
        w52 = self.week_52_high_low()
        vol_ma = self.volume_moving_average(20)
        latest_vol_ma = self._last_valid(vol_ma)

        support = sr["support"]
        resistance = sr["resistance"]
        high_52w = w52["high_52w"]
        low_52w = w52["low_52w"]

        above_resistance = (
            resistance is not None and latest_close > resistance
        )
        below_support = support is not None and latest_close < support
        volume_surge = (
            latest_vol_ma is not None
            and latest_vol_ma > 0
            and latest_volume >= latest_vol_ma * volume_surge_multiplier
        )

        near_52w_high = (
            high_52w is not None
            and high_52w > 0
            and ((high_52w - latest_close) / high_52w) * 100 <= near_extreme_pct
        )
        near_52w_low = (
            low_52w is not None
            and low_52w > 0
            and ((latest_close - low_52w) / low_52w) * 100 <= near_extreme_pct
        )

        if above_resistance and volume_surge:
            assessment = "bullish_breakout"
        elif below_support and volume_surge:
            assessment = "bearish_breakdown"
        elif above_resistance:
            assessment = "testing_resistance"
        elif below_support:
            assessment = "testing_support"
        elif near_52w_high:
            assessment = "near_52w_high"
        elif near_52w_low:
            assessment = "near_52w_low"
        else:
            assessment = "neutral"

        return {
            "above_resistance": above_resistance,
            "below_support": below_support,
            "volume_surge": volume_surge,
            "near_52w_high": near_52w_high,
            "near_52w_low": near_52w_low,
            "assessment": assessment,
        }

    def compute_all(self) -> dict:
        """Return a dict of all computed indicators for downstream agents."""
        trend = {
            "sma_20": self.sma(20),
            "sma_50": self.sma(50),
            "sma_100": self.sma(100),
            "sma_200": self.sma(200),
            "ema_20": self.ema(20),
            "ema_50": self.ema(50),
        }
        momentum = {
            "rsi_14": self.rsi(14),
            "macd": self.macd(),
            "stochastic": self.stochastic(),
        }
        volatility = {
            "atr_14": self.atr(14),
            "bollinger_bands": self.bollinger_bands(),
        }
        volume = {
            "volume_ma_20": self.volume_moving_average(20),
            "volume_change": self.volume_change(),
            "obv": self.obv(),
        }
        price_levels = {
            "week_52_high_low": self.week_52_high_low(),
            "support_resistance": self.support_resistance(),
            "drawdown": self.drawdown(),
            "breakout_conditions": self.breakout_conditions(),
        }

        latest = {
            "close": self._closes()[-1],
            "sma_20": self._last_valid(trend["sma_20"]),
            "sma_50": self._last_valid(trend["sma_50"]),
            "sma_200": self._last_valid(trend["sma_200"]),
            "ema_20": self._last_valid(trend["ema_20"]),
            "rsi_14": self._last_valid(momentum["rsi_14"]),
            "macd": self._last_valid(momentum["macd"]["macd"]),
            "macd_signal": self._last_valid(momentum["macd"]["signal"]),
            "atr_14": self._last_valid(volatility["atr_14"]),
            "volume_ma_20": self._last_valid(volume["volume_ma_20"]),
            **price_levels["week_52_high_low"],
            **price_levels["support_resistance"],
            **price_levels["drawdown"],
            "breakout_assessment": price_levels["breakout_conditions"]["assessment"],
        }

        return {
            "bar_count": self.bar_count,
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "volume": volume,
            "price_levels": price_levels,
            "latest": latest,
        }
