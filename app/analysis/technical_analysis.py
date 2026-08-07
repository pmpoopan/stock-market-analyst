"""Deterministic technical signal detection, scoring, and trend classification.

All numeric indicator values come from TechnicalIndicatorEngine — this module
only interprets pre-computed values into signals and scores.
"""

from __future__ import annotations

from typing import Any

from app.models.schemas import TechnicalSignal, TrendDirection

MIN_BARS_RECOMMENDED = 200


class TechnicalAnalysisEngine:
    """Build signals, trend, and score from pre-computed indicator dict."""

    def __init__(self, indicators: dict[str, Any]) -> None:
        self._indicators = indicators
        self._latest = indicators.get("latest", {})

    @property
    def bar_count(self) -> int:
        return int(self._indicators.get("bar_count", 0))

    @property
    def close(self) -> float | None:
        return self._latest.get("close")

    @property
    def support(self) -> float | None:
        return self._latest.get("support")

    @property
    def resistance(self) -> float | None:
        return self._latest.get("resistance")

    def build_signals(self) -> list[TechnicalSignal]:
        signals: list[TechnicalSignal] = []
        close = self.close
        if close is None:
            return signals

        sma20 = self._latest.get("sma_20")
        sma50 = self._latest.get("sma_50")
        sma200 = self._latest.get("sma_200")
        ema20 = self._latest.get("ema_20")
        rsi = self._latest.get("rsi_14")
        macd = self._latest.get("macd")
        macd_signal = self._latest.get("macd_signal")
        atr = self._latest.get("atr_14")
        high_52w = self._latest.get("high_52w")
        low_52w = self._latest.get("low_52w")
        support = self._latest.get("support")
        resistance = self._latest.get("resistance")
        breakout = self._indicators.get("price_levels", {}).get("breakout_conditions", {})

        # --- Trend: moving averages ---
        if sma20 is not None:
            signal = "bullish" if close > sma20 else "bearish"
            signals.append(
                TechnicalSignal(
                    name="SMA 20",
                    value=sma20,
                    signal=signal,
                    description=f"Price {'above' if close > sma20 else 'below'} 20-day SMA",
                )
            )

        if sma50 is not None:
            signal = "bullish" if close > sma50 else "bearish"
            signals.append(
                TechnicalSignal(
                    name="SMA 50",
                    value=sma50,
                    signal=signal,
                    description=f"Price {'above' if close > sma50 else 'below'} 50-day SMA",
                )
            )

        if sma200 is not None:
            signal = "bullish" if close > sma200 else "bearish"
            signals.append(
                TechnicalSignal(
                    name="SMA 200",
                    value=sma200,
                    signal=signal,
                    description=f"Price {'above' if close > sma200 else 'below'} 200-day SMA",
                )
            )

        if ema20 is not None and sma50 is not None:
            signal = "bullish" if ema20 > sma50 else "bearish"
            signals.append(
                TechnicalSignal(
                    name="EMA 20 vs SMA 50",
                    value=f"{ema20:.2f} / {sma50:.2f}",
                    signal=signal,
                    description="Short-term EMA relative to medium-term SMA",
                )
            )

        # --- Momentum ---
        if rsi is not None:
            if rsi >= 70:
                rsi_signal, rsi_desc = "bearish", "RSI indicates overbought conditions"
            elif rsi <= 30:
                rsi_signal, rsi_desc = "bullish", "RSI indicates oversold conditions"
            else:
                rsi_signal, rsi_desc = "neutral", "RSI in neutral range"
            signals.append(
                TechnicalSignal(
                    name="RSI 14",
                    value=rsi,
                    signal=rsi_signal,
                    description=rsi_desc,
                )
            )

        if macd is not None and macd_signal is not None:
            macd_sig = "bullish" if macd > macd_signal else "bearish"
            signals.append(
                TechnicalSignal(
                    name="MACD",
                    value=f"{macd:.4f} / {macd_signal:.4f}",
                    signal=macd_sig,
                    description="MACD line relative to signal line",
                )
            )

        stoch = self._indicators.get("momentum", {}).get("stochastic", {})
        stoch_k = self._last_valid(stoch.get("k", []))
        if stoch_k is not None:
            if stoch_k >= 80:
                st_sig, st_desc = "bearish", "Stochastic overbought"
            elif stoch_k <= 20:
                st_sig, st_desc = "bullish", "Stochastic oversold"
            else:
                st_sig, st_desc = "neutral", "Stochastic in neutral zone"
            signals.append(
                TechnicalSignal(
                    name="Stochastic %K",
                    value=stoch_k,
                    signal=st_sig,
                    description=st_desc,
                )
            )

        # --- Volatility ---
        bb = self._indicators.get("volatility", {}).get("bollinger_bands", {})
        bb_upper = self._last_valid(bb.get("upper", []))
        bb_lower = self._last_valid(bb.get("lower", []))
        if bb_upper is not None and bb_lower is not None:
            if close >= bb_upper:
                bb_sig, bb_desc = "bearish", "Price at or above upper Bollinger Band"
            elif close <= bb_lower:
                bb_sig, bb_desc = "bullish", "Price at or below lower Bollinger Band"
            else:
                bb_sig, bb_desc = "neutral", "Price within Bollinger Bands"
            signals.append(
                TechnicalSignal(
                    name="Bollinger Bands",
                    value=f"{bb_lower:.2f}–{bb_upper:.2f}",
                    signal=bb_sig,
                    description=bb_desc,
                )
            )

        if atr is not None and close > 0:
            atr_pct = (atr / close) * 100
            if atr_pct > 3:
                atr_sig, atr_desc = "bearish", f"Elevated ATR ({atr_pct:.1f}% of price)"
            elif atr_pct < 1:
                atr_sig, atr_desc = "neutral", f"Low ATR ({atr_pct:.1f}% of price)"
            else:
                atr_sig, atr_desc = "neutral", f"Moderate ATR ({atr_pct:.1f}% of price)"
            signals.append(
                TechnicalSignal(
                    name="ATR 14",
                    value=atr,
                    signal=atr_sig,
                    description=atr_desc,
                )
            )

        # --- Volume ---
        vol_surge = breakout.get("volume_surge", False)
        signals.append(
            TechnicalSignal(
                name="Volume",
                value=vol_surge,
                signal="bullish" if vol_surge else "neutral",
                description="Volume surge detected" if vol_surge else "No volume surge",
            )
        )

        # --- Price levels ---
        if support is not None:
            signals.append(
                TechnicalSignal(
                    name="Support",
                    value=support,
                    signal="bullish" if close > support else "bearish",
                    description=f"Nearest support at {support:.2f}",
                )
            )

        if resistance is not None:
            signals.append(
                TechnicalSignal(
                    name="Resistance",
                    value=resistance,
                    signal="bullish" if close > resistance else "bearish",
                    description=f"Nearest resistance at {resistance:.2f}",
                )
            )

        if high_52w is not None and close > 0:
            dist_high = ((high_52w - close) / close) * 100
            signals.append(
                TechnicalSignal(
                    name="52-week high",
                    value=high_52w,
                    signal="bullish" if dist_high <= 5 else "neutral",
                    description=f"{dist_high:.1f}% below 52-week high",
                )
            )

        if low_52w is not None and close > 0:
            dist_low = ((close - low_52w) / close) * 100
            signals.append(
                TechnicalSignal(
                    name="52-week low",
                    value=low_52w,
                    signal="bearish" if dist_low <= 5 else "neutral",
                    description=f"{dist_low:.1f}% above 52-week low",
                )
            )

        assessment = breakout.get("assessment", "neutral")
        if assessment != "neutral":
            bo_signal = (
                "bullish"
                if assessment in ("bullish_breakout", "near_52w_high", "testing_resistance")
                else "bearish"
            )
            signals.append(
                TechnicalSignal(
                    name="Breakout",
                    value=assessment,
                    signal=bo_signal,
                    description=f"Breakout assessment: {assessment.replace('_', ' ')}",
                )
            )

        return signals

    def determine_trend(self) -> TrendDirection:
        close = self.close
        if close is None:
            return TrendDirection.SIDEWAYS

        sma20 = self._latest.get("sma_20")
        sma50 = self._latest.get("sma_50")
        sma100 = self._last_valid(self._indicators.get("trend", {}).get("sma_100", []))
        sma200 = self._latest.get("sma_200")

        bullish_stack = (
            sma20 is not None
            and sma50 is not None
            and close > sma20
            and sma20 > sma50
        )
        bearish_stack = (
            sma20 is not None
            and sma50 is not None
            and close < sma20
            and sma20 < sma50
        )

        if bullish_stack and sma100 is not None and sma50 > sma100:
            if sma200 is not None and sma100 > sma200:
                return TrendDirection.STRONG_UPTREND
            return TrendDirection.UPTREND

        if bearish_stack and sma100 is not None and sma50 < sma100:
            if sma200 is not None and sma100 < sma200:
                return TrendDirection.STRONG_DOWNTREND
            return TrendDirection.DOWNTREND

        if sma50 is not None:
            if close > sma50:
                return TrendDirection.UPTREND
            if close < sma50:
                return TrendDirection.DOWNTREND

        return TrendDirection.SIDEWAYS

    def compute_score(self, signals: list[TechnicalSignal]) -> float:
        """Deterministic 0–100 technical score from signal balance and trend."""
        if not signals:
            return 50.0

        bullish = sum(1 for s in signals if s.signal == "bullish")
        bearish = sum(1 for s in signals if s.signal == "bearish")
        total_directional = bullish + bearish
        if total_directional == 0:
            balance_score = 50.0
        else:
            balance_score = (bullish / total_directional) * 100

        trend_bonus = {
            TrendDirection.STRONG_UPTREND: 15.0,
            TrendDirection.UPTREND: 8.0,
            TrendDirection.SIDEWAYS: 0.0,
            TrendDirection.DOWNTREND: -8.0,
            TrendDirection.STRONG_DOWNTREND: -15.0,
        }[self.determine_trend()]

        data_penalty = 0.0
        if self.bar_count < MIN_BARS_RECOMMENDED:
            data_penalty = 5.0

        score = balance_score + trend_bonus - data_penalty
        return round(max(0.0, min(100.0, score)), 2)

    def momentum_label(self) -> str:
        rsi = self._latest.get("rsi_14")
        macd = self._latest.get("macd")
        macd_signal = self._latest.get("macd_signal")

        parts: list[str] = []
        if rsi is not None:
            if rsi >= 70:
                parts.append(f"RSI at {rsi:.1f} suggests overbought momentum")
            elif rsi <= 30:
                parts.append(f"RSI at {rsi:.1f} suggests oversold momentum")
            else:
                parts.append(f"RSI at {rsi:.1f} is in a neutral momentum zone")

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                parts.append("MACD is above its signal line (bullish momentum)")
            else:
                parts.append("MACD is below its signal line (bearish momentum)")

        return ". ".join(parts) if parts else "Momentum data is limited for this lookback."

    def volatility_label(self) -> str:
        close = self.close
        atr = self._latest.get("atr_14")
        if close is None or atr is None or close <= 0:
            return "Volatility data is limited for this lookback."

        atr_pct = (atr / close) * 100
        dd = self._indicators.get("price_levels", {}).get("drawdown", {})
        max_dd = dd.get("max_drawdown_pct")

        level = "elevated" if atr_pct > 3 else "moderate" if atr_pct >= 1 else "low"
        text = f"ATR is {atr_pct:.1f}% of price, indicating {level} volatility"
        if max_dd is not None:
            text += f"; maximum drawdown in period was {max_dd:.1f}%"
        return text

    def build_summary(
        self,
        symbol: str,
        trend: TrendDirection,
        score: float,
        signals: list[TechnicalSignal],
    ) -> str:
        close = self.close
        support = self._latest.get("support")
        resistance = self._latest.get("resistance")

        bullish = sum(1 for s in signals if s.signal == "bullish")
        bearish = sum(1 for s in signals if s.signal == "bearish")

        price_text = f"at {close:.2f}" if close is not None else "at current levels"
        levels = ""
        if support is not None and resistance is not None:
            levels = f" Support near {support:.2f}, resistance near {resistance:.2f}."

        data_note = ""
        if self.bar_count < MIN_BARS_RECOMMENDED:
            data_note = (
                f" Note: only {self.bar_count} bars available; "
                f"some long-term indicators may be incomplete."
            )

        return (
            f"{symbol} is trading {price_text} with a {trend.value} trend "
            f"(technical score {score:.0f}/100). "
            f"Signal balance: {bullish} bullish vs {bearish} bearish.{levels}{data_note}"
        )

    def summarize_for_llm(self, signals: list[TechnicalSignal]) -> dict[str, Any]:
        """Compact payload for LLM interpretation — no raw OHLCV series."""
        return {
            "bar_count": self.bar_count,
            "latest": self._latest,
            "trend_classification": self.determine_trend().value,
            "signals": [
                {
                    "name": s.name,
                    "value": s.value,
                    "signal": s.signal,
                    "description": s.description,
                }
                for s in signals
            ],
            "breakout": self._indicators.get("price_levels", {}).get("breakout_conditions"),
            "drawdown": self._indicators.get("price_levels", {}).get("drawdown"),
        }

    @staticmethod
    def _last_valid(values: list[float | None]) -> float | None:
        for value in reversed(values):
            if value is not None:
                return value
        return None
