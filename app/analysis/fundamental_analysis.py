"""Deterministic fundamental scoring and narrative fallbacks.

Scores and rating are computed in Python; LLM only enriches qualitative text.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings, get_settings
from app.models.schemas import FinancialMetrics, Rating


class FundamentalAnalysisEngine:
    """Score and classify fundamentals from pre-computed FinancialMetrics."""

    def __init__(
        self,
        metrics: FinancialMetrics,
        settings: Settings | None = None,
    ) -> None:
        self._metrics = metrics
        self._settings = settings or get_settings()

    def compute_score(self) -> float:
        scores: list[tuple[float, float]] = []

        growth = self._growth_score()
        if growth is not None:
            scores.append((growth, 0.25))

        profitability = self._profitability_score()
        if profitability is not None:
            scores.append((profitability, 0.30))

        valuation = self._valuation_score()
        if valuation is not None:
            scores.append((valuation, 0.25))

        health = self._financial_health_score()
        if health is not None:
            scores.append((health, 0.20))

        if not scores:
            return 50.0

        total_weight = sum(weight for _, weight in scores)
        weighted = sum(score * weight for score, weight in scores)
        return round(max(0.0, min(100.0, weighted / total_weight)), 2)

    def score_to_rating(self, score: float) -> Rating:
        if score >= self._settings.rating_strong_buy_min:
            return Rating.STRONG_BUY
        if score >= self._settings.rating_buy_min:
            return Rating.BUY
        if score >= self._settings.rating_hold_min:
            return Rating.HOLD
        return Rating.AVOID

    def build_strengths(self) -> list[str]:
        strengths: list[str] = []
        m = self._metrics

        if m.revenue_growth is not None and m.revenue_growth >= 10:
            strengths.append(f"Revenue growth of {m.revenue_growth:.1f}% indicates expansion")
        if m.earnings_growth is not None and m.earnings_growth >= 10:
            strengths.append(f"Earnings growth of {m.earnings_growth:.1f}% supports profitability trend")
        if m.roe is not None and m.roe >= 15:
            strengths.append(f"ROE of {m.roe:.1f}% reflects strong return on equity")
        if m.roce is not None and m.roce >= 12:
            strengths.append(f"ROCE of {m.roce:.1f}% shows efficient capital use")
        if m.pe_ratio is not None and 5 < m.pe_ratio < 25:
            strengths.append(f"P/E of {m.pe_ratio:.1f} suggests reasonable valuation")
        if m.debt_to_equity is not None and m.debt_to_equity < 0.5:
            strengths.append(f"Low debt-to-equity ({m.debt_to_equity:.2f}) limits leverage risk")
        if m.free_cash_flow is not None and m.free_cash_flow > 0:
            strengths.append("Positive free cash flow supports financial flexibility")
        if m.dividend_yield is not None and m.dividend_yield >= 1.0:
            strengths.append(f"Dividend yield of {m.dividend_yield:.2f}% offers income component")

        return strengths[:5]

    def build_weaknesses(self) -> list[str]:
        weaknesses: list[str] = []
        m = self._metrics

        if m.revenue_growth is not None and m.revenue_growth < 0:
            weaknesses.append(f"Revenue declined {abs(m.revenue_growth):.1f}% year-over-year")
        if m.earnings_growth is not None and m.earnings_growth < 0:
            weaknesses.append(f"Earnings growth is negative at {m.earnings_growth:.1f}%")
        if m.roe is not None and m.roe < 8:
            weaknesses.append(f"ROE of {m.roe:.1f}% is below typical quality thresholds")
        if m.roce is not None and m.roce < 8:
            weaknesses.append(f"ROCE of {m.roce:.1f}% indicates modest capital efficiency")
        if m.pe_ratio is not None and m.pe_ratio > 35:
            weaknesses.append(f"Elevated P/E of {m.pe_ratio:.1f} may imply rich valuation")
        if m.debt_to_equity is not None and m.debt_to_equity > 1.0:
            weaknesses.append(f"High debt-to-equity ({m.debt_to_equity:.2f}) increases leverage risk")
        if m.free_cash_flow is not None and m.free_cash_flow < 0:
            weaknesses.append("Negative free cash flow pressures liquidity and reinvestment capacity")

        return weaknesses[:5]

    def build_risks(self) -> list[str]:
        risks: list[str] = []
        m = self._metrics

        if m.debt_to_equity is not None and m.debt_to_equity > 0.8:
            risks.append("Elevated leverage could amplify downside in a rising rate environment")
        if m.revenue_growth is not None and m.revenue_growth < 0:
            risks.append("Contracting revenue may signal demand or competitive pressure")
        if m.pe_ratio is not None and m.pe_ratio < 0:
            risks.append("Negative earnings create uncertainty around valuation and sustainability")
        if m.free_cash_flow is not None and m.free_cash_flow < 0:
            risks.append("Cash burn may require external funding or balance sheet stress")
        if m.roe is not None and m.roe < 5:
            risks.append("Weak ROE may indicate capital allocation or margin challenges")

        if not risks:
            risks.append("Standard market and sector risks apply; monitor quarterly results")

        return risks[:5]

    def build_summary(self, symbol: str, score: float, rating: Rating) -> str:
        m = self._metrics
        parts = [
            f"{symbol} fundamental score is {score:.0f}/100 ({rating.value}).",
        ]
        if m.revenue is not None:
            parts.append(f"Revenue: {self._format_large_number(m.revenue)}.")
        if m.revenue_growth is not None:
            parts.append(f"Revenue growth: {m.revenue_growth:.1f}%.")
        if m.pe_ratio is not None:
            parts.append(f"P/E: {m.pe_ratio:.1f}.")
        if m.roe is not None:
            parts.append(f"ROE: {m.roe:.1f}%.")
        return " ".join(parts)

    def summarize_for_llm(self, score: float, rating: Rating) -> dict[str, Any]:
        return {
            "symbol": self._metrics.symbol,
            "score": score,
            "rating": rating.value,
            "metrics": self._metrics.model_dump(),
            "deterministic_strengths": self.build_strengths(),
            "deterministic_weaknesses": self.build_weaknesses(),
            "deterministic_risks": self.build_risks(),
        }

    def _growth_score(self) -> float | None:
        m = self._metrics
        components: list[float] = []

        if m.revenue_growth is not None:
            components.append(self._map_growth(m.revenue_growth))
        if m.earnings_growth is not None:
            components.append(self._map_growth(m.earnings_growth))

        if not components:
            return None
        return sum(components) / len(components)

    def _profitability_score(self) -> float | None:
        m = self._metrics
        components: list[float] = []

        if m.roe is not None:
            components.append(self._map_ratio_high_good(m.roe, good=20, weak=8))
        if m.roce is not None:
            components.append(self._map_ratio_high_good(m.roce, good=18, weak=8))
        if m.net_profit is not None and m.revenue is not None and m.revenue > 0:
            margin = (m.net_profit / m.revenue) * 100
            components.append(self._map_ratio_high_good(margin, good=15, weak=5))

        if not components:
            return None
        return sum(components) / len(components)

    def _valuation_score(self) -> float | None:
        m = self._metrics
        components: list[float] = []

        if m.pe_ratio is not None:
            if m.pe_ratio <= 0:
                components.append(20.0)
            elif m.pe_ratio < 12:
                components.append(85.0)
            elif m.pe_ratio < 20:
                components.append(75.0)
            elif m.pe_ratio < 30:
                components.append(55.0)
            elif m.pe_ratio < 45:
                components.append(40.0)
            else:
                components.append(25.0)

        if m.pb_ratio is not None:
            if m.pb_ratio < 1:
                components.append(80.0)
            elif m.pb_ratio < 3:
                components.append(65.0)
            elif m.pb_ratio < 6:
                components.append(50.0)
            else:
                components.append(35.0)

        if not components:
            return None
        return sum(components) / len(components)

    def _financial_health_score(self) -> float | None:
        m = self._metrics
        components: list[float] = []

        if m.debt_to_equity is not None:
            if m.debt_to_equity < 0.3:
                components.append(90.0)
            elif m.debt_to_equity < 0.7:
                components.append(70.0)
            elif m.debt_to_equity < 1.2:
                components.append(50.0)
            else:
                components.append(30.0)

        if m.free_cash_flow is not None:
            components.append(75.0 if m.free_cash_flow > 0 else 35.0)

        if not components:
            return None
        return sum(components) / len(components)

    @staticmethod
    def _map_growth(growth_pct: float) -> float:
        if growth_pct >= 20:
            return 90.0
        if growth_pct >= 10:
            return 75.0
        if growth_pct >= 5:
            return 65.0
        if growth_pct >= 0:
            return 50.0
        if growth_pct >= -5:
            return 40.0
        return 25.0

    @staticmethod
    def _map_ratio_high_good(value: float, good: float, weak: float) -> float:
        if value >= good:
            return 90.0
        if value >= (good + weak) / 2:
            return 70.0
        if value >= weak:
            return 55.0
        return 35.0

    @staticmethod
    def _format_large_number(value: float) -> str:
        if abs(value) >= 1e12:
            return f"{value / 1e12:.2f}T"
        if abs(value) >= 1e9:
            return f"{value / 1e9:.2f}B"
        if abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M"
        return f"{value:,.0f}"
