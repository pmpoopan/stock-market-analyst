"""Comparison Analyst agent — side-by-side stock comparison."""

from __future__ import annotations

import json
import logging

from app.agents.llm_client import LLMClient
from app.analysis.comparison_metrics import ComparisonMetricsCalculator
from app.models.schemas import (
    ComparisonInterpretation,
    DecisionResult,
    FundamentalAnalysisResult,
    SentimentAnalysisResult,
    StockComparisonResult,
    TechnicalAnalysisResult,
)

logger = logging.getLogger(__name__)


class ComparisonAnalyst:
    """Compare multiple stocks using pre-computed analyst outputs."""

    SYSTEM_PROMPT = (
        "You are an equity research analyst comparing Indian stocks side by side. "
        "Use only the structured scores and metrics provided. "
        "Highlight relative strengths in valuation, growth, risk, and technical trends. "
        "Do not invent metrics, prices, or news."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._metrics = ComparisonMetricsCalculator()

    async def compare_from_state(
        self,
        stocks: list[str],
        decisions: dict[str, DecisionResult],
        fundamental_analysis: dict[str, FundamentalAnalysisResult],
        technical_analysis: dict[str, TechnicalAnalysisResult],
        sentiment_analysis: dict[str, SentimentAnalysisResult],
    ) -> StockComparisonResult:
        """Build comparison from LangGraph state after parallel analysts run."""
        symbols = [symbol.upper() for symbol in stocks]
        available = [
            symbol
            for symbol in symbols
            if symbol in decisions
            and symbol in fundamental_analysis
            and symbol in technical_analysis
            and symbol in sentiment_analysis
        ]

        if len(available) < 2:
            raise ValueError(
                "At least two stocks with complete analysis are required for comparison"
            )

        base = self._metrics.build_comparison(
            available,
            {symbol: decisions[symbol] for symbol in available},
            {symbol: fundamental_analysis[symbol] for symbol in available},
            {symbol: technical_analysis[symbol] for symbol in available},
        )

        narratives = await self._interpret_with_llm(
            base,
            {symbol: fundamental_analysis[symbol] for symbol in available},
            {symbol: technical_analysis[symbol] for symbol in available},
            {symbol: sentiment_analysis[symbol] for symbol in available},
        )

        return base.model_copy(update=narratives)

    async def _interpret_with_llm(
        self,
        comparison: StockComparisonResult,
        fundamentals: dict[str, FundamentalAnalysisResult],
        technical: dict[str, TechnicalAnalysisResult],
        sentiment: dict[str, SentimentAnalysisResult],
    ) -> dict[str, str]:
        payload = {
            "stocks": comparison.stocks,
            "fundamental_scores": comparison.fundamental_scores,
            "technical_scores": comparison.technical_scores,
            "sentiment_scores": comparison.sentiment_scores,
            "overall_scores": comparison.overall_scores,
            "winner": comparison.winner,
            "fundamentals": {
                symbol: {
                    "pe_ratio": fundamentals[symbol].metrics.pe_ratio,
                    "pb_ratio": fundamentals[symbol].metrics.pb_ratio,
                    "revenue_growth": fundamentals[symbol].metrics.revenue_growth,
                    "earnings_growth": fundamentals[symbol].metrics.earnings_growth,
                    "debt_to_equity": fundamentals[symbol].metrics.debt_to_equity,
                    "strengths": fundamentals[symbol].strengths[:2],
                    "weaknesses": fundamentals[symbol].weaknesses[:2],
                }
                for symbol in comparison.stocks
            },
            "technical": {
                symbol: {
                    "trend": technical[symbol].trend.value,
                    "score": technical[symbol].score,
                    "momentum": technical[symbol].momentum,
                }
                for symbol in comparison.stocks
            },
            "sentiment": {
                symbol: {
                    "score": sentiment[symbol].sentiment_score,
                    "classification": sentiment[symbol].sentiment_classification.value,
                }
                for symbol in comparison.stocks
            },
            "deterministic": {
                "valuation_comparison": comparison.valuation_comparison,
                "growth_comparison": comparison.growth_comparison,
                "risk_comparison": comparison.risk_comparison,
                "technical_trend_comparison": comparison.technical_trend_comparison,
                "relative_assessment": comparison.relative_assessment,
            },
        }

        prompt = (
            "Interpret the following pre-computed stock comparison for Indian equities. "
            "Return relative narratives for valuation, growth, risk, technical trends, "
            "and an overall relative assessment.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=ComparisonInterpretation,
            )
            if isinstance(interpretation, ComparisonInterpretation):
                return {
                    "valuation_comparison": interpretation.valuation_comparison,
                    "growth_comparison": interpretation.growth_comparison,
                    "risk_comparison": interpretation.risk_comparison,
                    "technical_trend_comparison": interpretation.technical_trend_comparison,
                    "relative_assessment": interpretation.relative_assessment,
                }
        except Exception as exc:
            logger.warning("Comparison LLM interpretation failed: %s", exc)

        return {
            "valuation_comparison": comparison.valuation_comparison,
            "growth_comparison": comparison.growth_comparison,
            "risk_comparison": comparison.risk_comparison,
            "technical_trend_comparison": comparison.technical_trend_comparison,
            "relative_assessment": comparison.relative_assessment,
        }
