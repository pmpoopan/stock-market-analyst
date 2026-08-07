"""Technical Analyst agent.

Flow:
  1. Data layer fetches OHLCV history
  2. TechnicalIndicatorEngine computes all indicators in Python
  3. TechnicalAnalysisEngine builds signals and score in Python
  4. LLM interprets structured indicator output (does NOT calculate indicators)
"""

from __future__ import annotations

import json
import logging

from app.agents.llm_client import LLMClient
from app.analysis.technical_analysis import MIN_BARS_RECOMMENDED, TechnicalAnalysisEngine
from app.analysis.technical_indicators import TechnicalIndicatorEngine
from app.data.exceptions import DataNotFoundError
from app.data.interfaces import MarketDataProvider
from app.models.schemas import TechnicalAnalysisResult, TechnicalInterpretation

logger = logging.getLogger(__name__)

MIN_BARS_REQUIRED = 30


class TechnicalAnalyst:
    """Analyze price action, trends, and technical signals."""

    SYSTEM_PROMPT = (
        "You are a technical equity analyst for Indian markets. "
        "Interpret the provided pre-computed indicators and signals only. "
        "Do not calculate technical indicators yourself. "
        "Assess trend, momentum, volatility, and key levels. "
        "Clearly distinguish data from interpretation. "
        "Do not invent missing indicator values."
    )

    def __init__(
        self,
        market_data: MarketDataProvider,
        llm: LLMClient,
    ) -> None:
        self._market_data = market_data
        self._llm = llm

    async def analyze(self, symbol: str, period: str = "1y") -> TechnicalAnalysisResult:
        normalized = symbol.strip().upper()
        historical = self._market_data.get_historical_data(normalized, period=period)

        if len(historical.bars) < MIN_BARS_REQUIRED:
            raise DataNotFoundError(
                f"Insufficient historical data for {normalized}: "
                f"need at least {MIN_BARS_REQUIRED} bars, got {len(historical.bars)}"
            )

        indicator_engine = TechnicalIndicatorEngine(historical.bars)
        indicators = indicator_engine.compute_all()
        analysis_engine = TechnicalAnalysisEngine(indicators)

        signals = analysis_engine.build_signals()
        trend = analysis_engine.determine_trend()
        score = analysis_engine.compute_score(signals)
        support = analysis_engine.support
        resistance = analysis_engine.resistance

        momentum, volatility, summary = await self._interpret_with_llm(
            symbol=normalized,
            analysis_engine=analysis_engine,
            signals=signals,
            trend=trend,
            score=score,
        )

        # Slim indicators for API response — latest values only, not full series
        response_indicators = {
            "bar_count": indicators["bar_count"],
            "latest": indicators["latest"],
            "breakout": indicators["price_levels"]["breakout_conditions"],
            "drawdown": indicators["price_levels"]["drawdown"],
            "data_quality": (
                "full"
                if indicators["bar_count"] >= MIN_BARS_RECOMMENDED
                else "partial_long_term"
            ),
        }

        return TechnicalAnalysisResult(
            stock=normalized,
            score=score,
            trend=trend,
            signals=signals,
            support=support,
            resistance=resistance,
            momentum=momentum,
            volatility=volatility,
            summary=summary,
            indicators=response_indicators,
        )

    async def _interpret_with_llm(
        self,
        symbol: str,
        analysis_engine: TechnicalAnalysisEngine,
        signals: list,
        trend,
        score: float,
    ) -> tuple[str, str, str]:
        """LLM interpretation with deterministic fallback."""
        llm_payload = analysis_engine.summarize_for_llm(signals)
        llm_payload["symbol"] = symbol
        llm_payload["score"] = score

        prompt = (
            "Interpret the following pre-computed technical analysis for an Indian equity. "
            "Return momentum assessment, volatility assessment, and a coherent summary.\n\n"
            + json.dumps(llm_payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=TechnicalInterpretation,
            )
            if isinstance(interpretation, TechnicalInterpretation):
                return interpretation.momentum, interpretation.volatility, interpretation.summary
        except Exception as exc:
            logger.warning("LLM interpretation failed for %s: %s", symbol, exc)

        return (
            analysis_engine.momentum_label(),
            analysis_engine.volatility_label(),
            analysis_engine.build_summary(symbol, trend, score, signals),
        )
