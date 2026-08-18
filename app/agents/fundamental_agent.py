"""Fundamental Analyst agent.

Flow:
  1. Data layer fetches raw financials
  2. FundamentalMetricsCalculator computes ratios
  3. FundamentalAnalysisEngine scores in Python
  4. LLM interprets structured metrics (does NOT calculate numbers)
"""

from __future__ import annotations

import json
import logging

from app.agents.llm_client import LLMClient
from app.analysis.fundamental_analysis import FundamentalAnalysisEngine
from app.config.settings import get_settings
from app.data.interfaces import MarketDataProvider
from app.models.schemas import FundamentalAnalysisResult, FundamentalInterpretation

logger = logging.getLogger(__name__)


class FundamentalAnalyst:
    """Analyze financial health of a company."""

    SYSTEM_PROMPT = (
        "You are a fundamental equity analyst for Indian markets. "
        "Interpret the provided structured metrics only. "
        "Do not invent or calculate financial numbers. "
        "Clearly distinguish data from interpretation. "
        "List concrete strengths, weaknesses, and risks based on the metrics provided."
    )

    def __init__(
        self,
        market_data: MarketDataProvider,
        llm: LLMClient,
    ) -> None:
        self._market_data = market_data
        self._llm = llm

    async def analyze(self, symbol: str) -> FundamentalAnalysisResult:
        normalized = symbol.strip().upper()
        metrics = self._market_data.get_financials(normalized)

        analysis_engine = FundamentalAnalysisEngine(metrics)
        score = analysis_engine.compute_score()
        rating = analysis_engine.score_to_rating(score)

        strengths, weaknesses, risks, summary = await self._interpret_with_llm(
            symbol=normalized,
            analysis_engine=analysis_engine,
            score=score,
            rating=rating,
        )

        return FundamentalAnalysisResult(
            stock=normalized,
            score=score,
            rating=rating,
            metrics=metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            summary=summary,
        )

    async def _interpret_with_llm(
        self,
        symbol: str,
        analysis_engine: FundamentalAnalysisEngine,
        score: float,
        rating,
    ) -> tuple[list[str], list[str], list[str], str]:
        llm_payload = analysis_engine.summarize_for_llm(score=score, rating=rating)

        prompt = (
            "Interpret the following pre-computed fundamental metrics for an Indian equity. "
            "Return strengths, weaknesses, risks, and a coherent summary. "
            "Use only the provided numbers.\n\n"
            + json.dumps(llm_payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=FundamentalInterpretation,
                max_tokens=get_settings().llm_max_tokens_fundamental,
            )
            if isinstance(interpretation, FundamentalInterpretation):
                return (
                    interpretation.strengths,
                    interpretation.weaknesses,
                    interpretation.risks,
                    interpretation.summary,
                )
        except Exception as exc:
            logger.warning("LLM interpretation failed for %s: %s", symbol, exc)

        return (
            analysis_engine.build_strengths(),
            analysis_engine.build_weaknesses(),
            analysis_engine.build_risks(),
            analysis_engine.build_summary(symbol, score, rating),
        )
