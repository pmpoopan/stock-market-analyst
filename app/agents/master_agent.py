"""Master Analyst agent.

Synthesizes fundamental, technical, and sentiment outputs.
Does NOT blindly average scores — explains agreement and conflicts.
"""

from __future__ import annotations

import json
import logging

from app.agents.llm_client import LLMClient
from app.analysis.master_synthesis import build_master_analysis_fallback, build_master_llm_payload
from app.config.settings import get_settings
from app.models.schemas import (
    FundamentalAnalysisResult,
    MasterAnalysisResult,
    MasterInterpretation,
    SentimentAnalysisResult,
    TechnicalAnalysisResult,
)

logger = logging.getLogger(__name__)


class MasterAnalyst:
    """Compare three analyst perspectives and produce coherent narrative."""

    SYSTEM_PROMPT = (
        "You are a senior equity research analyst synthesizing Indian equity perspectives. "
        "Compare fundamental, technical, and sentiment outputs only. "
        "Never invent data, metrics, or news. "
        "Return compact JSON: max 2 short items per list; narrative max 3 sentences."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        symbol: str,
        fundamental: FundamentalAnalysisResult,
        technical: TechnicalAnalysisResult,
        sentiment: SentimentAnalysisResult,
    ) -> MasterAnalysisResult:
        payload = build_master_llm_payload(symbol, fundamental, technical, sentiment)

        prompt = (
            "Synthesize the analyst outputs below into agreement, disagreement, "
            "risks, catalysts, narrative, and data-vs-interpretation fields. "
            "Use only provided data.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=MasterInterpretation,
                max_tokens=get_settings().llm_max_tokens_master,
            )
            if isinstance(interpretation, MasterInterpretation):
                return MasterAnalysisResult(
                    stock=symbol,
                    agreement_points=interpretation.agreement_points,
                    disagreement_points=interpretation.disagreement_points,
                    major_risks=interpretation.major_risks,
                    important_catalysts=interpretation.important_catalysts,
                    narrative=interpretation.narrative,
                    data_vs_interpretation=interpretation.data_vs_interpretation,
                )
        except Exception as exc:
            logger.warning("Master analyst LLM synthesis failed for %s: %s", symbol, exc)

        return build_master_analysis_fallback(symbol, fundamental, technical, sentiment)
