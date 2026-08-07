"""Portfolio Analyst agent.

Runs per-holding analysis via analyst agents, then aggregates portfolio-level insights.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.fundamental_agent import FundamentalAnalyst
from app.agents.llm_client import LLMClient
from app.agents.master_agent import MasterAnalyst
from app.agents.sentiment_agent import SentimentAnalyst
from app.agents.technical_agent import TechnicalAnalyst
from app.analysis.portfolio_metrics import PortfolioMetricsCalculator
from app.analysis.scoring import ScoringEngine
from app.data.interfaces import MarketDataProvider
from app.models.schemas import (
    DecisionResult,
    FundamentalAnalysisResult,
    PortfolioAnalysisResult,
    PortfolioHolding,
    PortfolioInterpretation,
)

logger = logging.getLogger(__name__)


class PortfolioAnalyst:
    """Analyze a portfolio of holdings."""

    SYSTEM_PROMPT = (
        "You are a portfolio analyst for Indian equity holdings. "
        "Summarize portfolio health, risk concentration, and holding quality. "
        "Base conclusions only on provided structured data. "
        "Do not invent holdings, prices, or scores."
    )

    def __init__(
        self,
        market_data: MarketDataProvider,
        fundamental_analyst: FundamentalAnalyst,
        technical_analyst: TechnicalAnalyst,
        sentiment_analyst: SentimentAnalyst,
        scoring_engine: ScoringEngine,
        llm: LLMClient,
        master_analyst: MasterAnalyst | None = None,
    ) -> None:
        self._market_data = market_data
        self._fundamental = fundamental_analyst
        self._technical = technical_analyst
        self._sentiment = sentiment_analyst
        self._scoring = scoring_engine
        self._llm = llm
        self._master = master_analyst
        self._metrics = PortfolioMetricsCalculator()

    async def analyze(self, holdings: list[PortfolioHolding]) -> PortfolioAnalysisResult:
        """Full pipeline — analyze each holding then aggregate."""
        if not holdings:
            raise ValueError("Portfolio must contain at least one holding")

        holding_analyses: list[Any] = []
        fundamentals: dict[str, FundamentalAnalysisResult] = {}

        for holding in holdings:
            analysis, fundamental = await self._analyze_single_holding(holding)
            holding_analyses.append(analysis)
            fundamentals[holding.symbol.upper()] = fundamental

        return await self._finalize_portfolio(holding_analyses, fundamentals)

    async def analyze_from_state(
        self,
        holdings: list[PortfolioHolding],
        decisions: dict[str, DecisionResult],
        market_data: dict[str, Any],
        fundamental_analysis: dict[str, FundamentalAnalysisResult],
    ) -> PortfolioAnalysisResult:
        """Aggregate portfolio from pre-computed graph state."""
        if not holdings:
            raise ValueError("Portfolio must contain at least one holding")

        holding_analyses = []
        for holding in holdings:
            symbol = holding.symbol.upper()
            decision = decisions.get(symbol)
            quote_data = market_data.get(symbol)
            if decision is None or not quote_data or quote_data.get("price") is None:
                logger.warning("Skipping portfolio holding missing data: %s", symbol)
                continue

            from app.models.schemas import Quote

            quote = Quote(
                symbol=symbol,
                name=quote_data.get("name"),
                price=float(quote_data["price"]),
                currency=quote_data.get("currency", "INR"),
            )
            holding_analyses.append(
                self._metrics.analyze_holding(holding, quote, decision)
            )

        if not holding_analyses:
            raise ValueError("No holdings could be analyzed — missing quotes or decisions")

        return await self._finalize_portfolio(holding_analyses, fundamental_analysis)

    async def _analyze_single_holding(
        self,
        holding: PortfolioHolding,
    ) -> tuple[Any, FundamentalAnalysisResult]:
        symbol = holding.symbol.upper()
        quote = self._market_data.get_quote(symbol)

        fundamental = await self._fundamental.analyze(symbol)
        technical = await self._technical.analyze(symbol)
        sentiment = await self._sentiment.analyze(symbol, company_name=quote.name)

        master = None
        if self._master is not None:
            master = await self._master.synthesize(symbol, fundamental, technical, sentiment)

        decision = self._scoring.compute_decision(
            symbol, fundamental, technical, sentiment, master=master
        )
        holding_analysis = self._metrics.analyze_holding(holding, quote, decision)
        return holding_analysis, fundamental

    async def _finalize_portfolio(
        self,
        holding_analyses: list[Any],
        fundamental_analysis: dict[str, FundamentalAnalysisResult],
    ) -> PortfolioAnalysisResult:
        sector_map = {
            symbol: (fund.metrics.extra.get("sector") if fund.metrics.extra else None)
            for symbol, fund in fundamental_analysis.items()
        }
        sector_concentration = self._metrics.sector_concentration(
            holding_analyses, sector_map
        )

        base = self._metrics.analyze_portfolio(
            holding_analyses,
            summary="",
            sector_concentration=sector_concentration,
        )

        portfolio_risk, summary = await self._interpret_with_llm(base)
        return base.model_copy(update={"portfolio_risk": portfolio_risk, "summary": summary})

    async def _interpret_with_llm(
        self,
        portfolio: PortfolioAnalysisResult,
    ) -> tuple[str, str]:
        payload = {
            "total_invested": portfolio.total_invested,
            "total_current_value": portfolio.total_current_value,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_percent": portfolio.total_pnl_percent,
            "portfolio_score": portfolio.portfolio_score,
            "strongest_holdings": portfolio.strongest_holdings,
            "weakest_holdings": portfolio.weakest_holdings,
            "sector_concentration": portfolio.sector_concentration,
            "holdings": [
                {
                    "symbol": h.holding.symbol,
                    "allocation_percent": h.allocation_percent,
                    "pnl_percent": h.pnl_percent,
                    "overall_score": h.decision.overall_score,
                    "rating": h.decision.rating.value,
                }
                for h in portfolio.holdings
            ],
            "deterministic_risk": portfolio.portfolio_risk,
            "deterministic_summary": portfolio.summary,
        }

        prompt = (
            "Interpret the following pre-computed portfolio metrics for Indian equities. "
            "Return portfolio risk assessment and summary.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=PortfolioInterpretation,
            )
            if isinstance(interpretation, PortfolioInterpretation):
                return interpretation.portfolio_risk, interpretation.summary
        except Exception as exc:
            logger.warning("Portfolio LLM interpretation failed: %s", exc)

        return portfolio.portfolio_risk, portfolio.summary
