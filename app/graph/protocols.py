"""Protocols for services injected into the LangGraph workflow.

The graph layer depends on these interfaces — not on concrete agent classes.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.models.schemas import (
    DecisionResult,
    FundamentalAnalysisResult,
    MasterAnalysisResult,
    ParsedQuery,
    PortfolioAnalysisResult,
    PortfolioHolding,
    Quote,
    SentimentAnalysisResult,
    StockComparisonResult,
    TechnicalAnalysisResult,
)


class QueryParserProtocol(Protocol):
    def parse(self, query: str) -> ParsedQuery: ...

    def parse_compare(self, stocks: list[str]) -> ParsedQuery: ...

    def parse_portfolio(self, holdings: list[PortfolioHolding]) -> ParsedQuery: ...


class MarketDataProtocol(Protocol):
    def get_quote(self, symbol: str) -> Quote: ...


class FundamentalAnalystProtocol(Protocol):
    async def analyze(self, symbol: str) -> FundamentalAnalysisResult: ...


class TechnicalAnalystProtocol(Protocol):
    async def analyze(self, symbol: str) -> TechnicalAnalysisResult: ...


class SentimentAnalystProtocol(Protocol):
    async def analyze(
        self,
        symbol: str,
        company_name: str | None = None,
    ) -> SentimentAnalysisResult: ...


class MasterAnalystProtocol(Protocol):
    async def synthesize(
        self,
        symbol: str,
        fundamental: FundamentalAnalysisResult,
        technical: TechnicalAnalysisResult,
        sentiment: SentimentAnalysisResult,
    ) -> MasterAnalysisResult: ...


class ComparisonAnalystProtocol(Protocol):
    async def compare_from_state(
        self,
        stocks: list[str],
        decisions: dict[str, DecisionResult],
        fundamental_analysis: dict[str, FundamentalAnalysisResult],
        technical_analysis: dict[str, TechnicalAnalysisResult],
        sentiment_analysis: dict[str, SentimentAnalysisResult],
    ) -> StockComparisonResult: ...


class PortfolioAnalystProtocol(Protocol):
    async def analyze_from_state(
        self,
        holdings: list[PortfolioHolding],
        decisions: dict[str, DecisionResult],
        market_data: dict[str, Any],
        fundamental_analysis: dict[str, FundamentalAnalysisResult],
    ) -> PortfolioAnalysisResult: ...


class ScoringEngineProtocol(Protocol):
    def compute_decision(
        self,
        symbol: str,
        fundamental: FundamentalAnalysisResult,
        technical: TechnicalAnalysisResult,
        sentiment: SentimentAnalysisResult,
        master: MasterAnalysisResult | None = None,
        risk_adjustment: float | None = None,
    ) -> DecisionResult: ...
