"""Injectable dependencies for LangGraph nodes."""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.protocols import (
    ComparisonAnalystProtocol,
    FundamentalAnalystProtocol,
    MasterAnalystProtocol,
    MarketDataProtocol,
    PortfolioAnalystProtocol,
    QueryParserProtocol,
    ScoringEngineProtocol,
    SentimentAnalystProtocol,
    TechnicalAnalystProtocol,
)


@dataclass
class GraphDependencies:
    """Services required by the analysis workflow."""

    query_parser: QueryParserProtocol
    fundamental_analyst: FundamentalAnalystProtocol
    technical_analyst: TechnicalAnalystProtocol
    sentiment_analyst: SentimentAnalystProtocol
    master_analyst: MasterAnalystProtocol
    comparison_analyst: ComparisonAnalystProtocol
    portfolio_analyst: PortfolioAnalystProtocol
    scoring_engine: ScoringEngineProtocol
    market_data: MarketDataProtocol
