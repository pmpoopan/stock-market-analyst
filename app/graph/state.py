"""LangGraph shared state for the Buddy analysis pipeline."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from app.models.schemas import (
    DecisionResult,
    ErrorDetail,
    FundamentalAnalysisResult,
    MasterAnalysisResult,
    ParsedQuery,
    PortfolioAnalysisResult,
    PortfolioHolding,
    SentimentAnalysisResult,
    StockAnalysisResponse,
    StockComparisonResult,
    TechnicalAnalysisResult,
)


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: merge two dicts (used for per-stock analysis maps)."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def append_errors(
    left: list[ErrorDetail] | None,
    right: list[ErrorDetail] | None,
) -> list[ErrorDetail]:
    """Reducer: append error lists."""
    return (left or []) + (right or [])


class StockAnalysisState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Flow:
        query → stocks/portfolio → parallel analyst outputs →
        master → decision → final response
    """

    # Input
    query: str
    parsed_query: ParsedQuery
    stocks: list[str]
    portfolio: list[PortfolioHolding]

    # Raw market data (keyed by symbol)
    market_data: Annotated[dict[str, Any], merge_dicts]

    # Per-stock analyst outputs (keyed by symbol)
    fundamental_analysis: Annotated[dict[str, FundamentalAnalysisResult], merge_dicts]
    technical_analysis: Annotated[dict[str, TechnicalAnalysisResult], merge_dicts]
    sentiment_analysis: Annotated[dict[str, SentimentAnalysisResult], merge_dicts]

    # Synthesis
    master_analysis: Annotated[dict[str, MasterAnalysisResult], merge_dicts]
    decision: Annotated[dict[str, DecisionResult], merge_dicts]

    # Workflow outputs
    stock_response: StockAnalysisResponse | None
    comparison_analysis: StockComparisonResult | None
    portfolio_analysis: PortfolioAnalysisResult | None
    final_analysis: dict[str, Any] | None

    # Diagnostics
    errors: Annotated[list[ErrorDetail], append_errors]
    messages: Annotated[list, add_messages]
