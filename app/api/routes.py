"""FastAPI route handlers."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from app import __version__
from app.api.schemas import (
    AnalyzeRequest,
    CompareRequest,
    HealthResponse,
    PortfolioRequest,
)
from app.config.settings import get_settings
from app.models.schemas import PortfolioAnalysisResult, StockAnalysisResponse, StockComparisonResult
from app.services.container import get_container

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness/readiness probe."""
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, version=__version__)


@router.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(request: AnalyzeRequest) -> StockAnalysisResponse:
    """Analyze a single stock from a natural language query."""
    query = request.query.strip()
    logger.info("Analyze request: query=%r", query[:120])
    start = time.perf_counter()

    container = get_container()
    state = await container.orchestrator.analyze(query)
    duration_ms = (time.perf_counter() - start) * 1000

    response = state.get("stock_response")
    if response is not None:
        logger.info(
            "Analyze complete: symbol=%s score=%.1f (%.1fms)",
            response.symbol,
            response.decision.overall_score,
            duration_ms,
        )
        return response

    errors = state.get("errors", [])
    detail = errors[0].message if errors else "Stock analysis failed"
    logger.warning("Analyze failed after %.1fms: %s", duration_ms, detail)
    raise HTTPException(status_code=400, detail=detail)


@router.post("/compare", response_model=StockComparisonResult)
async def compare_stocks(request: CompareRequest) -> StockComparisonResult:
    """Compare two or more stocks side by side."""
    stocks = [symbol.strip().upper() for symbol in request.stocks]
    logger.info("Compare request: stocks=%s", stocks)
    start = time.perf_counter()

    container = get_container()
    state = await container.orchestrator.compare(stocks)
    duration_ms = (time.perf_counter() - start) * 1000

    result = state.get("comparison_analysis")
    if result is not None:
        logger.info(
            "Compare complete: stocks=%s winner=%s (%.1fms)",
            result.stocks,
            result.winner,
            duration_ms,
        )
        return result

    errors = state.get("errors", [])
    detail = errors[0].message if errors else "Stock comparison failed"
    logger.warning("Compare failed after %.1fms: %s", duration_ms, detail)
    raise HTTPException(status_code=400, detail=detail)


@router.post("/portfolio", response_model=PortfolioAnalysisResult)
async def analyze_portfolio(request: PortfolioRequest) -> PortfolioAnalysisResult:
    """Analyze a portfolio of holdings."""
    symbols = [holding.symbol.upper() for holding in request.holdings]
    logger.info("Portfolio request: holdings=%s", symbols)
    start = time.perf_counter()

    container = get_container()
    state = await container.orchestrator.portfolio(request.holdings)
    duration_ms = (time.perf_counter() - start) * 1000

    result = state.get("portfolio_analysis")
    if result is not None:
        logger.info(
            "Portfolio complete: count=%d score=%.1f (%.1fms)",
            len(result.holdings),
            result.portfolio_score,
            duration_ms,
        )
        return result

    errors = state.get("errors", [])
    detail = errors[0].message if errors else "Portfolio analysis failed"
    logger.warning("Portfolio failed after %.1fms: %s", duration_ms, detail)
    raise HTTPException(status_code=400, detail=detail)
