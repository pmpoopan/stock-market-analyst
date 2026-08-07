"""FastAPI request/response schemas (API boundary models)."""

from pydantic import BaseModel, Field

from app.models.schemas import (
    PortfolioAnalysisResult,
    PortfolioHolding,
    StockAnalysisResponse,
    StockComparisonResult,
)


class AnalyzeRequest(BaseModel):
    query: str = Field(..., examples=["How is Reliance doing?"])


class CompareRequest(BaseModel):
    stocks: list[str] = Field(
        ...,
        min_length=2,
        examples=[["TATAMOTORS.NS", "M&M.NS"]],
    )


class PortfolioRequest(BaseModel):
    holdings: list[PortfolioHolding] = Field(
        ...,
        min_length=1,
        examples=[
            [
                {"symbol": "TATAMOTORS.NS", "quantity": 100, "buy_price": 700},
                {"symbol": "INFY.NS", "quantity": 50, "buy_price": 1500},
            ]
        ],
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    errors: list[str] = Field(default_factory=list)


# Re-export domain response models used by API
__all__ = [
    "AnalyzeRequest",
    "CompareRequest",
    "ErrorResponse",
    "HealthResponse",
    "PortfolioAnalysisResult",
    "PortfolioHolding",
    "PortfolioRequest",
    "StockAnalysisResponse",
    "StockComparisonResult",
]
