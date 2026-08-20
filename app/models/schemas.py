"""Core Pydantic domain models shared across agents and API layers."""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Rating(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    AVOID = "Avoid"


class SentimentClassification(str, Enum):
    VERY_POSITIVE = "Very Positive"
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"


class TrendDirection(str, Enum):
    STRONG_UPTREND = "Strong Uptrend"
    UPTREND = "Uptrend"
    SIDEWAYS = "Sideways"
    DOWNTREND = "Downtrend"
    STRONG_DOWNTREND = "Strong Downtrend"


class QueryIntent(str, Enum):
    ANALYZE_STOCK = "analyze_stock"
    COMPARE_STOCKS = "compare_stocks"
    ANALYZE_PORTFOLIO = "analyze_portfolio"


# ---------------------------------------------------------------------------
# Market & financial data
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    symbol: str
    name: str | None = None
    price: float
    currency: str = "INR"
    change: float | None = None
    change_percent: float | None = None
    market_cap: float | None = None
    timestamp: datetime | None = None


class OHLCVBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalData(BaseModel):
    symbol: str
    period: str
    bars: list[OHLCVBar]


class FinancialMetrics(BaseModel):
    """Raw and derived fundamental metrics computed by the data/analysis layer."""

    symbol: str
    revenue: float | None = None
    revenue_growth: float | None = None
    ebitda: float | None = None
    ebit: float | None = None
    net_profit: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    roe: float | None = None
    roce: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    earnings_growth: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class NewsArticle(BaseModel):
    title: str
    source: str
    url: HttpUrl
    published_at: date | None = None
    snippet: str | None = None


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------


class FundamentalAnalysisResult(BaseModel):
    stock: str
    score: float = Field(ge=0, le=100)
    rating: Rating
    metrics: FinancialMetrics
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]
    summary: str


class FundamentalInterpretation(BaseModel):
    """LLM output for fundamental narrative — metrics are pre-computed in Python."""

    strengths: list[str] = Field(..., description="Key fundamental strengths")
    weaknesses: list[str] = Field(..., description="Key fundamental weaknesses")
    risks: list[str] = Field(..., description="Major fundamental risks")
    summary: str = Field(..., description="Coherent fundamental summary")


class TechnicalSignal(BaseModel):
    name: str
    value: float | str | None = None
    signal: str  # e.g. "bullish", "bearish", "neutral"
    description: str | None = None


class TechnicalAnalysisResult(BaseModel):
    stock: str
    score: float = Field(ge=0, le=100)
    trend: TrendDirection
    signals: list[TechnicalSignal]
    support: float | None = None
    resistance: float | None = None
    momentum: str
    volatility: str
    summary: str
    indicators: dict[str, Any] = Field(default_factory=dict)


class TechnicalInterpretation(BaseModel):
    """LLM output for technical narrative — indicators are pre-computed in Python."""

    momentum: str = Field(..., description="Momentum assessment narrative")
    volatility: str = Field(..., description="Volatility assessment narrative")
    summary: str = Field(..., description="Coherent technical summary")


class SentimentAnalysisResult(BaseModel):
    stock: str
    sentiment_score: float = Field(ge=0, le=100)
    sentiment_classification: SentimentClassification
    positive_catalysts: list[str]
    negative_catalysts: list[str]
    key_events: list[str]
    sources: list[HttpUrl]
    publication_dates: list[date | None]
    articles: list[NewsArticle]
    summary: str


class SentimentInterpretation(BaseModel):
    """LLM output for sentiment narrative — score is pre-computed in Python."""

    positive_catalysts: list[str] = Field(..., description="Positive catalysts from articles")
    negative_catalysts: list[str] = Field(..., description="Negative catalysts from articles")
    key_events: list[str] = Field(..., description="Key recent events")
    summary: str = Field(..., description="Coherent sentiment summary")


class MasterAnalysisResult(BaseModel):
    stock: str
    agreement_points: list[str]
    disagreement_points: list[str]
    major_risks: list[str]
    important_catalysts: list[str]
    narrative: str
    data_vs_interpretation: str


class MasterInterpretation(BaseModel):
    """LLM output for master synthesis — scores come from analyst agents."""

    agreement_points: list[str] = Field(..., description="Max 2 short agreement points")
    disagreement_points: list[str] = Field(..., description="Max 2 short disagreement points")
    major_risks: list[str] = Field(..., description="Max 2 short risks")
    important_catalysts: list[str] = Field(..., description="Max 2 short catalysts")
    narrative: str = Field(..., description="1-2 short sentences of cross-perspective narrative")
    data_vs_interpretation: str = Field(
        ...,
        description="One short sentence distinguishing data vs interpretation",
    )


class DecisionResult(BaseModel):
    stock: str
    overall_score: float = Field(ge=0, le=100)
    rating: Rating
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    risk_adjustment: float = 0.0
    key_reasons: list[str]
    major_risks: list[str]


# ---------------------------------------------------------------------------
# Comparison & portfolio
# ---------------------------------------------------------------------------


class StockComparisonResult(BaseModel):
    stocks: list[str]
    fundamental_scores: dict[str, float]
    technical_scores: dict[str, float]
    sentiment_scores: dict[str, float]
    overall_scores: dict[str, float]
    valuation_comparison: str
    growth_comparison: str
    risk_comparison: str
    technical_trend_comparison: str
    winner: str | None = None
    relative_assessment: str


class ComparisonInterpretation(BaseModel):
    """LLM output for stock comparison — scores are pre-computed in Python."""

    valuation_comparison: str = Field(..., description="Relative valuation narrative")
    growth_comparison: str = Field(..., description="Relative growth narrative")
    risk_comparison: str = Field(..., description="Relative risk narrative")
    technical_trend_comparison: str = Field(..., description="Relative technical trend narrative")
    relative_assessment: str = Field(..., description="Overall relative assessment and winner context")


class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float
    buy_price: float


class HoldingAnalysis(BaseModel):
    holding: PortfolioHolding
    current_price: float
    invested_value: float
    current_value: float
    pnl: float
    pnl_percent: float
    allocation_percent: float
    decision: DecisionResult


class PortfolioAnalysisResult(BaseModel):
    holdings: list[HoldingAnalysis]
    total_invested: float
    total_current_value: float
    total_pnl: float
    total_pnl_percent: float
    portfolio_score: float = Field(ge=0, le=100)
    strongest_holdings: list[str]
    weakest_holdings: list[str]
    sector_concentration: dict[str, float] = Field(default_factory=dict)
    portfolio_risk: str
    summary: str


class PortfolioInterpretation(BaseModel):
    """LLM output for portfolio narrative — metrics are pre-computed in Python."""

    portfolio_risk: str = Field(..., description="Portfolio-level risk assessment")
    summary: str = Field(..., description="Coherent portfolio summary")


# ---------------------------------------------------------------------------
# Query parsing & final response
# ---------------------------------------------------------------------------


class ParsedQuery(BaseModel):
    raw_query: str
    intent: QueryIntent
    stocks: list[str] = Field(default_factory=list)
    portfolio: list[PortfolioHolding] = Field(default_factory=list)


class StockAnalysisResponse(BaseModel):
    symbol: str
    name: str | None = None
    current_price: float | None = None
    decision: DecisionResult
    fundamental: FundamentalAnalysisResult
    technical: TechnicalAnalysisResult
    sentiment: SentimentAnalysisResult
    master: MasterAnalysisResult
    sources: list[HttpUrl] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    component: str
    message: str
    recoverable: bool = True
