"""Fallback synthesis helpers and LLM payload builders for master analyst."""

from __future__ import annotations

from typing import Any

from app.models.schemas import (
    FundamentalAnalysisResult,
    MasterAnalysisResult,
    SentimentAnalysisResult,
    TechnicalAnalysisResult,
)


def build_master_llm_payload(
    symbol: str,
    fundamental: FundamentalAnalysisResult,
    technical: TechnicalAnalysisResult,
    sentiment: SentimentAnalysisResult,
) -> dict[str, Any]:
    """Compact cross-agent payload for LLM synthesis — no raw OHLCV or full indicator series."""
    metrics = fundamental.metrics
    return {
        "symbol": symbol,
        "fundamental": {
            "score": fundamental.score,
            "rating": fundamental.rating.value,
            "summary": fundamental.summary,
            "strengths": fundamental.strengths,
            "weaknesses": fundamental.weaknesses,
            "risks": fundamental.risks,
            "revenue_growth": metrics.revenue_growth,
            "pe_ratio": metrics.pe_ratio,
            "roe": metrics.roe,
            "debt_to_equity": metrics.debt_to_equity,
        },
        "technical": {
            "score": technical.score,
            "trend": technical.trend.value,
            "summary": technical.summary,
            "momentum": technical.momentum,
            "volatility": technical.volatility,
            "support": technical.support,
            "resistance": technical.resistance,
            "signals": [
                {"name": s.name, "signal": s.signal, "description": s.description}
                for s in technical.signals[:8]
            ],
        },
        "sentiment": {
            "sentiment_score": sentiment.sentiment_score,
            "classification": sentiment.sentiment_classification.value,
            "summary": sentiment.summary,
            "positive_catalysts": sentiment.positive_catalysts,
            "negative_catalysts": sentiment.negative_catalysts,
            "key_events": sentiment.key_events,
            "article_count": len(sentiment.articles),
        },
    }


def build_master_analysis_fallback(
    symbol: str,
    fundamental: FundamentalAnalysisResult,
    technical: TechnicalAnalysisResult,
    sentiment: SentimentAnalysisResult,
) -> MasterAnalysisResult:
    """Rule-based synthesis when LLM is unavailable."""
    agreement: list[str] = []
    disagreement: list[str] = []

    if fundamental.score >= 60 and technical.score >= 60:
        agreement.append("Fundamental and technical scores both lean positive")
    if technical.score >= 60 and sentiment.sentiment_score >= 60:
        agreement.append("Technical momentum aligns with positive sentiment")
    if fundamental.score >= 60 and sentiment.sentiment_score >= 60:
        agreement.append("Fundamentals and news sentiment are supportive")

    if fundamental.score >= 65 and technical.score < 45:
        disagreement.append(
            "Fundamentals are relatively strong, but technical momentum has weakened"
        )
    if technical.score >= 65 and fundamental.score < 45:
        disagreement.append(
            "Technical setup is strong despite weaker fundamental scores"
        )
    if sentiment.sentiment_score < 40 and (
        fundamental.score >= 55 or technical.score >= 55
    ):
        disagreement.append(
            "Price/fundamental signals diverge from negative news sentiment"
        )

    if not agreement:
        agreement.append("Analyst perspectives show a mixed setup without strong alignment")

    major_risks = list(dict.fromkeys(fundamental.risks + sentiment.negative_catalysts))[:5]
    catalysts = list(
        dict.fromkeys(sentiment.positive_catalysts + fundamental.strengths)
    )[:5]

    narrative = (
        f"{symbol}: fundamental score {fundamental.score:.0f}, "
        f"technical score {technical.score:.0f}, "
        f"sentiment score {sentiment.sentiment_score:.0f}. "
        f"Trend: {technical.trend.value}."
    )

    return MasterAnalysisResult(
        stock=symbol,
        agreement_points=agreement,
        disagreement_points=disagreement,
        major_risks=major_risks or ["Monitor standard market and company-specific risks"],
        important_catalysts=catalysts,
        narrative=narrative,
        data_vs_interpretation=(
            "Numeric scores and metrics are from structured agent outputs; "
            "narrative interpretation summarizes those results."
        ),
    )
