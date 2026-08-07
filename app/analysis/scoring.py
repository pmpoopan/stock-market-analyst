"""Deterministic decision/scoring engine.

Configurable weights; no LLM involvement in score computation.
Master analysis enriches reasons and risk — it does not change numeric weights.
"""

from __future__ import annotations

from app.config.settings import Settings, get_settings
from app.models.schemas import (
    DecisionResult,
    FundamentalAnalysisResult,
    MasterAnalysisResult,
    Rating,
    SentimentAnalysisResult,
    TechnicalAnalysisResult,
)


class ScoringEngine:
    """Combine agent scores into a final rating using configurable weights."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def weights(self) -> dict[str, float]:
        return {
            "fundamental": self._settings.weight_fundamental,
            "technical": self._settings.weight_technical,
            "sentiment": self._settings.weight_sentiment,
            "risk": self._settings.weight_risk,
        }

    def score_to_rating(self, overall_score: float) -> Rating:
        """Map numeric score to rating classification."""
        if overall_score >= self._settings.rating_strong_buy_min:
            return Rating.STRONG_BUY
        if overall_score >= self._settings.rating_buy_min:
            return Rating.BUY
        if overall_score >= self._settings.rating_hold_min:
            return Rating.HOLD
        return Rating.AVOID

    def compute_risk_adjustment(
        self,
        fundamental: FundamentalAnalysisResult,
        sentiment: SentimentAnalysisResult,
        master: MasterAnalysisResult | None = None,
    ) -> float:
        """Deterministic risk penalty capped at 10 points."""
        adjustment = (
            len(fundamental.risks) * 1.5
            + len(sentiment.negative_catalysts) * 1.0
        )
        if master is not None:
            adjustment += len(master.major_risks) * 0.5
            adjustment += len(master.disagreement_points) * 1.0
        return round(min(10.0, adjustment), 2)

    def build_key_reasons(
        self,
        fundamental: FundamentalAnalysisResult,
        technical: TechnicalAnalysisResult,
        sentiment: SentimentAnalysisResult,
        master: MasterAnalysisResult | None = None,
    ) -> list[str]:
        reasons = [
            f"Fundamental score: {fundamental.score:.0f}/100 ({fundamental.rating.value})",
            f"Technical score: {technical.score:.0f}/100 — {technical.trend.value}",
            f"Sentiment score: {sentiment.sentiment_score:.0f}/100 "
            f"({sentiment.sentiment_classification.value})",
        ]

        if master is not None:
            if master.agreement_points:
                reasons.append(f"Agreement: {master.agreement_points[0]}")
            if master.disagreement_points:
                reasons.append(f"Divergence: {master.disagreement_points[0]}")
            if master.important_catalysts:
                reasons.append(f"Catalyst: {master.important_catalysts[0]}")

        return reasons[:5]

    def build_major_risks(
        self,
        fundamental: FundamentalAnalysisResult,
        sentiment: SentimentAnalysisResult,
        master: MasterAnalysisResult | None = None,
    ) -> list[str]:
        risks = list(dict.fromkeys(fundamental.risks + sentiment.negative_catalysts))
        if master is not None:
            risks = list(dict.fromkeys(risks + master.major_risks))
        return risks[:5] or ["Standard market volatility applies"]

    def compute_decision(
        self,
        symbol: str,
        fundamental: FundamentalAnalysisResult,
        technical: TechnicalAnalysisResult,
        sentiment: SentimentAnalysisResult,
        master: MasterAnalysisResult | None = None,
        risk_adjustment: float | None = None,
    ) -> DecisionResult:
        """Produce deterministic overall score and rating from agent outputs."""
        weights = self.weights

        if risk_adjustment is None:
            risk_adjustment = self.compute_risk_adjustment(fundamental, sentiment, master)

        weighted_score = (
            weights["fundamental"] * fundamental.score
            + weights["technical"] * technical.score
            + weights["sentiment"] * sentiment.sentiment_score
        )
        overall = weighted_score - weights["risk"] * risk_adjustment
        overall = round(max(0.0, min(100.0, overall)), 2)

        rating = self.score_to_rating(overall)

        return DecisionResult(
            stock=symbol,
            overall_score=overall,
            rating=rating,
            fundamental_score=fundamental.score,
            technical_score=technical.score,
            sentiment_score=sentiment.sentiment_score,
            risk_adjustment=risk_adjustment,
            key_reasons=self.build_key_reasons(
                fundamental, technical, sentiment, master
            ),
            major_risks=self.build_major_risks(fundamental, sentiment, master),
        )

    def compare_scores(self, decisions: dict[str, DecisionResult]) -> str:
        """Generate relative assessment text for stock comparison."""
        from app.analysis.comparison_metrics import ComparisonMetricsCalculator

        overall_scores = {
            symbol: decision.overall_score for symbol, decision in decisions.items()
        }
        winner = ComparisonMetricsCalculator.determine_winner(overall_scores)
        return ComparisonMetricsCalculator.build_relative_assessment(decisions, winner)
