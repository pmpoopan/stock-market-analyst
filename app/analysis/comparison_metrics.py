"""Stock comparison metric calculations — deterministic relative analysis."""

from __future__ import annotations

from app.models.schemas import (
    DecisionResult,
    FundamentalAnalysisResult,
    StockComparisonResult,
    TechnicalAnalysisResult,
    TrendDirection,
)


class ComparisonMetricsCalculator:
    """Build side-by-side score maps and deterministic comparison narratives."""

    @staticmethod
    def score_maps(decisions: dict[str, DecisionResult]) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        fundamental_scores = {
            symbol: decision.fundamental_score for symbol, decision in decisions.items()
        }
        technical_scores = {
            symbol: decision.technical_score for symbol, decision in decisions.items()
        }
        sentiment_scores = {
            symbol: decision.sentiment_score for symbol, decision in decisions.items()
        }
        overall_scores = {
            symbol: decision.overall_score for symbol, decision in decisions.items()
        }
        return fundamental_scores, technical_scores, sentiment_scores, overall_scores

    @staticmethod
    def determine_winner(overall_scores: dict[str, float], margin: float = 2.0) -> str | None:
        if not overall_scores:
            return None

        ranked = sorted(overall_scores.items(), key=lambda item: item[1], reverse=True)
        top_symbol, top_score = ranked[0]
        if len(ranked) == 1:
            return top_symbol

        second_score = ranked[1][1]
        if top_score - second_score < margin:
            return None
        return top_symbol

    @staticmethod
    def compare_valuation(
        fundamentals: dict[str, FundamentalAnalysisResult],
        stocks: list[str],
    ) -> str:
        lines: list[str] = []
        for symbol in stocks:
            metrics = fundamentals.get(symbol)
            if metrics is None:
                continue
            pe = metrics.metrics.pe_ratio
            pb = metrics.metrics.pb_ratio
            pe_text = f"PE {pe:.1f}" if pe is not None else "PE unavailable"
            pb_text = f"PB {pb:.1f}" if pb is not None else "PB unavailable"
            lines.append(f"{symbol}: {pe_text}, {pb_text}")

        if not lines:
            return "Valuation metrics unavailable for the selected stocks."

        pe_values = {
            symbol: fundamentals[symbol].metrics.pe_ratio
            for symbol in stocks
            if symbol in fundamentals and fundamentals[symbol].metrics.pe_ratio is not None
        }
        if len(pe_values) >= 2:
            cheapest = min(pe_values, key=pe_values.get)
            richest = max(pe_values, key=pe_values.get)
            return (
                f"{'; '.join(lines)}. "
                f"{cheapest} trades at a lower PE ({pe_values[cheapest]:.1f}) "
                f"vs {richest} ({pe_values[richest]:.1f})."
            )
        return "; ".join(lines)

    @staticmethod
    def compare_growth(
        fundamentals: dict[str, FundamentalAnalysisResult],
        stocks: list[str],
    ) -> str:
        lines: list[str] = []
        growth_values: dict[str, float] = {}

        for symbol in stocks:
            metrics = fundamentals.get(symbol)
            if metrics is None:
                continue
            rev = metrics.metrics.revenue_growth
            earn = metrics.metrics.earnings_growth
            rev_text = f"revenue growth {rev:.1f}%" if rev is not None else "revenue growth unavailable"
            earn_text = (
                f"earnings growth {earn:.1f}%" if earn is not None else "earnings growth unavailable"
            )
            lines.append(f"{symbol}: {rev_text}, {earn_text}")
            if rev is not None:
                growth_values[symbol] = rev

        if not lines:
            return "Growth metrics unavailable for the selected stocks."

        if len(growth_values) >= 2:
            leader = max(growth_values, key=growth_values.get)
            return (
                f"{'; '.join(lines)}. "
                f"{leader} shows the strongest revenue growth ({growth_values[leader]:.1f}%)."
            )
        return "; ".join(lines)

    @staticmethod
    def compare_risk(
        decisions: dict[str, DecisionResult],
        fundamentals: dict[str, FundamentalAnalysisResult],
        stocks: list[str],
    ) -> str:
        lines: list[str] = []
        for symbol in stocks:
            decision = decisions.get(symbol)
            fundamental = fundamentals.get(symbol)
            if decision is None:
                continue
            risk_count = len(decision.major_risks)
            debt = (
                fundamental.metrics.debt_to_equity
                if fundamental and fundamental.metrics.debt_to_equity is not None
                else None
            )
            debt_text = f"debt/equity {debt:.2f}" if debt is not None else "debt/equity unavailable"
            lines.append(
                f"{symbol}: risk adjustment {decision.risk_adjustment:.1f}, "
                f"{risk_count} flagged risks, {debt_text}"
            )

        if not lines:
            return "Risk metrics unavailable for the selected stocks."

        risk_scores = {
            symbol: decisions[symbol].risk_adjustment
            for symbol in stocks
            if symbol in decisions
        }
        if len(risk_scores) >= 2:
            lowest_risk = min(risk_scores, key=risk_scores.get)
            return (
                f"{'; '.join(lines)}. "
                f"{lowest_risk} has the lowest risk adjustment ({risk_scores[lowest_risk]:.1f})."
            )
        return "; ".join(lines)

    @staticmethod
    def compare_technical_trends(
        technical: dict[str, TechnicalAnalysisResult],
        stocks: list[str],
    ) -> str:
        trend_labels = {
            TrendDirection.STRONG_UPTREND: "strong uptrend",
            TrendDirection.UPTREND: "uptrend",
            TrendDirection.SIDEWAYS: "sideways",
            TrendDirection.DOWNTREND: "downtrend",
            TrendDirection.STRONG_DOWNTREND: "strong downtrend",
        }
        lines: list[str] = []
        trend_scores: dict[str, float] = {}

        for symbol in stocks:
            result = technical.get(symbol)
            if result is None:
                continue
            trend_text = trend_labels.get(result.trend, result.trend.value)
            lines.append(
                f"{symbol}: {trend_text}, technical score {result.score:.0f}/100"
            )
            trend_scores[symbol] = result.score

        if not lines:
            return "Technical trend data unavailable for the selected stocks."

        if len(trend_scores) >= 2:
            leader = max(trend_scores, key=trend_scores.get)
            return (
                f"{'; '.join(lines)}. "
                f"{leader} has the strongest technical setup ({trend_scores[leader]:.0f}/100)."
            )
        return "; ".join(lines)

    @staticmethod
    def build_relative_assessment(
        decisions: dict[str, DecisionResult],
        winner: str | None,
    ) -> str:
        if not decisions:
            return "No stocks available for comparison."

        ranked = sorted(
            decisions.items(),
            key=lambda item: item[1].overall_score,
            reverse=True,
        )
        score_summary = ", ".join(
            f"{symbol} {decision.overall_score:.0f}/100 ({decision.rating.value})"
            for symbol, decision in ranked
        )

        if winner is None:
            return (
                f"Overall scores are closely matched: {score_summary}. "
                "No clear winner on aggregate score."
            )

        winner_decision = decisions[winner]
        return (
            f"{winner} leads the comparison at {winner_decision.overall_score:.0f}/100 "
            f"({winner_decision.rating.value}). Scores: {score_summary}."
        )

    @staticmethod
    def build_comparison(
        stocks: list[str],
        decisions: dict[str, DecisionResult],
        fundamentals: dict[str, FundamentalAnalysisResult],
        technical: dict[str, TechnicalAnalysisResult],
    ) -> StockComparisonResult:
        fundamental_scores, technical_scores, sentiment_scores, overall_scores = (
            ComparisonMetricsCalculator.score_maps(decisions)
        )
        winner = ComparisonMetricsCalculator.determine_winner(overall_scores)

        return StockComparisonResult(
            stocks=stocks,
            fundamental_scores=fundamental_scores,
            technical_scores=technical_scores,
            sentiment_scores=sentiment_scores,
            overall_scores=overall_scores,
            valuation_comparison=ComparisonMetricsCalculator.compare_valuation(
                fundamentals, stocks
            ),
            growth_comparison=ComparisonMetricsCalculator.compare_growth(fundamentals, stocks),
            risk_comparison=ComparisonMetricsCalculator.compare_risk(
                decisions, fundamentals, stocks
            ),
            technical_trend_comparison=ComparisonMetricsCalculator.compare_technical_trends(
                technical, stocks
            ),
            winner=winner,
            relative_assessment=ComparisonMetricsCalculator.build_relative_assessment(
                decisions, winner
            ),
        )
