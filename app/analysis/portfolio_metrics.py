"""Portfolio-level metric calculations."""

from __future__ import annotations

from app.models.schemas import (
    DecisionResult,
    FundamentalAnalysisResult,
    HoldingAnalysis,
    PortfolioAnalysisResult,
    PortfolioHolding,
    Quote,
)


class PortfolioMetricsCalculator:
    """Compute invested value, P&L, allocation, and portfolio-level scores."""

    @staticmethod
    def analyze_holding(
        holding: PortfolioHolding,
        quote: Quote,
        decision: DecisionResult,
    ) -> HoldingAnalysis:
        invested_value = round(holding.quantity * holding.buy_price, 4)
        current_value = round(holding.quantity * quote.price, 4)
        pnl = round(current_value - invested_value, 4)
        pnl_percent = round((pnl / invested_value) * 100, 4) if invested_value else 0.0

        return HoldingAnalysis(
            holding=holding,
            current_price=round(quote.price, 4),
            invested_value=invested_value,
            current_value=current_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            allocation_percent=0.0,
            decision=decision,
        )

    @staticmethod
    def apply_allocations(holdings: list[HoldingAnalysis]) -> list[HoldingAnalysis]:
        total_current = sum(h.current_value for h in holdings)
        if total_current <= 0:
            return holdings

        return [
            h.model_copy(
                update={
                    "allocation_percent": round((h.current_value / total_current) * 100, 2)
                }
            )
            for h in holdings
        ]

    @staticmethod
    def sector_concentration(
        holdings: list[HoldingAnalysis],
        sector_by_symbol: dict[str, str | None],
    ) -> dict[str, float]:
        total_current = sum(h.current_value for h in holdings)
        if total_current <= 0:
            return {}

        sector_values: dict[str, float] = {}
        for holding in holdings:
            symbol = holding.holding.symbol.upper()
            sector = sector_by_symbol.get(symbol) or "Unknown"
            sector_values[sector] = sector_values.get(sector, 0.0) + holding.current_value

        return {
            sector: round((value / total_current) * 100, 2)
            for sector, value in sorted(sector_values.items(), key=lambda x: -x[1])
        }

    @staticmethod
    def strongest_weakest(
        holdings: list[HoldingAnalysis],
        count: int = 3,
    ) -> tuple[list[str], list[str]]:
        if not holdings:
            return [], []

        ranked = sorted(
            holdings,
            key=lambda h: h.decision.overall_score,
            reverse=True,
        )
        strongest = [h.holding.symbol.upper() for h in ranked[:count]]
        weakest = [h.holding.symbol.upper() for h in ranked[-count:][::-1]]
        return strongest, weakest

    @staticmethod
    def portfolio_score(holdings: list[HoldingAnalysis]) -> float:
        if not holdings:
            return 50.0

        total_allocation = sum(h.allocation_percent for h in holdings)
        if total_allocation <= 0:
            scores = [h.decision.overall_score for h in holdings]
            return round(sum(scores) / len(scores), 2)

        weighted = sum(
            h.decision.overall_score * (h.allocation_percent / 100) for h in holdings
        )
        return round(max(0.0, min(100.0, weighted)), 2)

    @staticmethod
    def assess_portfolio_risk(
        holdings: list[HoldingAnalysis],
        sector_concentration: dict[str, float],
    ) -> str:
        if not holdings:
            return "No holdings to assess"

        max_sector = max(sector_concentration.values()) if sector_concentration else 0.0
        weak_count = sum(1 for h in holdings if h.decision.overall_score < 45)

        if max_sector > 50:
            top_sector = max(sector_concentration, key=sector_concentration.get)
            return (
                f"Elevated sector concentration — {max_sector:.0f}% in {top_sector}. "
                "Diversification may be limited."
            )
        if weak_count >= max(1, len(holdings) // 2):
            return (
                f"{weak_count} of {len(holdings)} holdings score below 45, "
                "weighing on overall portfolio quality."
            )
        if max_sector > 35:
            return "Moderate sector concentration; monitor overlap across holdings."
        return "Portfolio risk appears broadly diversified across holdings and sectors."

    @staticmethod
    def build_summary(
        holdings: list[HoldingAnalysis],
        total_pnl_percent: float,
        portfolio_score: float,
        strongest: list[str],
        weakest: list[str],
    ) -> str:
        if not holdings:
            return "Portfolio is empty."

        return (
            f"Portfolio of {len(holdings)} holdings: "
            f"P&L {total_pnl_percent:+.1f}%, score {portfolio_score:.0f}/100. "
            f"Strongest: {', '.join(strongest[:2])}. "
            f"Weakest: {', '.join(weakest[:2])}."
        )

    @staticmethod
    def analyze_portfolio(
        holdings: list[HoldingAnalysis],
        summary: str,
        sector_concentration: dict[str, float] | None = None,
        portfolio_risk: str | None = None,
    ) -> PortfolioAnalysisResult:
        holdings = PortfolioMetricsCalculator.apply_allocations(holdings)
        sector_concentration = sector_concentration or {}

        total_invested = round(sum(h.invested_value for h in holdings), 4)
        total_current_value = round(sum(h.current_value for h in holdings), 4)
        total_pnl = round(total_current_value - total_invested, 4)
        total_pnl_percent = (
            round((total_pnl / total_invested) * 100, 4) if total_invested else 0.0
        )

        portfolio_score = PortfolioMetricsCalculator.portfolio_score(holdings)
        strongest, weakest = PortfolioMetricsCalculator.strongest_weakest(holdings)
        risk = portfolio_risk or PortfolioMetricsCalculator.assess_portfolio_risk(
            holdings, sector_concentration
        )

        if not summary:
            summary = PortfolioMetricsCalculator.build_summary(
                holdings, total_pnl_percent, portfolio_score, strongest, weakest
            )

        return PortfolioAnalysisResult(
            holdings=holdings,
            total_invested=total_invested,
            total_current_value=total_current_value,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            portfolio_score=portfolio_score,
            strongest_holdings=strongest,
            weakest_holdings=weakest,
            sector_concentration=sector_concentration,
            portfolio_risk=risk,
            summary=summary,
        )
