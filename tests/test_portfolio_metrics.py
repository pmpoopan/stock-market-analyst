"""Tests for PortfolioMetricsCalculator — deterministic portfolio math."""

from app.analysis.portfolio_metrics import PortfolioMetricsCalculator
from app.models.schemas import (
    DecisionResult,
    PortfolioHolding,
    Quote,
    Rating,
)
from tests.fixtures.market_data import MOCK_SYMBOL, make_mock_quote


def _decision(score: float = 70.0) -> DecisionResult:
    return DecisionResult(
        stock=MOCK_SYMBOL,
        overall_score=score,
        rating=Rating.BUY,
        fundamental_score=score,
        technical_score=score,
        sentiment_score=score,
        risk_adjustment=0.0,
        key_reasons=["reason"],
        major_risks=["risk"],
    )


def test_analyze_holding_pnl():
    holding = PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)
    quote = make_mock_quote(MOCK_SYMBOL)
    analysis = PortfolioMetricsCalculator.analyze_holding(holding, quote, _decision())

    assert analysis.invested_value == 10000.0
    assert analysis.current_value == round(10 * quote.price, 4)
    assert analysis.pnl == round(analysis.current_value - 10000.0, 4)
    assert analysis.pnl_percent != 0


def test_apply_allocations_splits_by_current_value():
    holding = PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)
    quote = make_mock_quote(MOCK_SYMBOL)
    analyses = [
        PortfolioMetricsCalculator.analyze_holding(holding, quote, _decision(80)),
        PortfolioMetricsCalculator.analyze_holding(
            PortfolioHolding(symbol="INFY.NS", quantity=5, buy_price=1500),
            Quote(symbol="INFY.NS", price=1600.0, currency="INR"),
            _decision(60),
        ),
    ]
    allocated = PortfolioMetricsCalculator.apply_allocations(analyses)

    assert sum(h.allocation_percent for h in allocated) == 100.0
    assert all(h.allocation_percent > 0 for h in allocated)


def test_sector_concentration_and_portfolio_score():
    holding = PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)
    quote = make_mock_quote(MOCK_SYMBOL)
    analyses = PortfolioMetricsCalculator.apply_allocations(
        [PortfolioMetricsCalculator.analyze_holding(holding, quote, _decision(80))]
    )
    sectors = PortfolioMetricsCalculator.sector_concentration(
        analyses, {MOCK_SYMBOL: "Energy"}
    )

    assert sectors["Energy"] == 100.0
    assert PortfolioMetricsCalculator.portfolio_score(analyses) == 80.0


def test_analyze_portfolio_aggregates_totals():
    holding = PortfolioHolding(symbol=MOCK_SYMBOL, quantity=10, buy_price=1000)
    quote = make_mock_quote(MOCK_SYMBOL)
    base = PortfolioMetricsCalculator.analyze_holding(holding, quote, _decision(75))

    result = PortfolioMetricsCalculator.analyze_portfolio(
        [base],
        summary="",
        sector_concentration={"Energy": 100.0},
    )

    assert result.total_invested == base.invested_value
    assert result.total_current_value == base.current_value
    assert result.strongest_holdings == [MOCK_SYMBOL]
    assert result.weakest_holdings == [MOCK_SYMBOL]
    assert "Portfolio of 1 holdings" in result.summary
