"""Tests for QueryParser — deterministic, no LLM."""

import pytest

from app.agents.query_parser import QueryParser
from app.models.schemas import PortfolioHolding, QueryIntent


@pytest.fixture
def parser():
    return QueryParser()


def test_parse_single_stock_reliance(parser):
    parsed = parser.parse("How is Reliance doing?")
    assert parsed.intent == QueryIntent.ANALYZE_STOCK
    assert parsed.stocks == ["RELIANCE.NS"]


def test_parse_compare_tata_mahindra(parser):
    parsed = parser.parse("Compare Tata Motors and Mahindra. Which has the stronger setup?")
    assert parsed.intent == QueryIntent.COMPARE_STOCKS
    assert "TATAMOTORS.NS" in parsed.stocks
    assert "M&M.NS" in parsed.stocks


def test_parse_explicit_symbols(parser):
    parsed = parser.parse("Compare INFY.NS and TCS.NS")
    assert "INFY.NS" in parsed.stocks
    assert "TCS.NS" in parsed.stocks


def test_parse_portfolio_intent(parser):
    parsed = parser.parse("How is my portfolio doing?")
    assert parsed.intent == QueryIntent.ANALYZE_PORTFOLIO


def test_parse_compare_requires_two_symbols(parser):
    with pytest.raises(ValueError, match="at least two"):
        parser.parse("Compare Reliance")


def test_parse_compare_explicit(parser):
    parsed = parser.parse_compare(["TATAMOTORS.NS", "M&M.NS"])
    assert parsed.intent == QueryIntent.COMPARE_STOCKS
    assert len(parsed.stocks) == 2


def test_parse_portfolio_explicit(parser):
    holdings = [
        PortfolioHolding(symbol="TATAMOTORS.NS", quantity=100, buy_price=700),
    ]
    parsed = parser.parse_portfolio(holdings)
    assert parsed.intent == QueryIntent.ANALYZE_PORTFOLIO
    assert parsed.stocks == ["TATAMOTORS.NS"]
