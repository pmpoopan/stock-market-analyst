"""Tests for Streamlit UI formatting helpers."""

import pytest

from frontend.ui_helpers import (
    comparison_score_rows,
    format_inr,
    format_percent,
    format_score,
    normalize_holdings,
    portfolio_holdings_rows,
    rating_label,
)


def test_format_inr_and_percent():
    assert format_inr(1450.25) == "₹1,450.25"
    assert format_percent(12.5) == "12.5%"
    assert format_percent(2.3, signed=True) == "+2.3%"


def test_rating_label():
    assert "Strong Buy" in rating_label("Strong Buy")
    assert rating_label(None) == "—"


def test_comparison_score_rows():
    result = {
        "stocks": ["A.NS", "B.NS"],
        "overall_scores": {"A.NS": 80, "B.NS": 65},
        "fundamental_scores": {"A.NS": 78, "B.NS": 60},
        "technical_scores": {"A.NS": 72, "B.NS": 62},
        "sentiment_scores": {"A.NS": 65, "B.NS": 58},
    }
    rows = comparison_score_rows(result)
    assert len(rows) == 2
    assert rows[0]["Symbol"] == "A.NS"
    assert rows[0]["Overall"] == 80


def test_portfolio_holdings_rows():
    result = {
        "holdings": [
            {
                "holding": {"symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 1000},
                "current_price": 1450.25,
                "pnl_percent": 45.0,
                "allocation_percent": 100.0,
                "decision": {"overall_score": 75, "rating": "Buy"},
            }
        ]
    }
    rows = portfolio_holdings_rows(result)
    assert rows[0]["Symbol"] == "RELIANCE.NS"
    assert rows[0]["Score"] == 75


def test_normalize_holdings_validates_input():
    holdings = normalize_holdings(
        [{"symbol": "reliance.ns", "quantity": 10, "buy_price": 1000}]
    )
    assert holdings[0]["symbol"] == "RELIANCE.NS"

    with pytest.raises(ValueError):
        normalize_holdings([{"symbol": "A.NS", "quantity": 0, "buy_price": 100}])
