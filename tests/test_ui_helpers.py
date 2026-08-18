"""Tests for Streamlit UI formatting helpers."""

import pytest

from frontend.ui_helpers import (
    analyze_conclusion_card_html,
    build_analyst_bullets,
    comparison_score_rows,
    comparison_scorecard_html,
    crisp_analyst_card_html,
    display_name,
    format_decimal,
    format_inr,
    format_percent,
    format_ratio,
    format_score,
    format_stock_name,
    fundamental_view_bullets,
    is_local_development,
    is_streamlit_cloud,
    LOCAL_API_DEFAULT,
    PRODUCTION_API_DEFAULT,
    normalize_holdings,
    parse_symbol_segments,
    portfolio_holdings_rows,
    rating_label,
    resolve_api_base_url,
    resolve_stock_symbol,
    resolve_symbol,
    sentiment_view_bullets,
    stock_picker_labels,
    stock_picker_symbol,
    technical_view_bullets,
    user_friendly_error,
)


def test_format_inr_and_percent():
    assert format_inr(1450.25) == "₹1,450.25"
    assert format_percent(12.5) == "12.50%"
    assert format_percent(2.3, signed=True) == "+2.30%"
    assert format_percent(9.6789) == "9.68%"


def test_format_decimal_and_score():
    assert format_decimal(57.1234) == "57.12"
    assert format_decimal(23.8) == "23.80"
    assert format_decimal(2) == "2.00"
    assert format_score(57.123) == "57.12/100"
    assert format_ratio(23.8123) == "23.81"


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
    assert rows[0]["Symbol"] == "A"
    assert rows[0]["Overall"] == 80


def test_portfolio_holdings_rows():
    result = {
        "holdings": [
            {
                "holding": {"symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 1000},
                "current_price": 1450.25,
                "pnl": 4502.5,
                "pnl_percent": 45.0,
                "allocation_percent": 100.0,
                "decision": {"overall_score": 75, "rating": "Buy"},
            }
        ]
    }
    rows = portfolio_holdings_rows(result)
    assert rows[0]["Symbol"] == "Reliance"
    assert rows[0]["Score"] == "75.00/100"
    assert rows[0]["P&L %"] == "+45.00%"


def test_normalize_holdings_validates_input():
    holdings = normalize_holdings(
        [{"symbol": "reliance.ns", "quantity": 10, "buy_price": 1000}]
    )
    assert holdings[0]["symbol"] == "RELIANCE.NS"

    holdings = normalize_holdings(
        [{"symbol": "Infosys", "quantity": 10, "buy_price": 1500}]
    )
    assert holdings[0]["symbol"] == "INFY.NS"

    with pytest.raises(ValueError):
        normalize_holdings([{"symbol": "A.NS", "quantity": 0, "buy_price": 100}])


def test_is_local_development():
    assert is_local_development("http://localhost:8000/api")
    assert is_local_development("http://127.0.0.1:8000/api")
    assert not is_local_development("https://stock-market-analyst-api.onrender.com/api")


def test_is_local_development_false_on_streamlit_cloud(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("STREAMLIT_RUNTIME_ENV", "cloud")
    assert is_streamlit_cloud()
    assert not is_local_development()
    assert resolve_api_base_url() == PRODUCTION_API_DEFAULT


def test_resolve_api_base_url_defaults(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("STREAMLIT_RUNTIME_ENV", raising=False)
    monkeypatch.delenv("STREAMLIT_RUNTIME_ENVIRONMENT", raising=False)
    monkeypatch.delenv("STREAMLIT_GIT_REPO", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)

    assert resolve_api_base_url() == LOCAL_API_DEFAULT

    monkeypatch.setenv("STREAMLIT_GIT_REPO", "user/buddy")
    assert resolve_api_base_url() == PRODUCTION_API_DEFAULT
    assert not is_local_development()

    monkeypatch.delenv("STREAMLIT_GIT_REPO", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://stock-market-analyst-api.onrender.com/api")
    assert resolve_api_base_url() == "https://stock-market-analyst-api.onrender.com/api"
    assert not is_local_development()


def test_user_friendly_error_messages():
    assert "temporarily unavailable" in user_friendly_error(is_connection=True)
    assert "try again" in user_friendly_error(status_code=500)


def test_format_stock_name_and_display_name():
    assert format_stock_name("RELIANCE.NS") == "Reliance"
    assert display_name("RELIANCE.NS") == "Reliance"
    assert format_stock_name("INFY.NS") == "Infosys"
    assert display_name("INFY.NS") == "Infosys"
    assert format_stock_name("TATAMOTORS.NS") == "Tata Motors"
    assert format_stock_name("M&M.NS") == "Mahindra"
    assert format_stock_name("TCS.NS") == "TCS"
    assert format_stock_name("UNKNOWN.NS") == "UNKNOWN"


def test_resolve_stock_symbol_and_resolve_symbol():
    assert resolve_symbol("Infosys") == "INFY.NS"
    assert resolve_symbol("infosys") == "INFY.NS"
    assert resolve_symbol("INFY") == "INFY.NS"
    assert resolve_symbol("INFY.NS") == "INFY.NS"
    assert resolve_symbol("Infosys Limited") == "INFY.NS"
    assert resolve_symbol("Reliance") == "RELIANCE.NS"
    assert resolve_stock_symbol("tata motors") == "TATAMOTORS.NS"
    assert resolve_stock_symbol("Mahindra") == "M&M.NS"
    assert resolve_stock_symbol("RELIANCE") == "RELIANCE.NS"
    assert resolve_stock_symbol("SBI") == "SBIN.NS"
    assert resolve_stock_symbol("HDFC Bank") == "HDFCBANK.NS"


def test_normalize_holdings_resolves_infosys():
    holdings = normalize_holdings(
        [{"symbol": "Infosys", "quantity": 10, "buy_price": 1500}]
    )
    assert holdings[0]["symbol"] == "INFY.NS"


def test_parse_symbol_segments():
    text = (
        "RELIANCE.NS: PE 23.8, PB 2.0; INFY.NS: PE 14.8, PB 5.0. "
        "INFY.NS trades at a lower PE."
    )
    segments = parse_symbol_segments(text, ["RELIANCE.NS", "INFY.NS"])
    assert "PE 23.8" in segments["RELIANCE.NS"]
    assert segments["INFY.NS"] == "PE 14.8, PB 5.0"


def test_stock_picker_helpers():
    labels = stock_picker_labels()
    assert "Reliance" in labels
    assert "Infosys" in labels
    assert stock_picker_symbol("Reliance") == "RELIANCE.NS"
    assert stock_picker_symbol("Mahindra") == "M&M.NS"


def test_comparison_scorecard_html():
    result = {
        "stocks": ["RELIANCE.NS", "INFY.NS"],
        "overall_scores": {"RELIANCE.NS": 57, "INFY.NS": 57},
        "fundamental_scores": {"RELIANCE.NS": 53, "INFY.NS": 68},
        "technical_scores": {"RELIANCE.NS": 79, "INFY.NS": 65},
        "sentiment_scores": {"RELIANCE.NS": 53, "INFY.NS": 50},
    }
    html = comparison_scorecard_html(result)
    assert "Reliance" in html
    assert "Infosys" in html
    assert "Highest" in html
    assert "Tie" in html


def test_build_compare_stock_list():
    from frontend.ui_helpers import build_compare_stock_list

    stocks = build_compare_stock_list("Reliance", "Infosys", "TCS, HDFC Bank")
    assert stocks == ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]


def test_action_button_label_spacer():
    from frontend.ui_helpers import action_button_label_spacer

    html = action_button_label_spacer()
    assert "buddy-action-label-spacer" in html


def test_clean_display_text():
    from frontend.ui_helpers import clean_display_text

    assert clean_display_text("P/E 23.8, P/B 2.0;") == "P/E 23.8, P/B 2.0"
    assert clean_display_text("revenue growth 9.6%;") == "revenue growth 9.6%"


def test_build_analyst_bullets_limits_and_marks_risks():
    bullets = build_analyst_bullets(
        positives=["Revenue growth remains positive", "Valuation is reasonable"],
        risks=["Earnings trend is the key concern"],
        summary="Extra sentence that should not be needed.",
        max_bullets=3,
    )
    assert len(bullets) == 3
    assert bullets[0][0] == "✓"
    assert bullets[-1][0] == "⚠"


def test_fundamental_view_bullets_from_backend_fields():
    fundamental = {
        "strengths": ["Mock strength: solid revenue growth."],
        "weaknesses": ["Mock weakness: valuation is stretched."],
        "risks": ["Mock risk: leverage should be monitored."],
        "summary": "Mock summary: Fundamentals are broadly balanced based on provided metrics.",
    }
    bullets = fundamental_view_bullets(fundamental)
    assert len(bullets) <= 3
    assert any("revenue" in text.lower() for _, text in bullets)


def test_technical_and_sentiment_view_bullets():
    technical = {
        "trend": "Uptrend",
        "momentum": "Momentum is positive",
        "volatility": "Volatility remains elevated",
        "summary": "Technical setup is supportive.",
    }
    tech_bullets = technical_view_bullets(technical)
    assert len(tech_bullets) <= 3
    assert any("volatility" in text.lower() for _, text in tech_bullets)

    sentiment = {
        "positive_catalysts": ["Recent news flow is broadly positive"],
        "negative_catalysts": ["Some sector-specific concerns remain"],
        "summary": "Sentiment is mixed.",
    }
    sent_bullets = sentiment_view_bullets(sentiment)
    assert len(sent_bullets) <= 3


def test_crisp_analyst_card_html_renders_bullets():
    html = crisp_analyst_card_html(
        "Fundamental View",
        "Buy",
        72.5,
        [("✓", "Revenue growth remains positive")],
        "positive",
    )
    assert "Fundamental View" in html
    assert "✓" in html
    assert "72.50/100" in html


def test_analyze_conclusion_card_html():
    html = analyze_conclusion_card_html(
        "Hold",
        "Fundamentals are stable, while technical momentum remains supportive.",
        risks=["Valuation risk remains elevated"],
    )
    assert "AI Conclusion" in html
    assert "Hold" in html
    assert "Fundamentals are stable" in html
    assert "Key risk:" in html


def test_analyst_bullets_are_complete_not_truncated():
    long_sma = (
        "The price (4064.3) sits above the 20-day SMA (3972.37), "
        "50-day SMA (4006.30) and 200-day SMA (3953.49), and trend strength continues."
    )
    long_atr = (
        "ATR 14 is 59.80, roughly 1.5% of the current price, suggesting moderate price "
        "volatility. The current drawdown remains within normal bounds."
    )

    sma_bullets = build_analyst_bullets(positives=[long_sma], max_bullets=1)
    atr_bullets = build_analyst_bullets(risks=[long_atr], max_bullets=1)

    assert sma_bullets
    assert atr_bullets
    assert "…" not in sma_bullets[0][1]
    assert "…" not in atr_bullets[0][1]
    assert sma_bullets[0][1].endswith(".")
    assert atr_bullets[0][1].endswith(".")
    assert "20-day" in sma_bullets[0][1]
    assert "200-day" in sma_bullets[0][1]
    assert "59.80" in atr_bullets[0][1]
    assert "moderate" in atr_bullets[0][1].lower()


def test_technical_view_bullets_summarize_long_fields():
    technical = {
        "trend": "Uptrend",
        "momentum": (
            "The price (4064.3) sits above the 20-day SMA (3972.37), "
            "50-day SMA (4006.30) and 200-day SMA (3953.49), confirming trend strength."
        ),
        "volatility": (
            "ATR 14 is 59.80, roughly 1.5% of the current price, suggesting moderate price "
            "volatility. The current drawdown remains elevated."
        ),
        "summary": "Technical setup remains supportive.",
    }
    bullets = technical_view_bullets(technical)
    assert len(bullets) <= 3
    for _, text in bullets:
        assert "…" not in text
        assert text.endswith(".")
