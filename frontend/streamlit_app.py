"""Buddy — Stock Market Analyst Streamlit frontend."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Streamlit runs with CWD on frontend/ — ensure project root is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import requests
import streamlit as st

from ui_helpers import (
    action_button_label_spacer,
    ai_conclusion_cards_html,
    analyze_conclusion_card_html,
    build_compare_stock_list,
    comparison_detailed_analysis_html,
    comparison_scorecard_html,
    comparison_stock_cards_html,
    comparison_wins_html,
    crisp_analyst_card_html,
    display_name,
    format_inr,
    format_large_number,
    format_percent,
    format_ratio,
    format_score,
    format_stock_name,
    fundamental_view_bullets,
    global_css,
    holdings_table_html,
    is_local_development,
    metric_summary_table_html,
    normalize_holdings,
    portfolio_holdings_rows,
    portfolio_summary_cards_html,
    rating_tone,
    resolve_api_base_url,
    sanitize_display_text,
    score_delta_color,
    section_panel_html,
    sentiment_tone,
    sentiment_view_bullets,
    single_stock_summary_html,
    technical_view_bullets,
    user_friendly_error,
)

logger = logging.getLogger(__name__)

NAV_ITEMS = [
    ("analyze", "Analyze Stock", "👤"),
    ("compare", "Compare Stocks", "📊"),
    ("portfolio", "My Portfolio", "💼"),
]


def _secrets_dict() -> dict | None:
    try:
        return dict(st.secrets)
    except Exception:
        return None


def _get_api_base_url() -> str:
    return resolve_api_base_url(_secrets_dict())


def _api_post(endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    base_url = _get_api_base_url()
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            detail = None
            try:
                body = response.json()
                detail = body.get("detail", response.text)
            except ValueError:
                detail = response.text
            logger.warning("API POST %s failed (%s): %s", endpoint, response.status_code, detail)
            return None, user_friendly_error(response.status_code, str(detail))
        return response.json(), None
    except requests.Timeout:
        logger.warning("API POST %s timed out", endpoint)
        return None, user_friendly_error(is_timeout=True)
    except requests.RequestException as exc:
        logger.warning("API POST %s connection error: %s", endpoint, exc)
        return None, user_friendly_error(is_connection=True)


def _show_user_error(message: str) -> None:
    st.warning(message)


def _inject_styles() -> None:
    st.markdown(global_css(), unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
<div class="buddy-hero">
  <div class="buddy-hero-brand">BUDDY</div>
  <div class="buddy-hero-title">Stock Market Analyst</div>
  <div class="buddy-hero-sub">AI-powered analysis for Indian equities</div>
  <div class="buddy-hero-tags">Fundamentals · Technicals · Sentiment · AI Synthesis</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_navigation() -> str:
    if "buddy_page" not in st.session_state:
        st.session_state.buddy_page = "compare"

    _, nav_col, _ = st.columns([0.35, 2.9, 0.35])
    with nav_col:
        st.markdown('<div class="buddy-nav-marker"></div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="small")
        for col, (key, label, icon) in zip(cols, NAV_ITEMS):
            with col:
                active = st.session_state.buddy_page == key
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_{key}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.buddy_page = key
                    st.rerun()
    return st.session_state.buddy_page


def _render_local_debug() -> None:
    """Developer controls — only visible during genuine local development."""
    if not is_local_development(_get_api_base_url()):
        return

    with st.expander("Developer / Debug (local only)", expanded=False):
        st.caption(f"API base: {_get_api_base_url()}")


def _section_heading(title: str, description: str) -> None:
    st.markdown(f'<div class="buddy-section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="buddy-section-desc">{description}</div>', unsafe_allow_html=True)


def _run_with_status(label: str, endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    with st.spinner(label):
        return _api_post(endpoint, payload)


def _render_disclaimer() -> None:
    st.markdown(
        '<div class="buddy-disclaimer">'
        "AI-generated analysis for informational purposes only. Not investment advice."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_analyze_result(result: dict) -> None:
    symbol = result.get("symbol", "")
    name = result.get("name") or display_name(symbol)
    decision = result.get("decision", {})
    fundamental = result.get("fundamental", {})
    technical = result.get("technical", {})
    sentiment = result.get("sentiment", {})
    master = result.get("master", {})
    metrics = fundamental.get("metrics", {})

    rating = str(decision.get("rating", "—"))
    overall = decision.get("overall_score")
    narrative = sanitize_display_text(
        master.get("narrative") or fundamental.get("summary") or "",
        [symbol] if symbol else [],
    )

    st.markdown(
        single_stock_summary_html(name, symbol, overall, rating),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="buddy-panel-label">Scorecard</div>', unsafe_allow_html=True)
    score_rows = [
        ("Overall", format_score(overall)),
        ("Fundamental", format_score(decision.get("fundamental_score"))),
        ("Technical", format_score(decision.get("technical_score"))),
        ("Sentiment", format_score(decision.get("sentiment_score"))),
        ("Price", format_inr(result.get("current_price"))),
        ("P/E", format_ratio(metrics.get("pe_ratio"))),
        ("P/B", format_ratio(metrics.get("pb_ratio"))),
        ("Revenue Growth", format_percent(metrics.get("revenue_growth"))),
    ]
    st.markdown(metric_summary_table_html(score_rows), unsafe_allow_html=True)

    st.markdown('<div class="buddy-panel-label">Analyst Views</div>', unsafe_allow_html=True)
    cards = st.columns(3)
    fund_rating = fundamental.get("rating")
    sent_class = sentiment.get("sentiment_classification")

    with cards[0]:
        st.markdown(
            crisp_analyst_card_html(
                "Fundamental View",
                str(fund_rating) if fund_rating else None,
                fundamental.get("score"),
                fundamental_view_bullets(fundamental),
                rating_tone(str(fund_rating) if fund_rating else None),
            ),
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            crisp_analyst_card_html(
                "Technical View",
                str(technical.get("trend")) if technical.get("trend") else None,
                technical.get("score"),
                technical_view_bullets(technical),
                "neutral",
            ),
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            crisp_analyst_card_html(
                "Sentiment View",
                str(sent_class) if sent_class else None,
                sentiment.get("sentiment_score"),
                sentiment_view_bullets(sentiment),
                sentiment_tone(str(sent_class) if sent_class else None),
            ),
            unsafe_allow_html=True,
        )

    risks = decision.get("major_risks", []) or master.get("major_risks", [])
    key_reasons = decision.get("key_reasons", [])
    st.markdown(
        analyze_conclusion_card_html(rating, narrative, risks, key_reasons),
        unsafe_allow_html=True,
    )

    if risks:
        st.markdown(
            section_panel_html(
                "Risks",
                "<div class='buddy-card-body'>"
                + "<br>".join(f"• {r}" for r in risks[:6])
                + "</div>",
            ),
            unsafe_allow_html=True,
        )

    articles = sentiment.get("articles", [])
    if articles:
        with st.expander("News sources", expanded=False):
            for article in articles[:5]:
                title = article.get("title", "Article")
                url = article.get("url")
                source = article.get("source", "")
                if url:
                    st.markdown(f"- [{title}]({url}) ({source})")
                else:
                    st.markdown(f"- {title} ({source})")

    _render_disclaimer()


def _render_compare_result(result: dict) -> None:
    stocks = result.get("stocks", [])
    if len(stocks) < 2:
        st.warning("Comparison data is incomplete.")
        return

    st.markdown(comparison_stock_cards_html(result), unsafe_allow_html=True)

    st.markdown('<div class="buddy-panel-label">Scorecard</div>', unsafe_allow_html=True)
    st.markdown(comparison_scorecard_html(result), unsafe_allow_html=True)

    st.markdown('<div class="buddy-panel-label">Where Each Stock Wins</div>', unsafe_allow_html=True)
    st.markdown(comparison_wins_html(result), unsafe_allow_html=True)

    st.markdown('<div class="buddy-panel-label">Detailed Analysis</div>', unsafe_allow_html=True)
    st.markdown(comparison_detailed_analysis_html(result), unsafe_allow_html=True)

    st.markdown(ai_conclusion_cards_html(result), unsafe_allow_html=True)
    _render_disclaimer()


def _render_portfolio_result(result: dict) -> None:
    st.markdown(portfolio_summary_cards_html(result), unsafe_allow_html=True)

    holdings_rows = portfolio_holdings_rows(result)
    if holdings_rows:
        st.markdown('<div class="buddy-panel-label">Holdings</div>', unsafe_allow_html=True)
        st.markdown(holdings_table_html(result), unsafe_allow_html=True)

    strongest = result.get("strongest_holdings", [])
    weakest = result.get("weakest_holdings", [])

    st.markdown('<div class="buddy-panel-label">Portfolio Insights</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="buddy-conclusion-grid">
  <div class="buddy-summary-card">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">🏆</span> Strongest Holdings</div>
    <div class="buddy-summary-body">{", ".join(format_stock_name(s) for s in strongest[:5]) or "—"}</div>
  </div>
  <div class="buddy-summary-card">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">⚠</span> Weakest Holdings</div>
    <div class="buddy-summary-body">{", ".join(format_stock_name(s) for s in weakest[:5]) or "—"}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    sector = result.get("sector_concentration", {})
    if sector:
        sector_rows = [
            (sector_name, format_percent(value)) for sector_name, value in sector.items()
        ]
        st.markdown('<div class="buddy-panel-label">Sector Allocation</div>', unsafe_allow_html=True)
        st.markdown(metric_summary_table_html(sector_rows), unsafe_allow_html=True)

    holding_symbols = [
        h.get("holding", {}).get("symbol", "")
        for h in result.get("holdings", [])
        if h.get("holding", {}).get("symbol")
    ]
    summary = sanitize_display_text(result.get("summary", ""), holding_symbols)
    risk = result.get("portfolio_risk", "")
    if summary or risk:
        st.markdown(
            f"""
<div class="buddy-conclusion-grid">
  <div class="buddy-summary-card">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">✦</span> AI Conclusion</div>
    <div class="buddy-summary-body">{summary or "—"}</div>
  </div>
  <div class="buddy-summary-card preferred">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">◎</span> Risk Overview</div>
    <div class="buddy-summary-body">{risk or "No major risks flagged."}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    _render_disclaimer()


def render_analyze_tab() -> None:
    _section_heading(
        "Analyze a Stock",
        "Get a multi-factor view using fundamentals, technicals, sentiment and AI synthesis.",
    )

    st.markdown('<div class="buddy-input-marker"></div>', unsafe_allow_html=True)
    input_cols = st.columns([2.2, 0.8])
    with input_cols[0]:
        query = st.text_input(
            "Ask about a stock",
            placeholder="How is Reliance doing?",
            key="analyze_query",
            label_visibility="visible",
        )
    with input_cols[1]:
        st.markdown(action_button_label_spacer(), unsafe_allow_html=True)
        analyze_clicked = st.button(
            "👤  Analyze Stock",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )
    st.markdown(
        '<div class="buddy-input-hint compact">Try: "How is Reliance doing?" · "Is TCS a good buy?"</div>',
        unsafe_allow_html=True,
    )

    if analyze_clicked:
        if not query.strip():
            st.warning("Enter a stock name or question to analyze.")
            return

        label = query.strip()[:48]
        result, error = _run_with_status(
            f"Analyzing {label}…",
            "/analyze",
            {"query": query.strip()},
        )
        if error:
            _show_user_error(error)
            return
        if result:
            _render_analyze_result(result)


def render_compare_tab() -> None:
    _section_heading(
        "Compare Stocks",
        "Compare Indian equities across fundamentals, technicals, sentiment and overall setup.",
    )

    st.markdown('<div class="buddy-input-marker"></div>', unsafe_allow_html=True)
    input_cols = st.columns([1.1, 1.1, 1.4, 0.8])
    with input_cols[0]:
        stock_a = st.text_input("Stock 1", value="Reliance", key="compare_stock_1")
        st.markdown(
            '<div class="buddy-input-hint compact">e.g. Reliance, TCS</div>',
            unsafe_allow_html=True,
        )
    with input_cols[1]:
        stock_b = st.text_input("Stock 2", value="Infosys", key="compare_stock_2")
        st.markdown(
            '<div class="buddy-input-hint compact">e.g. Infosys, HDFC Bank</div>',
            unsafe_allow_html=True,
        )
    with input_cols[2]:
        combined = st.text_input(
            "Additional stocks (optional)",
            placeholder="TCS, HDFC Bank",
            key="compare_symbols_extra",
        )
        st.markdown(
            '<div class="buddy-input-hint compact">Comma-separated names</div>',
            unsafe_allow_html=True,
        )
    with input_cols[3]:
        st.markdown(action_button_label_spacer(), unsafe_allow_html=True)
        compare_clicked = st.button(
            "📊  Compare Stocks",
            type="primary",
            use_container_width=True,
            key="compare_btn",
        )

    if compare_clicked:
        stocks = build_compare_stock_list(stock_a, stock_b, combined)

        if len(stocks) < 2:
            st.warning("Enter at least two stocks to compare.")
            return

        result, error = _run_with_status(
            "Comparing stocks…",
            "/compare",
            {"stocks": stocks},
        )
        if error:
            _show_user_error(error)
            return
        if result:
            _render_compare_result(result)


def render_portfolio_tab() -> None:
    _section_heading(
        "My Portfolio",
        "Understand how your holdings are performing and what may require attention.",
    )

    st.markdown('<div class="buddy-input-marker"></div>', unsafe_allow_html=True)
    editor_cols = st.columns([4.2, 0.9])
    with editor_cols[0]:
        default_df = pd.DataFrame(
            [
                {"symbol": "Reliance", "quantity": 10.0, "buy_price": 1000.0},
                {"symbol": "Infosys", "quantity": 50.0, "buy_price": 1500.0},
            ]
        )
        edited = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "symbol": st.column_config.TextColumn("Stock", required=True),
                "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, format="%.2f"),
                "buy_price": st.column_config.NumberColumn("Buy price (₹)", min_value=0.0, format="%.2f"),
            },
            hide_index=True,
            key="portfolio_editor",
        )
    with editor_cols[1]:
        st.markdown(action_button_label_spacer(), unsafe_allow_html=True)
        portfolio_clicked = st.button(
            "💼  Analyze Portfolio",
            type="primary",
            use_container_width=True,
            key="portfolio_btn",
        )
    st.markdown(
        '<div class="buddy-input-hint compact">Enter company names or tickers (e.g. Infosys, RELIANCE, TCS)</div>',
        unsafe_allow_html=True,
    )

    if portfolio_clicked:
        try:
            holdings = normalize_holdings(edited.to_dict("records"))
        except ValueError as exc:
            logger.warning("Portfolio validation failed: %s", exc)
            _show_user_error(user_friendly_error(status_code=422))
            return

        result, error = _run_with_status(
            "Analyzing portfolio…",
            "/portfolio",
            {"holdings": holdings},
        )
        if error:
            _show_user_error(error)
            return
        if result:
            _render_portfolio_result(result)


def main() -> None:
    st.set_page_config(
        page_title="Buddy — Stock Market Analyst",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_styles()
    _render_hero()
    page_key = _render_navigation()

    if page_key == "analyze":
        render_analyze_tab()
    elif page_key == "compare":
        render_compare_tab()
    else:
        render_portfolio_tab()

    _render_local_debug()


if __name__ == "__main__":
    main()
