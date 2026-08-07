"""Buddy — Streamlit frontend.

Calls FastAPI backend; structured layouts for analyze, compare, and portfolio modes.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from ui_helpers import (
    comparison_score_rows,
    format_inr,
    format_percent,
    format_score,
    normalize_holdings,
    portfolio_holdings_rows,
    rating_label,
    score_delta_color,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")
DEFAULT_API_BASE_URL = API_BASE_URL.rstrip("/")


def _get_api_base_url() -> str:
    """Return API base URL from session state (sidebar can override env default)."""
    url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    return str(url).rstrip("/")


def _api_post(endpoint: str, payload: dict) -> dict | None:
    """POST to FastAPI and return JSON response."""
    base_url = _get_api_base_url()
    try:
        response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=120)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Request failed: {detail}")
            return None
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def _check_api_health() -> bool:
    """Show API connectivity in the sidebar. Returns True when health check succeeds."""
    base_url = _get_api_base_url()
    health_url = f"{base_url}/health"
    try:
        response = requests.get(health_url, timeout=5)
        if response.ok:
            data = response.json()
            st.sidebar.success(f"API online — {data.get('app', 'Buddy')}")
            return True
        st.sidebar.error(f"API error ({response.status_code}) at {health_url}")
    except requests.RequestException:
        st.sidebar.error("Cannot reach API")
        with st.sidebar.expander("Start the backend", expanded=True):
            st.markdown(
                "1. Open a terminal in the project folder\n"
                "2. Run: `.venv\\Scripts\\activate` then `python main.py`\n"
                "3. Verify: [health check](http://localhost:8000/api/health)\n\n"
                f"Current URL: `{base_url}`\n\n"
                "Note: `http://localhost:8000/api` has no page — use `/api/health` or `/docs`."
            )
    return False


def _render_api_sidebar() -> None:
    """Sidebar API settings and health probe."""
    st.sidebar.header("Backend")
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL

    st.sidebar.text_input(
        "API base URL",
        key="api_base_url",
        help="Default: http://localhost:8000/api (set API_BASE_URL env to override on load)",
    )
    api_online = _check_api_health()
    if api_online:
        base = _get_api_base_url()
        st.sidebar.caption(f"[OpenAPI docs]({base.replace('/api', '')}/docs)")


def _render_bullet_list(title: str, items: list[str]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def _render_analyze_result(result: dict) -> None:
    symbol = result.get("symbol", "")
    name = result.get("name") or symbol
    decision = result.get("decision", {})
    fundamental = result.get("fundamental", {})
    technical = result.get("technical", {})
    sentiment = result.get("sentiment", {})
    master = result.get("master", {})

    st.markdown(f"### {name}")
    st.caption(symbol)

    header_cols = st.columns([2, 1, 1, 1, 1])
    with header_cols[0]:
        st.metric("Current price", format_inr(result.get("current_price")))
    with header_cols[1]:
        overall = decision.get("overall_score", 0)
        st.metric("Overall score", format_score(overall), delta=rating_label(decision.get("rating")))
    with header_cols[2]:
        st.metric("Fundamental", format_score(decision.get("fundamental_score")))
    with header_cols[3]:
        st.metric("Technical", format_score(decision.get("technical_score")))
    with header_cols[4]:
        st.metric("Sentiment", format_score(decision.get("sentiment_score")))

    st.divider()

    tab_fund, tab_tech, tab_sent, tab_master = st.tabs(
        ["Fundamentals", "Technical", "Sentiment", "Master view"]
    )

    with tab_fund:
        metrics = fundamental.get("metrics", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("PE", metrics.get("pe_ratio") or "—")
        metric_cols[1].metric("PB", metrics.get("pb_ratio") or "—")
        metric_cols[2].metric("ROE", format_percent(metrics.get("roe")))
        metric_cols[3].metric("Rev growth", format_percent(metrics.get("revenue_growth")))

        _render_bullet_list("Strengths", fundamental.get("strengths", []))
        _render_bullet_list("Weaknesses", fundamental.get("weaknesses", []))
        _render_bullet_list("Risks", fundamental.get("risks", []))
        if fundamental.get("summary"):
            st.info(fundamental["summary"])

    with tab_tech:
        tech_cols = st.columns(3)
        tech_cols[0].metric("Trend", technical.get("trend", "—"))
        tech_cols[1].metric("Momentum", technical.get("momentum", "—"))
        tech_cols[2].metric("Volatility", technical.get("volatility", "—"))

        signals = technical.get("signals", [])
        if signals:
            signal_rows = [
                {
                    "Signal": signal.get("name"),
                    "Reading": signal.get("value"),
                    "Bias": signal.get("signal"),
                }
                for signal in signals[:8]
            ]
            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

        if technical.get("summary"):
            st.info(technical["summary"])

    with tab_sent:
        sent_cols = st.columns(2)
        sent_cols[0].metric(
            "Sentiment score",
            format_score(sentiment.get("sentiment_score")),
        )
        sent_cols[1].metric(
            "Classification",
            sentiment.get("sentiment_classification", "—"),
        )

        _render_bullet_list("Positive catalysts", sentiment.get("positive_catalysts", []))
        _render_bullet_list("Negative catalysts", sentiment.get("negative_catalysts", []))
        _render_bullet_list("Key events", sentiment.get("key_events", []))

        articles = sentiment.get("articles", [])
        if articles:
            st.markdown("**Recent articles**")
            for article in articles[:5]:
                title = article.get("title", "Article")
                url = article.get("url")
                source = article.get("source", "")
                if url:
                    st.markdown(f"- [{title}]({url}) ({source})")
                else:
                    st.markdown(f"- {title} ({source})")

        if sentiment.get("summary"):
            st.info(sentiment["summary"])

    with tab_master:
        _render_bullet_list("Agreement", master.get("agreement_points", []))
        _render_bullet_list("Disagreement", master.get("disagreement_points", []))
        _render_bullet_list("Major risks", master.get("major_risks", []))
        _render_bullet_list("Catalysts", master.get("important_catalysts", []))
        if master.get("narrative"):
            st.markdown(master["narrative"])
        if master.get("data_vs_interpretation"):
            st.caption(master["data_vs_interpretation"])

    _render_bullet_list("Key reasons", decision.get("key_reasons", []))
    _render_bullet_list("Major risks", decision.get("major_risks", []))

    sources = result.get("sources", [])
    if sources:
        st.markdown("**Sources**")
        for url in sources:
            st.markdown(f"- [{url}]({url})")


def _render_compare_result(result: dict) -> None:
    stocks = result.get("stocks", [])
    winner = result.get("winner")

    if winner:
        st.success(f"**Leader:** {winner}")
        st.caption(result.get("relative_assessment", ""))
    else:
        st.info(result.get("relative_assessment", "Scores are closely matched."))

    score_df = pd.DataFrame(comparison_score_rows(result))
    if not score_df.empty:
        st.dataframe(
            score_df.set_index("Symbol"),
            use_container_width=True,
        )

    if len(stocks) >= 2:
        chart_df = score_df.set_index("Symbol")[["Overall", "Fundamental", "Technical", "Sentiment"]]
        st.bar_chart(chart_df, height=320)

    compare_cols = st.columns(2)
    narratives = [
        ("Valuation", result.get("valuation_comparison")),
        ("Growth", result.get("growth_comparison")),
        ("Risk", result.get("risk_comparison")),
        ("Technical trends", result.get("technical_trend_comparison")),
    ]
    for index, (title, text) in enumerate(narratives):
        with compare_cols[index % 2]:
            if text:
                st.markdown(f"**{title}**")
                st.write(text)


def _render_portfolio_result(result: dict) -> None:
    summary_cols = st.columns(5)
    summary_cols[0].metric("Invested", format_inr(result.get("total_invested")))
    summary_cols[1].metric("Current value", format_inr(result.get("total_current_value")))
    pnl = result.get("total_pnl")
    pnl_pct = result.get("total_pnl_percent")
    summary_cols[2].metric(
        "P&L",
        format_inr(pnl),
        delta=format_percent(pnl_pct, signed=True),
        delta_color=score_delta_color(50 + (pnl_pct or 0)),
    )
    summary_cols[3].metric("Portfolio score", format_score(result.get("portfolio_score")))
    strongest = result.get("strongest_holdings", [])
    weakest = result.get("weakest_holdings", [])
    summary_cols[4].metric("Holdings", str(len(result.get("holdings", []))))

    if strongest or weakest:
        highlight_cols = st.columns(2)
        with highlight_cols[0]:
            st.markdown(f"**Strongest:** {', '.join(strongest[:3]) or '—'}")
        with highlight_cols[1]:
            st.markdown(f"**Weakest:** {', '.join(weakest[:3]) or '—'}")

    holdings_df = pd.DataFrame(portfolio_holdings_rows(result))
    if not holdings_df.empty:
        st.markdown("**Holdings**")
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    sector = result.get("sector_concentration", {})
    if sector:
        st.markdown("**Sector concentration**")
        sector_df = pd.DataFrame(
            {"Sector": list(sector.keys()), "Allocation %": list(sector.values())}
        ).set_index("Sector")
        st.bar_chart(sector_df, height=280)

    if result.get("portfolio_risk"):
        st.warning(result["portfolio_risk"])
    if result.get("summary"):
        st.info(result["summary"])


def render_analyze_mode() -> None:
    st.subheader("Analyze Stock")
    query = st.text_input("Ask about a stock", placeholder="How is Reliance doing?")
    if st.button("Analyze", key="analyze_btn", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        with st.spinner("Running fundamental, technical, and sentiment analysis…"):
            result = _api_post("/analyze", {"query": query})
        if result:
            _render_analyze_result(result)


def render_compare_mode() -> None:
    st.subheader("Compare Stocks")
    symbols = st.text_input(
        "Stock symbols (comma-separated)",
        placeholder="TATAMOTORS.NS, M&M.NS",
    )
    if st.button("Compare", key="compare_btn", type="primary"):
        stocks = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
        if len(stocks) < 2:
            st.warning("Enter at least two symbols.")
            return
        with st.spinner("Comparing stocks side by side…"):
            result = _api_post("/compare", {"stocks": stocks})
        if result:
            _render_compare_result(result)


def render_portfolio_mode() -> None:
    st.subheader("Analyze Portfolio")
    st.caption("Add holdings below — symbols should include the NSE suffix (e.g. RELIANCE.NS).")

    default_df = pd.DataFrame(
        [
            {"symbol": "TATAMOTORS.NS", "quantity": 100.0, "buy_price": 700.0},
            {"symbol": "INFY.NS", "quantity": 50.0, "buy_price": 1500.0},
        ]
    )
    edited = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, format="%.2f"),
            "buy_price": st.column_config.NumberColumn("Buy price (₹)", min_value=0.0, format="%.2f"),
        },
        hide_index=True,
    )

    if st.button("Analyze Portfolio", key="portfolio_btn", type="primary"):
        try:
            holdings = normalize_holdings(edited.to_dict("records"))
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.spinner("Analyzing each holding and aggregating portfolio metrics…"):
            result = _api_post("/portfolio", {"holdings": holdings})
        if result:
            _render_portfolio_result(result)


def main() -> None:
    st.set_page_config(page_title="Buddy — Stock Analyst", page_icon="📈", layout="wide")
    st.title("Buddy — Stock Market Analyst")
    st.caption("AI-powered analysis for Indian equities")

    st.sidebar.header("Navigation")
    mode = st.sidebar.radio(
        "Mode",
        ["Analyze Stock", "Compare Stocks", "Analyze Portfolio"],
    )
    _render_api_sidebar()

    if mode == "Analyze Stock":
        render_analyze_mode()
    elif mode == "Compare Stocks":
        render_compare_mode()
    else:
        render_portfolio_mode()


if __name__ == "__main__":
    main()
