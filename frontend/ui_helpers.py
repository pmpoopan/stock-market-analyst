"""Pure formatting helpers for the Streamlit UI — no Streamlit imports."""

from __future__ import annotations

from typing import Any


RATING_EMOJI: dict[str, str] = {
    "Strong Buy": "🟢",
    "Buy": "🔵",
    "Hold": "🟡",
    "Avoid": "🔴",
}


def format_inr(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 1, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}/100"


def rating_label(rating: str | None) -> str:
    if not rating:
        return "—"
    emoji = RATING_EMOJI.get(rating, "•")
    return f"{emoji} {rating}"


def score_delta_color(score: float) -> str:
    if score >= 70:
        return "normal"
    if score >= 50:
        return "off"
    return "inverse"


def comparison_score_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build tabular rows for side-by-side comparison scores."""
    stocks = result.get("stocks", [])
    rows: list[dict[str, Any]] = []
    for symbol in stocks:
        rows.append(
            {
                "Symbol": symbol,
                "Overall": result.get("overall_scores", {}).get(symbol),
                "Fundamental": result.get("fundamental_scores", {}).get(symbol),
                "Technical": result.get("technical_scores", {}).get(symbol),
                "Sentiment": result.get("sentiment_scores", {}).get(symbol),
            }
        )
    return rows


def portfolio_holdings_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten portfolio holdings for tabular display."""
    rows: list[dict[str, Any]] = []
    for item in result.get("holdings", []):
        holding = item.get("holding", {})
        decision = item.get("decision", {})
        rows.append(
            {
                "Symbol": holding.get("symbol"),
                "Qty": holding.get("quantity"),
                "Buy": holding.get("buy_price"),
                "Current": item.get("current_price"),
                "P&L %": item.get("pnl_percent"),
                "Allocation %": item.get("allocation_percent"),
                "Score": decision.get("overall_score"),
                "Rating": decision.get("rating"),
            }
        )
    return rows


def normalize_holdings(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Validate and normalize holdings from the portfolio editor."""
    holdings: list[dict[str, float | str]] = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        quantity = float(row.get("quantity", 0))
        buy_price = float(row.get("buy_price", 0))
        if quantity <= 0 or buy_price <= 0:
            raise ValueError(f"Invalid quantity or buy price for {symbol}")
        holdings.append(
            {"symbol": symbol, "quantity": quantity, "buy_price": buy_price}
        )
    if not holdings:
        raise ValueError("Add at least one holding with symbol, quantity, and buy price")
    return holdings
