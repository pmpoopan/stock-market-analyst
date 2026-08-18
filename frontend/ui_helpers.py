"""Pure formatting and environment helpers for the Streamlit UI — no Streamlit imports."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.agents.stock_aliases import INDIAN_STOCK_ALIASES
from app.agents.stock_aliases import display_name as _canonical_display_name
from app.agents.stock_aliases import resolve_symbol as _canonical_resolve_symbol

logger = logging.getLogger(__name__)

LOCAL_API_DEFAULT = "http://localhost:8000/api"
PRODUCTION_API_DEFAULT = "https://stock-market-analyst-api.onrender.com/api"

USER_ERROR_GENERIC = (
    "Unable to complete the analysis right now. Please try again."
)
USER_ERROR_SERVICE = (
    "Analysis service is temporarily unavailable. Please try again in a moment."
)
USER_ERROR_TIMEOUT = (
    "The analysis is taking longer than expected. Please try again in a moment."
)
USER_ERROR_BAD_REQUEST = (
    "We could not process that request. Try a different query or check your inputs."
)
USER_ERROR_VALIDATION = (
    "Please check your inputs and try again."
)


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud (not local `streamlit run`)."""
    for env_key in ("STREAMLIT_RUNTIME_ENV", "STREAMLIT_RUNTIME_ENVIRONMENT"):
        value = os.getenv(env_key, "").strip().lower()
        if value in {"cloud", "community-cloud", "streamlit-cloud"}:
            return True

    host_hints = " ".join(
        os.getenv(name, "")
        for name in ("HOSTNAME", "STREAMLIT_SERVER_ADDRESS")
    ).lower()
    if "streamlit.app" in host_hints:
        return True

    if os.getenv("STREAMLIT_GIT_REPO"):
        return True

    return False


def resolve_api_base_url(secrets: dict[str, Any] | None = None) -> str:
    """Resolve API base URL from Streamlit secrets, env, or environment default."""
    if secrets and secrets.get("API_BASE_URL"):
        return str(secrets["API_BASE_URL"]).rstrip("/")

    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    if is_streamlit_cloud():
        return PRODUCTION_API_DEFAULT

    return LOCAL_API_DEFAULT


def is_local_development(api_base_url: str | None = None) -> bool:
    """True only for genuine local backend development (never on Streamlit Cloud)."""
    if is_streamlit_cloud():
        return False

    url = (api_base_url or resolve_api_base_url()).lower()
    return "localhost" in url or "127.0.0.1" in url


def format_decimal(value: float | None, decimals: int = 2) -> str:
    """Format a numeric value with fixed decimal places for display."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def format_inr(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 2, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{format_decimal(value)}/100"


def format_ratio(value: float | None, decimals: int = 2) -> str:
    """Format ratio metrics such as P/E or P/B."""
    return format_decimal(value, decimals=decimals)


def format_large_number(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1_000_000_000_000:
        return f"₹{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"
    return format_inr(value)


def rating_label(rating: str | None) -> str:
    if not rating:
        return "—"
    return str(rating)


def rating_tone(rating: str | None) -> str:
    if not rating:
        return "neutral"
    normalized = str(rating).lower()
    if "strong buy" in normalized or normalized == "buy":
        return "positive"
    if normalized == "hold":
        return "neutral"
    if normalized == "avoid":
        return "negative"
    return "neutral"


def sentiment_tone(classification: str | None) -> str:
    if not classification:
        return "neutral"
    normalized = str(classification).lower()
    if "positive" in normalized:
        return "positive"
    if "negative" in normalized:
        return "negative"
    return "neutral"


def score_delta_color(score: float) -> str:
    if score >= 70:
        return "normal"
    if score >= 50:
        return "off"
    return "inverse"


def user_friendly_error(
    status_code: int | None = None,
    detail: str | None = None,
    is_timeout: bool = False,
    is_connection: bool = False,
) -> str:
    if is_connection:
        return USER_ERROR_SERVICE
    if is_timeout:
        return USER_ERROR_TIMEOUT
    if status_code == 400:
        return USER_ERROR_BAD_REQUEST
    if status_code == 422:
        return USER_ERROR_VALIDATION
    if status_code in (404, 500, 502, 503, 504):
        return USER_ERROR_SERVICE
    if detail:
        logger.warning("API error (%s): %s", status_code, detail)
    return USER_ERROR_GENERIC


def comparison_score_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    stocks = result.get("stocks", [])
    rows: list[dict[str, Any]] = []
    for symbol in stocks:
        rows.append(
            {
                "Symbol": format_stock_name(symbol),
                "Overall": result.get("overall_scores", {}).get(symbol),
                "Fundamental": result.get("fundamental_scores", {}).get(symbol),
                "Technical": result.get("technical_scores", {}).get(symbol),
                "Sentiment": result.get("sentiment_scores", {}).get(symbol),
            }
        )
    return rows


def comparison_metrics_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    stocks = result.get("stocks", [])
    if not stocks:
        return []

    metrics = [
        ("Overall score", result.get("overall_scores", {})),
        ("Fundamental", result.get("fundamental_scores", {})),
        ("Technical", result.get("technical_scores", {})),
        ("Sentiment", result.get("sentiment_scores", {})),
    ]
    rows: list[dict[str, Any]] = []
    for label, values in metrics:
        row: dict[str, Any] = {"Metric": label}
        for symbol in stocks:
            value = values.get(symbol)
            row[symbol] = format_decimal(value) if isinstance(value, (int, float)) else "—"
        rows.append(row)
    return rows


def portfolio_holdings_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.get("holdings", []):
        holding = item.get("holding", {})
        decision = item.get("decision", {})
        rows.append(
            {
                "Symbol": format_stock_name(holding.get("symbol", "")),
                "Qty": format_decimal(holding.get("quantity")),
                "Buy (₹)": format_inr(holding.get("buy_price")),
                "Current (₹)": format_inr(item.get("current_price")),
                "P&L (₹)": format_inr(item.get("pnl")),
                "P&L %": format_percent(item.get("pnl_percent"), signed=True),
                "Allocation %": format_percent(item.get("allocation_percent")),
                "Score": format_score(decision.get("overall_score")),
                "Rating": decision.get("rating"),
            }
        )
    return rows


def normalize_holdings(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    holdings: list[dict[str, float | str]] = []
    for row in rows:
        raw_symbol = str(row.get("symbol", "")).strip()
        if not raw_symbol:
            continue
        symbol = resolve_stock_symbol(raw_symbol)
        if not symbol:
            continue
        quantity = float(row.get("quantity", 0))
        buy_price = float(row.get("buy_price", 0))
        if quantity <= 0 or buy_price <= 0:
            raise ValueError(f"Invalid quantity or buy price for {display_name(symbol)}")
        holdings.append(
            {"symbol": symbol, "quantity": quantity, "buy_price": buy_price}
        )
    if not holdings:
        raise ValueError("Add at least one holding with symbol, quantity, and buy price")
    return holdings


def resolve_stock_symbol(raw: str) -> str:
    """Map user input (name or ticker) to NSE symbol for API calls."""
    return resolve_symbol(raw)


def resolve_symbol(raw: str) -> str:
    """Normalize user input to a canonical Yahoo Finance NSE symbol."""
    return _canonical_resolve_symbol(raw)


def parse_compare_symbols(raw: str) -> list[str]:
    parts = re.split(r"[,;\n]+", raw)
    return [resolve_stock_symbol(part) for part in parts if part.strip()]


def _build_symbol_display_map() -> dict[str, str]:
    """Build display names from backend stock alias registry."""
    by_symbol: dict[str, list[str]] = {}
    for alias, symbol in INDIAN_STOCK_ALIASES.items():
        by_symbol.setdefault(symbol.upper(), []).append(alias)

    display: dict[str, str] = {}
    for symbol in sorted(set(INDIAN_STOCK_ALIASES.values())):
        display[symbol] = _canonical_display_name(symbol)
    return display


_SYMBOL_DISPLAY_MAP = _build_symbol_display_map()
_STOCK_PICKER_CUSTOM_LABEL = "— Type custom name —"


def _stock_picker_map() -> dict[str, str]:
    """Map display label -> API symbol for dropdown pickers."""
    symbols = sorted(set(INDIAN_STOCK_ALIASES.values()))
    return {display_name(symbol): symbol for symbol in symbols}


def stock_picker_custom_label() -> str:
    return _STOCK_PICKER_CUSTOM_LABEL


def stock_picker_labels() -> list[str]:
    return sorted(_stock_picker_map().keys())


def stock_picker_symbol(display_label: str) -> str:
    return _stock_picker_map().get(display_label, resolve_stock_symbol(display_label))


def format_stock_name(symbol: str) -> str:
    """Display-friendly stock name — does not alter API symbols."""
    return display_name(symbol)


def display_name(symbol: str) -> str:
    """Human-readable company name for a Yahoo Finance symbol."""
    return _canonical_display_name(symbol)


def sanitize_display_text(text: str, symbols: list[str]) -> str:
    """Replace API symbols with display names in narrative text."""
    if not text:
        return text
    result = text
    for symbol in symbols:
        result = result.replace(symbol.upper(), format_stock_name(symbol))
        result = result.replace(symbol, format_stock_name(symbol))
    return result


def _trim_comparative_tail(segment: str) -> str:
    """Drop trailing summary sentences from per-stock metric fragments."""
    if not segment or ". " not in segment:
        return segment
    parts = segment.split(". ")
    if len(parts) < 2:
        return segment
    rest = ". ".join(parts[1:]).lower()
    summary_markers = (
        "trades at",
        "shows the",
        "has the",
        "has a lower",
        "strongest",
        "lowest risk",
    )
    if any(marker in rest for marker in summary_markers):
        return parts[0]
    return segment


def parse_symbol_segments(narrative: str, symbols: list[str]) -> dict[str, str]:
    """Split combined comparison narrative into per-symbol segments."""
    segments = {symbol: "" for symbol in symbols}
    if not narrative:
        return segments

    text = narrative.strip()
    for symbol in symbols:
        marker = f"{symbol.upper()}:"
        start = text.find(marker)
        if start == -1:
            continue
        content_start = start + len(marker)
        end = len(text)
        for other in symbols:
            if other.upper() == symbol.upper():
                continue
            other_marker = f"{other.upper()}:"
            pos = text.find(other_marker, content_start)
            if pos != -1:
                end = min(end, pos)
        segment = text[content_start:end].strip().strip(".")
        segment = _trim_comparative_tail(segment)
        segments[symbol] = segment
    return segments


def clean_display_text(text: str) -> str:
    """Remove awkward trailing punctuation from metric fragments."""
    if not text:
        return text
    cleaned = text.strip()
    cleaned = re.sub(r",\s*;", ",", cleaned)
    cleaned = re.sub(r";\s*;", ";", cleaned)
    cleaned = re.sub(r";\s*,", ",", cleaned)
    cleaned = re.sub(r";\s+", " ", cleaned)
    cleaned = re.sub(r";+\s*$", "", cleaned)
    cleaned = re.sub(r",\s*$", "", cleaned)
    return cleaned.strip()


def _format_metric_number(value: str) -> str:
    """Format numeric fragments extracted from narrative text."""
    text = value.strip().rstrip(";,")
    if not text or "unavailable" in text.lower():
        return "—"
    if text.endswith("%"):
        try:
            return format_percent(float(text[:-1]))
        except ValueError:
            return text
    try:
        return format_decimal(float(text))
    except ValueError:
        return text


def infer_rating_from_score(score: float | None) -> str:
    """Mirror backend rating bands for display-only labels."""
    if score is None:
        return "—"
    if score >= 80:
        return "Strong Buy"
    if score >= 65:
        return "Buy"
    if score >= 45:
        return "Hold"
    return "Avoid"


def parse_ratings_from_assessment(text: str, stocks: list[str]) -> dict[str, str]:
    ratings: dict[str, str] = {}
    if not text:
        return ratings
    for symbol in stocks:
        sym = symbol.upper()
        match = re.search(rf"{re.escape(sym)}\s+[\d.]+/100\s+\((\w+)", text, re.I)
        if match:
            ratings[symbol] = match.group(1)
            continue
        name = format_stock_name(symbol)
        match = re.search(rf"{re.escape(name)}\s+[\d.]+/100\s+\((\w+)", text, re.I)
        if match:
            ratings[symbol] = match.group(1)
    return ratings


def stock_rating_label(symbol: str, result: dict[str, Any]) -> str:
    assessment = result.get("relative_assessment", "")
    parsed = parse_ratings_from_assessment(assessment, [symbol])
    if symbol in parsed:
        return parsed[symbol]
    overall = result.get("overall_scores", {}).get(symbol)
    return infer_rating_from_score(overall if isinstance(overall, (int, float)) else None)


def build_compare_stock_list(stock_a: str, stock_b: str, additional: str) -> list[str]:
    stocks: list[str] = []
    if stock_a.strip():
        stocks.append(resolve_stock_symbol(stock_a))
    if stock_b.strip():
        stocks.append(resolve_stock_symbol(stock_b))
    stocks.extend(parse_compare_symbols(additional))
    return [symbol for symbol in dict.fromkeys(stocks) if symbol]


def _extract_metric(text: str, pattern: str) -> str:
    if not text:
        return "—"
    text = clean_display_text(text)
    match = re.search(pattern, text, re.I)
    if not match:
        return "—"
    value = match.group(1).strip()
    if "unavailable" in value.lower():
        return "—"
    return _format_metric_number(value)


def _value_tone_class(metric: str, value: str) -> str:
    if value == "—":
        return ""
    metric_lower = metric.lower()
    if "growth" in metric_lower or metric_lower == "trend":
        normalized = value.lower()
        if "up" in normalized or normalized == "strong":
            return "val-positive"
        if "down" in normalized or normalized == "weak":
            return "val-negative"
        try:
            num = float(value.replace("%", ""))
            if num < 0:
                return "val-negative"
            if num > 0:
                return "val-positive"
        except ValueError:
            return ""
    return ""


def format_ticker(symbol: str) -> str:
    """Short ticker for display (no exchange suffix)."""
    sym = symbol.upper().strip()
    return sym.replace(".NS", "").replace(".BO", "")


def score_tone_class(score: float | None) -> str:
    if not isinstance(score, (int, float)):
        return "score-mid"
    if score >= 65:
        return "score-high"
    if score >= 45:
        return "score-mid"
    return "score-low"


def comparison_overall_leader(result: dict[str, Any]) -> str | None:
    stocks = result.get("stocks", [])
    overall = result.get("overall_scores", {})
    numeric = {
        symbol: overall.get(symbol)
        for symbol in stocks
        if isinstance(overall.get(symbol), (int, float))
    }
    if not numeric:
        return None
    max_score = max(numeric.values())
    leaders = [symbol for symbol, value in numeric.items() if value == max_score]
    if len(leaders) != 1:
        return None
    return leaders[0]


def _best_value_indices(values: list[str], higher_is_better: bool = True) -> set[int]:
    """Return indices of best numeric values in a row."""
    parsed: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value == "—":
            continue
        try:
            num = float(value.replace("%", ""))
            parsed.append((index, num))
        except ValueError:
            continue
    if len(parsed) < 2:
        return set()
    best = max(parsed, key=lambda item: item[1])[1] if higher_is_better else min(parsed, key=lambda item: item[1])[1]
    worst = min(parsed, key=lambda item: item[1])[1] if higher_is_better else max(parsed, key=lambda item: item[1])[1]
    if best == worst:
        return set()
    return {index for index, num in parsed if num == best}


def _lower_is_better_metric(metric: str) -> bool:
    lowered = metric.lower()
    return "risk adjustment" in lowered or "flagged risks" in lowered or "debt" in lowered or "p/e" in lowered or "p/b" in lowered


def _rating_badge_class(rating: str) -> str:
    tone = rating_tone(rating)
    if tone == "positive":
        return "badge-buy"
    if tone == "negative":
        return "badge-avoid"
    return "badge-hold"


def _row_highest_label(stocks: list[str], values: dict[str, float | None]) -> str:
    numeric = {
        symbol: values[symbol]
        for symbol in stocks
        if isinstance(values.get(symbol), (int, float))
    }
    if len(numeric) < 2:
        return "—"
    max_val = max(numeric.values())
    min_val = min(numeric.values())
    leaders = [symbol for symbol, value in numeric.items() if value == max_val]
    if max_val == min_val:
        return "Tie"
    if len(leaders) == 1:
        return format_stock_name(leaders[0])
    return ", ".join(format_stock_name(symbol) for symbol in leaders)


def comparison_stock_cards_html(result: dict[str, Any]) -> str:
    stocks = result.get("stocks", [])
    if not stocks:
        return ""

    parts: list[str] = []
    for index, symbol in enumerate(stocks):
        if index > 0:
            parts.append('<div class="buddy-vs-badge">VS</div>')
        name = format_stock_name(symbol)
        ticker = format_ticker(symbol)
        overall = result.get("overall_scores", {}).get(symbol)
        score_text = (
            f"{format_decimal(overall)} / 100"
            if isinstance(overall, (int, float))
            else "—"
        )
        score_class = score_tone_class(overall if isinstance(overall, (int, float)) else None)
        rating = stock_rating_label(symbol, result)
        badge_class = _rating_badge_class(rating)
        initial = name[0].upper() if name else "?"
        parts.append(
            f"""
<div class="buddy-stock-card">
  <div class="buddy-stock-icon">{initial}</div>
  <div class="buddy-stock-name">{name}</div>
  <div class="buddy-stock-ticker">{ticker}</div>
  <div class="buddy-stock-score-label">Overall Score</div>
  <div class="buddy-stock-score {score_class}">{score_text}</div>
  <div class="buddy-rating-badge {badge_class}">{rating}</div>
</div>
"""
        )
    return f'<div class="buddy-stock-grid">{"".join(parts)}</div>'


def format_segment_bullets(segment: str) -> str:
    """Turn comma-separated metric fragments into readable lines."""
    if not segment:
        return "—"
    segment = clean_display_text(segment)
    parts = re.split(r"[,;]+", segment)
    parts = [part.strip() for part in parts if part.strip()]
    formatted: list[str] = []
    for part in parts:
        part = re.sub(r"\bPE\b", "P/E", part, flags=re.IGNORECASE)
        part = re.sub(r"\bPB\b", "P/B", part, flags=re.IGNORECASE)
        part = re.sub(r"\brevenue growth\b", "Revenue Growth", part, flags=re.IGNORECASE)
        part = re.sub(r"\bearnings growth\b", "Earnings Growth", part, flags=re.IGNORECASE)
        part = re.sub(r"\brisk adjustment\b", "Risk Adjustment", part, flags=re.IGNORECASE)
        part = re.sub(r"\bdebt/equity\b", "Debt/Equity", part, flags=re.IGNORECASE)
        part = re.sub(r"\bflagged risks\b", "Flagged Risks", part, flags=re.IGNORECASE)
        match = re.match(
            r"^(P/?E|P/?B|Revenue Growth|Earnings Growth|Risk Adjustment|Debt/Equity|RSI|Technical Score)\s+(.+)$",
            part,
            re.I,
        )
        if match:
            label = match.group(1)
            if label.lower() in ("pe", "p/e"):
                label = "P/E"
            elif label.lower() in ("pb", "p/b"):
                label = "P/B"
            else:
                label = label.title() if label.islower() else label
            formatted.append(f"{label} {_format_metric_number(match.group(2))}")
        else:
            formatted.append(part)
    return "<br>".join(formatted)


def comparison_category_wins(result: dict[str, Any]) -> dict[str, list[str]]:
    """Derive which stock leads each score category."""
    stocks = result.get("stocks", [])
    wins: dict[str, list[str]] = {symbol: [] for symbol in stocks}
    categories = [
        ("Overall", "overall_scores"),
        ("Fundamentals", "fundamental_scores"),
        ("Technicals", "technical_scores"),
        ("Sentiment", "sentiment_scores"),
    ]
    for label, key in categories:
        scores = result.get(key, {})
        values = {
            symbol: scores.get(symbol)
            for symbol in stocks
            if isinstance(scores.get(symbol), (int, float))
        }
        if len(values) < 2:
            continue
        max_score = max(values.values())
        min_score = min(values.values())
        if max_score == min_score:
            continue
        leaders = [symbol for symbol, value in values.items() if value == max_score]
        if len(leaders) == 1:
            wins[leaders[0]].append(label)
    return wins


def comparison_scorecard_html(result: dict[str, Any]) -> str:
    """HTML scorecard table with highest-score column."""
    stocks = result.get("stocks", [])
    if not stocks:
        return ""

    headers = [format_stock_name(symbol) for symbol in stocks]
    rows = [
        ("Overall Score", "overall_scores"),
        ("Fundamental", "fundamental_scores"),
        ("Technical", "technical_scores"),
        ("Sentiment", "sentiment_scores"),
    ]

    head_html = "".join(f"<th>{name}</th>" for name in headers)
    head_html += "<th>Highest</th>"
    body_rows: list[str] = []

    for label, key in rows:
        score_map = result.get(key, {})
        values = [score_map.get(symbol) for symbol in stocks]
        numeric = [v for v in values if isinstance(v, (int, float))]
        max_val = max(numeric) if numeric else None
        highest = _row_highest_label(stocks, score_map)
        cells: list[str] = []
        for symbol in stocks:
            value = score_map.get(symbol)
            if isinstance(value, (int, float)):
                is_win = (
                    max_val is not None
                    and value == max_val
                    and len(numeric) > 1
                    and highest != "Tie"
                )
                cell_class = "score-win" if is_win else ""
                cells.append(f'<td class="{cell_class}">{format_decimal(value)}</td>')
            else:
                cells.append("<td>—</td>")
        highest_class = "score-win" if highest not in ("—", "Tie") else ""
        body_rows.append(
            f'<tr><td class="metric-name">{label}</td>{"".join(cells)}'
            f'<td class="{highest_class}">{highest}</td></tr>'
        )

    return (
        '<div class="buddy-scorecard-wrap"><table class="buddy-scorecard">'
        f"<thead><tr><th>Metric</th>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def comparison_wins_html(result: dict[str, Any]) -> str:
    stocks = result.get("stocks", [])
    if len(stocks) < 2:
        return ""

    leader = comparison_overall_leader(result)
    wins = comparison_category_wins(result)
    cards: list[str] = []
    for symbol in stocks:
        name = format_stock_name(symbol)
        items = wins.get(symbol, [])
        leader_class = " leader" if symbol == leader else ""
        trophy = "🏆" if symbol == leader else "🏅"
        if items:
            items_html = "".join(f'<div class="buddy-wins-item">✓ {item}</div>' for item in items)
        else:
            items_html = '<div class="buddy-wins-item muted">Comparable across categories</div>'
        cards.append(
            f'<div class="buddy-wins-card{leader_class}">'
            f'<div class="buddy-wins-title">{trophy} {name}</div>{items_html}</div>'
        )

    columns = min(len(cards), 4)
    return f'<div class="buddy-wins-grid cols-{columns}">{"".join(cards)}</div>'


def _comparison_metric_table_html(
    title: str,
    icon_class: str,
    icon_char: str,
    metric_rows: list[tuple[str, list[str]]],
    stocks: list[str],
) -> str:
    headers = [format_stock_name(symbol) for symbol in stocks]
    head_html = "".join(f"<th>{name}</th>" for name in headers)
    body_rows: list[str] = []
    for metric, values in metric_rows:
        higher_better = not _lower_is_better_metric(metric)
        best_idxs = _best_value_indices(values, higher_is_better=higher_better)
        cells: list[str] = []
        for index, value in enumerate(values):
            tone = _value_tone_class(metric, value)
            if index in best_idxs:
                tone = "val-best"
            elif tone == "" and value.lower() in ("moderate", "sideways"):
                tone = "val-neutral"
            cells.append(f'<td class="{tone}">{value}</td>')
        body_rows.append(
            f'<tr><td class="metric-name">{metric}</td>{"".join(cells)}</tr>'
        )
    return (
        f'<div class="buddy-detail-card {icon_class}">'
        f'<div class="buddy-detail-title"><span class="buddy-detail-icon">{icon_char}</span>{title}</div>'
        '<div class="buddy-scorecard-wrap inner"><table class="buddy-scorecard compact">'
        f"<thead><tr><th>Metric</th>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div></div>"
    )


def comparison_fundamental_analysis_html(result: dict[str, Any]) -> str:
    stocks = result.get("stocks", [])
    val_segments = parse_symbol_segments(result.get("valuation_comparison", ""), stocks)
    growth_segments = parse_symbol_segments(result.get("growth_comparison", ""), stocks)
    rows: list[tuple[str, list[str]]] = []
    for metric, pattern, source in [
        ("P/E Ratio", r"P/?E\s*([\d.]+|unavailable)", val_segments),
        ("P/B Ratio", r"P/?B\s*([\d.]+|unavailable)", val_segments),
        ("Revenue Growth", r"revenue growth\s*([-\d.]+%?|unavailable)", growth_segments),
        ("Earnings Growth", r"earnings growth\s*([-\d.]+%?|unavailable)", growth_segments),
    ]:
        values = [_extract_metric(source.get(symbol, ""), pattern) for symbol in stocks]
        rows.append((metric, values))
    return _comparison_metric_table_html("Fundamental Analysis", "icon-fund", "◆", rows, stocks)


def comparison_technical_analysis_html(result: dict[str, Any]) -> str:
    stocks = result.get("stocks", [])
    tech_segments = parse_symbol_segments(result.get("technical_trend_comparison", ""), stocks)
    rows: list[tuple[str, list[str]]] = []

    trend_values = [
        _extract_metric(tech_segments.get(symbol, ""), r"(uptrend|downtrend|sideways)")
        for symbol in stocks
    ]
    if any(v != "—" for v in trend_values):
        rows.append(("Trend", [v.capitalize() if v != "—" else "—" for v in trend_values]))

    score_values = [
        _extract_metric(tech_segments.get(symbol, ""), r"technical score\s*(\d+)")
        for symbol in stocks
    ]
    if any(v != "—" for v in score_values):
        rows.append(("Technical Score", score_values))

    rsi_values = [
        _extract_metric(tech_segments.get(symbol, ""), r"rsi\s*([\d.]+|unavailable)")
        for symbol in stocks
    ]
    if any(v != "—" for v in rsi_values):
        rows.append(("RSI", rsi_values))

    setup_values: list[str] = []
    for symbol in stocks:
        score = result.get("technical_scores", {}).get(symbol)
        if isinstance(score, (int, float)):
            if score >= 70:
                setup_values.append("Strong")
            elif score >= 50:
                setup_values.append("Moderate")
            else:
                setup_values.append("Weak")
        else:
            setup_values.append("—")
    rows.append(("Setup Strength", setup_values))

    return _comparison_metric_table_html("Technical Analysis", "icon-tech", "↗", rows, stocks)


def comparison_sentiment_risk_html(result: dict[str, Any]) -> str:
    stocks = result.get("stocks", [])
    risk_segments = parse_symbol_segments(result.get("risk_comparison", ""), stocks)
    sentiment_values: list[str] = []
    for symbol in stocks:
        score = result.get("sentiment_scores", {}).get(symbol)
        sentiment_values.append(
            format_decimal(score) if isinstance(score, (int, float)) else "—"
        )

    rows: list[tuple[str, list[str]]] = [
        ("Sentiment Score", sentiment_values),
        (
            "Risk Adjustment",
            [
                _extract_metric(risk_segments.get(symbol, ""), r"risk adjustment\s*([\d.]+)")
                for symbol in stocks
            ],
        ),
        (
            "Flagged Risks",
            [
                _extract_metric(risk_segments.get(symbol, ""), r"(\d+)\s*flagged risks")
                for symbol in stocks
            ],
        ),
        (
            "Debt/Equity",
            [
                _extract_metric(risk_segments.get(symbol, ""), r"debt/equity\s*([\d.]+|unavailable)")
                for symbol in stocks
            ],
        ),
    ]
    return _comparison_metric_table_html("Sentiment & Risk", "icon-sentiment", "♡", rows, stocks)


def comparison_detailed_analysis_html(result: dict[str, Any]) -> str:
    return (
        '<div class="buddy-detail-grid">'
        + comparison_fundamental_analysis_html(result)
        + comparison_technical_analysis_html(result)
        + comparison_sentiment_risk_html(result)
        + "</div>"
    )


def ai_conclusion_cards_html(result: dict[str, Any]) -> str:
    winner = result.get("winner")
    stocks = result.get("stocks", [])
    preferred_name = format_stock_name(winner) if winner else "No clear winner"
    preferred_sub = (
        "Based on combined analyst scores."
        if winner
        else "Scores are too close to call."
    )
    assessment = sanitize_display_text(
        clean_display_text(result.get("relative_assessment", "Comparison complete.")),
        stocks,
    )
    return f"""
<div class="buddy-conclusion-grid">
  <div class="buddy-summary-card">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">✦</span> AI Conclusion</div>
    <div class="buddy-summary-body">{assessment}</div>
  </div>
  <div class="buddy-summary-card preferred">
    <div class="buddy-summary-head"><span class="buddy-detail-icon">◎</span> Preferred Setup</div>
    <div class="buddy-preferred-name">{preferred_name}</div>
    <div class="buddy-preferred-sub">{preferred_sub}</div>
  </div>
</div>
"""


def single_stock_summary_html(
    name: str,
    symbol: str,
    overall: float | None,
    rating: str,
) -> str:
    score_text = (
        f"{format_decimal(overall)} / 100"
        if isinstance(overall, (int, float))
        else "—"
    )
    score_class = score_tone_class(overall)
    badge_class = _rating_badge_class(rating)
    ticker = format_ticker(symbol)
    initial = name[0].upper() if name else "?"
    return f"""
<div class="buddy-stock-grid single">
  <div class="buddy-stock-card">
    <div class="buddy-stock-icon">{initial}</div>
    <div class="buddy-stock-name">{name}</div>
    <div class="buddy-stock-ticker">{ticker}</div>
    <div class="buddy-stock-score-label">Overall Score</div>
    <div class="buddy-stock-score {score_class}">{score_text}</div>
    <div class="buddy-rating-badge {badge_class}">{rating}</div>
  </div>
</div>
"""


def metric_summary_table_html(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<tr><td class="metric-name">{label}</td><td>{value}</td></tr>' for label, value in rows
    )
    return (
        '<div class="buddy-scorecard-wrap"><table class="buddy-scorecard">'
        f"<tbody>{body}</tbody></table></div>"
    )


def portfolio_summary_cards_html(result: dict[str, Any]) -> str:
    cards = [
        ("Portfolio Value", format_inr(result.get("total_current_value"))),
        ("Total P&L", format_inr(result.get("total_pnl"))),
        ("P&L %", format_percent(result.get("total_pnl_percent"), signed=True)),
        ("Portfolio Score", format_score(result.get("portfolio_score"))),
    ]
    html = "".join(
        f'<div class="buddy-mini-card"><div class="buddy-mini-label">{label}</div>'
        f'<div class="buddy-mini-value">{value}</div></div>'
        for label, value in cards
    )
    return f'<div class="buddy-mini-grid">{html}</div>'


def holdings_table_html(result: dict[str, Any]) -> str:
    rows = portfolio_holdings_rows(result)
    if not rows:
        return ""
    headers = list(rows[0].keys())
    head_html = "".join(f"<th>{header}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{row[header]}</td>" for header in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="buddy-scorecard-wrap"><table class="buddy-scorecard">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def ai_conclusion_hero_html(result: dict[str, Any]) -> str:
    """Alias for compare conclusion cards layout."""
    return ai_conclusion_cards_html(result)


def global_css() -> str:
    return """
<style>
    /* Hide sidebar completely */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

  /* Layout — compact dashboard */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        padding-left: 1.75rem;
        padding-right: 1.75rem;
        max-width: 1120px;
        margin: 0 auto;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Segmented tab navigation — marker styles the next Streamlit row */
    .buddy-nav-marker,
    .buddy-input-marker {
        display: none;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 0.25rem;
        margin: 0 auto 0.75rem auto;
        max-width: 580px;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        padding: 0 !important;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #8b9cb3 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 0.4rem !important;
        border-radius: 7px !important;
        box-shadow: none !important;
        border-bottom: 2px solid transparent !important;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        color: #c5d0de !important;
        background: rgba(255,255,255,0.03) !important;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: rgba(79,209,165,0.1) !important;
        color: #4fd1a5 !important;
        border-bottom: 2px solid #4fd1a5 !important;
    }
    div[data-testid="stMarkdown"]:has(.buddy-nav-marker) + div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"]:hover {
        background: rgba(79,209,165,0.14) !important;
        color: #4fd1a5 !important;
    }

    /* Input card — marker styles the next Streamlit row */
    div[data-testid="stMarkdown"]:has(.buddy-input-marker) + div[data-testid="stHorizontalBlock"],
    div[data-testid="stMarkdown"]:has(.buddy-input-marker) + div[data-testid="stVerticalBlock"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stMarkdown"]:has(.buddy-input-marker) + div label {
        font-size: 0.74rem !important;
        color: #8b9cb3 !important;
        font-weight: 600 !important;
        margin-bottom: 0.15rem !important;
    }
    div[data-testid="stMarkdown"]:has(.buddy-input-marker) + div .stTextInput {
        margin-bottom: 0 !important;
    }
    .buddy-btn-col {
        display: flex;
        align-items: flex-end;
        height: 100%;
        padding-bottom: 0.1rem;
    }
    .buddy-btn-col .stButton {
        width: 100%;
    }
    .buddy-action-label-spacer {
        display: block;
        font-size: 0.74rem;
        line-height: 1.25;
        min-height: 1.35rem;
        margin-bottom: 0.35rem;
        opacity: 0;
        pointer-events: none;
        user-select: none;
    }
    button[kind="primary"], div[data-testid="stFormSubmitButton"] > button {
        background: #e85d5d !important;
        border: none !important;
        color: #fff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.1rem !important;
    }
    button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #f06f6f !important;
    }

    .buddy-stock-grid {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .buddy-stock-grid.single { margin-top: 0.25rem; }
    .buddy-stock-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        text-align: center;
        min-width: 170px;
        flex: 1 1 170px;
        max-width: 240px;
    }
    .buddy-stock-icon {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(79,209,165,0.15);
        color: #4fd1a5;
        font-weight: 800;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.5rem auto;
    }
    .buddy-stock-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f3f6fb;
    }
    .buddy-stock-ticker {
        font-size: 0.72rem;
        color: #7d8da6;
        margin-top: 0.1rem;
        letter-spacing: 0.04em;
    }
    .buddy-stock-score-label {
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b9cb3;
        margin-top: 0.5rem;
    }
    .buddy-stock-score {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
    }
    .score-high { color: #4fd1a5; }
    .score-mid { color: #e8b339; }
    .score-low { color: #f56565; }
    .buddy-rating-badge {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-hold { background: rgba(232,179,57,0.15); color: #e8b339; border: 1px solid rgba(232,179,57,0.35); }
    .badge-buy { background: rgba(79,209,165,0.15); color: #4fd1a5; border: 1px solid rgba(79,209,165,0.35); }
    .badge-avoid { background: rgba(245,101,101,0.15); color: #f56565; border: 1px solid rgba(245,101,101,0.35); }
    .buddy-vs-badge {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: #8b9cb3;
        font-size: 0.68rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    /* Detailed analysis grid */
    .buddy-detail-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    @media (max-width: 1100px) {
        .buddy-detail-grid { grid-template-columns: 1fr; }
    }
    .buddy-detail-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 0.75rem 0.75rem 0.75rem;
    }
    .buddy-detail-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #eef2f7;
        margin-bottom: 0.5rem;
        padding-left: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .buddy-detail-icon {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .icon-fund .buddy-detail-title { color: #c4b5fd; }
    .icon-tech .buddy-detail-title { color: #4fd1a5; }
    .icon-sentiment .buddy-detail-title { color: #f687b3; }
    .buddy-scorecard.compact td, .buddy-scorecard.compact th {
        padding: 0.55rem 0.65rem;
        font-size: 0.82rem;
    }
    .buddy-scorecard-wrap.inner {
        margin-bottom: 0;
        padding: 0;
        border: none;
        background: transparent;
    }
    .val-positive { color: #4fd1a5 !important; font-weight: 600; }
    .val-negative { color: #f56565 !important; font-weight: 600; }
    .val-best { color: #4fd1a5 !important; font-weight: 700; }
    .val-neutral { color: #e8b339 !important; font-weight: 600; }

    /* Conclusion cards */
    .buddy-conclusion-grid {
        display: grid;
        grid-template-columns: 1.5fr 1fr;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    @media (max-width: 900px) {
        .buddy-conclusion-grid { grid-template-columns: 1fr; }
    }
    .buddy-summary-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.15rem;
    }
    .buddy-summary-card.preferred {
        border-color: rgba(196,181,253,0.25);
        background: rgba(196,181,253,0.05);
    }
    .buddy-summary-head {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8b9cb3;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .buddy-summary-body {
        color: #c5d0de;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Portfolio mini cards */
    .buddy-mini-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin-bottom: 1rem;
    }
    @media (max-width: 900px) {
        .buddy-mini-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .buddy-mini-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
    }
    .buddy-mini-label {
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8b9cb3;
        font-weight: 600;
    }
    .buddy-mini-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f4f8;
        margin-top: 0.25rem;
    }

    /* Wins grid responsive */
    .buddy-wins-grid.cols-2 { grid-template-columns: 1fr 1fr; }
    .buddy-wins-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
    .buddy-wins-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
    @media (max-width: 900px) {
        .buddy-wins-grid.cols-3, .buddy-wins-grid.cols-4 { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 600px) {
        .buddy-wins-grid { grid-template-columns: 1fr !important; }
        .buddy-stock-grid { flex-direction: column; }
        .buddy-vs-badge { margin: 0.25rem 0; }
    }
    .buddy-wins-item.muted { color: #7d8da6; }

    /* Hero — compact */
    .buddy-hero {
        text-align: center;
        padding: 0.15rem 0 0.5rem 0;
        margin-bottom: 0.15rem;
    }
    .buddy-hero-brand {
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #7d8da6;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .buddy-hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f3f6fb;
        line-height: 1.2;
        margin: 0;
    }
    .buddy-hero-sub {
        color: #9aa8bc;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }
    .buddy-hero-tags {
        color: #6f9fd8;
        font-size: 0.82rem;
    }

    /* Legacy tab styles (fallback) */
    div[data-testid="stTabs"] {
        width: 100%;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 0.25rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.92rem;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #4fd1a5 !important;
    }

    /* Section typography */
    .buddy-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f0f4f8;
        margin: 0 0 0.2rem 0;
    }
    .buddy-section-desc {
        color: #8b9cb3;
        font-size: 0.85rem;
        margin-bottom: 0.65rem;
        line-height: 1.45;
    }
    .buddy-panel-label {
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7d8da6;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }

    /* Cards & panels */
    .buddy-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .buddy-verdict {
        background: linear-gradient(145deg, rgba(79,209,165,0.1), rgba(255,255,255,0.02));
        border: 1px solid rgba(79,209,165,0.22);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }
    .buddy-verdict-rating {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.25rem 0;
    }
    .buddy-verdict-score {
        color: #8b9cb3;
        font-size: 0.88rem;
    }
    .buddy-verdict-text {
        color: #c5d0de;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.65rem;
    }
    .buddy-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 140px;
    }
    .buddy-card.compact {
        min-height: auto;
        padding: 0.85rem 1rem;
    }
    .buddy-card-bullets {
        list-style: none;
        padding: 0;
        margin: 0.4rem 0 0 0;
    }
    .buddy-card-bullets li {
        color: #b8c5d6;
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0.28rem 0;
    }
    .buddy-card-bullets .bullet-icon {
        display: inline-block;
        width: 1.1rem;
        font-weight: 700;
    }
    .buddy-decision-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .buddy-decision-rating {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0.35rem 0 0.5rem 0;
        letter-spacing: 0.04em;
    }
    .buddy-decision-summary {
        color: #c5d0de;
        font-size: 0.88rem;
        line-height: 1.5;
        max-width: 42rem;
        margin: 0 auto;
    }
    .buddy-decision-risk {
        color: #e8b339;
        font-size: 0.82rem;
        margin-top: 0.55rem;
        font-weight: 600;
    }
    .buddy-card-title {
        font-size: 0.75rem;
        color: #7d8da6;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .buddy-card-headline {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.4rem 0 0.2rem 0;
    }
    .buddy-card-score {
        color: #8b9cb3;
        font-size: 0.82rem;
    }
    .buddy-card-body {
        color: #b8c5d6;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.5rem;
    }
    .tone-positive { color: #4fd1a5; }
    .tone-negative { color: #f56565; }
    .tone-neutral { color: #e8b339; }
    .buddy-preferred-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: #c4b5fd;
        margin: 0.25rem 0;
    }
    .buddy-preferred-sub {
        color: #8b9cb3;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .buddy-disclaimer {
        color: #6b7c93;
        font-size: 0.72rem;
        margin-top: 1.25rem;
        text-align: center;
    }
    .buddy-input-hint {
        color: #6d7d94;
        font-size: 0.72rem;
        margin-top: 0.15rem;
        line-height: 1.3;
    }
    .buddy-input-hint.compact {
        margin-top: 0.1rem;
        margin-bottom: 0;
    }

    /* Compare scorecard */
    .buddy-scorecard-wrap {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 1rem;
        overflow-x: auto;
    }
    .buddy-scorecard {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .buddy-scorecard th {
        color: #8b9cb3;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.75rem 1rem;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .buddy-scorecard th:first-child { text-align: left; }
    .buddy-scorecard td {
        padding: 0.7rem 1rem;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #d5dde8;
    }
    .buddy-scorecard td.metric-name {
        text-align: left;
        font-weight: 600;
        color: #eef2f7;
    }
    .buddy-scorecard td.score-win {
        color: #4fd1a5;
        font-weight: 700;
    }
    .buddy-compare-hero {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .buddy-compare-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f3f6fb;
    }
    .buddy-compare-score {
        font-size: 1.55rem;
        font-weight: 800;
        color: #4fd1a5;
        margin: 0.35rem 0;
    }
    .buddy-compare-score-label {
        font-size: 0.78rem;
        color: #8b9cb3;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .buddy-wins-grid {
        display: grid;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .buddy-wins-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .buddy-wins-card.leader {
        border-color: rgba(232,179,57,0.45);
        background: rgba(232,179,57,0.06);
        box-shadow: 0 0 0 1px rgba(232,179,57,0.12);
    }
    .buddy-wins-title {
        font-weight: 700;
        color: #f0f4f8;
        margin-bottom: 0.5rem;
    }
    .buddy-wins-item {
        color: #b8c5d6;
        font-size: 0.88rem;
        margin: 0.2rem 0;
    }
    .buddy-ai-hero {
        background: linear-gradient(145deg, rgba(79,209,165,0.12), rgba(255,255,255,0.02));
        border: 1px solid rgba(79,209,165,0.28);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0;
    }
    .buddy-ai-hero-text {
        color: #d5dde8;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-top: 0.5rem;
    }
    .buddy-compare-insight {
        color: #b8c5d6;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .buddy-compare-insight strong {
        color: #e8edf4;
    }

    /* Streamlit widget polish */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.65rem 0.75rem;
    }
</style>
"""


def action_button_label_spacer() -> str:
    """Invisible spacer matching Streamlit text-input label height for column alignment."""
    return '<div class="buddy-action-label-spacer" aria-hidden="true">&nbsp;</div>'


def section_panel_html(label: str, content_html: str) -> str:
    return f"""
<div class="buddy-panel">
  <div class="buddy-panel-label">{label}</div>
  {content_html}
</div>
"""


_BULLET_PREFIX_RE = re.compile(
    r"^(mock\s+)?(summary|strength|weakness|risk|momentum|volatility|catalyst|agreement|event)\s*:\s*",
    re.IGNORECASE,
)
_BOILERPLATE_PHRASES = (
    "based on provided",
    "based on the provided",
    "according to provided",
    "no summary available",
)
_POSITIVE_HINTS = (
    "positive",
    "strong",
    "growth",
    "support",
    "uptrend",
    "bullish",
    "reasonable",
    "solid",
    "improve",
    "above",
    "aligned",
    "stable",
    "favorable",
)
_RISK_HINTS = (
    "risk",
    "concern",
    "weak",
    "volatile",
    "volatility",
    "down",
    "pressure",
    "stretched",
    "elevated",
    "headwind",
    "negative",
    "caution",
    "mixed",
    "monitor",
)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BOILERPLATE_PHRASES)


def _clean_bullet_text(text: str) -> str:
    cleaned = _BULLET_PREFIX_RE.sub("", text.strip())
    cleaned = cleaned.strip(" -•")
    if not cleaned or _is_boilerplate(cleaned):
        return ""
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _format_sma_periods(periods: list[int]) -> str:
    if len(periods) == 1:
        return f"{periods[0]}-day SMA"
    if len(periods) == 2:
        return f"{periods[0]}-day and {periods[1]}-day SMAs"
    return ", ".join(f"{period}-day" for period in periods[:-1]) + f", and {periods[-1]}-day SMAs"


def _summarize_sma_bullet(text: str) -> str | None:
    if "sma" not in text.lower():
        return None
    periods = sorted({int(match) for match in re.findall(r"(\d+)-day\s+SMA", text, re.I)})
    if not periods:
        return None

    price_match = re.search(
        r"price\s*(?:\(([\d.]+)\)|(?:is|sits)?\s*(?:at|around)?\s*([\d.]+))",
        text,
        re.I,
    )
    price = price_match.group(1) or price_match.group(2) if price_match else None
    direction = (
        "above"
        if re.search(r"\babove\b", text, re.I)
        else "below"
        if re.search(r"\bbelow\b", text, re.I)
        else "relative to"
    )
    sma_phrase = _format_sma_periods(periods)
    if price:
        return f"Price ({price}) is {direction} the {sma_phrase}."
    return f"Price is {direction} the {sma_phrase}."


def _summarize_atr_bullet(text: str) -> str | None:
    if "atr" not in text.lower():
        return None
    atr_match = re.search(r"ATR\s*(?:\d+\s*)?(?:is\s*)?([\d.]+)", text, re.I)
    if not atr_match:
        return None

    period_match = re.search(r"ATR\s*(\d+)", text, re.I)
    period = period_match.group(1) if period_match else "14"
    pct_match = re.search(
        r"([\d.]+)%\s*(?:of\s*(?:the\s*)?current\s*price|of\s*price)",
        text,
        re.I,
    )
    pct_part = (
        f", around {pct_match.group(1)}% of the current price" if pct_match else ""
    )

    lowered = text.lower()
    if "moderate" in lowered:
        severity = "moderate"
    elif "elevated" in lowered or "high volatility" in lowered:
        severity = "elevated"
    elif "low" in lowered:
        severity = "low"
    else:
        severity = "moderate"

    return (
        f"ATR {period} is {atr_match.group(1)}{pct_part}, "
        f"indicating {severity} price volatility."
    )


def _summarize_bullet_text(text: str) -> str:
    """Rewrite long analyst fragments into complete concise bullets."""
    cleaned = _clean_bullet_text(text)
    if not cleaned:
        return ""

    for summarizer in (_summarize_sma_bullet, _summarize_atr_bullet):
        summary = summarizer(cleaned)
        if summary:
            return summary

    sentences = _split_sentences(cleaned)
    if sentences:
        return _clean_bullet_text(sentences[0])

    return cleaned


def _bullet_marker(text: str, forced: str | None = None) -> str:
    if forced == "positive":
        return "✓"
    if forced == "risk":
        return "⚠"
    lowered = text.lower()
    if any(hint in lowered for hint in _RISK_HINTS):
        return "⚠"
    if any(hint in lowered for hint in _POSITIVE_HINTS):
        return "✓"
    return "✓"


def build_analyst_bullets(
    *,
    positives: list[str] | None = None,
    risks: list[str] | None = None,
    summary: str | None = None,
    max_bullets: int = 3,
) -> list[tuple[str, str]]:
    """Build concise analyst bullets from structured backend fields."""
    bullets: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(marker: str, raw: str) -> None:
        if len(bullets) >= max_bullets:
            return
        text = _summarize_bullet_text(raw)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            bullets.append((marker, text))

    for item in positives or []:
        _add(_bullet_marker(item, "positive"), item)
    for item in risks or []:
        _add(_bullet_marker(item, "risk"), item)

    if len(bullets) < max_bullets and summary:
        for sentence in _split_sentences(summary):
            _add(_bullet_marker(sentence), sentence)

    return bullets[:max_bullets]


def fundamental_view_bullets(fundamental: dict[str, Any]) -> list[tuple[str, str]]:
    weaknesses = list(fundamental.get("weaknesses") or [])
    weaknesses.extend(fundamental.get("risks") or [])
    return build_analyst_bullets(
        positives=fundamental.get("strengths"),
        risks=weaknesses,
        summary=fundamental.get("summary"),
    )


def technical_view_bullets(technical: dict[str, Any]) -> list[tuple[str, str]]:
    positives: list[str] = []
    risks: list[str] = []
    trend = technical.get("trend")
    if trend:
        trend_text = str(trend)
        if any(word in trend_text.lower() for word in ("down", "weak", "bear")):
            risks.append(f"Trend is {trend_text}")
        else:
            positives.append(f"Trend is {trend_text}")
    if technical.get("momentum"):
        positives.append(str(technical["momentum"]))
    if technical.get("volatility"):
        risks.append(str(technical["volatility"]))
    return build_analyst_bullets(
        positives=positives,
        risks=risks,
        summary=technical.get("summary"),
    )


def sentiment_view_bullets(sentiment: dict[str, Any]) -> list[tuple[str, str]]:
    return build_analyst_bullets(
        positives=sentiment.get("positive_catalysts"),
        risks=sentiment.get("negative_catalysts"),
        summary=sentiment.get("summary"),
    )


def _short_summary(text: str | None, max_sentences: int = 2) -> str:
    if not text:
        return "Analysis complete."
    sentences = [
        _clean_bullet_text(sentence)
        for sentence in _split_sentences(text)
    ]
    sentences = [sentence for sentence in sentences if sentence and not _is_boilerplate(sentence)]
    if not sentences:
        cleaned = _clean_bullet_text(text)
        return cleaned or "Analysis complete."
    return " ".join(sentences[:max_sentences])


def crisp_analyst_card_html(
    title: str,
    headline: str | None,
    score: float | None,
    bullets: list[tuple[str, str]],
    tone: str = "neutral",
) -> str:
    headline_text = headline or "—"
    score_text = format_score(score) if score is not None else "—"
    if bullets:
        bullet_html = "".join(
            f'<li><span class="bullet-icon">{marker}</span> {text}</li>'
            for marker, text in bullets
        )
        body = f'<ul class="buddy-card-bullets">{bullet_html}</ul>'
    else:
        body = '<div class="buddy-card-body">No highlights available.</div>'
    return f"""
<div class="buddy-card compact">
  <div class="buddy-card-title">{title}</div>
  <div class="buddy-card-headline tone-{tone}">{headline_text}</div>
  <div class="buddy-card-score">Score: {score_text}</div>
  {body}
</div>
"""


def analyze_conclusion_card_html(
    rating: str,
    narrative: str | None,
    risks: list[str] | None = None,
    key_reasons: list[str] | None = None,
) -> str:
    """Compact decision card for single-stock AI conclusion."""
    summary = _short_summary(narrative, max_sentences=2)
    key_risk = ""
    for candidate in (risks or []) + (key_reasons or []):
        cleaned = _summarize_bullet_text(candidate)
        if cleaned:
            key_risk = cleaned
            break
    risk_html = (
        f'<div class="buddy-decision-risk">Key risk: {key_risk}</div>'
        if key_risk
        else ""
    )
    tone = rating_tone(rating)
    return f"""
<div class="buddy-decision-card">
  <div class="buddy-summary-head"><span class="buddy-detail-icon">✦</span> AI Conclusion</div>
  <div class="buddy-decision-rating tone-{tone}">{rating}</div>
  <div class="buddy-decision-summary">{summary}</div>
  {risk_html}
</div>
"""


def analyst_card_html(
    title: str,
    headline: str | None,
    score: float | None,
    summary: str | None,
    tone: str = "neutral",
) -> str:
    headline_text = headline or "—"
    score_text = format_score(score) if score is not None else "—"
    summary_text = summary or "No summary available."
    return f"""
<div class="buddy-card">
  <div class="buddy-card-title">{title}</div>
  <div class="buddy-card-headline tone-{tone}">{headline_text}</div>
  <div class="buddy-card-score">Score: {score_text}</div>
  <div class="buddy-card-body">{summary_text}</div>
</div>
"""
