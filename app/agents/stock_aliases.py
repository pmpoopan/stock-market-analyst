"""Known Indian equity aliases and canonical symbol resolution."""

from __future__ import annotations

import re

INDIAN_STOCK_ALIASES: dict[str, str] = {
    "reliance industries": "RELIANCE.NS",
    "reliance": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS",
    "mahindra and mahindra": "M&M.NS",
    "mahindra": "M&M.NS",
    "m&m": "M&M.NS",
    "infosys": "INFY.NS",
    "infosys limited": "INFY.NS",
    "infy": "INFY.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfcbank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icicibank": "ICICIBANK.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "wipro": "WIPRO.NS",
    "itc": "ITC.NS",
    "sbi": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "sbin": "SBIN.NS",
    "larsen and toubro": "LT.NS",
    "larsen & toubro": "LT.NS",
    "l&t": "LT.NS",
    "lt": "LT.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani": "ADANIENT.NS",
}

COMPARE_KEYWORDS = ("compare", "versus", " vs ", " vs.", "which is better")
PORTFOLIO_KEYWORDS = ("portfolio", "my holdings", "my stocks")

_CORPORATE_SUFFIX_RE = re.compile(
    r"\s+(limited|ltd\.?|inc\.?|corporation|corp\.?)$",
    re.IGNORECASE,
)
_EXCHANGE_SUFFIX_RE = re.compile(r"\.(NS|BO)$", re.IGNORECASE)

_DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "M&M.NS": "Mahindra",
    "RELIANCE.NS": "Reliance",
    "INFY.NS": "Infosys",
    "TATAMOTORS.NS": "Tata Motors",
    "TCS.NS": "TCS",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "BHARTIARTL.NS": "Airtel",
    "LT.NS": "L&T",
    "ADANIENT.NS": "Adani",
    "SBIN.NS": "SBI",
}


def _normalize_lookup_key(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = _CORPORATE_SUFFIX_RE.sub("", text).strip()
    return text


def _canonical_exchange_symbol(text: str) -> str | None:
    """Return normalized symbol when input already includes .NS or .BO."""
    match = _EXCHANGE_SUFFIX_RE.search(text)
    if not match:
        return None
    base = text[: match.start()].strip().upper()
    if not base:
        return None
    suffix = match.group(1).upper()
    return f"{base}.{suffix}"


def resolve_symbol(raw: str) -> str:
    """Map user input (company name or ticker) to a canonical Yahoo Finance NSE symbol."""
    text = raw.strip()
    if not text:
        return ""

    exchange_symbol = _canonical_exchange_symbol(text)
    if exchange_symbol:
        return exchange_symbol

    lookup = _normalize_lookup_key(text)
    if lookup in INDIAN_STOCK_ALIASES:
        return INDIAN_STOCK_ALIASES[lookup]

    for alias, symbol in sorted(INDIAN_STOCK_ALIASES.items(), key=lambda item: -len(item[0])):
        if lookup == alias or lookup.startswith(f"{alias} "):
            return symbol

    ticker = text.upper().strip()
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker


def _build_display_name_map() -> dict[str, str]:
    by_symbol: dict[str, list[str]] = {}
    for alias, symbol in INDIAN_STOCK_ALIASES.items():
        by_symbol.setdefault(symbol.upper(), []).append(alias)

    display = dict(_DISPLAY_NAME_OVERRIDES)
    for symbol, aliases in by_symbol.items():
        if symbol in display:
            continue
        spaced = [alias for alias in aliases if " " in alias]
        pick = spaced[0] if spaced else aliases[0]
        display[symbol] = pick.title()
    return display


_SYMBOL_DISPLAY_MAP = _build_display_name_map()


def display_name(symbol: str) -> str:
    """Human-readable company name for a Yahoo Finance symbol."""
    sym = symbol.upper().strip()
    if sym in _SYMBOL_DISPLAY_MAP:
        return _SYMBOL_DISPLAY_MAP[sym]
    base = sym.replace(".NS", "").replace(".BO", "")
    if base == "M&M":
        return "Mahindra"
    return base
