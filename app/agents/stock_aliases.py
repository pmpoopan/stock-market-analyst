"""Known Indian equity aliases for deterministic query parsing."""

INDIAN_STOCK_ALIASES: dict[str, str] = {
    "reliance industries": "RELIANCE.NS",
    "reliance": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS",
    "mahindra and mahindra": "M&M.NS",
    "mahindra": "M&M.NS",
    "m&m": "M&M.NS",
    "infosys": "INFY.NS",
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
    "larsen and toubro": "LT.NS",
    "l&t": "LT.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani": "ADANIENT.NS",
}

COMPARE_KEYWORDS = ("compare", "versus", " vs ", " vs.", "which is better")
PORTFOLIO_KEYWORDS = ("portfolio", "my holdings", "my stocks")
