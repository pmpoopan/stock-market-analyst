"""Query parser — classifies user intent and extracts entities.

Deterministic parsing where possible; LLM fallback reserved for later phases.
"""

from __future__ import annotations

import re

from app.agents.llm_client import LLMClient
from app.agents.stock_aliases import COMPARE_KEYWORDS, INDIAN_STOCK_ALIASES, PORTFOLIO_KEYWORDS
from app.models.schemas import ParsedQuery, PortfolioHolding, QueryIntent

SYMBOL_SUFFIX_PATTERN = re.compile(r"\b([A-Za-z&]{2,15})\.(NS|BO)\b", re.IGNORECASE)


class QueryParser:
    """Parse natural language or structured input into a ParsedQuery."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def parse(self, query: str) -> ParsedQuery:
        """Classify intent and extract stock symbols from a natural language query."""
        raw = query.strip()
        if not raw:
            raise ValueError("Query cannot be empty")

        lower = raw.lower()

        if any(keyword in lower for keyword in PORTFOLIO_KEYWORDS):
            return ParsedQuery(
                raw_query=raw,
                intent=QueryIntent.ANALYZE_PORTFOLIO,
                stocks=[],
            )

        stocks = self._extract_symbols(lower)

        if any(keyword in lower for keyword in COMPARE_KEYWORDS):
            if len(stocks) < 2:
                raise ValueError("Comparison queries require at least two stock symbols")
            return ParsedQuery(
                raw_query=raw,
                intent=QueryIntent.COMPARE_STOCKS,
                stocks=stocks,
            )

        if not stocks:
            raise ValueError(f"Could not identify a stock symbol in query: {raw}")

        return ParsedQuery(
            raw_query=raw,
            intent=QueryIntent.ANALYZE_STOCK,
            stocks=[stocks[0]],
        )

    def parse_compare(self, stocks: list[str]) -> ParsedQuery:
        """Build a comparison query from explicit symbol list."""
        normalized = [self._normalize_symbol(s) for s in stocks]
        if len(normalized) < 2:
            raise ValueError("At least two stocks are required for comparison")
        return ParsedQuery(
            raw_query=f"Compare {', '.join(normalized)}",
            intent=QueryIntent.COMPARE_STOCKS,
            stocks=normalized,
        )

    def parse_portfolio(self, holdings: list[PortfolioHolding]) -> ParsedQuery:
        """Build a portfolio analysis query from holdings."""
        if not holdings:
            raise ValueError("Portfolio must contain at least one holding")
        return ParsedQuery(
            raw_query="Analyze portfolio",
            intent=QueryIntent.ANALYZE_PORTFOLIO,
            portfolio=holdings,
            stocks=[h.symbol.upper() for h in holdings],
        )

    def _extract_symbols(self, text: str) -> list[str]:
        found: list[str] = []

        for alias, symbol in sorted(INDIAN_STOCK_ALIASES.items(), key=lambda item: -len(item[0])):
            if alias in text and symbol not in found:
                found.append(symbol)

        for match in SYMBOL_SUFFIX_PATTERN.finditer(text):
            symbol = f"{match.group(1).upper()}.{match.group(2).upper()}"
            if symbol not in found:
                found.append(symbol)

        return found

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        symbol = symbol.strip().upper()
        if "." not in symbol:
            return f"{symbol}.NS"
        return symbol
