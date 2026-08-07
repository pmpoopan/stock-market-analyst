"""Mock fundamental data for tests — no live API or LLM calls."""

from app.models.schemas import FinancialMetrics

MOCK_FUNDAMENTAL_INFO: dict = {
    "shortName": "Reliance Industries Limited",
    "longName": "Reliance Industries Limited",
    "currency": "INR",
    "sector": "Energy",
    "industry": "Oil & Gas Refining & Marketing",
    "totalRevenue": 2_500_000_000_000,
    "revenueGrowth": 0.12,
    "ebitda": 450_000_000_000,
    "netIncomeToCommon": 180_000_000_000,
    "trailingEps": 28.5,
    "trailingPE": 22.5,
    "priceToBook": 2.4,
    "returnOnEquity": 0.14,
    "debtToEquity": 0.35,
    "freeCashflow": 120_000_000_000,
    "dividendYield": 0.0035,
    "earningsGrowth": 0.15,
    "marketCap": 9_800_000_000_000,
}


def make_mock_financial_metrics(symbol: str = "RELIANCE.NS") -> FinancialMetrics:
    return FinancialMetrics(
        symbol=symbol.upper(),
        revenue=2_500_000_000_000,
        revenue_growth=12.0,
        ebitda=450_000_000_000,
        ebit=400_000_000_000,
        net_profit=180_000_000_000,
        eps=28.5,
        pe_ratio=22.5,
        pb_ratio=2.4,
        roe=14.0,
        roce=16.5,
        debt_to_equity=0.35,
        free_cash_flow=120_000_000_000,
        dividend_yield=0.35,
        market_cap=9_800_000_000_000,
        earnings_growth=15.0,
        extra={
            "sector": "Energy",
            "industry": "Oil & Gas Refining & Marketing",
            "currency": "INR",
            "data_sources": ["yahoo_finance_info"],
        },
    )


def make_mock_fundamental_raw_data(symbol: str = "RELIANCE.NS") -> dict:
    return {
        "symbol": symbol.upper(),
        "info": MOCK_FUNDAMENTAL_INFO.copy(),
        "income_stmt": None,
        "balance_sheet": None,
        "cashflow": None,
        "data_sources": ["yahoo_finance_info"],
    }
