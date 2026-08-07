"""Deterministic fundamental metric calculations.

Raw financial data is fetched by the data layer; derived ratios are computed here.
"""

from __future__ import annotations

from typing import Any

from app.models.schemas import FinancialMetrics


class FundamentalMetricsCalculator:
    """Compute and enrich fundamental metrics from raw provider data."""

    @staticmethod
    def compute(raw_data: dict[str, Any]) -> FinancialMetrics:
        """Transform raw provider data into a structured FinancialMetrics object."""
        symbol = raw_data.get("symbol", "")
        info = raw_data.get("info") or {}
        income_stmt = raw_data.get("income_stmt")
        balance_sheet = raw_data.get("balance_sheet")
        cashflow = raw_data.get("cashflow")

        revenue, revenue_prev = FundamentalMetricsCalculator._stmt_latest_two(
            income_stmt,
            ["Total Revenue", "Revenue", "Total Revenues"],
        )
        ebit, _ = FundamentalMetricsCalculator._stmt_latest_two(
            income_stmt,
            ["EBIT", "Operating Income"],
        )
        net_profit, net_prev = FundamentalMetricsCalculator._stmt_latest_two(
            income_stmt,
            ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"],
        )
        equity, _ = FundamentalMetricsCalculator._stmt_latest_two(
            balance_sheet,
            [
                "Total Stockholder Equity",
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity",
            ],
        )
        total_assets, _ = FundamentalMetricsCalculator._stmt_latest_two(
            balance_sheet,
            ["Total Assets"],
        )
        current_liabilities, _ = FundamentalMetricsCalculator._stmt_latest_two(
            balance_sheet,
            ["Current Liabilities", "Total Current Liabilities"],
        )
        fcf_stmt, _ = FundamentalMetricsCalculator._stmt_latest_two(
            cashflow,
            ["Free Cash Flow", "Free Cashflow"],
        )

        revenue = revenue or FundamentalMetricsCalculator._info_float(
            info, ["totalRevenue", "revenue"]
        )
        ebitda = FundamentalMetricsCalculator._info_float(info, ["ebitda", "EBITDA"])
        ebit = ebit or FundamentalMetricsCalculator._info_float(info, ["ebit", "EBIT"])
        net_profit = net_profit or FundamentalMetricsCalculator._info_float(
            info,
            ["netIncomeToCommon", "netIncome", "Net Income"],
        )
        eps = FundamentalMetricsCalculator._info_float(
            info, ["trailingEps", "epsTrailingTwelveMonths", "epsCurrentYear"]
        )
        pe_ratio = FundamentalMetricsCalculator._info_float(
            info, ["trailingPE", "forwardPE", "priceEpsCurrentYear"]
        )
        pb_ratio = FundamentalMetricsCalculator._info_float(info, ["priceToBook"])
        roe_info = FundamentalMetricsCalculator._info_float(info, ["returnOnEquity"])
        debt_to_equity = FundamentalMetricsCalculator._info_float(info, ["debtToEquity"])
        free_cash_flow = fcf_stmt or FundamentalMetricsCalculator._info_float(
            info, ["freeCashflow", "freeCashFlow"]
        )
        dividend_yield = FundamentalMetricsCalculator._normalize_yield(
            FundamentalMetricsCalculator._info_float(info, ["dividendYield", "yield"])
        )
        market_cap = FundamentalMetricsCalculator._info_float(info, ["marketCap"])
        earnings_growth = FundamentalMetricsCalculator._normalize_growth(
            FundamentalMetricsCalculator._info_float(
                info,
                [
                    "earningsGrowth",
                    "earningsQuarterlyGrowth",
                    "earningsAnnualGrowth",
                ],
            )
        )

        revenue_growth = FundamentalMetricsCalculator.revenue_growth(revenue, revenue_prev)
        if revenue_growth is None:
            revenue_growth = FundamentalMetricsCalculator._normalize_growth(
                FundamentalMetricsCalculator._info_float(info, ["revenueGrowth"])
            )

        if earnings_growth is None and net_profit is not None and net_prev is not None:
            earnings_growth = FundamentalMetricsCalculator.revenue_growth(net_profit, net_prev)

        roe = roe_info
        if roe is not None and abs(roe) <= 1:
            roe = roe * 100
        if roe is None:
            roe = FundamentalMetricsCalculator.roe(net_profit, equity)

        capital_employed = None
        if total_assets is not None and current_liabilities is not None:
            capital_employed = total_assets - current_liabilities
        roce = FundamentalMetricsCalculator.roce(ebit, capital_employed)

        extra = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": info.get("currency"),
            "book_value": FundamentalMetricsCalculator._info_float(
                info, ["bookValue", "bookValuePerShare"]
            ),
            "profit_margins": FundamentalMetricsCalculator._info_float(
                info, ["profitMargins", "operatingMargins"]
            ),
            "data_sources": raw_data.get("data_sources", ["yahoo_finance_info"]),
        }

        return FinancialMetrics(
            symbol=symbol,
            revenue=revenue,
            revenue_growth=revenue_growth,
            ebitda=ebitda,
            ebit=ebit,
            net_profit=net_profit,
            eps=eps,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            roe=roe,
            roce=roce,
            debt_to_equity=debt_to_equity,
            free_cash_flow=free_cash_flow,
            dividend_yield=dividend_yield,
            market_cap=market_cap,
            earnings_growth=earnings_growth,
            extra=extra,
        )

    @staticmethod
    def revenue_growth(current: float | None, previous: float | None) -> float | None:
        if current is None or previous is None or previous == 0:
            return None
        return round(((current - previous) / abs(previous)) * 100, 4)

    @staticmethod
    def roe(net_income: float | None, equity: float | None) -> float | None:
        if net_income is None or equity is None or equity == 0:
            return None
        return round((net_income / equity) * 100, 4)

    @staticmethod
    def roce(ebit: float | None, capital_employed: float | None) -> float | None:
        if ebit is None or capital_employed is None or capital_employed == 0:
            return None
        return round((ebit / capital_employed) * 100, 4)

    @staticmethod
    def _info_float(info: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            value = info.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _normalize_growth(value: float | None) -> float | None:
        if value is None:
            return None
        if abs(value) <= 1:
            return round(value * 100, 4)
        return round(value, 4)

    @staticmethod
    def _normalize_yield(value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 1:
            return round(value * 100, 4)
        return round(value, 4)

    @staticmethod
    def _stmt_latest_two(
        statement: Any,
        row_names: list[str],
    ) -> tuple[float | None, float | None]:
        if statement is None:
            return None, None

        try:
            if hasattr(statement, "empty") and statement.empty:
                return None, None
        except Exception:
            return None, None

        for name in row_names:
            try:
                if name not in statement.index:
                    continue
                series = statement.loc[name]
                values = []
                for col in series.index:
                    val = series[col]
                    if val is not None and str(val) != "nan":
                        try:
                            values.append(float(val))
                        except (TypeError, ValueError):
                            continue
                if not values:
                    return None, None
                latest = values[0]
                previous = values[1] if len(values) > 1 else None
                return latest, previous
            except Exception:
                continue

        return None, None
