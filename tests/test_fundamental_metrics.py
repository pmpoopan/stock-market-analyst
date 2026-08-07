"""Tests for FundamentalMetricsCalculator — no API or LLM calls."""

import pandas as pd
import pytest

from app.analysis.fundamental_metrics import FundamentalMetricsCalculator
from tests.fixtures.fundamental_data import make_mock_fundamental_raw_data


def test_compute_from_info_dict():
    metrics = FundamentalMetricsCalculator.compute(make_mock_fundamental_raw_data())

    assert metrics.symbol == "RELIANCE.NS"
    assert metrics.revenue == 2_500_000_000_000
    assert metrics.revenue_growth == pytest.approx(12.0)
    assert metrics.pe_ratio == pytest.approx(22.5)
    assert metrics.roe == pytest.approx(14.0)
    assert metrics.debt_to_equity == pytest.approx(0.35)
    assert metrics.earnings_growth == pytest.approx(15.0)


def test_revenue_growth_calculation():
    growth = FundamentalMetricsCalculator.revenue_growth(120.0, 100.0)
    assert growth == pytest.approx(20.0)


def test_roe_calculation():
    roe = FundamentalMetricsCalculator.roe(20.0, 100.0)
    assert roe == pytest.approx(20.0)


def test_roce_calculation():
    roce = FundamentalMetricsCalculator.roce(15.0, 100.0)
    assert roce == pytest.approx(15.0)


def test_compute_from_income_statement():
    income = pd.DataFrame(
        {
            "2025-03-31": [2_800_000_000_000, 400_000_000_000, 200_000_000_000],
            "2024-03-31": [2_500_000_000_000, 350_000_000_000, 170_000_000_000],
        },
        index=["Total Revenue", "EBIT", "Net Income"],
    )
    balance = pd.DataFrame(
        {
            "2025-03-31": [1_500_000_000_000, 500_000_000_000, 300_000_000_000],
            "2024-03-31": [1_400_000_000_000, 480_000_000_000, 280_000_000_000],
        },
        index=["Total Assets", "Total Stockholder Equity", "Current Liabilities"],
    )
    cashflow = pd.DataFrame(
        {"2025-03-31": [130_000_000_000], "2024-03-31": [110_000_000_000]},
        index=["Free Cash Flow"],
    )

    raw = {
        "symbol": "RELIANCE.NS",
        "info": {},
        "income_stmt": income,
        "balance_sheet": balance,
        "cashflow": cashflow,
        "data_sources": ["yahoo_income_statement"],
    }
    metrics = FundamentalMetricsCalculator.compute(raw)

    assert metrics.revenue == 2_800_000_000_000
    assert metrics.revenue_growth == pytest.approx(12.0)
    assert metrics.net_profit == 200_000_000_000
    assert metrics.roe == pytest.approx(40.0)
    assert metrics.roce == pytest.approx(33.3333, rel=1e-3)
    assert metrics.free_cash_flow == 130_000_000_000
