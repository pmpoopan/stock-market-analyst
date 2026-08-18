"""Tests for canonical Indian equity symbol resolution."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.stock_aliases import display_name, resolve_symbol
from app.main import create_app


@pytest.mark.parametrize(
    "user_input,expected",
    [
        ("Infosys", "INFY.NS"),
        ("infosys", "INFY.NS"),
        ("INFOSYS", "INFY.NS"),
        ("Infy", "INFY.NS"),
        ("infy", "INFY.NS"),
        ("INFY", "INFY.NS"),
        ("INFY.NS", "INFY.NS"),
        ("infy.ns", "INFY.NS"),
        ("Infosys Limited", "INFY.NS"),
        ("Reliance", "RELIANCE.NS"),
        ("reliance", "RELIANCE.NS"),
        ("RELIANCE", "RELIANCE.NS"),
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("Tata Motors", "TATAMOTORS.NS"),
        ("Tata motors", "TATAMOTORS.NS"),
        ("TATAMOTORS", "TATAMOTORS.NS"),
        ("TATAMOTORS.NS", "TATAMOTORS.NS"),
        ("HDFC Bank", "HDFCBANK.NS"),
        ("HDFC BANK", "HDFCBANK.NS"),
        ("HDFCBANK", "HDFCBANK.NS"),
        ("HDFCBANK.NS", "HDFCBANK.NS"),
    ],
)
def test_resolve_symbol_company_names_and_tickers(user_input: str, expected: str) -> None:
    assert resolve_symbol(user_input) == expected


def test_display_name_maps_canonical_symbols() -> None:
    assert display_name("INFY.NS") == "Infosys"
    assert display_name("RELIANCE.NS") == "Reliance"


def test_normalize_holdings_resolves_infosys_from_frontend_module() -> None:
    """Portfolio path must resolve company names even when imported like Streamlit does."""
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    frontend_path = str(frontend_dir)
    if frontend_path not in sys.path:
        sys.path.insert(0, frontend_path)

    from ui_helpers import normalize_holdings

    holdings = normalize_holdings(
        [{"symbol": "Infosys", "quantity": 10, "buy_price": 1500}]
    )
    assert holdings == [{"symbol": "INFY.NS", "quantity": 10.0, "buy_price": 1500.0}]


def test_portfolio_api_resolves_infosys_holding(graph_deps) -> None:
    from unittest.mock import MagicMock, patch

    from app.graph.workflow import AnalysisOrchestrator

    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    mock_container = MagicMock()
    mock_container.orchestrator = orchestrator

    with patch("app.api.routes.get_container", return_value=mock_container):
        client = TestClient(create_app())
        response = client.post(
            "/api/portfolio",
            json={
                "holdings": [
                    {"symbol": "Infosys", "quantity": 10, "buy_price": 1500},
                ]
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["holdings"][0]["holding"]["symbol"] == "INFY.NS"
