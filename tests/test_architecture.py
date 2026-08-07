"""Architecture-level tests — verify skeleton imports and health endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.graph.workflow import AnalysisOrchestrator
from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Buddy"


def test_analyze_endpoint(graph_deps):
    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    mock_container = MagicMock()
    mock_container.orchestrator = orchestrator

    with patch("app.api.routes.get_container", return_value=mock_container):
        client = TestClient(create_app())
        response = client.post("/api/analyze", json={"query": "How is Reliance doing?"})

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert "decision" in data
    assert "fundamental" in data
    assert "technical" in data
    assert "sentiment" in data


def test_compare_endpoint(graph_deps):
    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    mock_container = MagicMock()
    mock_container.orchestrator = orchestrator

    with patch("app.api.routes.get_container", return_value=mock_container):
        client = TestClient(create_app())
        response = client.post(
            "/api/compare",
            json={"stocks": ["TATAMOTORS.NS", "M&M.NS"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["stocks"]) == 2
    assert "overall_scores" in data
    assert data["valuation_comparison"]
    assert data["relative_assessment"]


def test_portfolio_endpoint(graph_deps):
    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    mock_container = MagicMock()
    mock_container.orchestrator = orchestrator

    with patch("app.api.routes.get_container", return_value=mock_container):
        client = TestClient(create_app())
        response = client.post(
            "/api/portfolio",
            json={
                "holdings": [
                    {"symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 1000}
                ]
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["holding"]["symbol"] == "RELIANCE.NS"
    assert "portfolio_score" in data
    assert data["summary"]


def test_imports():
    """Ensure all architecture modules are importable."""
    from app.agents import (
        ComparisonAnalyst,
        FundamentalAnalyst,
        MasterAnalyst,
        PortfolioAnalyst,
        QueryParser,
        SentimentAnalyst,
        TechnicalAnalyst,
    )
    from app.analysis import ScoringEngine, TechnicalIndicatorEngine
    from app.data import YahooFinanceProvider, create_cache
    from app.graph import AnalysisOrchestrator, StockAnalysisState
    from app.models import DecisionResult, Rating
    from app.services import get_container

    assert FundamentalAnalyst is not None
    assert StockAnalysisState is not None
    assert get_container() is not None
