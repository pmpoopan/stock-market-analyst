"""API validation and error-path tests."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.graph.workflow import AnalysisOrchestrator
from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_analyze_invalid_query_returns_400(graph_deps):
    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    mock_container = MagicMock()
    mock_container.orchestrator = orchestrator

    with patch("app.api.routes.get_container", return_value=mock_container):
        client = TestClient(create_app())
        response = client.post("/api/analyze", json={"query": "What is the weather today?"})

    assert response.status_code == 400
    assert "detail" in response.json()


def test_compare_requires_two_symbols(client):
    response = client.post("/api/compare", json={"stocks": ["RELIANCE.NS"]})
    assert response.status_code == 422


def test_analyze_requires_query(client):
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422


def test_request_logging_middleware_emits_log(client, caplog):
    with caplog.at_level(logging.INFO, logger="app.api.middleware"):
        response = client.get("/api/health")

    assert response.status_code == 200
    assert any("GET /api/health" in record.message for record in caplog.records)
