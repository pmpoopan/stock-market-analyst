"""Tests for ComparisonAnalyst — mock data and mock LLM."""

import pytest

from app.agents.comparison_agent import ComparisonAnalyst
from app.agents.llm_client import MockLLMClient
from app.graph.deps import GraphDependencies
from app.graph.workflow import AnalysisOrchestrator
from app.models.schemas import ComparisonInterpretation
from tests.fixtures.market_data import MOCK_SYMBOL, MOCK_SYMBOL_2


@pytest.fixture
def comparison_analyst(graph_deps):
    return graph_deps.comparison_analyst


@pytest.mark.asyncio
async def test_compare_from_state(graph_deps):
    orchestrator = AnalysisOrchestrator(deps=graph_deps)
    state = await orchestrator.compare([MOCK_SYMBOL, MOCK_SYMBOL_2])

    result = state["comparison_analysis"]
    assert len(result.stocks) == 2
    assert result.overall_scores[MOCK_SYMBOL] >= 0
    assert result.valuation_comparison
    assert result.relative_assessment
    assert "PE" in result.valuation_comparison
    assert "technical score" in result.technical_trend_comparison


@pytest.mark.asyncio
async def test_compare_uses_custom_llm_interpretation(graph_deps):
    custom = ComparisonInterpretation(
        valuation_comparison="Custom valuation narrative.",
        growth_comparison="Custom growth narrative.",
        risk_comparison="Custom risk narrative.",
        technical_trend_comparison="Custom technical narrative.",
        relative_assessment="Custom relative assessment.",
    )
    llm = MockLLMClient(structured_responses={ComparisonInterpretation: custom})
    deps = GraphDependencies(
        query_parser=graph_deps.query_parser,
        fundamental_analyst=graph_deps.fundamental_analyst,
        technical_analyst=graph_deps.technical_analyst,
        sentiment_analyst=graph_deps.sentiment_analyst,
        master_analyst=graph_deps.master_analyst,
        comparison_analyst=ComparisonAnalyst(llm),
        portfolio_analyst=graph_deps.portfolio_analyst,
        scoring_engine=graph_deps.scoring_engine,
        market_data=graph_deps.market_data,
    )
    orchestrator = AnalysisOrchestrator(deps=deps)
    state = await orchestrator.compare([MOCK_SYMBOL, MOCK_SYMBOL_2])
    result = state["comparison_analysis"]

    assert result.relative_assessment == "Custom relative assessment."
    assert "PE" in result.valuation_comparison
    assert "Custom valuation narrative." not in result.valuation_comparison
    assert "technical score" in result.technical_trend_comparison
    assert result.growth_comparison != "Custom growth narrative."
    assert result.risk_comparison != "Custom risk narrative."
