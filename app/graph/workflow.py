"""LangGraph workflow definition.

Graph topology:

    START → parse_query → orchestrator
      → [fundamental_analyst, technical_analyst, sentiment_analyst]  (parallel)
      → master_analyst → decision_engine
      → [comparison | portfolio | final_response] → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.deps import GraphDependencies
from app.graph import nodes
from app.graph.state import StockAnalysisState
from app.models.schemas import QueryIntent


def _route_after_decision(state: StockAnalysisState) -> str:
    parsed = state.get("parsed_query")
    if parsed is None:
        return "final"

    if parsed.intent == QueryIntent.COMPARE_STOCKS:
        return "comparison"
    if parsed.intent == QueryIntent.ANALYZE_PORTFOLIO:
        return "portfolio"
    return "final"


def build_analysis_graph(deps: GraphDependencies) -> StateGraph:
    """Construct the LangGraph analysis workflow with injected dependencies."""
    graph = StateGraph(StockAnalysisState)

    graph.add_node("parse_query", nodes.make_parse_query_node(deps))
    graph.add_node("orchestrator", nodes.make_orchestrator_node(deps))
    graph.add_node("fundamental_analyst", nodes.make_fundamental_analyst_node(deps))
    graph.add_node("technical_analyst", nodes.make_technical_analyst_node(deps))
    graph.add_node("sentiment_analyst", nodes.make_sentiment_analyst_node(deps))
    graph.add_node("master_analyst", nodes.make_master_analyst_node(deps))
    graph.add_node("decision_engine", nodes.make_decision_engine_node(deps))
    graph.add_node("comparison", nodes.make_comparison_node(deps))
    graph.add_node("portfolio", nodes.make_portfolio_node(deps))
    graph.add_node("final_response", nodes.make_final_response_node(deps))

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "orchestrator")

    # Parallel analyst fan-out
    graph.add_edge("orchestrator", "fundamental_analyst")
    graph.add_edge("orchestrator", "technical_analyst")
    graph.add_edge("orchestrator", "sentiment_analyst")

    # Fan-in to master analyst
    graph.add_edge("fundamental_analyst", "master_analyst")
    graph.add_edge("technical_analyst", "master_analyst")
    graph.add_edge("sentiment_analyst", "master_analyst")

    graph.add_edge("master_analyst", "decision_engine")

    graph.add_conditional_edges(
        "decision_engine",
        _route_after_decision,
        {
            "comparison": "comparison",
            "portfolio": "portfolio",
            "final": "final_response",
        },
    )

    graph.add_edge("comparison", "final_response")
    graph.add_edge("portfolio", "final_response")
    graph.add_edge("final_response", END)

    return graph


def compile_workflow(deps: GraphDependencies):
    """Return a compiled, invokable LangGraph workflow."""
    return build_analysis_graph(deps).compile()


class AnalysisOrchestrator:
    """High-level facade over the LangGraph workflow."""

    def __init__(self, deps: GraphDependencies) -> None:
        self._deps = deps
        self._workflow = compile_workflow(deps)

    async def run(self, initial_state: dict) -> StockAnalysisState:
        result = await self._workflow.ainvoke(initial_state)
        return result

    async def analyze(self, query: str) -> StockAnalysisState:
        return await self.run({"query": query})

    async def compare(self, stocks: list[str]) -> StockAnalysisState:
        parsed = self._deps.query_parser.parse_compare(stocks)
        return await self.run({"query": parsed.raw_query, "parsed_query": parsed})

    async def portfolio(self, holdings: list) -> StockAnalysisState:
        parsed = self._deps.query_parser.parse_portfolio(holdings)
        return await self.run({"query": parsed.raw_query, "parsed_query": parsed})
