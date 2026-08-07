"""LangGraph node functions.

Each node delegates to agents/services. The orchestrator coordinates only —
it does not perform financial analysis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from app.graph.deps import GraphDependencies
from app.graph.state import StockAnalysisState
from app.models.schemas import ErrorDetail, PortfolioAnalysisResult, QueryIntent, StockAnalysisResponse

logger = logging.getLogger(__name__)


async def _run_per_stock(
    stocks: list[str],
    component: str,
    runner: Callable[[str], Coroutine[Any, Any, Any]],
) -> tuple[dict[str, Any], list[ErrorDetail]]:
    results: dict[str, Any] = {}
    errors: list[ErrorDetail] = []

    if not stocks:
        return results, errors

    async def _run_one(symbol: str) -> None:
        try:
            results[symbol] = await runner(symbol)
        except Exception as exc:
            logger.exception("%s failed for %s", component, symbol)
            errors.append(
                ErrorDetail(
                    component=component,
                    message=f"{symbol}: {exc}",
                    recoverable=True,
                )
            )

    await asyncio.gather(*[_run_one(symbol) for symbol in stocks])
    return results, errors


def make_parse_query_node(deps: GraphDependencies):
    async def parse_query_node(state: StockAnalysisState) -> dict[str, Any]:
        if state.get("parsed_query") is not None:
            parsed = state["parsed_query"]
            return {
                "query": state.get("query", parsed.raw_query),
                "parsed_query": parsed,
            }

        query = state.get("query", "").strip()
        if not query:
            return {
                "errors": [
                    ErrorDetail(
                        component="query_parser",
                        message="Query is required",
                        recoverable=False,
                    )
                ]
            }

        try:
            parsed = deps.query_parser.parse(query)
        except ValueError as exc:
            return {
                "errors": [
                    ErrorDetail(
                        component="query_parser",
                        message=str(exc),
                        recoverable=False,
                    )
                ]
            }

        return {"query": query, "parsed_query": parsed}

    return parse_query_node


def make_orchestrator_node(deps: GraphDependencies):
    async def orchestrator_node(state: StockAnalysisState) -> dict[str, Any]:
        parsed = state.get("parsed_query")
        if parsed is None:
            return {}

        stocks: list[str] = []
        portfolio = parsed.portfolio

        if parsed.intent == QueryIntent.ANALYZE_PORTFOLIO:
            stocks = [h.symbol.upper() for h in portfolio]
        else:
            stocks = [s.upper() for s in parsed.stocks]

        market_data: dict[str, Any] = {}
        errors: list[ErrorDetail] = []

        for symbol in stocks:
            try:
                quote = deps.market_data.get_quote(symbol)
                market_data[symbol] = quote.model_dump(mode="json")
            except Exception as exc:
                logger.warning("Quote fetch failed for %s: %s", symbol, exc)
                errors.append(
                    ErrorDetail(
                        component="market_data",
                        message=f"Quote unavailable for {symbol}: {exc}",
                        recoverable=True,
                    )
                )

        return {
            "stocks": stocks,
            "portfolio": portfolio,
            "market_data": market_data,
            "errors": errors,
        }

    return orchestrator_node


def make_fundamental_analyst_node(deps: GraphDependencies):
    async def fundamental_analyst_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        results, errors = await _run_per_stock(
            stocks,
            "fundamental_analyst",
            deps.fundamental_analyst.analyze,
        )
        return {"fundamental_analysis": results, "errors": errors}

    return fundamental_analyst_node


def make_technical_analyst_node(deps: GraphDependencies):
    async def technical_analyst_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        results, errors = await _run_per_stock(
            stocks,
            "technical_analyst",
            deps.technical_analyst.analyze,
        )
        return {"technical_analysis": results, "errors": errors}

    return technical_analyst_node


def make_sentiment_analyst_node(deps: GraphDependencies):
    async def sentiment_analyst_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        market_data = state.get("market_data", {})

        async def _analyze(symbol: str):
            company_name = None
            quote_data = market_data.get(symbol)
            if quote_data:
                company_name = quote_data.get("name")
            return await deps.sentiment_analyst.analyze(symbol, company_name=company_name)

        results, errors = await _run_per_stock(
            stocks,
            "sentiment_analyst",
            _analyze,
        )
        return {"sentiment_analysis": results, "errors": errors}

    return sentiment_analyst_node


def make_master_analyst_node(deps: GraphDependencies):
    async def master_analyst_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        fundamental = state.get("fundamental_analysis", {})
        technical = state.get("technical_analysis", {})
        sentiment = state.get("sentiment_analysis", {})

        master: dict[str, Any] = {}
        errors: list[ErrorDetail] = []

        async def _synthesize(symbol: str) -> None:
            if symbol not in fundamental or symbol not in technical or symbol not in sentiment:
                errors.append(
                    ErrorDetail(
                        component="master_analyst",
                        message=f"Incomplete analyst outputs for {symbol}",
                        recoverable=True,
                    )
                )
                return
            try:
                master[symbol] = await deps.master_analyst.synthesize(
                    symbol,
                    fundamental[symbol],
                    technical[symbol],
                    sentiment[symbol],
                )
            except Exception as exc:
                logger.exception("Master analyst failed for %s", symbol)
                errors.append(
                    ErrorDetail(
                        component="master_analyst",
                        message=f"{symbol}: {exc}",
                        recoverable=True,
                    )
                )

        if stocks:
            await asyncio.gather(*[_synthesize(symbol) for symbol in stocks])

        return {"master_analysis": master, "errors": errors}

    return master_analyst_node


def make_decision_engine_node(deps: GraphDependencies):
    async def decision_engine_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        fundamental = state.get("fundamental_analysis", {})
        technical = state.get("technical_analysis", {})
        sentiment = state.get("sentiment_analysis", {})
        master_analysis = state.get("master_analysis", {})

        decisions: dict[str, Any] = {}
        errors: list[ErrorDetail] = []

        for symbol in stocks:
            if symbol not in fundamental or symbol not in technical or symbol not in sentiment:
                errors.append(
                    ErrorDetail(
                        component="decision_engine",
                        message=f"Missing inputs for decision on {symbol}",
                        recoverable=True,
                    )
                )
                continue
            decisions[symbol] = deps.scoring_engine.compute_decision(
                symbol,
                fundamental[symbol],
                technical[symbol],
                sentiment[symbol],
                master=master_analysis.get(symbol),
            )

        return {"decision": decisions, "errors": errors}

    return decision_engine_node


def make_comparison_node(deps: GraphDependencies):
    async def comparison_node(state: StockAnalysisState) -> dict[str, Any]:
        stocks = state.get("stocks", [])
        if len(stocks) < 2:
            return {
                "errors": [
                    ErrorDetail(
                        component="comparison",
                        message="At least two stocks are required for comparison",
                        recoverable=False,
                    )
                ]
            }

        try:
            comparison_analysis = await deps.comparison_analyst.compare_from_state(
                stocks=stocks,
                decisions=state.get("decision", {}),
                fundamental_analysis=state.get("fundamental_analysis", {}),
                technical_analysis=state.get("technical_analysis", {}),
                sentiment_analysis=state.get("sentiment_analysis", {}),
            )
            return {"comparison_analysis": comparison_analysis}
        except Exception as exc:
            logger.exception("Comparison analysis failed")
            return {
                "errors": [
                    ErrorDetail(
                        component="comparison",
                        message=str(exc),
                        recoverable=False,
                    )
                ]
            }

    return comparison_node


def make_portfolio_node(deps: GraphDependencies):
    async def portfolio_node(state: StockAnalysisState) -> dict[str, Any]:
        holdings = state.get("portfolio", [])
        if not holdings:
            return {
                "errors": [
                    ErrorDetail(
                        component="portfolio",
                        message="No portfolio holdings in state",
                        recoverable=False,
                    )
                ]
            }

        try:
            portfolio_analysis = await deps.portfolio_analyst.analyze_from_state(
                holdings=holdings,
                decisions=state.get("decision", {}),
                market_data=state.get("market_data", {}),
                fundamental_analysis=state.get("fundamental_analysis", {}),
            )
            return {"portfolio_analysis": portfolio_analysis}
        except Exception as exc:
            logger.exception("Portfolio analysis failed")
            return {
                "errors": [
                    ErrorDetail(
                        component="portfolio",
                        message=str(exc),
                        recoverable=False,
                    )
                ]
            }

    return portfolio_node


def make_final_response_node(deps: GraphDependencies):
    async def final_response_node(state: StockAnalysisState) -> dict[str, Any]:
        parsed = state.get("parsed_query")
        if parsed is None:
            return {}

        intent = parsed.intent
        stocks = state.get("stocks", [])
        fundamental = state.get("fundamental_analysis", {})
        technical = state.get("technical_analysis", {})
        sentiment = state.get("sentiment_analysis", {})
        master = state.get("master_analysis", {})
        decisions = state.get("decision", {})
        market_data = state.get("market_data", {})

        if intent == QueryIntent.COMPARE_STOCKS:
            comparison_result = state.get("comparison_analysis")
            if comparison_result is not None:
                return {
                    "comparison_analysis": comparison_result,
                    "final_analysis": {
                        "intent": intent.value,
                        "stocks": comparison_result.stocks,
                        "winner": comparison_result.winner,
                    },
                }
            return {
                "errors": [
                    ErrorDetail(
                        component="final_response",
                        message="Comparison analysis result unavailable",
                        recoverable=False,
                    )
                ]
            }

        if intent == QueryIntent.ANALYZE_PORTFOLIO:
            portfolio_result = state.get("portfolio_analysis")
            if portfolio_result is not None:
                return {
                    "portfolio_analysis": portfolio_result,
                    "final_analysis": {
                        "intent": intent.value,
                        "holdings_count": len(portfolio_result.holdings),
                    },
                }
            return {
                "errors": [
                    ErrorDetail(
                        component="final_response",
                        message="Portfolio analysis result unavailable",
                        recoverable=False,
                    )
                ]
            }

        if intent == QueryIntent.ANALYZE_STOCK and len(stocks) == 1:
            symbol = stocks[0]
            if symbol not in decisions:
                return {
                    "errors": [
                        ErrorDetail(
                            component="final_response",
                            message=f"Decision unavailable for {symbol}",
                            recoverable=False,
                        )
                    ]
                }

            quote_data = market_data.get(symbol, {})
            sent = sentiment.get(symbol)
            sources = list(sent.sources) if sent else []

            stock_response = StockAnalysisResponse(
                symbol=symbol,
                name=quote_data.get("name"),
                current_price=quote_data.get("price"),
                decision=decisions[symbol],
                fundamental=fundamental[symbol],
                technical=technical[symbol],
                sentiment=sent,
                master=master[symbol],
                sources=sources,
            )
            return {
                "stock_response": stock_response,
                "final_analysis": {"intent": intent.value, "symbol": symbol},
            }

        return {
            "final_analysis": {
                "intent": intent.value,
                "stocks": stocks,
                "decisions": {k: v.model_dump() for k, v in decisions.items()},
            }
        }

    return final_response_node
