"""Tests for rate-limit resilience across LLM, market data, and news providers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.comparison_agent import ComparisonAnalyst
from app.agents.llm_client import GroqLLMClient, MockLLMClient
from app.agents.llm_exceptions import LLMRateLimitError
from app.api.error_messages import select_primary_error, user_facing_api_error
from app.config.settings import Settings
from app.data.exceptions import DataNotFoundError, DataProviderError
from app.data.web_search import DuckDuckGoSearchProvider
from app.data.yahoo_finance import CACHE_NS_QUOTES, YahooFinanceProvider
from app.graph.deps import GraphDependencies
from app.graph.workflow import AnalysisOrchestrator
from app.models.schemas import ErrorDetail, MasterInterpretation, TechnicalInterpretation
from app.util.retry import (
    async_retry_with_backoff,
    is_rate_limit_error,
    is_token_exhaustion_error,
    is_transient_timeout_error,
    parse_retry_after_seconds,
    sync_retry_with_backoff,
)
from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.market_data import (
    MOCK_SYMBOL,
    MOCK_SYMBOL_2,
    MockMarketDataProvider,
    make_mock_historical_long,
    make_mock_quote,
)


@pytest.fixture
def provider(cache, test_settings):
    return YahooFinanceProvider(cache=cache, settings=test_settings)


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, message: str = "Rate limit reached") -> None:
        super().__init__(message)
        self.response = MagicMock(headers={"retry-after": "2"})


def test_is_rate_limit_error_detects_common_patterns():
    assert is_rate_limit_error(RateLimitError())
    assert is_rate_limit_error(Exception("Too Many Requests. Rate limited."))
    assert not is_rate_limit_error(Exception("network down"))


def test_sync_retry_with_backoff_retries_rate_limits():
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RateLimitError()
        return "ok"

    with patch("app.util.retry.time.sleep"):
        result = sync_retry_with_backoff(
            operation,
            max_attempts=3,
            base_delay=0.01,
            max_delay=1.0,
            operation_name="test-op",
        )

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_async_retry_with_backoff_retries_rate_limits():
    attempts = {"count": 0}

    async def operation():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RateLimitError("429 rate limit")
        return "done"

    with patch("app.util.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await async_retry_with_backoff(
            operation,
            max_attempts=3,
            base_delay=0.01,
            max_delay=1.0,
            operation_name="async-test",
        )

    assert result == "done"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_groq_client_retries_then_raises_clean_rate_limit_error():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens=512,
        llm_retry_max_attempts=2,
        llm_retry_base_delay_seconds=0.01,
        llm_retry_max_delay_seconds=0.05,
        llm_max_concurrent_requests=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    groq_client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("Rate limit reached for model")
    )
    client._client = groq_client

    with patch("app.util.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMRateLimitError, match="temporarily busy"):
            await client.generate("prompt")

    assert groq_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_groq_client_retries_429_once_and_respects_retry_after():
    settings = Settings(
        groq_api_key="test-key",
        llm_retry_max_attempts=2,
        llm_retry_base_delay_seconds=10.0,
        llm_retry_max_delay_seconds=30.0,
        llm_max_concurrent_requests=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    success = MagicMock()
    success.choices = [MagicMock(message=MagicMock(content="ok"))]
    groq_client.chat.completions.create = AsyncMock(
        side_effect=[RateLimitError("Rate limit reached for model"), success]
    )
    client._client = groq_client

    sleep_mock = AsyncMock()
    with patch("app.util.retry.asyncio.sleep", sleep_mock):
        result = await client.generate("prompt")

    assert result == "ok"
    assert groq_client.chat.completions.create.await_count == 2
    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args.args[0] == 2.0


@pytest.mark.asyncio
async def test_groq_client_default_retry_budget_is_one_retry():
    settings = Settings(
        groq_api_key="test-key",
        llm_retry_max_attempts=2,
        llm_retry_base_delay_seconds=0.01,
        llm_retry_max_delay_seconds=0.05,
        llm_max_concurrent_requests=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    groq_client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("Rate limit reached for model")
    )
    client._client = groq_client

    with patch("app.util.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMRateLimitError, match="temporarily busy"):
            await client.generate("prompt")

    assert groq_client.chat.completions.create.await_count == 2
    assert client._get_semaphore()._value == 1


@pytest.mark.asyncio
async def test_groq_client_passes_agent_max_tokens():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens=1024,
        llm_max_tokens_technical=768,
        llm_retry_max_attempts=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"momentum":"m","volatility":"v","summary":"s"}'))]
    groq_client.chat.completions.create = AsyncMock(return_value=response)
    client._client = groq_client

    await client.generate(
        "prompt",
        structured_output=TechnicalInterpretation,
        max_tokens=settings.llm_max_tokens_technical,
    )

    kwargs = groq_client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 768


@pytest.mark.asyncio
async def test_mock_llm_records_max_tokens():
    llm = MockLLMClient()
    await llm.generate("prompt", max_tokens=640)
    assert llm.calls[0]["max_tokens"] == 640


def test_llm_concurrency_and_retry_defaults():
    assert Settings.model_fields["llm_max_concurrent_requests"].default == 1
    assert Settings.model_fields["llm_retry_max_attempts"].default == 2


def test_parse_retry_after_seconds_from_headers_and_message():
    header_exc = RateLimitError()
    assert parse_retry_after_seconds(header_exc) == 2.0
    message_exc = Exception("Please retry after 3 seconds")
    assert parse_retry_after_seconds(message_exc) == 3.0
    ms_exc = Exception("retry-after: 1500 ms")
    assert parse_retry_after_seconds(ms_exc) == 1.5
    assert parse_retry_after_seconds(Exception("no delay")) is None


def test_token_budget_settings_defaults():
    settings = Settings()
    assert settings.llm_max_tokens == 1024
    assert settings.llm_max_tokens_fundamental == 640
    assert settings.llm_max_tokens_technical == 768
    assert settings.llm_max_tokens_sentiment == 640
    assert settings.llm_max_tokens_master == 1024
    assert settings.llm_max_tokens_comparison == 1024
    assert settings.llm_max_tokens_portfolio == 768
    assert settings.llm_structured_output_retry_boost == 256
    assert settings.llm_structured_output_retry_max_tokens == 1280


@patch("app.data.yahoo_finance.yf.Ticker")
def test_yahoo_quote_retries_on_rate_limit(mock_ticker_cls, cache, test_settings):
    mock_ticker = MagicMock()
    mock_ticker.info = {"currentPrice": 100.0, "shortName": "Test"}
    mock_ticker_cls.side_effect = [RateLimitError("Too Many Requests"), mock_ticker]

    provider = YahooFinanceProvider(
        cache=cache,
        settings=Settings(
            cache_enabled=True,
            cache_db_path=test_settings.cache_db_path,
            yahoo_retry_max_attempts=2,
            yahoo_retry_base_delay_seconds=0.01,
            yahoo_retry_max_delay_seconds=0.05,
        ),
    )

    with patch("app.util.retry.time.sleep"):
        quote = provider.get_quote(MOCK_SYMBOL)

    assert quote.price == 100.0
    assert mock_ticker_cls.call_count == 2


@patch("app.data.web_search.DDGS")
def test_web_search_returns_empty_on_rate_limit(mock_ddgs_cls, cache, test_settings):
    mock_ddgs_cls.return_value.__enter__.side_effect = RateLimitError("403 Ratelimit")

    provider = DuckDuckGoSearchProvider(
        cache=cache,
        settings=Settings(
            cache_enabled=True,
            cache_db_path=test_settings.cache_db_path,
            web_search_retry_max_attempts=1,
            web_search_rate_limit_cache_seconds=120,
        ),
        max_results=5,
    )

    articles = provider.search_news("Reliance stock India news")
    assert articles == []

    cached_skip = provider.search_news("Reliance stock India news")
    assert cached_skip == []
    assert mock_ddgs_cls.call_count == 1


@patch("app.data.web_search.DDGS")
def test_web_search_still_raises_non_rate_limit_errors(mock_ddgs_cls, cache, test_settings):
    provider = DuckDuckGoSearchProvider(cache=cache, settings=test_settings, max_results=5)
    mock_ddgs_cls.return_value.__enter__.side_effect = RuntimeError("network down")
    with pytest.raises(DataProviderError, match="News search failed"):
        provider.search_news("Reliance news")


def test_user_facing_api_error_sanitizes_provider_messages():
    assert "temporarily busy" in user_facing_api_error(
        "Rate limit reached for model openai/gpt-oss-120b on tokens per minute"
    )
    assert "market data" in user_facing_api_error(
        "Quote unavailable for RELIANCE.NS: Too Many Requests. Rate limited."
    ).lower()
    assert "fewer than two" in user_facing_api_error(
        "At least two stocks with complete analysis are required for comparison"
    ).lower()


def test_select_primary_error_prefers_comparison_error():
    errors = [
        ErrorDetail(component="market_data", message="Quote unavailable for X: rate limited", recoverable=True),
        ErrorDetail(
            component="comparison",
            message="At least two stocks with complete analysis are required for comparison",
            recoverable=False,
        ),
    ]
    detail = select_primary_error(errors)
    assert detail is not None
    assert "fewer than two" in detail.lower()


@pytest.mark.asyncio
async def test_comparison_succeeds_when_one_stock_fails(graph_deps):
    failing_symbol = "M&M.NS"
    market = MockMarketDataProvider(
        historical={
            f"{MOCK_SYMBOL}:1y": make_mock_historical_long(MOCK_SYMBOL),
            f"{MOCK_SYMBOL_2}:1y": make_mock_historical_long(MOCK_SYMBOL_2),
            f"{failing_symbol}:1y": make_mock_historical_long(failing_symbol),
        },
        financials={
            MOCK_SYMBOL: make_mock_financial_metrics(MOCK_SYMBOL),
            MOCK_SYMBOL_2: make_mock_financial_metrics(MOCK_SYMBOL_2),
            failing_symbol: make_mock_financial_metrics(failing_symbol),
        },
        quotes={
            MOCK_SYMBOL: make_mock_quote(MOCK_SYMBOL),
            MOCK_SYMBOL_2: make_mock_quote(MOCK_SYMBOL_2),
            failing_symbol: make_mock_quote(failing_symbol),
        },
    )

    original_get_financials = market.get_financials

    def get_financials(symbol: str):
        if symbol.upper() == failing_symbol:
            raise DataProviderError("Failed to fetch financials: Too Many Requests")
        return original_get_financials(symbol)

    market.get_financials = get_financials  # type: ignore[method-assign]

    deps = GraphDependencies(
        query_parser=graph_deps.query_parser,
        fundamental_analyst=graph_deps.fundamental_analyst.__class__(market, graph_deps.fundamental_analyst._llm),
        technical_analyst=graph_deps.technical_analyst.__class__(market, graph_deps.technical_analyst._llm),
        sentiment_analyst=graph_deps.sentiment_analyst,
        master_analyst=graph_deps.master_analyst,
        comparison_analyst=ComparisonAnalyst(graph_deps.comparison_analyst._llm),
        portfolio_analyst=graph_deps.portfolio_analyst,
        scoring_engine=graph_deps.scoring_engine,
        market_data=market,
    )

    orchestrator = AnalysisOrchestrator(deps=deps)
    state = await orchestrator.compare([MOCK_SYMBOL, MOCK_SYMBOL_2, failing_symbol])

    result = state.get("comparison_analysis")
    assert result is not None
    assert len(result.stocks) == 2
    assert failing_symbol not in result.stocks


@pytest.mark.asyncio
async def test_comparison_still_requires_two_complete_stocks(graph_deps):
    analyst = ComparisonAnalyst(MockLLMClient())
    with pytest.raises(ValueError, match="At least two stocks with complete analysis"):
        await analyst.compare_from_state(
            stocks=[MOCK_SYMBOL, MOCK_SYMBOL_2],
            decisions={MOCK_SYMBOL: MagicMock()},
            fundamental_analysis={MOCK_SYMBOL: MagicMock()},
            technical_analysis={},
            sentiment_analysis={},
        )


class TokenExhaustionError(Exception):
    status_code = 400

    def __init__(self) -> None:
        super().__init__(
            'Error code: 400 - {"error":{"message":"max completion tokens reached '
            'before generating a valid document","type":"json_validate_failed"}}'
        )


def test_is_token_exhaustion_error_detects_groq_message():
    assert is_token_exhaustion_error(TokenExhaustionError())
    assert not is_token_exhaustion_error(Exception("network down"))


def test_is_transient_timeout_error():
    assert is_transient_timeout_error(TimeoutError())
    assert is_transient_timeout_error(Exception("read timed out"))
    assert not is_transient_timeout_error(Exception("network down"))


@pytest.mark.asyncio
async def test_groq_client_retries_structured_output_with_boosted_tokens():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens_technical=768,
        llm_structured_output_retry_boost=256,
        llm_structured_output_retry_max_tokens=1280,
        llm_retry_max_attempts=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    success = MagicMock()
    success.choices = [
        MagicMock(message=MagicMock(content='{"momentum":"m","volatility":"v","summary":"s"}'))
    ]
    groq_client.chat.completions.create = AsyncMock(
        side_effect=[TokenExhaustionError(), success]
    )
    client._client = groq_client

    result = await client.generate(
        "prompt",
        structured_output=TechnicalInterpretation,
        max_tokens=settings.llm_max_tokens_technical,
    )

    assert isinstance(result, TechnicalInterpretation)
    assert groq_client.chat.completions.create.await_count == 2
    first_tokens = groq_client.chat.completions.create.await_args_list[0].kwargs["max_tokens"]
    second_tokens = groq_client.chat.completions.create.await_args_list[1].kwargs["max_tokens"]
    assert first_tokens == 768
    assert second_tokens == 1024


@pytest.mark.asyncio
async def test_groq_client_master_token_budget():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens_master=1024,
        llm_retry_max_attempts=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    success = MagicMock()
    success.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"agreement_points":["a"],"disagreement_points":["b"],'
                    '"major_risks":["r"],"important_catalysts":["c"],'
                    '"narrative":"n","data_vs_interpretation":"d"}'
                )
            )
        )
    ]
    groq_client.chat.completions.create = AsyncMock(return_value=success)
    client._client = groq_client

    result = await client.generate(
        "prompt",
        structured_output=MasterInterpretation,
        max_tokens=settings.llm_max_tokens_master,
    )

    assert isinstance(result, MasterInterpretation)
    assert groq_client.chat.completions.create.await_args.kwargs["max_tokens"] == 1024


@patch("app.data.web_search.DDGS")
def test_web_search_returns_empty_on_timeout(mock_ddgs_cls, cache, test_settings):
    provider = DuckDuckGoSearchProvider(cache=cache, settings=test_settings, max_results=5)
    mock_ddgs_cls.return_value.__enter__.side_effect = TimeoutError("timed out")
    articles = provider.search_news("Reliance stock India news")
    assert articles == []


@patch("app.data.yahoo_finance.yf.Ticker")
def test_get_quote_retries_empty_price_then_raises(mock_ticker_cls, provider):
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker_cls.return_value = mock_ticker

    with patch("app.util.retry.time.sleep"):
        with pytest.raises(DataNotFoundError, match="No price data"):
            provider.get_quote(MOCK_SYMBOL)


@patch("app.data.yahoo_finance.yf.Ticker")
@patch.object(YahooFinanceProvider, "_cache_get", return_value=None)
def test_get_quote_uses_stale_cache_after_empty_response(
    mock_cache_get, mock_ticker_cls, provider
):
    cached_quote = make_mock_quote(MOCK_SYMBOL)
    provider._cache_set(CACHE_NS_QUOTES, MOCK_SYMBOL, cached_quote, ttl_seconds=300)

    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker_cls.return_value = mock_ticker

    with patch("app.util.retry.time.sleep"):
        quote = provider.get_quote(MOCK_SYMBOL)

    assert quote.price == cached_quote.price


@patch("app.data.yahoo_finance.yf.Ticker")
@patch.object(YahooFinanceProvider, "_cache_get", return_value=None)
def test_get_quote_uses_stale_cache_after_rate_limit(
    mock_cache_get, mock_ticker_cls, provider
):
    cached_quote = make_mock_quote(MOCK_SYMBOL)
    provider._cache_set(CACHE_NS_QUOTES, MOCK_SYMBOL, cached_quote, ttl_seconds=300)

    mock_ticker_cls.side_effect = RateLimitError("Too Many Requests")

    with patch("app.util.retry.time.sleep"):
        quote = provider.get_quote(MOCK_SYMBOL)

    assert quote.price == cached_quote.price
