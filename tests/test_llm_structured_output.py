"""Tests for structured-output resilience and LLM client behavior."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.llm_client import GroqLLMClient
from app.config.settings import Settings
from app.models.schemas import TechnicalInterpretation


class TokenExhaustionError(Exception):
    status_code = 400

    def __init__(self) -> None:
        super().__init__(
            'Error code: 400 - {"error":{"message":"max completion tokens reached '
            'before generating a valid document","type":"json_validate_failed"}}'
        )


@pytest.mark.asyncio
async def test_groq_does_not_retry_token_boost_more_than_once():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens_technical=768,
        llm_structured_output_retry_boost=256,
        llm_structured_output_retry_max_tokens=1280,
        llm_retry_max_attempts=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    groq_client.chat.completions.create = AsyncMock(side_effect=TokenExhaustionError())
    client._client = groq_client

    with pytest.raises(TokenExhaustionError):
        await client.generate(
            "prompt",
            structured_output=TechnicalInterpretation,
            max_tokens=settings.llm_max_tokens_technical,
        )

    assert groq_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_groq_retries_once_on_invalid_json_with_boosted_tokens():
    settings = Settings(
        groq_api_key="test-key",
        llm_max_tokens_technical=768,
        llm_structured_output_retry_boost=256,
        llm_structured_output_retry_max_tokens=1280,
        llm_retry_max_attempts=1,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    invalid = MagicMock()
    invalid.choices = [MagicMock(message=MagicMock(content='{"momentum":"only partial'))]
    valid = MagicMock()
    valid.choices = [
        MagicMock(message=MagicMock(content='{"momentum":"m","volatility":"v","summary":"s"}'))
    ]
    groq_client.chat.completions.create = AsyncMock(side_effect=[invalid, valid])
    client._client = groq_client

    result = await client.generate(
        "prompt",
        structured_output=TechnicalInterpretation,
        max_tokens=settings.llm_max_tokens_technical,
    )

    assert isinstance(result, TechnicalInterpretation)
    assert groq_client.chat.completions.create.await_count == 2
    assert groq_client.chat.completions.create.await_args_list[1].kwargs["max_tokens"] == 1024


def test_groq_sdk_automatic_retries_disabled():
    import inspect

    from groq import AsyncGroq

    settings = Settings(groq_api_key="test-key")
    client = GroqLLMClient(settings=settings)
    groq_client = client._get_client()

    if "max_retries" not in inspect.signature(AsyncGroq.__init__).parameters:
        pytest.skip("Installed Groq SDK does not expose max_retries")

    assert groq_client.max_retries == 0


@pytest.mark.asyncio
async def test_sentiment_interpretation_parses_instance_json():
    from app.models.schemas import SentimentInterpretation

    settings = Settings(groq_api_key="test-key", llm_retry_max_attempts=1)
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    payload = {
        "positive_catalysts": ["Earnings beat"],
        "negative_catalysts": ["Margin pressure"],
        "key_events": ["Q1 results"],
        "summary": "Mixed sentiment from recent coverage.",
    }
    groq_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))])
    )
    client._client = groq_client

    result = await client.generate(
        "prompt",
        structured_output=SentimentInterpretation,
        max_tokens=640,
    )

    assert isinstance(result, SentimentInterpretation)
    assert result.positive_catalysts == ["Earnings beat"]
    assert result.negative_catalysts == ["Margin pressure"]
    assert result.key_events == ["Q1 results"]
    assert result.summary.startswith("Mixed sentiment")
    system_msg = groq_client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "Positive catalysts from articles" not in system_msg
    assert "properties" not in system_msg


@pytest.mark.asyncio
async def test_sentiment_schema_echo_does_not_retry():
    from pydantic import ValidationError

    from app.models.schemas import SentimentInterpretation

    settings = Settings(
        groq_api_key="test-key",
        llm_retry_max_attempts=1,
        llm_max_tokens_sentiment=640,
        llm_structured_output_retry_boost=256,
        llm_structured_output_retry_max_tokens=1280,
    )
    client = GroqLLMClient(settings=settings)
    groq_client = MagicMock()
    schema_json = json.dumps(SentimentInterpretation.model_json_schema())
    groq_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=schema_json))])
    )
    client._client = groq_client

    with pytest.raises(ValidationError):
        await client.generate(
            "prompt",
            structured_output=SentimentInterpretation,
            max_tokens=640,
        )

    assert groq_client.chat.completions.create.await_count == 1
