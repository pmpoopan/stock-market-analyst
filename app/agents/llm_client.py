"""Configurable LLM client (Groq default).

Agents use this for interpretation/reasoning — not for numeric calculations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from app.agents.llm_exceptions import LLMRateLimitError
from app.config.settings import Settings, get_settings
from app.util.retry import (
    async_retry_with_backoff,
    is_rate_limit_error,
    is_token_exhaustion_error,
)

logger = logging.getLogger(__name__)

USER_FACING_RATE_LIMIT_MESSAGE = (
    "Analysis service is temporarily busy. Please try again in a minute."
)


class LLMClient(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        structured_output: type | None = None,
        max_tokens: int | None = None,
    ) -> str | Any:
        """Generate a completion; optionally parse into a Pydantic model."""
        ...


class MockLLMClient(LLMClient):
    """In-memory LLM for tests — never calls external APIs."""

    def __init__(
        self,
        default_text: str = "Mock LLM response",
        structured_responses: dict[type, BaseModel] | None = None,
    ) -> None:
        self._default_text = default_text
        self._structured_responses = structured_responses or {}
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        structured_output: type | None = None,
        max_tokens: int | None = None,
    ) -> str | Any:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "structured_output": structured_output,
                "max_tokens": max_tokens,
            }
        )

        if structured_output is not None:
            if structured_output in self._structured_responses:
                return self._structured_responses[structured_output]
            return self._default_structured(structured_output)

        return self._default_text

    def _default_structured(self, model_type: type) -> BaseModel:
        from app.models.schemas import (
            ComparisonInterpretation,
            FundamentalInterpretation,
            MasterInterpretation,
            PortfolioInterpretation,
            SentimentInterpretation,
            TechnicalInterpretation,
        )

        if model_type is TechnicalInterpretation:
            return TechnicalInterpretation(
                momentum="Mock momentum: RSI and MACD indicate balanced momentum.",
                volatility="Mock volatility: ATR suggests moderate price swings.",
                summary="Mock summary: Technical setup is neutral based on provided indicators.",
            )
        if model_type is FundamentalInterpretation:
            return FundamentalInterpretation(
                strengths=["Mock strength: solid revenue growth."],
                weaknesses=["Mock weakness: valuation is stretched."],
                risks=["Mock risk: leverage should be monitored."],
                summary="Mock summary: Fundamentals are broadly balanced based on provided metrics.",
            )
        if model_type is SentimentInterpretation:
            return SentimentInterpretation(
                positive_catalysts=["Mock positive catalyst from news."],
                negative_catalysts=["Mock negative catalyst from news."],
                key_events=["Mock key event headline."],
                summary="Mock summary: Sentiment is mixed based on provided articles.",
            )
        if model_type is MasterInterpretation:
            return MasterInterpretation(
                agreement_points=["Mock agreement: fundamentals and technicals align."],
                disagreement_points=["Mock divergence: sentiment is mixed."],
                major_risks=["Mock risk: monitor macro headwinds."],
                important_catalysts=["Mock catalyst: earnings momentum."],
                narrative="Mock narrative: Combined perspectives show a balanced setup.",
                data_vs_interpretation="Scores are from structured agents; narrative is interpretation.",
            )
        if model_type is PortfolioInterpretation:
            return PortfolioInterpretation(
                portfolio_risk="Mock portfolio risk: moderate sector concentration.",
                summary="Mock portfolio summary: holdings show mixed quality with positive aggregate P&L.",
            )
        if model_type is ComparisonInterpretation:
            return ComparisonInterpretation(
                valuation_comparison="Mock valuation: lower PE stock appears cheaper on earnings basis.",
                growth_comparison="Mock growth: one stock shows stronger revenue momentum.",
                risk_comparison="Mock risk: leverage and flagged risks differ across names.",
                technical_trend_comparison="Mock technical: one name has a clearer uptrend.",
                relative_assessment="Mock assessment: overall leader emerges on combined score.",
            )
        raise ValueError(f"No mock structured response registered for {model_type}")


def _example_value_for_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        inner = args[0] if args else str
        return [_example_value_for_annotation(inner)]
    if origin is dict:
        return {"key": "example"}
    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if non_none:
            return _example_value_for_annotation(non_none[0])
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    return "example"


def structured_output_example(model_type: type[BaseModel]) -> dict[str, Any]:
    """Compact instance example — not a JSON Schema with Field descriptions."""
    return {
        name: _example_value_for_annotation(field.annotation)
        for name, field in model_type.model_fields.items()
    }


def json_looks_truncated_or_invalid(content: str | None) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return True
    return False


def json_looks_like_schema(content: str | None) -> bool:
    try:
        parsed = json.loads((content or "").strip())
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "properties" in parsed and (
        parsed.get("type") == "object" or "title" in parsed or "$defs" in parsed
    )


class GroqLLMClient(LLMClient):
    """Groq-backed LLM client for production interpretation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq LLM client")
        self._client = None
        self._semaphore: asyncio.Semaphore | None = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            # Application retry policy owns 429 handling. The Groq SDK otherwise
            # retries with Retry-After (6s, 13s, 23s, …) on top of our attempts.
            self._client = AsyncGroq(
                api_key=self._settings.groq_api_key,
                max_retries=0,
            )
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            limit = max(1, self._settings.llm_max_concurrent_requests)
            self._semaphore = asyncio.Semaphore(limit)
        return self._semaphore

    def _boost_token_budget(self, current_tokens: int) -> int | None:
        boosted = min(
            current_tokens + self._settings.llm_structured_output_retry_boost,
            self._settings.llm_structured_output_retry_max_tokens,
        )
        if boosted <= current_tokens:
            return None
        return boosted

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        structured_output: type | None = None,
        max_tokens: int | None = None,
    ) -> str | Any:
        system_content = system or "You are a helpful financial analysis assistant."
        if structured_output is not None:
            example = json.dumps(
                structured_output_example(structured_output),
                separators=(",", ":"),
            )
            system_content += (
                "\nRespond with a JSON object only, using these keys and value types. "
                "Do not return a JSON Schema, field descriptions, or metadata.\n"
                + example
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        completion_tokens = max_tokens or self._settings.llm_max_tokens
        base_kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
        }
        if structured_output is not None:
            base_kwargs["response_format"] = {"type": "json_object"}

        async def _call_groq(tokens: int) -> str:
            client = self._get_client()
            async with self._get_semaphore():
                response = await client.chat.completions.create(
                    **base_kwargs,
                    max_tokens=tokens,
                )
            return response.choices[0].message.content or ""

        async def _invoke_with_rate_limit_retry(tokens: int) -> str:
            return await async_retry_with_backoff(
                lambda: _call_groq(tokens),
                max_attempts=self._settings.llm_retry_max_attempts,
                base_delay=self._settings.llm_retry_base_delay_seconds,
                max_delay=self._settings.llm_retry_max_delay_seconds,
                operation_name="Groq LLM",
                retry_on=is_rate_limit_error,
            )

        tokens = completion_tokens
        content: str | None = None
        token_boost_attempted = False

        while True:
            try:
                content = await _invoke_with_rate_limit_retry(tokens)
                break
            except Exception as exc:
                if is_rate_limit_error(exc):
                    logger.error(
                        "Groq rate limit exceeded after %d attempts: %s",
                        self._settings.llm_retry_max_attempts,
                        exc,
                    )
                    raise LLMRateLimitError(USER_FACING_RATE_LIMIT_MESSAGE) from exc

                if (
                    structured_output is not None
                    and not token_boost_attempted
                    and is_token_exhaustion_error(exc)
                ):
                    boosted_tokens = self._boost_token_budget(tokens)
                    if boosted_tokens is not None:
                        logger.warning(
                            "Structured output token exhaustion at %d tokens; "
                            "retrying once with %d tokens: %s",
                            tokens,
                            boosted_tokens,
                            exc,
                        )
                        tokens = boosted_tokens
                        token_boost_attempted = True
                        continue

                raise

        if structured_output is not None:
            try:
                return structured_output.model_validate_json(content or "")
            except ValidationError as exc:
                if json_looks_like_schema(content):
                    logger.warning(
                        "Structured output looked like a JSON Schema instead of an instance: %s",
                        exc,
                    )
                    raise
                if not token_boost_attempted and json_looks_truncated_or_invalid(content):
                    boosted_tokens = self._boost_token_budget(tokens)
                    if boosted_tokens is not None:
                        logger.warning(
                            "Structured output JSON validation failed at %d tokens; "
                            "retrying once with %d tokens: %s",
                            tokens,
                            boosted_tokens,
                            exc,
                        )
                        token_boost_attempted = True
                        content = await _invoke_with_rate_limit_retry(boosted_tokens)
                        return structured_output.model_validate_json(content or "")
                raise

        return content


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory — selects provider based on settings.llm_provider."""
    settings = settings or get_settings()
    if settings.llm_provider == "groq":
        if settings.groq_api_key:
            return GroqLLMClient(settings=settings)
        logger.warning("GROQ_API_KEY not set — using MockLLMClient")
        return MockLLMClient()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
