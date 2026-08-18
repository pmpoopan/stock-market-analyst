"""Configurable LLM client (Groq default).

Agents use this for interpretation/reasoning — not for numeric calculations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.agents.llm_exceptions import LLMRateLimitError
from app.config.settings import Settings, get_settings
from app.util.retry import async_retry_with_backoff, is_rate_limit_error

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

            self._client = AsyncGroq(api_key=self._settings.groq_api_key)
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            limit = max(1, self._settings.llm_max_concurrent_requests)
            self._semaphore = asyncio.Semaphore(limit)
        return self._semaphore

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        structured_output: type | None = None,
        max_tokens: int | None = None,
    ) -> str | Any:
        system_content = system or "You are a helpful financial analysis assistant."
        if structured_output is not None:
            schema_hint = json.dumps(
                structured_output.model_json_schema(),
                separators=(",", ":"),
            )
            system_content += (
                "\nRespond with valid JSON only, matching this schema:\n" + schema_hint
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        completion_tokens = max_tokens or self._settings.llm_max_tokens
        kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_tokens": completion_tokens,
        }
        if structured_output is not None:
            kwargs["response_format"] = {"type": "json_object"}

        async def _call_groq() -> str:
            client = self._get_client()
            async with self._get_semaphore():
                response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        try:
            content = await async_retry_with_backoff(
                _call_groq,
                max_attempts=self._settings.llm_retry_max_attempts,
                base_delay=self._settings.llm_retry_base_delay_seconds,
                max_delay=self._settings.llm_retry_max_delay_seconds,
                operation_name="Groq LLM",
                retry_on=is_rate_limit_error,
            )
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.error(
                    "Groq rate limit exceeded after %d attempts: %s",
                    self._settings.llm_retry_max_attempts,
                    exc,
                )
                raise LLMRateLimitError(USER_FACING_RATE_LIMIT_MESSAGE) from exc
            raise

        if structured_output is not None:
            return structured_output.model_validate_json(content)

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
