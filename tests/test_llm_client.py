"""Tests for MockLLMClient — no external API calls."""

import pytest

from app.agents.llm_client import MockLLMClient
from app.models.schemas import TechnicalInterpretation


@pytest.mark.asyncio
async def test_mock_llm_text_response():
    llm = MockLLMClient(default_text="hello")
    result = await llm.generate("test prompt")
    assert result == "hello"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_mock_llm_structured_default():
    llm = MockLLMClient()
    result = await llm.generate(
        "interpret",
        structured_output=TechnicalInterpretation,
    )
    assert isinstance(result, TechnicalInterpretation)
    assert result.momentum
    assert result.summary


@pytest.mark.asyncio
async def test_mock_llm_structured_custom():
    custom = TechnicalInterpretation(
        momentum="m",
        volatility="v",
        summary="s",
    )
    llm = MockLLMClient(structured_responses={TechnicalInterpretation: custom})
    result = await llm.generate("x", structured_output=TechnicalInterpretation)
    assert result.summary == "s"
