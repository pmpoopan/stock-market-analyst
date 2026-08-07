"""Tests for MasterAnalyst — mock LLM, no live API."""

import pytest

from app.agents.llm_client import MockLLMClient
from app.agents.master_agent import MasterAnalyst
from app.analysis.master_synthesis import build_master_analysis_fallback
from app.models.schemas import MasterInterpretation
from tests.fixtures.fundamental_data import make_mock_financial_metrics
from tests.fixtures.market_data import MOCK_SYMBOL
from app.models.schemas import (
    FundamentalAnalysisResult,
    Rating,
    SentimentAnalysisResult,
    SentimentClassification,
    TechnicalAnalysisResult,
    TrendDirection,
)


def _sample_inputs():
    metrics = make_mock_financial_metrics()
    fundamental = FundamentalAnalysisResult(
        stock=MOCK_SYMBOL,
        score=75.0,
        rating=Rating.BUY,
        metrics=metrics,
        strengths=["Strong revenue growth"],
        weaknesses=["High valuation"],
        risks=["Leverage risk"],
        summary="Solid fundamentals.",
    )
    technical = TechnicalAnalysisResult(
        stock=MOCK_SYMBOL,
        score=70.0,
        trend=TrendDirection.UPTREND,
        signals=[],
        momentum="Positive",
        volatility="Moderate",
        summary="Uptrend intact.",
    )
    sentiment = SentimentAnalysisResult(
        stock=MOCK_SYMBOL,
        sentiment_score=60.0,
        sentiment_classification=SentimentClassification.POSITIVE,
        positive_catalysts=["Earnings beat"],
        negative_catalysts=["Regulatory noise"],
        key_events=["Q3 results"],
        sources=[],
        publication_dates=[],
        articles=[],
        summary="Mixed sentiment.",
    )
    return fundamental, technical, sentiment


@pytest.mark.asyncio
async def test_synthesize_uses_mock_llm():
    custom = MasterInterpretation(
        agreement_points=["Custom agreement"],
        disagreement_points=["Custom disagreement"],
        major_risks=["Custom risk"],
        important_catalysts=["Custom catalyst"],
        narrative="Custom narrative.",
        data_vs_interpretation="Custom distinction.",
    )
    llm = MockLLMClient(structured_responses={MasterInterpretation: custom})
    analyst = MasterAnalyst(llm)

    fundamental, technical, sentiment = _sample_inputs()
    result = await analyst.synthesize(MOCK_SYMBOL, fundamental, technical, sentiment)

    assert result.stock == MOCK_SYMBOL
    assert result.narrative == "Custom narrative."
    assert result.agreement_points == ["Custom agreement"]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_synthesize_fallback_on_llm_failure():
    class FailingLLM(MockLLMClient):
        async def generate(self, prompt, system=None, structured_output=None):
            raise RuntimeError("LLM down")

    analyst = MasterAnalyst(FailingLLM())
    fundamental, technical, sentiment = _sample_inputs()
    result = await analyst.synthesize(MOCK_SYMBOL, fundamental, technical, sentiment)

    fallback = build_master_analysis_fallback(MOCK_SYMBOL, fundamental, technical, sentiment)
    assert result.agreement_points == fallback.agreement_points
    assert MOCK_SYMBOL in result.narrative


@pytest.mark.asyncio
async def test_synthesize_default_mock_response():
    analyst = MasterAnalyst(MockLLMClient())
    fundamental, technical, sentiment = _sample_inputs()
    result = await analyst.synthesize(MOCK_SYMBOL, fundamental, technical, sentiment)

    assert "Mock narrative" in result.narrative
    assert len(result.major_risks) > 0
