"""Sentiment Analyst agent.

Flow:
  1. Web search collects real news articles
  2. Deduplicate and filter irrelevant results
  3. SentimentAnalysisEngine scores in Python
  4. LLM analyzes sentiment from provided articles only
  5. Never fabricate sources or news
"""

from __future__ import annotations

import json
import logging

from pydantic import HttpUrl

from app.analysis.sentiment_analysis import SentimentAnalysisEngine, WORD_PATTERN
from app.agents.llm_client import LLMClient
from app.data.interfaces import NewsSearchProvider
from app.models.schemas import NewsArticle, SentimentAnalysisResult, SentimentInterpretation

logger = logging.getLogger(__name__)


class SentimentAnalyst:
    """Analyze news, sentiment, catalysts, and risks from web search."""

    SYSTEM_PROMPT = (
        "You are a sentiment analyst for Indian equities. "
        "Analyze only the news articles provided. "
        "Never fabricate sources, URLs, or publication dates. "
        "Identify positive/negative catalysts and key events from the articles. "
        "Clearly distinguish reported facts from interpretation."
    )

    def __init__(
        self,
        news_search: NewsSearchProvider,
        llm: LLMClient,
        max_articles: int = 15,
    ) -> None:
        self._news_search = news_search
        self._llm = llm
        self._max_articles = max_articles

    def _build_search_queries(self, symbol: str, company_name: str | None) -> list[str]:
        """Build targeted search queries for news collection."""
        base_name = company_name or symbol.replace(".NS", "").replace(".BO", "")
        return [
            f"{base_name} stock India news",
            f"{base_name} earnings results India",
            f"{base_name} management regulatory India",
        ]

    def _deduplicate_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[NewsArticle] = []

        for article in articles:
            url_key = str(article.url).lower().rstrip("/")
            title_key = article.title.lower().strip()
            if url_key in seen_urls or title_key in seen_titles:
                continue
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            unique.append(article)

        return unique

    def _filter_relevant(
        self,
        articles: list[NewsArticle],
        symbol: str,
        company_name: str | None,
        queries: list[str],
    ) -> list[NewsArticle]:
        terms: set[str] = set()
        base_symbol = symbol.replace(".NS", "").replace(".BO", "").lower()
        if base_symbol:
            terms.add(base_symbol)
        if company_name:
            terms.add(company_name.lower())
            for part in company_name.lower().split():
                if len(part) > 3:
                    terms.add(part)
        for query in queries:
            for word in query.lower().split():
                if len(word) > 3:
                    terms.add(word)

        relevant: list[NewsArticle] = []
        for article in articles:
            text = f"{article.title} {article.snippet or ''}".lower()
            words = set(WORD_PATTERN.findall(text))
            if any(term in words for term in terms):
                relevant.append(article)

        return relevant

    async def _collect_articles(
        self,
        symbol: str,
        company_name: str | None,
    ) -> list[NewsArticle]:
        queries = self._build_search_queries(symbol, company_name)
        collected: list[NewsArticle] = []

        per_query = max(3, self._max_articles // len(queries))
        for query in queries:
            try:
                batch = self._news_search.search_news(query, max_results=per_query)
                collected.extend(batch)
            except Exception as exc:
                logger.warning("News search failed for query '%s': %s", query, exc)

        deduped = self._deduplicate_articles(collected)
        filtered = self._filter_relevant(deduped, symbol, company_name, queries)
        return filtered[:self._max_articles]

    async def analyze(self, symbol: str, company_name: str | None = None) -> SentimentAnalysisResult:
        normalized = symbol.strip().upper()
        articles = await self._collect_articles(normalized, company_name)

        analysis_engine = SentimentAnalysisEngine(articles)
        score = analysis_engine.compute_score()
        classification = analysis_engine.classify(score)

        if not articles:
            return SentimentAnalysisResult(
                stock=normalized,
                sentiment_score=score,
                sentiment_classification=classification,
                positive_catalysts=[],
                negative_catalysts=[],
                key_events=[],
                sources=[],
                publication_dates=[],
                articles=[],
                summary=analysis_engine.build_summary(normalized, score, classification),
            )

        positive, negative, key_events, summary = await self._interpret_with_llm(
            symbol=normalized,
            analysis_engine=analysis_engine,
            score=score,
            classification=classification,
        )

        sources = [HttpUrl(str(article.url)) for article in articles]
        publication_dates = [article.published_at for article in articles]

        return SentimentAnalysisResult(
            stock=normalized,
            sentiment_score=score,
            sentiment_classification=classification,
            positive_catalysts=positive,
            negative_catalysts=negative,
            key_events=key_events,
            sources=sources,
            publication_dates=publication_dates,
            articles=articles,
            summary=summary,
        )

    async def _interpret_with_llm(
        self,
        symbol: str,
        analysis_engine: SentimentAnalysisEngine,
        score: float,
        classification,
    ) -> tuple[list[str], list[str], list[str], str]:
        llm_payload = analysis_engine.summarize_for_llm(score=score, classification=classification)
        llm_payload["symbol"] = symbol

        prompt = (
            "Analyze sentiment for the following Indian equity using only the provided articles. "
            "Return positive catalysts, negative catalysts, key events, and a summary. "
            "Do not invent articles or URLs.\n\n"
            + json.dumps(llm_payload, indent=2, default=str)
        )

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                structured_output=SentimentInterpretation,
            )
            if isinstance(interpretation, SentimentInterpretation):
                return (
                    interpretation.positive_catalysts,
                    interpretation.negative_catalysts,
                    interpretation.key_events,
                    interpretation.summary,
                )
        except Exception as exc:
            logger.warning("LLM interpretation failed for %s: %s", symbol, exc)

        return (
            analysis_engine.build_positive_catalysts(),
            analysis_engine.build_negative_catalysts(),
            analysis_engine.build_key_events(),
            analysis_engine.build_summary(symbol, score, classification),
        )
