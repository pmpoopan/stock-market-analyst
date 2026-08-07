"""Deterministic sentiment scoring from news articles.

Keyword-based scoring lives here — LLM enriches catalysts and narrative only.
"""

from __future__ import annotations

import re
from datetime import date

from app.models.schemas import NewsArticle, SentimentClassification

POSITIVE_TERMS = frozenset(
    {
        "growth",
        "profit",
        "beat",
        "beats",
        "upgrade",
        "expansion",
        "dividend",
        "record",
        "strong",
        "rally",
        "acquisition",
        "partnership",
        "approval",
        "outperform",
        "surge",
        "gain",
        "positive",
        "bullish",
    }
)

NEGATIVE_TERMS = frozenset(
    {
        "loss",
        "decline",
        "downgrade",
        "miss",
        "fraud",
        "lawsuit",
        "penalty",
        "weak",
        "cut",
        "slowdown",
        "breach",
        "investigation",
        "bearish",
        "slump",
        "drop",
        "negative",
        "concern",
        "risk",
    }
)

WORD_PATTERN = re.compile(r"[a-zA-Z]+")


class SentimentAnalysisEngine:
    """Score sentiment and classify from collected news articles."""

    def __init__(self, articles: list[NewsArticle]) -> None:
        self._articles = articles

    @property
    def article_count(self) -> int:
        return len(self._articles)

    def compute_score(self) -> float:
        if not self._articles:
            return 50.0

        scores = [self._article_score(article) for article in self._articles]
        return round(sum(scores) / len(scores), 2)

    def classify(self, score: float) -> SentimentClassification:
        if score >= 75:
            return SentimentClassification.VERY_POSITIVE
        if score >= 60:
            return SentimentClassification.POSITIVE
        if score >= 40:
            return SentimentClassification.NEUTRAL
        if score >= 25:
            return SentimentClassification.NEGATIVE
        return SentimentClassification.VERY_NEGATIVE

    def build_positive_catalysts(self) -> list[str]:
        catalysts: list[str] = []
        for article in self._articles:
            text = self._article_text(article)
            if self._count_terms(text, POSITIVE_TERMS) > self._count_terms(text, NEGATIVE_TERMS):
                catalysts.append(article.title)
        return catalysts[:5]

    def build_negative_catalysts(self) -> list[str]:
        catalysts: list[str] = []
        for article in self._articles:
            text = self._article_text(article)
            if self._count_terms(text, NEGATIVE_TERMS) > self._count_terms(text, POSITIVE_TERMS):
                catalysts.append(article.title)
        return catalysts[:5]

    def build_key_events(self) -> list[str]:
        return [article.title for article in self._articles[:5]]

    def build_summary(self, symbol: str, score: float, classification: SentimentClassification) -> str:
        if not self._articles:
            return (
                f"No recent news articles were found for {symbol}. "
                f"Sentiment is neutral (score {score:.0f}/100) due to limited coverage."
            )
        pos = len(self.build_positive_catalysts())
        neg = len(self.build_negative_catalysts())
        return (
            f"{symbol} sentiment score is {score:.0f}/100 ({classification.value}) "
            f"based on {self.article_count} articles "
            f"({pos} positive-leaning, {neg} negative-leaning headlines)."
        )

    def summarize_for_llm(self, score: float, classification: SentimentClassification) -> dict:
        return {
            "article_count": self.article_count,
            "sentiment_score": score,
            "sentiment_classification": classification.value,
            "articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "url": str(a.url),
                    "published_at": a.published_at.isoformat() if a.published_at else None,
                    "snippet": a.snippet,
                }
                for a in self._articles
            ],
            "deterministic_positive_catalysts": self.build_positive_catalysts(),
            "deterministic_negative_catalysts": self.build_negative_catalysts(),
            "deterministic_key_events": self.build_key_events(),
        }

    def _article_score(self, article: NewsArticle) -> float:
        text = self._article_text(article)
        positive = self._count_terms(text, POSITIVE_TERMS)
        negative = self._count_terms(text, NEGATIVE_TERMS)
        score = 50.0 + (positive - negative) * 8.0
        return max(0.0, min(100.0, score))

    def _article_text(self, article: NewsArticle) -> str:
        return f"{article.title} {article.snippet or ''}".lower()

    def _count_terms(self, text: str, terms: frozenset[str]) -> int:
        words = WORD_PATTERN.findall(text)
        return sum(1 for word in words if word in terms)
