"""Mock news articles for tests — no live search or LLM calls."""

from datetime import date

from app.models.schemas import NewsArticle

MOCK_NEWS_POSITIVE = NewsArticle(
    title="Reliance reports strong quarterly profit growth",
    source="Economic Times",
    url="https://economictimes.indiatimes.com/markets/stocks/news/reliance-profit-growth",
    published_at=date(2026, 1, 10),
    snippet="Reliance Industries posted record profit growth driven by retail and energy segments.",
)

MOCK_NEWS_NEGATIVE = NewsArticle(
    title="Reliance faces regulatory scrutiny over compliance concerns",
    source="Business Standard",
    url="https://www.business-standard.com/markets/reliance-regulatory-scrutiny",
    published_at=date(2026, 1, 8),
    snippet="Regulators raised compliance concern and investigation risk impacting sentiment.",
)

MOCK_NEWS_NEUTRAL = NewsArticle(
    title="Reliance Industries announces board meeting schedule",
    source="Moneycontrol",
    url="https://www.moneycontrol.com/news/business/reliance-board-meeting",
    published_at=date(2026, 1, 5),
    snippet="The company announced dates for the upcoming board meeting.",
)

MOCK_NEWS_DUPLICATE = NewsArticle(
    title="Reliance reports strong quarterly profit growth",
    source="Economic Times",
    url="https://economictimes.indiatimes.com/markets/stocks/news/reliance-profit-growth",
    published_at=date(2026, 1, 10),
    snippet="Duplicate URL article should be removed during deduplication.",
)

MOCK_NEWS_IRRELEVANT = NewsArticle(
    title="Global tech stocks rally on AI optimism",
    source="Reuters",
    url="https://www.reuters.com/markets/global-tech-rally",
    published_at=date(2026, 1, 7),
    snippet="Technology stocks gained amid broader AI enthusiasm.",
)


def make_mock_articles() -> list[NewsArticle]:
    return [MOCK_NEWS_POSITIVE, MOCK_NEWS_NEGATIVE, MOCK_NEWS_NEUTRAL]


class MockNewsSearchProvider:
    """In-memory news search for unit/integration tests."""

    def __init__(self, articles_by_query: dict[str, list[NewsArticle]] | None = None) -> None:
        if articles_by_query is None:
            default = make_mock_articles()
            self._articles_by_query = {
                "reliance stock india news": default,
                "reliance earnings results india": default,
                "reliance management regulatory india": default,
            }
        else:
            self._articles_by_query = articles_by_query
        self.calls: list[dict] = []

    def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
        self.calls.append({"query": query, "max_results": max_results})
        articles = self._articles_by_query.get(query.lower(), [])
        if not articles:
            for key, value in self._articles_by_query.items():
                if query.lower() in key or key in query.lower():
                    articles = value
                    break
        if not articles and self._articles_by_query:
            articles = next(iter(self._articles_by_query.values()))
        return articles[:max_results]
