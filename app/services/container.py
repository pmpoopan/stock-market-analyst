"""Dependency injection container — wires data providers, agents, and orchestrator.

Central place to construct and share service instances across API and graph layers.
"""

from functools import lru_cache

from app.agents.comparison_agent import ComparisonAnalyst
from app.agents.fundamental_agent import FundamentalAnalyst
from app.agents.llm_client import create_llm_client
from app.agents.master_agent import MasterAnalyst
from app.agents.portfolio_agent import PortfolioAnalyst
from app.agents.query_parser import QueryParser
from app.agents.sentiment_agent import SentimentAnalyst
from app.agents.technical_agent import TechnicalAnalyst
from app.analysis.scoring import ScoringEngine
from app.config.settings import Settings, get_settings
from app.data.cache import create_cache
from app.data.web_search import DuckDuckGoSearchProvider
from app.data.yahoo_finance import YahooFinanceProvider
from app.graph.deps import GraphDependencies
from app.graph.workflow import AnalysisOrchestrator


class ServiceContainer:
    """Lazy-initialized service registry."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache = None
        self._market_data = None
        self._news_search = None
        self._llm = None
        self._orchestrator = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = create_cache(self.settings.cache_db_path)
        return self._cache

    @property
    def market_data(self) -> YahooFinanceProvider:
        if self._market_data is None:
            cache = self.cache if self.settings.cache_enabled else None
            self._market_data = YahooFinanceProvider(cache=cache)
        return self._market_data

    @property
    def news_search(self) -> DuckDuckGoSearchProvider:
        if self._news_search is None:
            cache = self.cache if self.settings.cache_enabled else None
            self._news_search = DuckDuckGoSearchProvider(
                cache=cache,
                max_results=self.settings.web_search_max_results,
            )
        return self._news_search

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm_client(self.settings)
        return self._llm

    @property
    def query_parser(self) -> QueryParser:
        return QueryParser(llm=self.llm)

    @property
    def fundamental_analyst(self) -> FundamentalAnalyst:
        return FundamentalAnalyst(market_data=self.market_data, llm=self.llm)

    @property
    def technical_analyst(self) -> TechnicalAnalyst:
        return TechnicalAnalyst(market_data=self.market_data, llm=self.llm)

    @property
    def sentiment_analyst(self) -> SentimentAnalyst:
        return SentimentAnalyst(news_search=self.news_search, llm=self.llm)

    @property
    def master_analyst(self) -> MasterAnalyst:
        return MasterAnalyst(llm=self.llm)

    @property
    def scoring_engine(self) -> ScoringEngine:
        return ScoringEngine(settings=self.settings)

    @property
    def comparison_analyst(self) -> ComparisonAnalyst:
        return ComparisonAnalyst(llm=self.llm)

    @property
    def portfolio_analyst(self) -> PortfolioAnalyst:
        return PortfolioAnalyst(
            market_data=self.market_data,
            fundamental_analyst=self.fundamental_analyst,
            technical_analyst=self.technical_analyst,
            sentiment_analyst=self.sentiment_analyst,
            scoring_engine=self.scoring_engine,
            llm=self.llm,
            master_analyst=self.master_analyst,
        )

    def graph_dependencies(self) -> GraphDependencies:
        return GraphDependencies(
            query_parser=self.query_parser,
            fundamental_analyst=self.fundamental_analyst,
            technical_analyst=self.technical_analyst,
            sentiment_analyst=self.sentiment_analyst,
            master_analyst=self.master_analyst,
            comparison_analyst=self.comparison_analyst,
            portfolio_analyst=self.portfolio_analyst,
            scoring_engine=self.scoring_engine,
            market_data=self.market_data,
        )

    @property
    def orchestrator(self) -> AnalysisOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = AnalysisOrchestrator(deps=self.graph_dependencies())
        return self._orchestrator


@lru_cache
def get_container() -> ServiceContainer:
    return ServiceContainer()
