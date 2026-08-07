"""Data layer exceptions."""


class DataProviderError(Exception):
    """Base exception for data provider failures."""


class DataNotFoundError(DataProviderError):
    """Raised when requested market data is unavailable."""


class CacheError(DataProviderError):
    """Raised when cache operations fail."""
