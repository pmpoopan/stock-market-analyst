"""LLM client exceptions."""


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class LLMRateLimitError(LLMClientError):
    """Raised when the LLM provider rate limit is exceeded after retries."""
