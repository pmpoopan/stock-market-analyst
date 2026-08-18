"""LLM client exceptions."""


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class LLMRateLimitError(LLMClientError):
    """Raised when the LLM provider rate limit is exceeded after retries."""


class LLMStructuredOutputError(LLMClientError):
    """Raised when structured JSON output cannot be produced after retries."""
