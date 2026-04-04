"""Error types for the LLM client (SPEC.md §2 — Error Handling)."""

from __future__ import annotations

from enum import Enum


class LLMErrorCategory(Enum):
    """Classification of LLM errors, controlling retry behaviour."""
    MISSING_CREDENTIALS = "missing_credentials"
    AUTHENTICATION = "authentication"
    TRANSPORT = "transport"
    RATE_LIMIT = "rate_limit"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    DESERIALIZATION = "deserialization"
    RETRIES_EXHAUSTED = "retries_exhausted"


# Categories that may be retried.
RETRYABLE_CATEGORIES = frozenset({
    LLMErrorCategory.TRANSPORT,
    LLMErrorCategory.RATE_LIMIT,
    LLMErrorCategory.SERVER_ERROR,
})


class LLMError(Exception):
    """Typed error from the LLM client layer."""

    def __init__(
        self,
        message: str,
        category: LLMErrorCategory,
        *,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES

    def __repr__(self) -> str:
        return (
            f"LLMError(category={self.category.value!r}, "
            f"status={self.status_code}, msg={str(self)!r})"
        )


def classify_http_status(status: int) -> LLMErrorCategory:
    """Map an HTTP status code to an error category."""
    if status == 401 or status == 403:
        return LLMErrorCategory.AUTHENTICATION
    if status == 429:
        return LLMErrorCategory.RATE_LIMIT
    if 400 <= status < 500:
        return LLMErrorCategory.BAD_REQUEST
    if status >= 500:
        return LLMErrorCategory.SERVER_ERROR
    return LLMErrorCategory.TRANSPORT
