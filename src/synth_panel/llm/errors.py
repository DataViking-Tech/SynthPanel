"""Error types for the LLM client (SPEC.md §2 — Error Handling)."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


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
RETRYABLE_CATEGORIES = frozenset(
    {
        LLMErrorCategory.TRANSPORT,
        LLMErrorCategory.RATE_LIMIT,
        LLMErrorCategory.SERVER_ERROR,
    }
)


class LLMError(Exception):
    """Typed error from the LLM client layer."""

    def __init__(
        self,
        message: str,
        category: LLMErrorCategory,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retry_after = retry_after
        self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES

    def __repr__(self) -> str:
        return (
            f"LLMError(category={self.category.value!r}, status={self.status_code}, "
            f"retry_after={self.retry_after}, msg={str(self)!r})"
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


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value.

    The header may be an integer/float number of seconds or an HTTP-date.
    Returns seconds-to-wait as a float, or None if the value is missing
    or unparseable. Negative values are clamped to 0.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)

    try:
        from datetime import datetime, timezone

        when = parsedate_to_datetime(value)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def retry_after_from_headers(headers: object) -> float | None:
    """Extract a retry-after hint from an HTTP response's headers.

    Accepts any httpx-style case-insensitive ``headers`` mapping. Looks at
    ``Retry-After`` first (RFC 7231), then common provider extensions
    (``x-ratelimit-reset``, ``anthropic-ratelimit-requests-reset``).
    Returns seconds-to-wait as a float, or None.
    """
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if get is None:
        return None
    for name in (
        "retry-after",
        "x-ratelimit-reset",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-reset",
    ):
        raw = get(name)
        parsed = parse_retry_after(raw)
        if parsed is not None:
            return parsed
    return None


def llm_error_from_response(
    resp: httpx.Response,
    provider_name: str,
) -> LLMError:
    """Build an LLMError from a non-2xx ``httpx.Response``.

    Classifies by status code and extracts a retry-after hint from the
    response headers. Use this helper from provider ``send``/``stream``
    paths to keep retry-after propagation consistent across providers.
    """
    status = resp.status_code
    category = classify_http_status(status)
    retry_after = retry_after_from_headers(resp.headers) if category == LLMErrorCategory.RATE_LIMIT else None
    return LLMError(
        f"{provider_name} API error {status}: {resp.text[:500]}",
        category,
        status_code=status,
        retry_after=retry_after,
    )
