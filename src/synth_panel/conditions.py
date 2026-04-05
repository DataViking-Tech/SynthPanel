"""Condition evaluation for conditional follow-ups.

Evaluates whether a follow-up question should be asked based on a condition
string and the response text from the main question.

Structured responses (dicts/JSON from response_schema) are serialized to string
via json.dumps before condition evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from synth_panel.llm.client import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------

def _eval_contains(keyword: str, response_text: str) -> bool:
    """Case-insensitive substring match."""
    return keyword.lower() in response_text.lower()


EVALUATORS: dict[str, Callable[[str, str], bool]] = {
    "always": lambda _kw, _resp: True,
    "never": lambda _kw, _resp: False,
    "response_contains": _eval_contains,
}

_SENTIMENT_PROMPT = (
    "Classify the sentiment of the following text as exactly one of: "
    "positive, negative, neutral.\n\n"
    "Text: {text}\n\n"
    "Respond with a single word: positive, negative, or neutral."
)

_VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Sentiment evaluator (LLM-based)
# ---------------------------------------------------------------------------

def _eval_sentiment(
    target: str,
    response_text: str,
    client: LLMClient,
    sentiment_cache: dict[str, str] | None = None,
) -> bool:
    """Classify response sentiment via a haiku LLM call.

    Args:
        target: Expected sentiment (positive, negative, neutral).
        response_text: Text to classify.
        client: LLM client for making the classification call.
        sentiment_cache: Optional cache mapping response_text -> sentiment.
            Avoids duplicate LLM calls for the same response.

    Returns:
        True if the classified sentiment matches *target*.
    """
    from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock

    target = target.lower().strip()

    # Check cache first
    if sentiment_cache is not None and response_text in sentiment_cache:
        cached = sentiment_cache[response_text]
        log.debug("sentiment cache hit: %r -> %s", response_text[:40], cached)
        return cached == target

    prompt = _SENTIMENT_PROMPT.format(text=response_text)
    request = CompletionRequest(
        model="haiku",
        max_tokens=16,
        messages=[InputMessage(role="user", content=[TextBlock(text=prompt)])],
    )
    response = client.send(request)
    classification = response.text.strip().lower()

    # Validate — fall back to neutral on unexpected output
    if classification not in _VALID_SENTIMENTS:
        log.warning(
            "Unexpected sentiment classification %r for text %r, defaulting to neutral",
            classification, response_text[:60],
        )
        classification = "neutral"

    # Populate cache
    if sentiment_cache is not None:
        sentiment_cache[response_text] = classification

    return classification == target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    response_text: Any,
    *,
    client: LLMClient | None = None,
    sentiment_cache: dict[str, str] | None = None,
) -> bool:
    """Evaluate whether a follow-up should fire.

    Args:
        condition: Condition string. Supported forms:
            - ``"always"`` -- always fire (default)
            - ``"never"`` -- never fire
            - ``"response_contains: <keyword>"`` -- case-insensitive substring
            - ``"response_sentiment: <positive|negative|neutral>"`` -- LLM-based
            - Unknown types default to True (forward-compatible)
        response_text: The response from the main question. If a dict or other
            non-string type, it is serialized to JSON before evaluation.
        client: Optional LLM client, required for ``response_sentiment``.
        sentiment_cache: Optional cache to avoid duplicate LLM calls for
            the same response text.

    Returns:
        True if the follow-up should be asked.
    """
    if not isinstance(response_text, str):
        response_text = json.dumps(response_text)

    condition = condition.strip()

    # Check for parameterised conditions ("type: arg")
    if ":" in condition:
        ctype, _, arg = condition.partition(":")
        ctype = ctype.strip().lower()
        arg = arg.strip()
    else:
        ctype = condition.lower()
        arg = ""

    # Handle response_sentiment separately (needs LLM client)
    if ctype == "response_sentiment":
        if client is None:
            # Graceful degradation: no client → default to True
            log.debug("response_sentiment: no client, defaulting to True")
            return True
        return _eval_sentiment(arg, response_text, client, sentiment_cache)

    evaluator = EVALUATORS.get(ctype)
    if evaluator is None:
        # Unknown condition type -- default to always (forward-compat)
        return True

    return evaluator(arg, response_text)


def normalize_follow_up(follow_up: str | dict) -> dict:
    """Normalize a follow-up to ``{"text": ..., "condition": "always"}``.

    Args:
        follow_up: Either a plain string (the question text) or a dict that
            must contain a ``"text"`` key and optionally a ``"condition"`` key.

    Returns:
        A dict with at least ``"text"`` and ``"condition"`` keys.
    """
    if isinstance(follow_up, str):
        return {"text": follow_up, "condition": "always"}

    # Dict -- ensure condition defaults to "always"
    result = dict(follow_up)
    result.setdefault("condition", "always")
    return result
