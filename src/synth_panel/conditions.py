"""Condition evaluation for conditional follow-ups (EXECUTION-PLAN-030.md F2-A).

Evaluates whether a follow-up question should be asked based on a condition
string and the response text from the main question.

Structured responses (dicts/JSON from response_schema) are serialized to string
via json.dumps before condition evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Callable


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_condition(condition: str, response_text: Any) -> bool:
    """Evaluate whether a follow-up should fire.

    Args:
        condition: Condition string. Supported forms:
            - ``"always"`` -- always fire (default)
            - ``"never"`` -- never fire
            - ``"response_contains: <keyword>"`` -- case-insensitive substring
            - Unknown types default to True (forward-compatible)
        response_text: The response from the main question. If a dict or other
            non-string type, it is serialized to JSON before evaluation.

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
