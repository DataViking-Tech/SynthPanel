"""Predicate engine and round router for v3 branching instruments.

Predicates are structured dicts on disk — no parser, no eval. Three
operators (``contains``, ``equals``, ``matches``) over the fields of a
``SynthesisResult``-shaped context. The router walks ``route_when``
clauses in order; first matching ``if`` wins, otherwise the ``else``
clause (validated to exist by the instrument parser). The reserved
sentinel ``__end__`` marks a terminal target — the orchestrator runs
final synthesis on the path traversed so far.
"""

from __future__ import annotations

import re
from typing import Any

END_SENTINEL = "__end__"

_VALID_FIELDS = frozenset({"themes", "recommendation", "disagreements", "summary", "agreements", "surprises"})


class RoutingError(ValueError):
    """Raised when a route_when clause cannot be resolved."""


def evaluate_predicate(predicate: dict[str, Any], context: dict[str, Any]) -> bool:
    """Evaluate a structured predicate against a synthesis context.

    ``predicate`` shape: ``{"field": str, "op": str, "value": str}``.
    Supported ops: ``contains`` (substring against any list entry or
    against a string field), ``equals`` (exact string match), ``matches``
    (``re.search``). Unknown field raises ``KeyError`` with the offending
    name; unknown op raises ``ValueError``.
    """
    field = predicate["field"]
    op = predicate["op"]
    value = predicate["value"]

    if field not in _VALID_FIELDS:
        raise KeyError(field)

    target = context[field]

    if op == "contains":
        if isinstance(target, list):
            return any(isinstance(item, str) and value in item for item in target)
        if isinstance(target, str):
            return value in target
        return False

    if op == "equals":
        if isinstance(target, list):
            return any(item == value for item in target)
        return target == value

    if op == "matches":
        pattern = re.compile(value)
        if isinstance(target, list):
            return any(isinstance(item, str) and pattern.search(item) for item in target)
        if isinstance(target, str):
            return bool(pattern.search(target))
        return False

    raise ValueError(f"unknown predicate op: {op!r}")


def route_round(route_when: list[dict[str, Any]], context: dict[str, Any]) -> str:
    """Walk a ``route_when`` block, returning the next round target.

    Each clause is either ``{"if": <predicate>, "goto": <round_name>}``
    or ``{"else": <round_name>}``. The first clause whose predicate
    matches wins; otherwise the trailing ``else`` clause's target is
    returned. The instrument parser guarantees an ``else`` exists, so
    a missing one here is a programmer error and raises
    :class:`RoutingError`. The terminal sentinel ``__end__`` may appear
    as any target.
    """
    for clause in route_when:
        if "if" in clause:
            if evaluate_predicate(clause["if"], context):
                return clause["goto"]
        elif "else" in clause:
            return clause["else"]
    raise RoutingError("route_when block has no matching clause and no else")
