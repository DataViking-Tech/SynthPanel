"""Predicate engine and round router for v3 branching instruments.

Predicates are structured dicts on disk — no parser, no eval. Six
operators (``contains``, ``equals``, ``matches``, ``gte``, ``lte``,
``in``) over an open or closed set of fields. The router walks
``route_when`` clauses in order; first matching ``if`` wins, otherwise
the ``else`` clause (validated to exist by the instrument parser). The
reserved sentinel ``__end__`` marks a terminal target — the
orchestrator runs final synthesis on the path traversed so far.

The predicate engine is shared with the attachments stratification
filter (``synth_panel.attachments.filter``). Routing closes ``field``
to the SynthesisResult schema via ``_VALID_FIELDS``; attachments pass
``valid_fields=None`` to allow any persona key.
"""

from __future__ import annotations

import re
from typing import Any

END_SENTINEL = "__end__"

_VALID_FIELDS = frozenset({"themes", "recommendation", "disagreements", "summary", "agreements", "surprises"})


class RoutingError(ValueError):
    """Raised when a route_when clause cannot be resolved."""


def _evaluate_predicate(
    predicate: dict[str, Any],
    context: dict[str, Any],
    *,
    valid_fields: frozenset[str] | None,
) -> bool:
    """Evaluate a structured predicate against a context dict.

    ``predicate`` shape: ``{"field": str, "op": str, "value": Any}``.

    Supported ops:

    - ``contains`` — substring against any list entry or a string field.
    - ``equals`` — exact match; on list targets, "any element equals".
    - ``matches`` — ``re.search``; on list targets, "any element matches".
    - ``gte`` / ``lte`` — coerce both sides to ``float``; raise on
      non-numeric.
    - ``in`` — ``value`` is a list; passes if ``target`` equals any
      element, or for list targets, if any element appears in ``value``.

    When ``valid_fields`` is a frozenset the field name is enforced
    against that allowlist (routing). When ``None`` any key is allowed
    and a missing key on ``context`` evaluates as ``False`` (attachments
    against open-shape personas).
    """
    field = predicate["field"]
    op = predicate["op"]
    value = predicate["value"]

    if valid_fields is not None:
        if field not in valid_fields:
            raise KeyError(field)
        target = context[field]
    else:
        if field not in context:
            return False
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

    if op == "gte":
        return float(target) >= float(value)

    if op == "lte":
        return float(target) <= float(value)

    if op == "in":
        if not isinstance(value, list):
            raise ValueError(f"'in' op requires a list value, got {type(value).__name__}")
        if isinstance(target, list):
            return any(item in value for item in target)
        return target in value

    raise ValueError(f"unknown predicate op: {op!r}")


def evaluate_predicate(predicate: dict[str, Any], context: dict[str, Any]) -> bool:
    """Evaluate a routing predicate against a synthesis context.

    Thin wrapper that closes ``field`` to the SynthesisResult-shaped
    allowlist (``_VALID_FIELDS``). Unknown field raises ``KeyError``;
    unknown op raises ``ValueError``.
    """
    return _evaluate_predicate(predicate, context, valid_fields=_VALID_FIELDS)


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
