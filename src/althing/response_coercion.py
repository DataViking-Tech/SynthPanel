"""Post-hoc coercion of free-text panelist answers to a declared
``response_schema`` (sy-547).

A question may declare a typed ``response_schema`` — ``enum`` (a fixed set
of options) or ``scale`` (an integer range ``[min, max]``). Historically
this was validated at instrument-load and factored into the dry-run
token estimate, but never enforced or checked against the model's output:
panelists free-answered in prose (``"Blue."`` for an ``["red","green",
"blue"]`` enum), and downstream tooling (``poll-summary``) saw ``kind=text``.

This module performs the minimum-viable enforcement layer described in
issue #547 (b): given a declared schema and the raw free-text answer,
map the answer back to the nearest declared option (enum) or to an
integer inside the scale range (scale). It returns a :class:`CoercionResult`
carrying the typed value (or ``None`` when nothing maps), so the caller
can persist BOTH the raw text and the typed value and surface a
per-response warning + run-level failure count for unmappable answers.

The matching is deliberately conservative — it normalizes case,
whitespace, and surrounding punctuation, and accepts an exact normalized
match or an unambiguous single-option substring hit. It does NOT attempt
fuzzy/edit-distance matching: silently snapping ``"navy"`` to ``"blue"``
would manufacture data the panelist never produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoercionResult:
    """Outcome of coercing one free-text answer against a typed schema.

    ``kind`` is the declared schema type (``"enum"`` / ``"scale"``).
    ``value`` is the typed result (the canonical option string for enum,
    an ``int`` for scale) or ``None`` when the answer could not be mapped.
    ``mapped`` is ``True`` iff a value was recovered. ``raw`` echoes the
    original free-text answer for persistence.
    """

    kind: str
    raw: str
    value: Any | None
    mapped: bool


# Punctuation/whitespace stripped from both the answer and each option
# before comparison. Keeps internal characters (e.g. "price-band-a") so
# hyphenated options still match.
_EDGE_PUNCT = " \t\r\n.,;:!?\"'`()[]{}*_-—–"  # noqa: RUF001 - em/en dashes are intentional strip chars


def _normalize(text: str) -> str:
    """Lowercase, collapse internal whitespace, strip edge punctuation."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.strip(_EDGE_PUNCT).lower()


def is_typed_schema(response_schema: Any) -> bool:
    """True iff ``response_schema`` declares an enforceable typed shape.

    Only ``enum`` and ``scale`` are enforceable here; ``text`` and
    ``tagged_themes`` (and legacy inline JSON Schemas) are not coerced.
    """
    if not isinstance(response_schema, dict):
        return False
    return response_schema.get("type") in {"enum", "scale"}


def coerce_enum(raw: str, options: list[str]) -> CoercionResult:
    """Map a free-text answer to one of *options* (case/punctuation-insensitive).

    Resolution order:
      1. Exact normalized equality with an option.
      2. Unique substring hit — exactly one option's normalized form
         appears as a token-bounded substring of the normalized answer
         (catches ``"Blue."`` → ``"blue"`` and ``"I'd pick green"`` →
         ``"green"``). Ambiguous matches (the answer contains two
         options) do NOT map, so a hedging answer is flagged rather
         than arbitrarily resolved.
    """
    norm_answer = _normalize(raw)
    norm_options = [(opt, _normalize(opt)) for opt in options]

    # 1. Exact normalized match.
    for canonical, norm in norm_options:
        if norm and norm_answer == norm:
            return CoercionResult(kind="enum", raw=raw, value=canonical, mapped=True)

    # 2. Unique token-bounded substring match.
    hits = [canonical for canonical, norm in norm_options if norm and _contains_token(norm_answer, norm)]
    # De-dupe by canonical option (options are unique per schema validation,
    # but normalization could collapse two — guard anyway).
    unique_hits = list(dict.fromkeys(hits))
    if len(unique_hits) == 1:
        return CoercionResult(kind="enum", raw=raw, value=unique_hits[0], mapped=True)

    return CoercionResult(kind="enum", raw=raw, value=None, mapped=False)


def _contains_token(haystack: str, needle: str) -> bool:
    """True iff *needle* appears in *haystack* on word boundaries.

    Both are already normalized (lowercased, edge-stripped). Using word
    boundaries avoids ``"red"`` matching inside ``"predisposed"`` while
    still catching ``"blue."`` (normalized to ``"blue"``) and multi-word
    options like ``"price band a"``.
    """
    if needle == haystack:
        return True
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def coerce_scale(raw: str, lo: int, hi: int) -> CoercionResult:
    """Map a free-text answer to an integer in ``[lo, hi]``.

    Pulls the first integer token from the answer and accepts it only
    when it lands inside the declared range. ``"7"``, ``"I'd say 7 out
    of 10"`` → ``7``; ``"eleven"`` or ``"42"`` (out of range) → unmapped.
    """
    m = re.search(r"-?\d+", raw)
    if m is None:
        return CoercionResult(kind="scale", raw=raw, value=None, mapped=False)
    try:
        n = int(m.group())
    except ValueError:  # pragma: no cover - regex guarantees parseable
        return CoercionResult(kind="scale", raw=raw, value=None, mapped=False)
    if lo <= n <= hi:
        return CoercionResult(kind="scale", raw=raw, value=n, mapped=True)
    return CoercionResult(kind="scale", raw=raw, value=None, mapped=False)


def coerce_response(response_schema: Any, raw: Any) -> CoercionResult | None:
    """Coerce *raw* against a declared typed ``response_schema``.

    Returns ``None`` when the schema is not an enforceable typed shape
    (``is_typed_schema`` is false) or when *raw* is not a usable string —
    the caller leaves the response untouched in those cases. Otherwise
    returns a :class:`CoercionResult` (``mapped`` may be ``False``).
    """
    if not is_typed_schema(response_schema):
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None

    kind = response_schema["type"]
    if kind == "enum":
        options = response_schema.get("options")
        if not isinstance(options, list) or not options:
            return None
        return coerce_enum(raw, [o for o in options if isinstance(o, str)])
    if kind == "scale":
        lo = response_schema.get("min")
        hi = response_schema.get("max")
        if not isinstance(lo, int) or not isinstance(hi, int):
            return None
        return coerce_scale(raw, lo, hi)
    return None
