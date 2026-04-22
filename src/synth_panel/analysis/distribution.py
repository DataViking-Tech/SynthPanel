"""Deterministic distributions over structured panelist responses (sp-2hpi).

Given a question's ``response_schema`` and the per-panelist raw responses,
this module computes a distribution summary without any LLM call:

* ``scale``  — frequency table, mean, median, stdev, min, max
* ``enum``   — frequency table over the declared options (zero-filled)
* ``tagged_themes`` — per-tag frequency over the declared taxonomy
  (zero-filled), with a bucket for off-taxonomy ``other`` themes
* ``text``   — only ``n`` and ``n_nonempty`` (text carries no
  distribution on its own; extraction lives upstream)

The output is intentionally plain dicts/lists so the result can be
serialized directly into the panel result's ``analysis`` section.
This module is the scaffolding the narrative-over-stats and
``--analysis-mode`` layers will ride on top of in follow-up work.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

__all__ = [
    "InvalidResponseSchemaError",
    "coerce_enum_value",
    "coerce_scale_value",
    "coerce_tagged_themes",
    "distribution_for_question",
]


_RECOGNIZED_TYPES: frozenset[str] = frozenset({"text", "scale", "enum", "tagged_themes"})


class InvalidResponseSchemaError(ValueError):
    """Raised when a response_schema lacks a recognized ``type``.

    Matches the contract in :mod:`synth_panel.instrument`: a ``response_schema``
    dict without a recognized ``type`` is treated as a legacy inline JSON
    Schema by the parser. This module only computes distributions for the
    *semantic* types, so callers must pre-filter.
    """


def distribution_for_question(
    responses: list[Any],
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Compute a distribution summary for one question's responses.

    Args:
        responses: Raw per-panelist response values. Shape depends on
            ``response_schema['type']``; unparseable entries are counted
            as ``n_invalid``, not dropped silently.
        response_schema: Question-level schema as validated by
            :func:`synth_panel.instrument._validate_response_schema`.

    Returns:
        A plain dict with at least ``type``, ``n`` (total responses),
        ``n_valid``, and ``n_invalid``. Shape beyond that is type-specific
        — see module docstring.

    Raises:
        InvalidResponseSchemaError: If ``response_schema`` is missing a
            recognized ``type``.
    """
    t = response_schema.get("type") if isinstance(response_schema, dict) else None
    if t not in _RECOGNIZED_TYPES:
        raise InvalidResponseSchemaError(f"response_schema must have type in {sorted(_RECOGNIZED_TYPES)}, got {t!r}")

    n = len(responses)
    if t == "scale":
        return _scale_distribution(responses, response_schema, n)
    if t == "enum":
        return _enum_distribution(responses, response_schema, n)
    if t == "tagged_themes":
        return _tagged_themes_distribution(responses, response_schema, n)
    return _text_distribution(responses, n)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def coerce_scale_value(value: Any, lo: int, hi: int) -> int | None:
    """Return a clamped integer scale value, or ``None`` if uncoerceable.

    Accepts ints, floats (rounded half-to-even per ``round()``),
    and numeric strings. Values outside [lo, hi] return ``None`` —
    the caller decides whether to clamp or count as invalid.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        iv = value
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        iv = round(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            iv = round(float(s))
        except ValueError:
            return None
    else:
        return None
    if iv < lo or iv > hi:
        return None
    return iv


def _scale_distribution(responses: list[Any], schema: dict[str, Any], n: int) -> dict[str, Any]:
    lo = int(schema["min"])
    hi = int(schema["max"])
    coerced = [coerce_scale_value(r, lo, hi) for r in responses]
    valid = [v for v in coerced if v is not None]
    n_valid = len(valid)
    n_invalid = n - n_valid

    # Zero-filled frequency table across the full integer range so callers
    # can render a histogram without post-processing.
    freq_counter = Counter(valid)
    distribution = [{"value": v, "count": freq_counter.get(v, 0)} for v in range(lo, hi + 1)]

    stats: dict[str, Any] = {
        "min": lo,
        "max": hi,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
    }
    if n_valid:
        mean = sum(valid) / n_valid
        stats["mean"] = round(mean, 4)
        stats["median"] = _median(valid)
        if n_valid > 1:
            var = sum((v - mean) ** 2 for v in valid) / (n_valid - 1)
            stats["stdev"] = round(math.sqrt(var), 4)
        else:
            stats["stdev"] = 0.0
        stats["observed_min"] = min(valid)
        stats["observed_max"] = max(valid)

    return {
        "type": "scale",
        "n": n,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "distribution": distribution,
        "stats": stats,
    }


def _median(values: list[int]) -> float:
    s = sorted(values)
    m = len(s)
    mid = m // 2
    if m % 2:
        return float(s[mid])
    return round((s[mid - 1] + s[mid]) / 2, 4)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def coerce_enum_value(value: Any, options: list[str]) -> str | None:
    """Match a free response to an enum option (case-insensitive, trimmed).

    Returns the canonical option string as declared in ``options`` when a
    match is found, or ``None`` otherwise. An exact match always wins over
    a case-insensitive one.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s in options:
        return s
    lowered = s.lower()
    for opt in options:
        if opt.lower() == lowered:
            return opt
    return None


def _enum_distribution(responses: list[Any], schema: dict[str, Any], n: int) -> dict[str, Any]:
    options: list[str] = list(schema["options"])
    coerced = [coerce_enum_value(r, options) for r in responses]
    valid = [v for v in coerced if v is not None]
    n_valid = len(valid)
    n_invalid = n - n_valid

    counter = Counter(valid)
    distribution = [{"option": opt, "count": counter.get(opt, 0)} for opt in options]

    top_option: str | None = None
    if n_valid:
        # Stable: options are iterated in declaration order — the first
        # option that ties for the max count is returned.
        top_option = max(options, key=lambda o: counter.get(o, 0))
        if counter.get(top_option, 0) == 0:
            top_option = None

    return {
        "type": "enum",
        "n": n,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "options": options,
        "distribution": distribution,
        "top_option": top_option,
    }


# ---------------------------------------------------------------------------
# Tagged themes
# ---------------------------------------------------------------------------


def coerce_tagged_themes(value: Any, taxonomy: list[str], *, multi: bool) -> tuple[list[str], list[str]] | None:
    """Split a response into (in-taxonomy tags, off-taxonomy tags).

    Accepts a list of strings, a single string (which becomes a 1-element
    list), or a delimited string (comma / semicolon / pipe / newline). When
    ``multi`` is False, only the first tag is kept. Returns ``None`` if the
    input cannot be coerced into any tag list.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts: list[str] = []
        buf = value
        for delim in (",", ";", "|", "\n"):
            buf = buf.replace(delim, "\x00")
        parts = [p.strip() for p in buf.split("\x00") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
    else:
        return None

    if not parts:
        return None

    if not multi:
        parts = parts[:1]

    lowered = {tag.lower(): tag for tag in taxonomy}
    in_tax: list[str] = []
    off_tax: list[str] = []
    for p in parts:
        canon = lowered.get(p.lower())
        if canon is not None:
            if canon not in in_tax:
                in_tax.append(canon)
        else:
            if p not in off_tax:
                off_tax.append(p)
    return in_tax, off_tax


def _tagged_themes_distribution(responses: list[Any], schema: dict[str, Any], n: int) -> dict[str, Any]:
    taxonomy: list[str] = list(schema["taxonomy"])
    multi: bool = bool(schema.get("multi", False))

    tag_counter: Counter[str] = Counter()
    off_counter: Counter[str] = Counter()
    n_valid = 0
    total_mentions = 0
    for r in responses:
        coerced = coerce_tagged_themes(r, taxonomy, multi=multi)
        if coerced is None:
            continue
        in_tax, off_tax = coerced
        if in_tax or off_tax:
            n_valid += 1
            total_mentions += len(in_tax) + len(off_tax)
        for tag in in_tax:
            tag_counter[tag] += 1
        for tag in off_tax:
            off_counter[tag] += 1

    n_invalid = n - n_valid
    distribution = [{"theme": tag, "count": tag_counter.get(tag, 0)} for tag in taxonomy]
    off_taxonomy = [
        {"theme": tag, "count": count} for tag, count in sorted(off_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return {
        "type": "tagged_themes",
        "n": n,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "multi": multi,
        "taxonomy": taxonomy,
        "distribution": distribution,
        "off_taxonomy": off_taxonomy,
        "total_mentions": total_mentions,
    }


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _text_distribution(responses: list[Any], n: int) -> dict[str, Any]:
    n_nonempty = sum(
        1 for r in responses if (isinstance(r, str) and r.strip()) or (r is not None and not isinstance(r, str))
    )
    return {
        "type": "text",
        "n": n,
        "n_valid": n_nonempty,
        "n_invalid": n - n_nonempty,
    }
