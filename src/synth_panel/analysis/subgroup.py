"""Subgroup breakdowns over structured responses (sp-2hpi).

Pair a persona-field key with per-panelist responses and this module
returns a nested distribution: one bucket per observed value of the
persona field, each bucket carrying the same distribution shape
:func:`synth_panel.analysis.distribution.distribution_for_question`
produces for the whole panel.

This is the scaffold the ``--analysis-mode structured`` layer will use
to emit ``subgroup_breakdowns`` alongside the top-level ``distribution``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from synth_panel.analysis.distribution import distribution_for_question

__all__ = [
    "UnknownPersonaFieldError",
    "subgroup_breakdown",
]


class UnknownPersonaFieldError(KeyError):
    """Raised when a requested persona field is missing from every persona."""


def subgroup_breakdown(
    responses: list[Any],
    personas: list[dict[str, Any]],
    *,
    field: str,
    response_schema: dict[str, Any],
    age_bands: list[tuple[int, int]] | None = None,
    missing_label: str = "unknown",
) -> dict[str, Any]:
    """Return per-subgroup distributions for one question.

    Responses and personas are paired positionally — caller is responsible
    for keeping the two lists aligned (``results[i].persona`` matches
    ``responses[i]``).

    Args:
        responses: Per-panelist responses to aggregate.
        personas: Persona dicts in the same order as ``responses``.
        field: Persona key to group by (e.g. ``"occupation"``, ``"age"``).
        response_schema: Question-level schema (forwarded unchanged to the
            inner :func:`distribution_for_question`).
        age_bands: Optional list of inclusive ``(lo, hi)`` pairs to bucket
            numeric age values. Applied only when ``field == "age"`` and
            the persona value is an int/float. Out-of-band ages fall into
            ``missing_label``.
        missing_label: Bucket label for personas that lack the field or
            whose value is ``None``.

    Returns:
        Dict with ``field``, ``n_buckets``, and ``buckets`` — a list of
        ``{"label", "n", "distribution": {...}}`` entries sorted by bucket
        label for deterministic output (age bands keep declaration order).

    Raises:
        ValueError: If ``len(responses) != len(personas)``.
        UnknownPersonaFieldError: If *no* persona carries the requested
            field (a partial presence is tolerated — missing personas go
            into ``missing_label``).
    """
    if len(responses) != len(personas):
        raise ValueError(f"responses and personas must be the same length ({len(responses)} vs {len(personas)})")

    use_bands = field == "age" and age_bands is not None
    if use_bands:
        bands = _validate_age_bands(age_bands or [])

    buckets: OrderedDict[str, list[Any]] = OrderedDict()
    if use_bands:
        for lo, hi in bands:
            buckets[_band_label(lo, hi)] = []
    any_present = False

    for resp, persona in zip(responses, personas):
        if field in persona:
            any_present = True
        raw = persona.get(field)
        if use_bands and _is_numeric(raw):
            label = _band_for_age(float(raw), bands) or missing_label
        else:
            label = _label_for(raw, missing_label)
        buckets.setdefault(label, []).append(resp)

    if not any_present:
        raise UnknownPersonaFieldError(f"persona field {field!r} missing from every persona")

    bucket_list: list[dict[str, Any]] = []
    labels: list[str]
    if use_bands:
        labels = list(buckets.keys())  # preserve declared band order + tail
        # Ensure missing_label goes last if present
        if missing_label in labels:
            labels = [lbl for lbl in labels if lbl != missing_label] + [missing_label]
    else:
        # Deterministic ordering: non-missing labels alphabetically, then missing_label.
        named = sorted(lbl for lbl in buckets if lbl != missing_label)
        labels = named + ([missing_label] if missing_label in buckets else [])

    for label in labels:
        values = buckets.get(label, [])
        bucket_list.append(
            {
                "label": label,
                "n": len(values),
                "distribution": distribution_for_question(values, response_schema),
            }
        )

    return {
        "field": field,
        "n_buckets": len(bucket_list),
        "buckets": bucket_list,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_age_bands(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for i, band in enumerate(bands):
        if (
            not isinstance(band, tuple)
            or len(band) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in band)
        ):
            raise ValueError(f"age_bands[{i}] must be a (lo, hi) tuple of ints")
        lo, hi = band
        if lo > hi:
            raise ValueError(f"age_bands[{i}]: lo ({lo}) must be <= hi ({hi})")
        cleaned.append((lo, hi))
    return cleaned


def _band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _band_for_age(age: float, bands: list[tuple[int, int]]) -> str | None:
    for lo, hi in bands:
        if lo <= age <= hi:
            return _band_label(lo, hi)
    return None


def _label_for(raw: Any, missing_label: str) -> str:
    if raw is None:
        return missing_label
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (list, tuple)):
        # Stringify lists deterministically so callers get a stable label.
        return "|".join(str(x) for x in raw)
    return str(raw)
