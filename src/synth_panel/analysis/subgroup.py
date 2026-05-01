"""Subgroup breakdowns over structured responses (sp-2hpi).

Pair a persona-field key with per-panelist responses and this module
returns a nested distribution: one bucket per observed value of the
persona field, each bucket carrying the same distribution shape
:func:`synth_panel.analysis.distribution.distribution_for_question`
produces for the whole panel.

This is the scaffold the ``--analysis-mode structured`` layer will use
to emit ``subgroup_breakdowns`` alongside the top-level ``distribution``.

Higher-level functions :func:`analyze_subgroup` and
:func:`format_subgroup_text` consume a saved panel result and produce
the per-question CLI-shaped report (effect size, F / Cramer's V,
p-value, sparse-group warnings) used by ``synthpanel analyze subgroup``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from synth_panel.analysis.distribution import (
    coerce_enum_value,
    coerce_scale_value,
    coerce_tagged_themes,
    distribution_for_question,
)
from synth_panel.stats import OneWayANOVAResult, one_way_anova

__all__ = [
    "AUTO_BIN_FIELDS",
    "UnknownPersonaFieldError",
    "analyze_subgroup",
    "auto_bin_value",
    "format_subgroup_text",
    "subgroup_breakdown",
]


# Map of "virtual" field names to (source_field, binner). Adding a new
# auto-bin is a single entry: pair the virtual key with the persona
# field it derives from and a callable that coerces a raw value to a
# bucket label string. Returning ``None`` puts the persona in
# ``missing_label``.
def _bin_age_decade(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    age = int(value)
    if age < 0:
        return None
    decade = (age // 10) * 10
    return f"{decade}s"


def _bin_age_decade_5y(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    age = int(value)
    if age < 0:
        return None
    lo = (age // 5) * 5
    hi = lo + 4
    return f"{lo}-{hi}"


AUTO_BIN_FIELDS: dict[str, tuple[str, Any]] = {
    "age_decade": ("age", _bin_age_decade),
    "age_decade_5y": ("age", _bin_age_decade_5y),
}


def auto_bin_value(field: str, persona: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve ``field`` against ``persona`` to (source_field, label).

    Returns the source persona field used for grouping (so the renderer
    can name it accurately) and the bucket label. ``label`` is ``None``
    when the persona's value is missing or out of range — the caller
    treats that as the ``missing`` bucket.

    For non-virtual fields, the source is ``field`` itself and the
    label is the stringified value (or ``None`` if missing).
    """
    if field in AUTO_BIN_FIELDS:
        source, binner = AUTO_BIN_FIELDS[field]
        return source, binner(persona.get(source))
    raw = persona.get(field)
    if raw is None:
        return field, None
    if isinstance(raw, bool):
        return field, "true" if raw else "false"
    if isinstance(raw, (list, tuple)):
        return field, "|".join(str(x) for x in raw)
    return field, str(raw)


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


# ---------------------------------------------------------------------------
# High-level CLI entry: per-question subgroup analysis with effect sizes
# ---------------------------------------------------------------------------

# Effect-size labels follow Cohen's conventions. They're advisory in
# the rendered output; the raw value remains primary.
_ETA_SQUARED_BANDS = ((0.14, "large"), (0.06, "medium"), (0.01, "small"))
_CRAMERS_V_BANDS = ((0.5, "large"), (0.3, "medium"), (0.1, "small"))


def _label_effect_size(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    """Return the qualitative band ('small'/'medium'/'large'/'negligible')."""
    for threshold, label in bands:
        if value >= threshold:
            return label
    return "negligible"


def _coerce_for_anova(value: Any, response_schema: dict[str, Any]) -> float | None:
    """Coerce a raw response into a numeric value for ANOVA.

    Scale responses use the existing ``coerce_scale_value``; everything
    else is treated as non-numeric (returns ``None``) so the caller
    falls back to a categorical statistic.
    """
    rtype = response_schema.get("type") if isinstance(response_schema, dict) else None
    if rtype != "scale":
        return None
    lo = int(response_schema["min"])
    hi = int(response_schema["max"])
    coerced = coerce_scale_value(value, lo, hi)
    if coerced is None:
        return None
    return float(coerced)


def _categorical_label(value: Any, response_schema: dict[str, Any]) -> str | None:
    """Extract a single categorical label from a typed response.

    Used for chi-squared / Cramer's V on ``enum`` and ``tagged_themes``
    responses. ``tagged_themes`` collapses to the first in-taxonomy tag
    when ``multi=True`` (so each respondent contributes one observation).
    Returns ``None`` for unparseable values; the caller treats that as
    a non-contributing observation (it's still counted in the group size
    but doesn't enter the statistic).
    """
    rtype = response_schema.get("type") if isinstance(response_schema, dict) else None
    if rtype == "enum":
        return coerce_enum_value(value, list(response_schema["options"]))
    if rtype == "tagged_themes":
        taxonomy = list(response_schema["taxonomy"])
        multi = bool(response_schema.get("multi", False))
        coerced = coerce_tagged_themes(value, taxonomy, multi=multi)
        if coerced is None:
            return None
        in_tax, _off = coerced
        if not in_tax:
            return None
        # Take the first in-taxonomy tag so each respondent contributes
        # exactly one observation. Counting all tags double-counts
        # respondents in the chi-squared denominator.
        return in_tax[0]
    return None


def _resolve_run_persona_attrs(
    panel_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pair each panelist with their persona attributes.

    Returns ``(persona_dicts, missing_names)`` aligned with the
    ``results`` list in ``panel_data``. ``missing_names`` lists
    panelists whose persona attributes couldn't be resolved (so the
    renderer can warn). Resolution order:

    1. The new ``personas`` field on the saved result (preferred).
    2. The legacy panelist entry's nested ``persona`` dict (if the
       panelist was saved with attributes inline).
    3. ``{"name": "<name>"}`` as a last-resort placeholder so the
       analysis can still run (the persona ends up in the "missing"
       bucket for any non-name field).
    """
    panelists = panel_data.get("results") or []
    saved = panel_data.get("personas")
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(saved, list):
        for p in saved:
            if isinstance(p, dict) and isinstance(p.get("name"), str):
                by_name[p["name"]] = p

    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in panelists:
        if not isinstance(entry, dict):
            resolved.append({})
            continue
        name = entry.get("persona", "")
        if name in by_name:
            resolved.append(by_name[name])
            continue
        # Some legacy results saved the persona dict inline under
        # 'persona_obj' or similar — try a couple of shapes before
        # falling back to the placeholder.
        for key in ("persona_obj", "persona_dict", "persona_attrs"):
            inline = entry.get(key)
            if isinstance(inline, dict):
                resolved.append(inline)
                break
        else:
            resolved.append({"name": name})
            if name:
                missing.append(name)
    return resolved, missing


def _resolve_questions(panel_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-question definition list.

    Saved runs persist the instrument's questions in
    ``panel_data['questions']``. Older runs may not — we synthesize a
    minimal stub from the first panelist's ``responses`` block so the
    renderer still has something to label questions with (text only,
    no schema, so subgroup analysis falls back to text mode).
    """
    questions = panel_data.get("questions")
    if isinstance(questions, list) and questions:
        return [q if isinstance(q, dict) else {} for q in questions]
    panelists = panel_data.get("results") or []
    if not panelists:
        return []
    first = panelists[0]
    if not isinstance(first, dict):
        return []
    out: list[dict[str, Any]] = []
    for r in first.get("responses") or []:
        if isinstance(r, dict):
            out.append({"text": r.get("question", "")})
        else:
            out.append({})
    return out


def _gather_response(panelist: dict[str, Any], question_index: int) -> Any:
    """Extract the raw response value for one (panelist, question)."""
    responses = panelist.get("responses") or []
    if question_index >= len(responses):
        return None
    entry = responses[question_index]
    if not isinstance(entry, dict):
        return None
    if entry.get("error"):
        return None
    if "extraction" in entry and entry["extraction"] is not None:
        # Structured extraction is the parsed form — prefer it for
        # subgroup analysis when present.
        return entry["extraction"]
    return entry.get("response")


def analyze_subgroup(
    panel_data: dict[str, Any],
    *,
    by: str,
    min_subgroup_n: int = 3,
    missing_label: str = "missing",
) -> dict[str, Any]:
    """Run subgroup analysis over a saved panel result.

    Args:
        panel_data: Loaded panel result dict (from
            :func:`synth_panel.mcp.data.get_panel_result` or a result
            JSON file).
        by: Persona field to group by. Recognises virtual fields like
            ``age_decade`` (auto-binned from ``age``).
        min_subgroup_n: Subgroups smaller than this still appear in
            output but with significance tests suppressed.
        missing_label: Bucket label for personas missing the field.

    Returns:
        A dict with ``field``, ``source_field`` (where the values were
        actually read from), ``n_panelists``, ``subgroups``
        (label → size), ``warnings``, and ``per_question`` — a list of
        per-question entries, each carrying:

        * ``question_index`` / ``question_text``
        * ``response_type`` (``scale``/``enum``/``tagged_themes``/``text``)
        * ``buckets`` — per-subgroup summary (n, mean+/-stdev for
          scale, top option for enum, top tag for tagged_themes)
        * ``effect_size`` — ``{"metric": "eta_squared"|"cramers_v",
          "value": float, "label": "small"|...}``
        * ``test`` — ``{"name": "one_way_anova"|"chi_squared"|None,
          "statistic": float, "df": ..., "p_value": float}``
          (omitted when significance can't be computed)
        * ``warnings`` — per-question caveats (sparse groups,
          non-numeric responses, etc.)

    Raises:
        UnknownPersonaFieldError: If no persona carries the source
            field (a partial presence is tolerated — those personas
            land in the missing bucket).
    """
    persona_attrs, missing_names = _resolve_run_persona_attrs(panel_data)
    panelists = panel_data.get("results") or []

    if not panelists:
        return {
            "field": by,
            "source_field": by,
            "n_panelists": 0,
            "subgroups": {},
            "warnings": ["No panelist results to analyze."],
            "per_question": [],
        }

    # Resolve the bucket label for each panelist once.
    source_field = AUTO_BIN_FIELDS[by][0] if by in AUTO_BIN_FIELDS else by
    labels: list[str] = []
    any_present = False
    for persona in persona_attrs:
        # auto_bin_value handles both virtual and direct fields.
        # ``source_used`` is identical across iterations because ``by``
        # is fixed; we capture it once for the warning message.
        _src, label = auto_bin_value(by, persona)
        if source_field in persona and persona.get(source_field) is not None:
            any_present = True
        labels.append(label or missing_label)

    if not any_present:
        raise UnknownPersonaFieldError(
            f"persona field {source_field!r} missing from every persona "
            f"(cannot group by {by!r})"
        )

    # Tally subgroup sizes in label-encounter order. Float
    # ``missing_label`` to the end of the natural ordering.
    bucket_indices: OrderedDict[str, list[int]] = OrderedDict()
    for idx, label in enumerate(labels):
        bucket_indices.setdefault(label, []).append(idx)
    ordered_labels: list[str]
    named = sorted(lbl for lbl in bucket_indices if lbl != missing_label)
    ordered_labels = named + ([missing_label] if missing_label in bucket_indices else [])
    subgroup_sizes: dict[str, int] = {lbl: len(bucket_indices[lbl]) for lbl in ordered_labels}

    warnings: list[str] = []
    if missing_names:
        warnings.append(
            f"Could not resolve persona attributes for {len(missing_names)} panelist(s); "
            f"they fall into the {missing_label!r} bucket. Re-run with --personas to attach attributes."
        )

    questions = _resolve_questions(panel_data)
    per_question: list[dict[str, Any]] = []

    for q_idx, q_def in enumerate(questions):
        q_def = q_def if isinstance(q_def, dict) else {}
        q_text = q_def.get("text") or q_def.get("question") or f"Question {q_idx + 1}"
        rs = q_def.get("response_schema") if isinstance(q_def.get("response_schema"), dict) else {}
        rtype = rs.get("type") if isinstance(rs, dict) else None

        per_q_warnings: list[str] = []
        buckets_out: list[dict[str, Any]] = []
        bucket_numeric: dict[str, list[float]] = {}
        bucket_categorical: dict[str, list[str]] = {}

        for label in ordered_labels:
            indices = bucket_indices[label]
            raw_values: list[Any] = []
            numeric_values: list[float] = []
            categorical_values: list[str] = []
            for i in indices:
                if i >= len(panelists):
                    continue
                entry = panelists[i] if isinstance(panelists[i], dict) else {}
                raw = _gather_response(entry, q_idx)
                raw_values.append(raw)
                if rtype == "scale":
                    nv = _coerce_for_anova(raw, rs)
                    if nv is not None:
                        numeric_values.append(nv)
                elif rtype in ("enum", "tagged_themes"):
                    cat = _categorical_label(raw, rs)
                    if cat is not None:
                        categorical_values.append(cat)
            bucket_numeric[label] = numeric_values
            bucket_categorical[label] = categorical_values

            distribution = (
                distribution_for_question(raw_values, rs) if rtype in {"scale", "enum", "tagged_themes", "text"} else None
            )
            bucket_summary: dict[str, Any] = {
                "label": label,
                "n": len(raw_values),
            }
            if rtype == "scale" and numeric_values:
                mean = sum(numeric_values) / len(numeric_values)
                if len(numeric_values) > 1:
                    var = sum((v - mean) ** 2 for v in numeric_values) / (len(numeric_values) - 1)
                    stdev = var**0.5
                else:
                    stdev = 0.0
                bucket_summary["mean"] = round(mean, 4)
                bucket_summary["stdev"] = round(stdev, 4)
                bucket_summary["n_valid"] = len(numeric_values)
            elif rtype == "enum" and distribution is not None:
                bucket_summary["top_option"] = distribution.get("top_option")
                bucket_summary["n_valid"] = distribution.get("n_valid", 0)
            elif rtype == "tagged_themes" and distribution is not None:
                tags = distribution.get("distribution") or []
                top = max(tags, key=lambda d: d.get("count", 0)) if tags else None
                bucket_summary["top_theme"] = top.get("theme") if top and top.get("count", 0) else None
                bucket_summary["n_valid"] = distribution.get("n_valid", 0)
            elif rtype == "text" and distribution is not None:
                bucket_summary["n_nonempty"] = distribution.get("n_valid", 0)
            buckets_out.append(bucket_summary)

        # Effect-size + significance test, sized to the response type.
        effect_size: dict[str, Any] | None = None
        test_result: dict[str, Any] | None = None

        scale_groups = {
            lbl: vals for lbl, vals in bucket_numeric.items() if vals
        }
        cat_groups = {lbl: vals for lbl, vals in bucket_categorical.items() if vals}

        non_missing_scale_groups = {
            lbl: vals for lbl, vals in scale_groups.items() if lbl != missing_label
        }
        non_missing_cat_groups = {
            lbl: vals for lbl, vals in cat_groups.items() if lbl != missing_label
        }

        if rtype == "scale":
            if len(non_missing_scale_groups) >= 2:
                anova: OneWayANOVAResult = one_way_anova(non_missing_scale_groups)
                effect_size = {
                    "metric": "eta_squared",
                    "value": round(anova.eta_squared, 4),
                    "label": _label_effect_size(anova.eta_squared, _ETA_SQUARED_BANDS),
                }
                if not anova.insufficient_data:
                    sparse = [lbl for lbl, n in anova.group_sizes.items() if n < min_subgroup_n]
                    if sparse:
                        per_q_warnings.append(
                            f"Subgroup(s) with n<{min_subgroup_n} ({', '.join(sparse)}); "
                            f"interpret F-test cautiously."
                        )
                    test_result = {
                        "name": "one_way_anova",
                        "statistic": round(anova.f_statistic, 4),
                        "df_between": anova.df_between,
                        "df_within": anova.df_within,
                        "p_value": round(anova.p_value, 6),
                    }
                else:
                    if anova.warning:
                        per_q_warnings.append(anova.warning)
            else:
                per_q_warnings.append("Need at least 2 non-missing subgroups with valid scale responses.")
        elif rtype in ("enum", "tagged_themes"):
            # Build the contingency table: rows are subgroups, cols are
            # categorical outcomes. Chi-squared with Cramer's V.
            if len(non_missing_cat_groups) >= 2:
                from collections import Counter

                col_counter: "Counter[str]" = Counter()
                row_counters: dict[str, Counter[str]] = {}
                for lbl, vals in non_missing_cat_groups.items():
                    rc: Counter[str] = Counter(vals)
                    row_counters[lbl] = rc
                    col_counter.update(vals)
                if col_counter and len(col_counter) >= 2:
                    n_total = sum(col_counter.values())
                    rows = list(row_counters.keys())
                    cols = list(col_counter.keys())
                    row_totals = {r: sum(row_counters[r].values()) for r in rows}
                    chi2 = 0.0
                    min_expected = float("inf")
                    for r in rows:
                        for c in cols:
                            expected = row_totals[r] * col_counter[c] / n_total
                            min_expected = min(min_expected, expected)
                            if expected > 0:
                                chi2 += (row_counters[r].get(c, 0) - expected) ** 2 / expected
                    df = (len(rows) - 1) * (len(cols) - 1)
                    from synth_panel.stats import _chi2_sf

                    p_val = _chi2_sf(chi2, df) if df > 0 else 1.0
                    # Cramer's V for r×c: sqrt(chi2 / (N * min(r-1, c-1)))
                    denom_dim = min(len(rows) - 1, len(cols) - 1)
                    cramers_v = (chi2 / (n_total * denom_dim)) ** 0.5 if denom_dim > 0 and n_total > 0 else 0.0
                    effect_size = {
                        "metric": "cramers_v",
                        "value": round(cramers_v, 4),
                        "label": _label_effect_size(cramers_v, _CRAMERS_V_BANDS),
                    }
                    sparse = [lbl for lbl, vals in non_missing_cat_groups.items() if len(vals) < min_subgroup_n]
                    if sparse:
                        per_q_warnings.append(
                            f"Subgroup(s) with n<{min_subgroup_n} ({', '.join(sparse)}); "
                            f"interpret chi-squared cautiously."
                        )
                    if min_expected < 5:
                        per_q_warnings.append(
                            f"Some expected cell count(s) < 5 (min={min_expected:.1f}); "
                            f"chi-squared approximation may be unreliable."
                        )
                    test_result = {
                        "name": "chi_squared",
                        "statistic": round(chi2, 4),
                        "df": df,
                        "p_value": round(p_val, 6),
                    }
                else:
                    per_q_warnings.append("Need at least 2 distinct categorical outcomes.")
            else:
                per_q_warnings.append("Need at least 2 non-missing subgroups with valid categorical responses.")
        else:
            # text or unknown response type — no effect-size metric
            per_q_warnings.append(
                "Free-text responses are not supported by effect-size analysis. "
                "Use ``analyze`` for narrative summaries."
            )

        per_question.append(
            {
                "question_index": q_idx,
                "question_text": q_text,
                "response_type": rtype or "unknown",
                "buckets": buckets_out,
                "effect_size": effect_size,
                "test": test_result,
                "warnings": per_q_warnings,
            }
        )

    return {
        "field": by,
        "source_field": source_field,
        "n_panelists": len(panelists),
        "subgroups": subgroup_sizes,
        "warnings": warnings,
        "per_question": per_question,
    }


def format_subgroup_text(report: dict[str, Any]) -> str:
    """Render an :func:`analyze_subgroup` result as a plain-text report."""
    lines: list[str] = []
    field = report.get("field", "?")
    source = report.get("source_field", field)
    n = report.get("n_panelists", 0)
    subgroups = report.get("subgroups", {})

    lines.append(f"Subgroup analysis (by {field}, n={n})")
    if source != field:
        lines.append(f"  Source field: {source}")
    if subgroups:
        size_parts = ", ".join(f"{lbl} (n={cnt})" for lbl, cnt in subgroups.items())
        lines.append(f"  Subgroups: {size_parts}")
    for w in report.get("warnings", []) or []:
        lines.append(f"  ! {w}")
    lines.append("")

    for q in report.get("per_question", []) or []:
        q_idx = q.get("question_index", 0) + 1
        text = q.get("question_text", "")
        lines.append(f"Q{q_idx}: {text!r}")
        lines.append(f"  By {field}:")

        rtype = q.get("response_type")
        for b in q.get("buckets", []) or []:
            lbl = b.get("label", "?")
            n_b = b.get("n", 0)
            if rtype == "scale" and "mean" in b:
                lines.append(
                    f"    {lbl:<10}  (n={n_b})  mean={b['mean']} ± {b.get('stdev', 0)}"
                )
            elif rtype == "enum":
                top = b.get("top_option") or "—"
                lines.append(f"    {lbl:<10}  (n={n_b})  top: {top}")
            elif rtype == "tagged_themes":
                top = b.get("top_theme") or "—"
                lines.append(f"    {lbl:<10}  (n={n_b})  top theme: {top}")
            elif rtype == "text":
                ne = b.get("n_nonempty", 0)
                lines.append(f"    {lbl:<10}  (n={n_b})  non-empty: {ne}")
            else:
                lines.append(f"    {lbl:<10}  (n={n_b})")

        es = q.get("effect_size") or {}
        if es:
            metric = es.get("metric", "?")
            sym = "η²" if metric == "eta_squared" else "Cramer's V" if metric == "cramers_v" else metric
            lines.append(f"  Effect size: {sym} = {es.get('value')} ({es.get('label')})")
        test = q.get("test") or {}
        if test:
            name = test.get("name")
            if name == "one_way_anova":
                lines.append(
                    f"  Statistical reliability: F={test.get('statistic')}, p={test.get('p_value')}"
                )
            elif name == "chi_squared":
                lines.append(
                    f"  Statistical reliability: χ²={test.get('statistic')}, df={test.get('df')}, p={test.get('p_value')}"
                )
        for w in q.get("warnings", []) or []:
            lines.append(f"  ! {w}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
