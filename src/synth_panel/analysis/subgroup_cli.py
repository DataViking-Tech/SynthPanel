"""CLI-level subgroup analysis for ``synthpanel analyze subgroup``.

Takes a saved panel result + a persona YAML/pack (because results store
persona names, not full dicts) and produces per-subgroup distributions
for each question that has a ``response_schema``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from synth_panel.analysis.distribution import InvalidResponseSchemaError
from synth_panel.analysis.subgroup import UnknownPersonaFieldError, subgroup_breakdown

__all__ = [
    "AGE_DECADE_5Y_BANDS",
    "AGE_DECADE_BANDS",
    "FieldBreakdown",
    "QuestionSubgroupResult",
    "SubgroupAnalysisResult",
    "analyze_subgroup",
    "format_subgroup_json",
    "format_subgroup_text",
]

# ---------------------------------------------------------------------------
# Age auto-bin strategies
# ---------------------------------------------------------------------------

AGE_DECADE_BANDS: list[tuple[int, int]] = [
    (18, 27),
    (28, 37),
    (38, 47),
    (48, 57),
    (58, 67),
    (68, 77),
    (78, 99),
]

AGE_DECADE_5Y_BANDS: list[tuple[int, int]] = [
    (18, 22),
    (23, 27),
    (28, 32),
    (33, 37),
    (38, 42),
    (43, 47),
    (48, 52),
    (53, 57),
    (58, 62),
    (63, 67),
    (68, 72),
    (73, 99),
]

# Maps special --by token → (actual persona field, age_bands or None)
_AUTO_BIN_STRATEGIES: dict[str, tuple[str, list[tuple[int, int]] | None]] = {
    "age_decade": ("age", AGE_DECADE_BANDS),
    "age_decade_5y": ("age", AGE_DECADE_5Y_BANDS),
}

_SPARSE_N = 3  # subgroups with n < this trigger a warning


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FieldBreakdown:
    field: str
    n_buckets: int
    buckets: list[dict[str, Any]]
    effect_size: float | None
    effect_size_type: str  # "eta-squared" | "cramers-v" | "density-ratio" | "none"
    p_value: float | None
    warnings: list[str]


@dataclass
class QuestionSubgroupResult:
    question_index: int
    question_text: str
    response_schema_type: str
    n_responses: int
    by_field: list[FieldBreakdown]


@dataclass
class SubgroupAnalysisResult:
    result_id: str
    fields: list[str]
    per_question: list[QuestionSubgroupResult]
    global_warnings: list[str]


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------


def _eta_squared(groups: list[list[float]]) -> float | None:
    """One-way ANOVA eta-squared from grouped numeric values."""
    all_vals = [v for g in groups for v in g]
    n = len(all_vals)
    if n == 0:
        return None
    grand_mean = sum(all_vals) / n
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total == 0.0:
        return 0.0
    ss_between = sum(len(g) * ((sum(g) / len(g)) - grand_mean) ** 2 for g in groups if g)
    return round(ss_between / ss_total, 4)


def _cramers_v_contingency(rows: list[list[str]]) -> tuple[float, float, int]:
    """Cramer's V for a groups x categories contingency table.

    ``rows[i]`` is the list of categorical responses in group i.
    Returns (cramers_v, chi2, df).
    """
    categories: list[str] = sorted({c for row in rows for c in row})
    n_cols = len(categories)
    n_rows = len(rows)
    if n_rows < 2 or n_cols < 2:
        return 0.0, 0.0, 0

    cat_idx = {c: j for j, c in enumerate(categories)}
    table = [[0] * n_cols for _ in range(n_rows)]
    for i, row in enumerate(rows):
        for val in row:
            j = cat_idx.get(val)
            if j is not None:
                table[i][j] += 1

    row_totals = [sum(table[i]) for i in range(n_rows)]
    col_totals = [sum(table[i][j] for i in range(n_rows)) for j in range(n_cols)]
    n_total = sum(row_totals)
    if n_total == 0:
        return 0.0, 0.0, 0

    chi2 = 0.0
    for i in range(n_rows):
        if row_totals[i] == 0:
            continue
        for j in range(n_cols):
            if col_totals[j] == 0:
                continue
            exp = (row_totals[i] * col_totals[j]) / n_total
            diff = table[i][j] - exp
            chi2 += diff * diff / exp

    df = (n_rows - 1) * (n_cols - 1)
    if df <= 0 or n_total <= 0:
        return 0.0, chi2, df

    denom = n_total * min(n_rows - 1, n_cols - 1)
    if denom <= 0:
        return 0.0, chi2, df

    v = math.sqrt(chi2 / denom)
    return round(max(0.0, min(1.0, v)), 4), round(chi2, 4), df


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution."""
    from synth_panel.stats import _chi2_sf as _stats_chi2_sf  # type: ignore[attr-defined]

    return _stats_chi2_sf(x, df)


# ---------------------------------------------------------------------------
# Response value extraction
# ---------------------------------------------------------------------------


def _response_value(resp: dict[str, Any]) -> Any:
    """Extract the typed response value from a stored response dict."""
    if resp.get("structured"):
        return resp.get("response")
    extraction = resp.get("extraction")
    if extraction is not None:
        return extraction
    return resp.get("response")


def _flatten_structured_value(val: Any, schema_type: str) -> Any:
    """Try to unwrap a dict-wrapped structured response to a scalar/list."""
    if not isinstance(val, dict):
        return val
    if schema_type == "scale":
        for key in ("score", "rating", "value", "answer", "response"):
            v = val.get(key)
            if isinstance(v, (int, float)):
                return v
    elif schema_type == "enum":
        for key in ("option", "answer", "value", "choice", "response"):
            v = val.get(key)
            if isinstance(v, str):
                return v
    elif schema_type == "tagged_themes":
        for key in ("themes", "tags", "values", "response"):
            v = val.get(key)
            if isinstance(v, list):
                return v
    return val


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _resolve_field_and_bands(by_token: str) -> tuple[str, list[tuple[int, int]] | None]:
    """Map a --by token to (persona_field, age_bands or None)."""
    if by_token in _AUTO_BIN_STRATEGIES:
        return _AUTO_BIN_STRATEGIES[by_token]
    return by_token, None


def _build_field_breakdown(
    by_token: str,
    responses_per_panelist: list[Any],
    personas: list[dict[str, Any]],
    response_schema: dict[str, Any],
) -> FieldBreakdown:
    persona_field, age_bands = _resolve_field_and_bands(by_token)
    schema_type = response_schema.get("type", "text")
    warnings: list[str] = []

    try:
        breakdown = subgroup_breakdown(
            responses_per_panelist,
            personas,
            field=persona_field,
            response_schema=response_schema,
            age_bands=age_bands,
        )
    except UnknownPersonaFieldError:
        return FieldBreakdown(
            field=by_token,
            n_buckets=0,
            buckets=[],
            effect_size=None,
            effect_size_type="none",
            p_value=None,
            warnings=[f"Persona field '{persona_field}' not found in any persona."],
        )
    except InvalidResponseSchemaError as exc:
        return FieldBreakdown(
            field=by_token,
            n_buckets=0,
            buckets=[],
            effect_size=None,
            effect_size_type="none",
            p_value=None,
            warnings=[f"Cannot analyze question with response_schema: {exc}"],
        )

    buckets = breakdown["buckets"]

    # Sparse subgroup warnings
    sparse = [b["label"] for b in buckets if b["n"] < _SPARSE_N and b["n"] > 0]
    if sparse:
        warnings.append(
            f"Sparse subgroup(s) (n < {_SPARSE_N}): {', '.join(sparse)}. "
            "Effect-size and significance tests suppressed for sparse groups."
        )

    non_sparse_buckets = [b for b in buckets if b["n"] >= _SPARSE_N]
    suppress_stats = len(non_sparse_buckets) < 2

    effect_size: float | None = None
    effect_size_type = "none"
    p_value: float | None = None

    if not suppress_stats:
        if schema_type == "scale":
            effect_size, p_value = _scale_effect_size(non_sparse_buckets, response_schema)
            effect_size_type = "eta-squared"
        elif schema_type == "enum":
            effect_size, p_value = _enum_effect_size(non_sparse_buckets)
            effect_size_type = "cramers-v"
        elif schema_type == "tagged_themes":
            effect_size_type = "density-ratio"
            # No single effect-size for multi-tag; leave None (per-tag density in buckets)

    return FieldBreakdown(
        field=by_token,
        n_buckets=breakdown["n_buckets"],
        buckets=buckets,
        effect_size=effect_size,
        effect_size_type=effect_size_type,
        p_value=p_value,
        warnings=warnings,
    )


def _scale_effect_size(
    non_sparse_buckets: list[dict[str, Any]],
    response_schema: dict[str, Any],
) -> tuple[float | None, float | None]:
    groups: list[list[float]] = []
    for bucket in non_sparse_buckets:
        dist = bucket.get("distribution", {})
        vals: list[float] = []
        for entry in dist.get("distribution", []):
            v = entry.get("value")
            c = entry.get("count", 0)
            if isinstance(v, (int, float)) and c > 0:
                vals.extend([float(v)] * c)
        groups.append(vals)
    eta_sq = _eta_squared(groups)
    return eta_sq, None  # F-test p-value requires F-distribution; omitted


def _enum_effect_size(
    non_sparse_buckets: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    rows: list[list[str]] = []
    for bucket in non_sparse_buckets:
        dist = bucket.get("distribution", {})
        row: list[str] = []
        for entry in dist.get("distribution", []):
            opt = entry.get("option", "")
            cnt = entry.get("count", 0)
            row.extend([opt] * cnt)
        rows.append(row)
    v, chi2, df = _cramers_v_contingency(rows)
    p_value: float | None = None
    if df > 0 and chi2 > 0:
        import contextlib

        with contextlib.suppress(Exception):
            p_value = round(_chi2_sf(chi2, df), 4)
    return v if v > 0 else None, p_value


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_subgroup(
    data: dict[str, Any],
    personas: list[dict[str, Any]],
    by_tokens: list[str],
    *,
    question_indices: list[int] | None = None,
) -> SubgroupAnalysisResult:
    """Run subgroup analysis on a saved panel result.

    Args:
        data: Panel result dict (from result JSON file or MCP data).
        personas: Full persona dicts to match against result by name.
        by_tokens: List of persona fields (or auto-bin tokens like
            ``age_decade``) to group by.
        question_indices: Optional filter — only analyze these question
            indices (0-based). None means all questions.

    Returns:
        SubgroupAnalysisResult with per-question breakdowns.
    """
    result_id = data.get("id", "unknown")
    raw_results = data.get("results") or []
    saved_questions = data.get("questions") or []
    global_warnings: list[str] = []

    # Build name → persona dict lookup
    persona_by_name: dict[str, dict[str, Any]] = {p.get("name", ""): p for p in personas if p.get("name")}

    # Align panelists with their full persona dicts
    ordered_personas: list[dict[str, Any]] = []
    ordered_raw_responses: list[list[dict[str, Any]]] = []

    matched = 0
    for entry in raw_results:
        name = entry.get("persona", "")
        persona_dict = persona_by_name.get(name)
        if persona_dict is None:
            global_warnings.append(f"Persona '{name}' not found in provided --personas file. Skipped.")
            continue
        matched += 1
        ordered_personas.append(persona_dict)
        ordered_raw_responses.append(entry.get("responses") or [])

    if not matched:
        global_warnings.append(
            "No panelist names matched the provided personas. Check that --personas is the same file used for the run."
        )
        return SubgroupAnalysisResult(
            result_id=result_id,
            fields=by_tokens,
            per_question=[],
            global_warnings=global_warnings,
        )

    # Derive question list (from saved or from first panelist)
    if saved_questions:
        questions = saved_questions
    else:
        questions = _infer_questions(ordered_raw_responses)

    per_question: list[QuestionSubgroupResult] = []
    for qi, q in enumerate(questions):
        if question_indices is not None and qi not in question_indices:
            continue

        q_text = q.get("text", f"Question {qi + 1}") if isinstance(q, dict) else str(q)
        response_schema: dict[str, Any] = {}
        if isinstance(q, dict):
            response_schema = q.get("response_schema") or q.get("extraction_schema") or {}

        schema_type = response_schema.get("type", "text") if isinstance(response_schema, dict) else "text"
        if not isinstance(response_schema, dict) or schema_type not in (
            "scale",
            "enum",
            "tagged_themes",
        ):
            # text-only questions: nothing meaningful to aggregate
            continue

        # Extract the typed response value for each matched panelist
        per_panelist_values: list[Any] = []
        for panelist_responses in ordered_raw_responses:
            if qi < len(panelist_responses):
                resp = panelist_responses[qi]
                if resp.get("error") or resp.get("follow_up"):
                    per_panelist_values.append(None)
                else:
                    raw_val = _response_value(resp)
                    per_panelist_values.append(_flatten_structured_value(raw_val, schema_type))
            else:
                per_panelist_values.append(None)

        by_breakdowns: list[FieldBreakdown] = []
        for token in by_tokens:
            bd = _build_field_breakdown(
                token,
                per_panelist_values,
                ordered_personas,
                response_schema,
            )
            by_breakdowns.append(bd)

        per_question.append(
            QuestionSubgroupResult(
                question_index=qi,
                question_text=q_text,
                response_schema_type=schema_type,
                n_responses=sum(1 for v in per_panelist_values if v is not None),
                by_field=by_breakdowns,
            )
        )

    return SubgroupAnalysisResult(
        result_id=result_id,
        fields=by_tokens,
        per_question=per_question,
        global_warnings=global_warnings,
    )


def _infer_questions(ordered_raw_responses: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Derive question list from first panelist's responses."""
    if not ordered_raw_responses:
        return []
    seen: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for resp in ordered_raw_responses[0]:
        if resp.get("follow_up"):
            continue
        q_text = resp.get("question", "")
        if q_text and q_text not in seen_texts:
            seen.append({"text": q_text})
            seen_texts.add(q_text)
    return seen


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

_EFFECT_LABELS = {
    "eta-squared": "η²",
    "cramers-v": "V",
    "density-ratio": "density",
    "none": "",
}


def format_subgroup_text(result: SubgroupAnalysisResult) -> str:
    lines: list[str] = []
    lines.append(f"Subgroup analysis — {result.result_id}")
    lines.append(f"Grouping by: {', '.join(result.fields)}")
    if result.global_warnings:
        for w in result.global_warnings:
            lines.append(f"  WARNING: {w}")
    lines.append("")

    if not result.per_question:
        lines.append("No structured questions found. Run the panel with response_schema to enable subgroup analysis.")
        return "\n".join(lines)

    for qr in result.per_question:
        lines.append(f"Q{qr.question_index + 1}: {qr.question_text}  [{qr.response_schema_type}] (n={qr.n_responses})")
        for bd in qr.by_field:
            label = _EFFECT_LABELS.get(bd.effect_size_type, "")
            if bd.effect_size is not None:
                eff_str = f"{label}={bd.effect_size:.3f}"
                if bd.p_value is not None:
                    eff_str += f"  p={bd.p_value:.4f}"
                lines.append(f"  --by {bd.field}  ({eff_str})")
            else:
                lines.append(f"  --by {bd.field}")
            for w in bd.warnings:
                lines.append(f"    ! {w}")
            for bucket in bd.buckets:
                dist = bucket.get("distribution", {})
                lines.append(_format_bucket(bucket, dist, qr.response_schema_type))
        lines.append("")

    return "\n".join(lines)


def _format_bucket(bucket: dict[str, Any], dist: dict[str, Any], schema_type: str) -> str:
    label = bucket.get("label", "?")
    n = bucket.get("n", 0)
    parts: list[str] = [f"    {label} (n={n})"]

    if schema_type == "scale":
        stats = dist.get("stats", {})
        mean = stats.get("mean")
        stdev = stats.get("stdev")
        if mean is not None:
            parts.append(f"  mean={mean:.2f}")
        if stdev is not None:
            parts.append(f"  sd={stdev:.2f}")
    elif schema_type == "enum":
        entries = dist.get("distribution", [])
        total = sum(e.get("count", 0) for e in entries)
        parts.append(
            "  "
            + "  ".join(
                f"{e['option']}={e['count']}" + (f"({e['count'] / total:.0%})" if total else "")
                for e in entries
                if e.get("count", 0) > 0
            )
        )
    elif schema_type == "tagged_themes":
        entries = dist.get("distribution", [])
        total_resp = dist.get("n_valid", n)
        theme_parts = []
        for e in entries:
            cnt = e.get("count", 0)
            tag = e.get("tag", "?")
            if cnt > 0:
                density = cnt / total_resp if total_resp > 0 else 0
                theme_parts.append(f"{tag}={density:.0%}")
        parts.append("  " + "  ".join(theme_parts) if theme_parts else "  (no themes)")

    return "".join(parts)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def format_subgroup_json(result: SubgroupAnalysisResult) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "fields": result.fields,
        "global_warnings": result.global_warnings,
        "per_question": [
            {
                "question_index": qr.question_index,
                "question_text": qr.question_text,
                "response_schema_type": qr.response_schema_type,
                "n_responses": qr.n_responses,
                "by_field": [
                    {
                        "field": bd.field,
                        "n_buckets": bd.n_buckets,
                        "effect_size_type": bd.effect_size_type,
                        "effect_size": bd.effect_size,
                        "p_value": bd.p_value,
                        "warnings": bd.warnings,
                        "buckets": bd.buckets,
                    }
                    for bd in qr.by_field
                ],
            }
            for qr in result.per_question
        ],
    }
