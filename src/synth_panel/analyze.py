"""Panel result analysis engine.

Runs four analysis sections over a saved panel result:
  1. Descriptive — frequency tables, ranking distributions, Borda count
  2. Inferential — bootstrap CIs, chi-squared, Kendall's W, Cramer's V
  3. Cross-model — Krippendorff's alpha (when multi-model run)
  4. Clusters   — persona groupings (when N >= 30)

Plus a warnings section that flags statistical concerns.
"""

from __future__ import annotations

import contextlib as _contextlib
import math
from dataclasses import dataclass
from typing import Any

from synth_panel.stats import (
    ChiSquaredResult,
    ClusterResult,
    ConvergenceReport,
    FrequencyTable,
    KendallWResult,
    borda_count,
    cluster_personas,
    convergence_report,
    frequency_table,
    kendall_w,
)
from synth_panel.text_width import pad, truncate

# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class QuestionAnalysis:
    """Analysis results for a single question."""

    question_index: int
    question_text: str
    n_responses: int
    frequency: FrequencyTable
    chi_squared: ChiSquaredResult | None
    kendall_w: KendallWResult | None
    borda: dict[str, float] | None  # scores from borda_count


@dataclass
class AnalysisResult:
    """Full analysis output."""

    result_id: str
    persona_count: int
    question_count: int
    model: str
    is_multi_model: bool
    per_question: list[QuestionAnalysis]
    convergence: ConvergenceReport | None
    clusters: ClusterResult | None
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLUSTER_MIN_N = 30


def _extract_responses_per_question(
    result_data: dict[str, Any],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Extract per-question response lists from a panel result.

    Returns (question_texts, responses_per_question, persona_names).
    responses_per_question[q_idx] = [response_str for each non-error persona].
    """
    panelists = result_data.get("results", [])
    if not panelists:
        return [], [], []

    # Determine question texts from the first panelist (all responses,
    # including errors — errors affect data, not the question list).
    first = panelists[0]
    question_texts = [r["question"] for r in first.get("responses", [])]
    n_questions = len(question_texts)

    responses_per_q: list[list[str]] = [[] for _ in range(n_questions)]
    persona_names: list[str] = []

    for p in panelists:
        persona_names.append(p.get("persona", "unknown"))
        all_resps = p.get("responses", [])
        for qi in range(min(n_questions, len(all_resps))):
            if not all_resps[qi].get("error"):
                responses_per_q[qi].append(all_resps[qi].get("response", ""))

    return question_texts, responses_per_q, persona_names


def _detect_multi_model(result_data: dict[str, Any]) -> tuple[bool, dict[str, list[int]]]:
    """Check if the panel result used multiple models.

    Returns (is_multi_model, model_to_panelist_indices).
    """
    panelists = result_data.get("results", [])
    model_map: dict[str, list[int]] = {}
    base_model = result_data.get("model", "unknown")

    for idx, p in enumerate(panelists):
        m = p.get("model", base_model)
        model_map.setdefault(m, []).append(idx)

    return len(model_map) > 1, model_map


def _build_multi_model_responses(
    result_data: dict[str, Any],
    model_map: dict[str, list[int]],
    n_questions: int,
) -> dict[str, list[list[str]]]:
    """Build model_name -> list[list[str]] for convergence_report.

    Shape: model -> [persona_idx][question_idx] -> response.
    convergence_report expects outer=personas, inner=questions.
    """
    panelists = result_data.get("results", [])
    out: dict[str, list[list[str]]] = {}

    for model_name, indices in model_map.items():
        personas: list[list[str]] = []
        for idx in indices:
            resps = [r for r in panelists[idx].get("responses", []) if not r.get("error")]
            persona_answers = [resps[qi].get("response", "") if qi < len(resps) else "" for qi in range(n_questions)]
            personas.append(persona_answers)
        out[model_name] = personas

    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_panel_result(result_data: dict[str, Any]) -> AnalysisResult:
    """Run the full 4-section analysis on a panel result."""
    result_id = result_data.get("id", "unknown")
    persona_count = result_data.get("persona_count", 0)
    question_count = result_data.get("question_count", 0)
    model = result_data.get("model", "unknown")

    question_texts, responses_per_q, _persona_names = _extract_responses_per_question(result_data)
    is_multi_model, model_map = _detect_multi_model(result_data)

    warnings: list[str] = []
    per_question: list[QuestionAnalysis] = []

    # --- Section 1 & 2: Descriptive + Inferential (per question) ---
    for qi, q_text in enumerate(question_texts):
        responses = responses_per_q[qi] if qi < len(responses_per_q) else []
        n = len(responses)

        if n < 5:
            warnings.append(f"Q{qi + 1}: only {n} responses — statistical tests unreliable")

        # Frequency table (includes bootstrap CIs and chi-squared)
        freq = frequency_table(responses, bootstrap_ci_conf=0.95 if n >= 5 else None)

        chi2 = freq.chi_squared

        # Check for wide CIs
        if freq.rows:
            for row in freq.rows:
                ci_width = row.ci_upper - row.ci_lower
                if ci_width > 0.5 and n >= 5:
                    warnings.append(f"Q{qi + 1}, '{row.category}': wide CI ({row.ci_lower:.2f}-{row.ci_upper:.2f})")

        # Kendall's W — needs ranking data; approximate from response frequencies
        # Build per-persona rankings from unique responses
        kw: KendallWResult | None = None
        unique_responses = list({r for r in responses if r})
        if len(unique_responses) >= 2 and n >= 3:
            # Each persona "ranks" items by whether they chose them (1=chosen, 2+=not)
            rankings: list[list[int]] = []
            for resp in responses:
                rank = []
                for ur in unique_responses:
                    rank.append(1 if resp == ur else 2)
                rankings.append(rank)
            with _contextlib.suppress(ValueError, ZeroDivisionError):
                kw = kendall_w(rankings)

        # Borda count — meaningful when responses are categorical
        borda_scores: dict[str, float] | None = None
        if len(unique_responses) >= 2:
            # Build ranking dicts: each persona ranks their choice as 1, others as len
            borda_rankings: list[dict[str, int]] = []
            for resp in responses:
                ranking_dict = {}
                for ur in unique_responses:
                    ranking_dict[ur] = 1 if resp == ur else len(unique_responses)
                borda_rankings.append(ranking_dict)
            try:
                br = borda_count(borda_rankings)
                borda_scores = br.scores
            except (ValueError, ZeroDivisionError):
                pass

        per_question.append(
            QuestionAnalysis(
                question_index=qi,
                question_text=q_text,
                n_responses=n,
                frequency=freq,
                chi_squared=chi2,
                kendall_w=kw,
                borda=borda_scores,
            )
        )

    # --- Section 3: Cross-model convergence ---
    conv: ConvergenceReport | None = None
    if is_multi_model and question_texts:
        # convergence_report requires each model to have the same N personas
        # (same personas run through each model). Ensemble runs assign different
        # personas to different models, so check if N is uniform.
        model_ns = {m: len(idxs) for m, idxs in model_map.items()}
        all_same_n = len(set(model_ns.values())) == 1

        if all_same_n and next(iter(model_ns.values())) >= 2:
            multi_resps = _build_multi_model_responses(result_data, model_map, len(question_texts))
            try:
                conv = convergence_report(multi_resps, question_texts)
            except (ValueError, ZeroDivisionError):
                warnings.append("Cross-model convergence could not be computed")
        else:
            # Ensemble run — personas are split across models, not shared.
            # Krippendorff's alpha requires shared observations; report
            # per-model distributions instead via the per-question frequency
            # tables (already computed above).
            warnings.append(
                "Cross-model convergence (Krippendorff's alpha) requires "
                "equal persona counts per model; ensemble run has unequal "
                f"splits ({model_ns}). Per-model distributions are in the "
                "descriptive section."
            )

    # --- Section 4: Persona clustering ---
    clusters: ClusterResult | None = None
    if persona_count >= _CLUSTER_MIN_N and responses_per_q:
        # Build persona_responses: name -> [response per question]
        persona_responses: dict[str, list[str]] = {}
        panelists = result_data.get("results", [])
        for p in panelists:
            name = p.get("persona", "unknown")
            resps = [r for r in p.get("responses", []) if not r.get("error")]
            persona_responses[name] = [r.get("response", "") for r in resps[: len(question_texts)]]

        try:
            clusters = cluster_personas(persona_responses)
        except (ValueError, ZeroDivisionError):
            warnings.append("Persona clustering could not be computed")
    elif persona_count < _CLUSTER_MIN_N:
        warnings.append(f"Persona clustering skipped: N={persona_count} (minimum {_CLUSTER_MIN_N})")

    # --- Additional warnings from last question's chi-squared ---
    for qa in per_question:
        if qa.chi_squared is not None and qa.chi_squared.warning:
            warnings.append(f"Chi-squared: {qa.chi_squared.warning}")
            break  # One warning is enough

    return AnalysisResult(
        result_id=result_id,
        persona_count=persona_count,
        question_count=question_count,
        model=model,
        is_multi_model=is_multi_model,
        per_question=per_question,
        convergence=conv,
        clusters=clusters,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _freq_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "category": row.category,
        "count": row.count,
        "proportion": round(row.proportion, 4),
        "ci_lower": round(row.ci_lower, 4),
        "ci_upper": round(row.ci_upper, 4),
    }


def _safe_round(v: float | None, n: int = 4) -> float | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(v, n)


def analysis_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Convert AnalysisResult to a JSON-serializable dict."""
    questions = []
    for qa in result.per_question:
        q: dict[str, Any] = {
            "question_index": qa.question_index,
            "question_text": qa.question_text,
            "n_responses": qa.n_responses,
            "frequency_table": [_freq_row_to_dict(r) for r in qa.frequency.rows],
            "total": qa.frequency.total,
        }
        if qa.chi_squared:
            q["chi_squared"] = {
                "statistic": _safe_round(qa.chi_squared.statistic),
                "df": qa.chi_squared.df,
                "p_value": _safe_round(qa.chi_squared.p_value),
                "cramers_v": _safe_round(qa.chi_squared.cramers_v),
            }
        if qa.kendall_w:
            q["kendall_w"] = {
                "w": _safe_round(qa.kendall_w.w),
                "chi_squared": _safe_round(qa.kendall_w.chi_squared),
                "df": qa.kendall_w.df,
                "p_value": _safe_round(qa.kendall_w.p_value),
            }
        if qa.borda:
            q["borda_scores"] = {k: _safe_round(v) for k, v in qa.borda.items()}
        questions.append(q)

    out: dict[str, Any] = {
        "result_id": result.result_id,
        "persona_count": result.persona_count,
        "question_count": result.question_count,
        "model": result.model,
        "is_multi_model": result.is_multi_model,
        "per_question": questions,
        "warnings": result.warnings,
    }

    if result.convergence:
        findings = []
        for f in result.convergence.findings:
            findings.append(
                {
                    "question_index": f.question_index,
                    "question_text": f.question_text,
                    "alpha": _safe_round(f.alpha),
                    "level": f.level.value,
                    "top_choice_agreement": f.top_choice_agreement,
                    "divergent_models": f.divergent_models,
                }
            )
        out["convergence"] = {
            "overall_alpha": _safe_round(result.convergence.overall_alpha),
            "overall_level": result.convergence.overall_level.value,
            "n_convergent": result.convergence.n_convergent,
            "n_divergent": result.convergence.n_divergent,
            "n_models": result.convergence.n_models,
            "model_names": result.convergence.model_names,
            "findings": findings,
        }

    if result.clusters:
        cluster_list = []
        for c in result.clusters.clusters:
            cluster_list.append(
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "persona_names": c.persona_names,
                    "dominant_responses": {str(k): v for k, v in c.dominant_responses.items()},
                }
            )
        out["clusters"] = {
            "n_clusters": result.clusters.n_clusters,
            "silhouette_score": _safe_round(result.clusters.silhouette_score),
            "clusters": cluster_list,
        }

    return out


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def format_text(result: AnalysisResult) -> str:
    """Format analysis as human-readable text."""
    lines: list[str] = []
    lines.append(f"Panel Analysis: {result.result_id}")
    lines.append(f"  Model: {result.model}" + (" (multi-model)" if result.is_multi_model else ""))
    lines.append(f"  Personas: {result.persona_count}  Questions: {result.question_count}")
    lines.append("")

    # Section 1: Descriptive
    lines.append("=" * 60)
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append("=" * 60)

    for qa in result.per_question:
        lines.append("")
        lines.append(f"Q{qa.question_index + 1}: {qa.question_text}")
        lines.append(f"  N = {qa.n_responses}")
        lines.append("")

        # Frequency table
        lines.append(f"  {pad('Category', 30)} {'Count':>5} {'Prop':>6} {'95% CI':>16}")
        lines.append(f"  {'-' * 30} {'-' * 5} {'-' * 6} {'-' * 16}")
        for row in qa.frequency.rows:
            cat = truncate(row.category, 30)
            ci = f"[{row.ci_lower:.2f}, {row.ci_upper:.2f}]"
            lines.append(f"  {pad(cat, 30)} {row.count:>5} {row.proportion:>6.2f} {ci:>16}")

        # Borda scores
        if qa.borda:
            lines.append("")
            lines.append("  Borda scores:")
            for item, score in sorted(qa.borda.items(), key=lambda x: -x[1]):
                lines.append(f"    {item}: {score:.1f}")

    # Section 2: Inferential
    lines.append("")
    lines.append("=" * 60)
    lines.append("INFERENTIAL STATISTICS")
    lines.append("=" * 60)

    for qa in result.per_question:
        lines.append("")
        lines.append(f"Q{qa.question_index + 1}: {qa.question_text}")

        if qa.chi_squared:
            lines.append(
                f"  Chi-squared: X2={qa.chi_squared.statistic:.3f}, "
                f"df={qa.chi_squared.df}, p={qa.chi_squared.p_value:.4f}"
            )
            lines.append(f"  Cramer's V: {qa.chi_squared.cramers_v:.3f}")

        if qa.kendall_w:
            lines.append(
                f"  Kendall's W: W={qa.kendall_w.w:.3f}, "
                f"X2={qa.kendall_w.chi_squared:.3f}, "
                f"df={qa.kendall_w.df}, p={qa.kendall_w.p_value:.4f}"
            )

    # Section 3: Cross-model
    if result.convergence:
        lines.append("")
        lines.append("=" * 60)
        lines.append("CROSS-MODEL CONVERGENCE")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Models: {', '.join(result.convergence.model_names)} (N={result.convergence.n_models})")
        lines.append(
            f"  Overall alpha: {result.convergence.overall_alpha:.3f} ({result.convergence.overall_level.value})"
        )
        lines.append(f"  Convergent: {result.convergence.n_convergent}  Divergent: {result.convergence.n_divergent}")
        for f in result.convergence.findings:
            lines.append("")
            lines.append(f"  Q{f.question_index + 1}: alpha={f.alpha:.3f} ({f.level.value})")
            if f.divergent_models:
                lines.append(f"    Divergent: {', '.join(f.divergent_models)}")

    # Section 4: Clusters
    if result.clusters:
        lines.append("")
        lines.append("=" * 60)
        lines.append("PERSONA CLUSTERS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Clusters: {result.clusters.n_clusters}  Silhouette: {result.clusters.silhouette_score:.3f}")
        for c in result.clusters.clusters:
            lines.append("")
            lines.append(f"  Cluster {c.cluster_id} (N={c.size}):")
            lines.append(f"    Personas: {', '.join(c.persona_names[:5])}")
            if len(c.persona_names) > 5:
                lines.append(f"    ... and {len(c.persona_names) - 5} more")

    # Warnings
    if result.warnings:
        lines.append("")
        lines.append("=" * 60)
        lines.append("WARNINGS")
        lines.append("=" * 60)
        for w in result.warnings:
            lines.append(f"  ! {w}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------


def format_csv(result: AnalysisResult) -> str:
    """Format key metrics as CSV for spreadsheet export."""
    lines: list[str] = []

    # Frequency table CSV
    lines.append("section,question_index,question_text,category,count,proportion,ci_lower,ci_upper")
    for qa in result.per_question:
        for row in qa.frequency.rows:
            q_text = qa.question_text.replace('"', '""')
            cat = row.category.replace('"', '""')
            lines.append(
                f'frequency,{qa.question_index},"{q_text}","{cat}",'
                f"{row.count},{row.proportion:.4f},{row.ci_lower:.4f},{row.ci_upper:.4f}"
            )

    # Inferential stats CSV
    lines.append("")
    lines.append("section,question_index,test,statistic,df,p_value,effect_size")
    for qa in result.per_question:
        if qa.chi_squared:
            lines.append(
                f"inferential,{qa.question_index},chi_squared,"
                f"{qa.chi_squared.statistic:.4f},{qa.chi_squared.df},"
                f"{qa.chi_squared.p_value:.4f},{qa.chi_squared.cramers_v:.4f}"
            )
        if qa.kendall_w:
            lines.append(
                f"inferential,{qa.question_index},kendall_w,"
                f"{qa.kendall_w.w:.4f},{qa.kendall_w.df},"
                f"{qa.kendall_w.p_value:.4f},"
            )

    # Convergence CSV
    if result.convergence:
        lines.append("")
        lines.append("section,question_index,alpha,level,top_choice_agreement")
        for f in result.convergence.findings:
            lines.append(f"convergence,{f.question_index},{f.alpha:.4f},{f.level.value},{f.top_choice_agreement}")

    return "\n".join(lines)
