"""Post-run variant aggregation and robustness scoring (sp-5on.16).

Groups panelist results by the ``_variant_of`` metadata field (set by the
perturbation module, sp-5on.14) and computes two-level nested robustness
scores: intra-persona agreement across variants (inner) and inter-persona
agreement across the panel (outer).

Formula: R(F) = (1/N) * sum_i[(1/K_i) * sum_j 1{variant_ij agrees with F}]

Thresholds: R >= 0.8 robust, 0.6-0.8 moderate, 0.4-0.6 sensitive, < 0.4 fragile.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from synth_panel.orchestrator import PanelistResult

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VariantGroup:
    """Results for a source persona and all its variants."""

    source_name: str
    original: PanelistResult | None
    variants: list[PanelistResult] = field(default_factory=list)

    @property
    def k(self) -> int:
        """Number of variants."""
        return len(self.variants)


@dataclass(frozen=True)
class FindingRobustness:
    """Robustness of responses to a single question across all personas."""

    question_index: int
    score: float
    classification: str
    per_persona: dict[str, float]


@dataclass(frozen=True)
class RobustnessReport:
    """Complete two-level robustness report."""

    findings: list[FindingRobustness]
    aggregate_robustness: float
    fragile_findings: list[FindingRobustness]
    n_personas: int
    per_persona_robustness: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_extractor(response: dict[str, Any]) -> str:
    """Extract a comparable string value from a response dict."""
    val = response.get("response", "")
    return str(val)


def _classify_robustness(r: float) -> str:
    """Classify a robustness score into a named tier."""
    if r >= 0.8:
        return "robust"
    elif r >= 0.6:
        return "moderate"
    elif r >= 0.4:
        return "sensitive"
    return "fragile"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def aggregate_variants(
    results: list[PanelistResult],
    personas: list[dict[str, Any]],
) -> list[VariantGroup]:
    """Group panelist results by ``_variant_of`` metadata.

    Matches results to personas by name.  Personas whose dict contains a
    ``_variant_of`` key are variants; all others are originals.

    Returns one :class:`VariantGroup` per source persona that has at least
    one variant.  Groups are sorted by source name for deterministic order.

    Args:
        results: Panelist results from a panel run that included both
            original personas and their perturbation variants.
        personas: The persona dicts used in the panel run.  Variant
            personas must carry the ``_variant_of`` field set by the
            perturbation module.

    Returns:
        List of :class:`VariantGroup`, one per source persona with variants.
    """
    persona_by_name: dict[str, dict[str, Any]] = {p["name"]: p for p in personas if p.get("name")}

    originals: dict[str, PanelistResult] = {}
    variant_map: dict[str, list[PanelistResult]] = {}

    for r in results:
        persona = persona_by_name.get(r.persona_name)
        if persona and "_variant_of" in persona:
            source = persona["_variant_of"]
            variant_map.setdefault(source, []).append(r)
        else:
            originals[r.persona_name] = r

    groups: list[VariantGroup] = []
    for source_name in sorted(set(originals) | set(variant_map)):
        variants = variant_map.get(source_name, [])
        if not variants:
            continue
        groups.append(
            VariantGroup(
                source_name=source_name,
                original=originals.get(source_name),
                variants=variants,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def robustness_report(
    groups: list[VariantGroup],
    *,
    response_extractor: Callable[[dict[str, Any]], str] | None = None,
) -> RobustnessReport:
    """Compute two-level nested robustness scores from variant groups.

    For each question, the original persona's response is the reference.
    Each variant's response is compared for agreement (exact match on the
    extracted string value).

    **Inner level** (intra-persona): for persona *i* with *K_i* variants,
    the agreement fraction is ``agreements / K_i``.

    **Outer level** (inter-persona): for question *q*,
    ``R(q) = (1/N) * sum_i(agreement_fraction_i)``.

    Args:
        groups: Variant groups from :func:`aggregate_variants`.
        response_extractor: Extracts a comparable string from a response
            dict.  Defaults to ``str(response["response"])``.

    Returns:
        :class:`RobustnessReport` with per-finding and per-persona scores.

    Raises:
        ValueError: If *groups* is empty or no group has both an original
            result and at least one variant.
    """
    if not groups:
        raise ValueError("groups must not be empty")

    extract = response_extractor or _default_extractor

    scorable = [g for g in groups if g.original is not None and g.k > 0]
    if not scorable:
        raise ValueError("no groups with both an original and variants")

    n_questions = max(len(g.original.responses) for g in scorable)

    findings: list[FindingRobustness] = []
    per_persona_all: dict[str, list[float]] = {}

    for qi in range(n_questions):
        per_persona: dict[str, float] = {}

        for group in scorable:
            if qi >= len(group.original.responses):
                continue

            ref = extract(group.original.responses[qi])

            k = 0
            agreements = 0
            for v in group.variants:
                if qi >= len(v.responses):
                    continue
                k += 1
                if extract(v.responses[qi]) == ref:
                    agreements += 1

            score = agreements / k if k > 0 else 0.0
            per_persona[group.source_name] = score
            per_persona_all.setdefault(group.source_name, []).append(score)

        if per_persona:
            finding_score = sum(per_persona.values()) / len(per_persona)
            findings.append(
                FindingRobustness(
                    question_index=qi,
                    score=finding_score,
                    classification=_classify_robustness(finding_score),
                    per_persona=per_persona,
                )
            )

    aggregate = sum(f.score for f in findings) / len(findings) if findings else 0.0
    fragile = [f for f in findings if f.score < 0.6]

    persona_robustness: dict[str, float] = {}
    for name, scores in per_persona_all.items():
        persona_robustness[name] = sum(scores) / len(scores) if scores else 0.0

    return RobustnessReport(
        findings=findings,
        aggregate_robustness=aggregate,
        fragile_findings=fragile,
        n_personas=len(scorable),
        per_persona_robustness=persona_robustness,
    )
