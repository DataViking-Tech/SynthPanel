"""Shared synchronous panel runners.

Factored out of ``synth_panel.mcp.server`` so that both the MCP server
handlers and the public Python SDK (``synth_panel.sdk``) can drive the
same underlying logic without one depending on the other — the MCP
server requires the optional ``mcp`` extra, whereas the SDK has to
work in a plain ``pip install synthpanel`` environment.

Nothing here imports the ``mcp`` library.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from synth_panel.cost import ZERO_USAGE, resolve_cost
from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.instrument import Instrument
from synth_panel.llm.client import LLMClient
from synth_panel.orchestrator import (
    MultiRoundResult,
    PanelistResult,
    run_multi_round_panel,
    run_panel_parallel,
)
from synth_panel.perturbation import generate_panel_variants
from synth_panel.prompts import build_question_prompt, persona_system_prompt
from synth_panel.stats import robustness_score
from synth_panel.synthesis import synthesize_panel

# sp-efip: max number of per-persona error samples to surface in the
# diagnostic output when a run fails wholesale. Keeps CLI banners and
# MCP error envelopes bounded while still naming real errors.
_MAX_SAMPLE_ERRORS = 3


class PanelTotalFailureError(RuntimeError):
    """Every panelist failed — no usable Q/A data was produced.

    Raised by ``run_panel_sync`` (and equivalents) when the orchestrator
    returns results where every panelist either errored wholesale or
    produced zero tokens with only errored responses. Callers should
    surface the message verbatim so operators see the failing model(s)
    and upstream provider error (e.g. "OpenRouter API error 400: …").
    """

    def __init__(self, message: str, *, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def _first_error_message(pr: Any) -> str | None:
    """Return a representative error string for *pr*, or None if clean."""
    err = getattr(pr, "error", None)
    if err:
        return str(err)
    for resp in getattr(pr, "responses", []) or []:
        if isinstance(resp, dict) and resp.get("error"):
            text = resp.get("response")
            if isinstance(text, str) and text:
                return text
            return "unspecified per-question error"
    return None


def detect_total_failure(panelist_results: list[Any]) -> dict[str, Any] | None:
    """Return a diagnostic dict when every panelist failed, else None.

    Total failure means that *no* panelist produced a usable Q/A pair.
    A panelist counts as failed when any of the following hold:

    * ``pr.error`` is set (wholesale panelist exception), OR
    * every response in ``pr.responses`` is flagged ``error: True``
      (per-question exceptions caught inside ``_run_panelist``), OR
    * the panelist produced zero tokens *and* recorded no primary
      non-errored response (the 400-on-every-call scenario from
      sp-efip, where tokens stayed at 0 because every request was
      rejected upstream).

    The returned dict carries the models exercised and a few sample
    per-persona error strings so downstream banners / MCP errors can
    name the upstream failure (e.g. the bad model name and HTTP 400
    the bead explicitly calls out).
    """
    if not panelist_results:
        return {
            "panelists": 0,
            "models": [],
            "sample_errors": [],
            "summary": "orchestrator returned no panelist results",
        }

    models: set[str] = set()
    sample_errors: list[tuple[str, str]] = []
    all_failed = True

    for pr in panelist_results:
        model = getattr(pr, "model", None)
        if model:
            models.add(str(model))

        pr_error = getattr(pr, "error", None)
        responses = getattr(pr, "responses", []) or []
        primary = [r for r in responses if isinstance(r, dict) and not r.get("follow_up")]
        has_clean_primary = any(isinstance(r, dict) and not r.get("error") and r.get("response") for r in primary)

        # A panelist is considered usable iff it produced at least one
        # non-errored primary response. The bead explicitly calls out
        # the "tokens=0 AND error set" short-circuit, which is already
        # subsumed by "no clean primary response" — zero-token panelists
        # whose only responses are error=True rows all fall here.
        panelist_failed = bool(pr_error) or not has_clean_primary
        if not panelist_failed:
            all_failed = False
            continue

        err_msg = _first_error_message(pr)
        if err_msg and len(sample_errors) < _MAX_SAMPLE_ERRORS:
            persona = getattr(pr, "persona_name", "unknown")
            sample_errors.append((str(persona), err_msg))

    if not all_failed:
        return None

    return {
        "panelists": len(panelist_results),
        "models": sorted(models),
        "sample_errors": sample_errors,
        "summary": (f"{len(panelist_results)} panelist(s) produced no usable Q/A pairs"),
    }


def format_total_failure_message(diagnostic: dict[str, Any]) -> str:
    """Render a single-line message naming failing models + sample error."""
    models = diagnostic.get("models") or []
    sample_errors = diagnostic.get("sample_errors") or []
    parts = [f"Panel run produced no usable results: {diagnostic.get('summary', 'every panelist failed')}"]
    if models:
        parts.append(f"models: {', '.join(models)}")
    if sample_errors:
        persona, err = sample_errors[0]
        parts.append(f"first error ({persona}): {err}")
    return ". ".join(parts)


logger = logging.getLogger(__name__)

# Caps mirrored from the MCP server — the SDK inherits the same guardrails.
MAX_PERSONAS = 100
MAX_QUESTIONS = 50

# Per-panelist timeout budget used by async wrappers (seconds).
PANELIST_TIMEOUT = 30


# Built-in extraction schema registry — keyed by short name.
EXTRACT_SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "sentiment": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
                "description": "Overall sentiment of the response.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score (0-1) for the sentiment classification.",
            },
        },
        "required": ["sentiment", "confidence"],
    },
    "themes": {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key themes or topics mentioned in the response.",
            },
            "primary_theme": {
                "type": "string",
                "description": "The single most dominant theme.",
            },
        },
        "required": ["themes", "primary_theme"],
    },
    "rating": {
        "type": "object",
        "properties": {
            "rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Numeric rating (1-10) implied by the response.",
            },
            "explanation": {
                "type": "string",
                "description": "Brief rationale for the assigned rating.",
            },
        },
        "required": ["rating", "explanation"],
    },
}


def resolve_extract_schema(
    value: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve extract_schema: pass dicts through, look up strings in the registry."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if value not in EXTRACT_SCHEMA_REGISTRY:
            names = ", ".join(sorted(EXTRACT_SCHEMA_REGISTRY))
            raise ValueError(
                f"Unknown extract_schema name {value!r}. Available: {names}. Or pass an inline JSON Schema dict."
            )
        return EXTRACT_SCHEMA_REGISTRY[value]
    raise TypeError(f"extract_schema must be a string or dict, got {type(value).__name__}")


def format_panelist_result(pr: PanelistResult, model: str) -> dict[str, Any]:
    """Render a :class:`PanelistResult` into the serialisable dict shape callers expect."""
    pr_model = pr.model or model
    # sp-kvpx: route through resolve_cost so per-panelist ``cost`` honors
    # sp-j3vk's precedence (provider-reported → local fallback). Prior to
    # this, ensemble/mixed-model runs quoted per-persona cost from the
    # local pricing table regardless of what the provider actually billed,
    # so ``per_model_results[*].results[i].cost`` drifted from the top-level
    # ``total_cost`` by the same ratio sp-j3vk fixed at the aggregate layer.
    persona_cost = resolve_cost(pr.usage, pr_model)
    rd: dict[str, Any] = {
        "persona": pr.persona_name,
        "responses": pr.responses,
        "usage": pr.usage.to_dict(),
        "cost": persona_cost.format_usd(),
        "error": pr.error,
    }
    if pr.model:
        rd["model"] = pr.model
    return rd


def compute_variant_data(
    result_dicts: list[dict[str, Any]],
    variant_names: set[str],
    variant_mapping: dict[str, str],
    variant_count: int,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute robustness scores from base + variant panel results."""
    # Separate base and variant results
    base_results = [r for r in result_dicts if r["persona"] not in variant_names]
    variant_results = [r for r in result_dicts if r["persona"] in variant_names]

    # Build per-question robustness scores
    robustness_scores: list[dict[str, Any]] = []
    per_persona_robustness: list[dict[str, Any]] = []
    n_questions = len(questions)

    for qi in range(n_questions):
        # Group variant responses by source persona
        variant_resps: dict[str, list[str]] = {}
        for vr in variant_results:
            source = variant_mapping.get(vr["persona"], "")
            if not source:
                continue
            resps = vr.get("responses", [])
            if qi < len(resps) and not resps[qi].get("error"):
                variant_resps.setdefault(source, []).append(resps[qi].get("response", ""))

        # Get base persona responses for this question
        for br in base_results:
            resps = br.get("responses", [])
            if qi >= len(resps) or resps[qi].get("error"):
                continue

            base_response = resps[qi].get("response", "")
            persona_name = br["persona"]
            v_resps = variant_resps.get(persona_name, [])

            if v_resps:
                try:
                    rs = robustness_score({persona_name: v_resps}, base_response)
                    per_persona_robustness.append(
                        {
                            "persona": persona_name,
                            "question_index": qi,
                            "robustness": round(rs.per_persona.get(persona_name, 0.0), 4),
                            "k_variants": len(v_resps),
                            "finding_value": base_response,
                            "interpretation": rs.interpretation,
                        }
                    )
                except (ValueError, ZeroDivisionError):
                    pass

        # Overall robustness for this question across all base personas
        if variant_resps:
            base_responses_q = [
                br["responses"][qi].get("response", "")
                for br in base_results
                if qi < len(br.get("responses", [])) and not br["responses"][qi].get("error")
            ]
            if base_responses_q:
                most_common = Counter(base_responses_q).most_common(1)[0][0]
                all_v: dict[str, list[str]] = {}
                for name, resps_list in variant_resps.items():
                    all_v[name] = resps_list
                try:
                    agg = robustness_score(all_v, most_common)
                    robustness_scores.append(
                        {
                            "question_index": qi,
                            "question_text": questions[qi].get("text", ""),
                            "finding_value": most_common,
                            "overall_robustness": round(agg.overall_robustness, 4),
                            "interpretation": agg.interpretation,
                        }
                    )
                except (ValueError, ZeroDivisionError):
                    pass

    return {
        "variant_count": variant_count,
        "robustness_scores": robustness_scores,
        "per_persona_robustness": per_persona_robustness,
    }


def run_panel_sync(
    client: LLMClient,
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str,
    response_schema: dict[str, Any] | None = None,
    *,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    persona_models: dict[str, str] | None = None,
    extract_schema: dict[str, Any] | None = None,
    synthesis_temperature: float | None = None,
    variants: int = 0,
) -> tuple[
    list[PanelistResult],
    list[dict[str, Any]],
    CostTokenUsage,
    Any,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Run a single-round panel synchronously.

    Returns ``(panelist_results, result_dicts, panelist_usage,
    panelist_cost, synthesis_dict, variant_data)``. ``variant_data`` is
    ``None`` when ``variants == 0``.
    """
    all_personas = list(personas)
    variant_names: set[str] = set()
    variant_mapping: dict[str, str] = {}
    variant_count = 0

    if variants > 0:
        logger.info("Generating %d variants per persona", variants)
        variant_sets = generate_panel_variants(personas, client, k=variants, model=model)
        for vs in variant_sets:
            for v in vs.variants:
                all_personas.append(v.persona)
                variant_names.add(v.variant_name)
                variant_mapping[v.variant_name] = v.source_persona_name
                variant_count += 1
        logger.info("Running panel with %d base + %d variant personas", len(personas), variant_count)

    panelist_results, _registry, _sessions = run_panel_parallel(
        client=client,
        personas=all_personas,
        questions=questions,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        response_schema=response_schema,
        temperature=temperature,
        top_p=top_p,
        persona_models=persona_models,
        extract_schema=extract_schema,
    )

    # sp-efip: fail loud when every panelist produced no usable data.
    # Without this, MCP callers see a normally-shaped "panel complete"
    # result even when every request 400'd upstream.
    total_failure = detect_total_failure(panelist_results)
    if total_failure is not None:
        raise PanelTotalFailureError(
            format_total_failure_message(total_failure),
            diagnostic=total_failure,
        )

    panelist_usage = ZERO_USAGE
    result_dicts: list[dict[str, Any]] = []
    for pr in panelist_results:
        result_dicts.append(format_panelist_result(pr, model))
        panelist_usage = panelist_usage + pr.usage

    # sp-kvpx: use resolve_cost so panelist_cost honors sp-j3vk precedence
    # (provider-reported → local fallback) for the MCP/SDK sync path too.
    panelist_cost = resolve_cost(panelist_usage, model)

    base_results = [pr for pr in panelist_results if pr.persona_name not in variant_names]
    synthesis_dict: dict[str, Any] | None = None
    if synthesis:
        try:
            synthesis_result = synthesize_panel(
                client,
                base_results,
                questions,
                model=synthesis_model,
                panelist_model=model,
                custom_prompt=synthesis_prompt,
                panelist_cost=panelist_cost,
                temperature=synthesis_temperature,
            )
            synthesis_dict = synthesis_result.to_dict()
        except Exception:
            logger.error("Synthesis failed (non-fatal)", exc_info=True)
            synthesis_dict = {"synthesis_error": "Synthesis failed — see logs for details."}

    variant_data: dict[str, Any] | None = None
    if variants > 0 and variant_mapping:
        variant_data = compute_variant_data(
            result_dicts,
            variant_names,
            variant_mapping,
            variant_count,
            questions,
        )

    return panelist_results, result_dicts, panelist_usage, panelist_cost, synthesis_dict, variant_data


def run_multi_round_sync(
    client: LLMClient,
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    response_schema: dict[str, Any] | None,
    *,
    synthesis: bool,
    synthesis_model: str | None,
    synthesis_prompt: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
    persona_models: dict[str, str] | None = None,
    extract_schema: dict[str, Any] | None = None,
    synthesis_temperature: float | None = None,
) -> MultiRoundResult:
    """Drive :func:`run_multi_round_panel` for v1/v2/v3 instruments."""

    def _round_synth(
        c: LLMClient,
        panelist_results: list[PanelistResult],
        questions: list[dict[str, Any]],
        *,
        model: str,
    ):
        return synthesize_panel(
            c,
            panelist_results,
            questions,
            model=synthesis_model,
            panelist_model=model,
            custom_prompt=synthesis_prompt,
            temperature=synthesis_temperature,
        )

    final_fn = _round_synth if synthesis else None
    return run_multi_round_panel(
        client=client,
        personas=personas,
        instrument=instrument,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        synthesize_round_fn=_round_synth if synthesis else (lambda *a, **kw: None),
        synthesize_final_fn=final_fn,
        response_schema=response_schema,
        temperature=temperature,
        top_p=top_p,
        persona_models=persona_models,
        extract_schema=extract_schema,
    )
