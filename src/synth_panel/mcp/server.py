"""MCP server implementation for synthpanel.

Exposes 12 tools, 4 resource URI patterns, and 3 prompt templates.
Uses stdio transport. Default model is haiku for MCP mode.

Tools:
    run_prompt             - Send a single prompt to an LLM (no personas)
    run_panel              - Run a full synthetic focus group panel
    run_quick_poll         - Quick single-question poll across personas
    extend_panel           - Append one ad-hoc round to a saved panel result
    list_persona_packs     - List saved persona packs
    get_persona_pack       - Get a specific persona pack
    save_persona_pack      - Save a persona pack
    list_instrument_packs  - List installed instrument packs
    get_instrument_pack    - Get an installed instrument pack
    save_instrument_pack   - Save (install) an instrument pack
    list_panel_results     - List saved panel results
    get_panel_result       - Get a specific panel result

Resources (URI patterns):
    persona-pack://{pack_id}         - A specific persona pack
    persona-pack://                  - List all persona packs
    panel-result://{result_id}       - A specific panel result
    panel-result://                  - List all panel results

Prompts:
    focus_group   - Run a focus group discussion
    name_test     - Test product/feature names with personas
    concept_test  - Test a concept or idea with personas
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from synth_panel.cost import ZERO_USAGE, estimate_cost, lookup_pricing
from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.instrument import Instrument, parse_instrument
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock
from synth_panel.mcp.data import (
    get_panel_result as _data_get_panel_result,
)
from synth_panel.mcp.data import (
    get_persona_pack as _data_get_persona_pack,
)
from synth_panel.mcp.data import (
    list_instrument_packs as _data_list_instrument_packs,
)
from synth_panel.mcp.data import (
    list_panel_results as _data_list_panel_results,
)
from synth_panel.mcp.data import (
    list_persona_packs as _data_list_persona_packs,
)
from synth_panel.mcp.data import (
    load_instrument_pack as _data_load_instrument_pack,
)
from synth_panel.mcp.data import (
    load_panel_sessions,
    save_panel_result,
    update_panel_result,
)
from synth_panel.mcp.data import (
    save_instrument_pack as _data_save_instrument_pack,
)
from synth_panel.mcp.data import (
    save_persona_pack as _data_save_persona_pack,
)
from synth_panel.metadata import PanelTimer, build_metadata
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

logger = logging.getLogger(__name__)

# Default model for MCP mode
MCP_DEFAULT_MODEL = "haiku"

# Per-panelist timeout (seconds)
PANELIST_TIMEOUT = 30

# Input caps to prevent runaway resource usage
MAX_PERSONAS = 100
MAX_QUESTIONS = 50

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


def _resolve_extract_schema(
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


mcp = FastMCP(
    "synthpanel",
    instructions=(
        "Synthetic focus group server. Run panels of AI personas to get "
        "structured qualitative feedback on products, concepts, and names."
    ),
)

# Shared LLM client — reused across tool calls to avoid rebuilding the
# provider cache on every invocation.  Thread-safe by design (see LLMClient).
_shared_client = LLMClient()


# ---------------------------------------------------------------------------
# Internal panel runner (bridges threads to async)
# ---------------------------------------------------------------------------


def _run_panel_sync(
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
    list[PanelistResult], list[dict[str, Any]], CostTokenUsage, Any, dict[str, Any] | None, dict[str, Any] | None
]:
    """Run panel synchronously.

    Returns (results, result_dicts, panelist_usage, panelist_cost, synthesis_dict, variant_data).
    variant_data is None when variants == 0.
    """
    client = _shared_client

    # Generate persona variants if requested
    all_personas = list(personas)
    variant_names: set[str] = set()
    variant_mapping: dict[str, str] = {}  # variant_name -> source_name
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

    panelist_usage = ZERO_USAGE
    result_dicts: list[dict[str, Any]] = []
    for pr in panelist_results:
        pr_model = pr.model or model
        pricing, _ = lookup_pricing(pr_model)
        persona_cost = estimate_cost(pr.usage, pricing)
        rd: dict[str, Any] = {
            "persona": pr.persona_name,
            "responses": pr.responses,
            "usage": pr.usage.to_dict(),
            "cost": persona_cost.format_usd(),
            "error": pr.error,
        }
        if pr.model:
            rd["model"] = pr.model
        result_dicts.append(rd)
        panelist_usage = panelist_usage + pr.usage

    pricing, _ = lookup_pricing(model)
    panelist_cost = estimate_cost(panelist_usage, pricing)

    # Synthesis (base personas only)
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
            synthesis_dict = {"synthesis_error": "Synthesis failed — see server logs for details."}

    # Compute robustness scores when variants were used
    variant_data: dict[str, Any] | None = None
    if variants > 0 and variant_mapping:
        variant_data = _compute_variant_data(
            result_dicts,
            variant_names,
            variant_mapping,
            variant_count,
            questions,
        )

    return panelist_results, result_dicts, panelist_usage, panelist_cost, synthesis_dict, variant_data


def _compute_variant_data(
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
            # Find most common base response as the "finding"
            base_responses_q = [
                br["responses"][qi].get("response", "")
                for br in base_results
                if qi < len(br.get("responses", [])) and not br["responses"][qi].get("error")
            ]
            if base_responses_q:
                from collections import Counter

                most_common = Counter(base_responses_q).most_common(1)[0][0]
                # Merge all variant responses for aggregate score
                all_v = {}
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


def _run_multi_round_sync(
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
    """Drive run_multi_round_panel for v1/v2/v3 instruments."""
    client = _shared_client

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


def _format_panelist_result(pr: PanelistResult, model: str) -> dict[str, Any]:
    pr_model = pr.model or model
    pricing, _ = lookup_pricing(pr_model)
    persona_cost = estimate_cost(pr.usage, pricing)
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


def _run_ensemble_sync(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    models: list[str],
    response_schema: dict[str, Any] | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """Run the same panel with each model and return comparative results."""
    from synth_panel.ensemble import ensemble_run

    client = LLMClient()
    ens = ensemble_run(
        personas,
        questions,
        models,
        client,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        response_schema=response_schema,
        extract_schema=extract_schema,
        temperature=temperature,
        top_p=top_p,
    )
    return {
        "per_model_results": {
            mr.model: [_format_panelist_result(pr, mr.model) for pr in mr.panelist_results] for mr in ens.model_results
        },
        "cost_breakdown": ens.per_model_cost,
        "models": ens.models,
        "total_usage": ens.total_usage.to_dict(),
    }


async def _run_panel_async_instrument(
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    ctx: Context,
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
) -> dict[str, Any]:
    """Run a (possibly branching) instrument and return v3-shaped response."""
    total = len(personas)
    timer = PanelTimer()
    await ctx.report_progress(0, total)

    mr: MultiRoundResult = await asyncio.wait_for(
        asyncio.to_thread(
            _run_multi_round_sync,
            personas,
            instrument,
            model,
            response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
            temperature=temperature,
            top_p=top_p,
            persona_models=persona_models,
            extract_schema=extract_schema,
            synthesis_temperature=synthesis_temperature,
        ),
        # Multi-round can chain N rounds; budget per panelist scales with rounds.
        timeout=PANELIST_TIMEOUT * max(total, 1) * max(len(instrument.rounds), 1),
    )

    await ctx.report_progress(total, total)

    rounds_payload: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []
    total_question_count = 0
    for rr in mr.rounds:
        round_dict_results = [_format_panelist_result(pr, model) for pr in rr.panelist_results]
        questions_for_round = next((r.questions for r in instrument.rounds if r.name == rr.name), [])
        total_question_count += len(questions_for_round)
        rounds_payload.append(
            {
                "name": rr.name,
                "results": round_dict_results,
                "synthesis": rr.synthesis.to_dict() if hasattr(rr.synthesis, "to_dict") else None,
                "usage": rr.usage.to_dict(),
            }
        )
        # Flat results for back-compat / persistence: use the *last* round per persona.
        flat_results = round_dict_results

    pricing, _ = lookup_pricing(model)
    total_cost = estimate_cost(mr.usage, pricing)

    timer.stop()

    final_synth_dict = (
        mr.final_synthesis.to_dict()
        if mr.final_synthesis is not None and hasattr(mr.final_synthesis, "to_dict")
        else None
    )

    # Compute per-model cost breakdown for metadata
    panelist_usage = ZERO_USAGE
    for rr in mr.rounds:
        for pr in rr.panelist_results:
            panelist_usage = panelist_usage + pr.usage
    panelist_cost_est = estimate_cost(panelist_usage, pricing)

    synthesis_usage_for_meta: CostTokenUsage | None = None
    synthesis_cost_for_meta = None
    if mr.final_synthesis is not None and hasattr(mr.final_synthesis, "usage"):
        synthesis_usage_for_meta = mr.final_synthesis.usage
        synth_model = getattr(mr.final_synthesis, "model", model)
        synth_pricing, _ = lookup_pricing(synth_model)
        synthesis_cost_for_meta = estimate_cost(synthesis_usage_for_meta, synth_pricing)

    inst_metadata = build_metadata(
        panelist_model=model,
        synthesis_model=getattr(mr.final_synthesis, "model", None) if mr.final_synthesis else None,
        panelist_usage=panelist_usage,
        panelist_cost=panelist_cost_est,
        synthesis_usage=synthesis_usage_for_meta,
        synthesis_cost=synthesis_cost_for_meta,
        total_usage=mr.usage,
        total_cost=total_cost,
        persona_count=len(personas),
        question_count=total_question_count,
        timer=timer,
    )

    result_id = save_panel_result(
        results=flat_results,
        model=model,
        total_usage=mr.usage.to_dict(),
        total_cost=total_cost.format_usd(),
        persona_count=len(personas),
        question_count=total_question_count,
    )

    return {
        "result_id": result_id,
        "model": model,
        "persona_count": len(personas),
        "question_count": total_question_count,
        "rounds": rounds_payload,
        "path": mr.path,
        "terminal_round": mr.terminal_round,
        "warnings": mr.warnings,
        "synthesis": final_synth_dict,
        "total_cost": total_cost.format_usd(),
        "total_usage": mr.usage.to_dict(),
        # Back-compat: ``results`` mirrors the terminal round's flat panelist
        # list so v1/v2 callers see the same shape they did pre-0.5.0.
        "results": flat_results,
        "metadata": inst_metadata,
    }


async def _run_panel_async(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str,
    ctx: Context,
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
) -> dict[str, Any]:
    """Run panel via asyncio.to_thread with progress notifications."""
    total = len(personas)
    timer = PanelTimer()
    await ctx.report_progress(0, total)

    # Run the blocking panel execution in a thread
    (
        _panelist_results,
        result_dicts,
        panelist_usage,
        panelist_cost,
        synthesis_dict,
        variant_data,
    ) = await asyncio.wait_for(
        asyncio.to_thread(
            _run_panel_sync,
            personas,
            questions,
            model,
            response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
            temperature=temperature,
            top_p=top_p,
            persona_models=persona_models,
            extract_schema=extract_schema,
            synthesis_temperature=synthesis_temperature,
            variants=variants,
        ),
        timeout=PANELIST_TIMEOUT * total * (1 + variants),
    )

    await ctx.report_progress(total, total)

    # Compute total cost (panelist + synthesis)
    synthesis_usage_obj: CostTokenUsage | None = None
    synthesis_cost_obj = None
    if synthesis_dict:
        synthesis_usage_obj = CostTokenUsage.from_dict(synthesis_dict["usage"])
        synthesis_pricing, _ = lookup_pricing(synthesis_dict.get("model"))
        synthesis_cost_obj = estimate_cost(synthesis_usage_obj, synthesis_pricing)
        total_usage = panelist_usage + synthesis_usage_obj
        total_cost = panelist_cost + synthesis_cost_obj
    else:
        total_usage = panelist_usage
        total_cost = panelist_cost

    timer.stop()
    metadata = build_metadata(
        panelist_model=model,
        synthesis_model=synthesis_dict.get("model") if synthesis_dict else None,
        panelist_usage=panelist_usage,
        panelist_cost=panelist_cost,
        synthesis_usage=synthesis_usage_obj,
        synthesis_cost=synthesis_cost_obj,
        total_usage=total_usage,
        total_cost=total_cost,
        persona_count=len(personas),
        question_count=len(questions),
        timer=timer,
    )

    # Save result
    variant_count = variant_data["variant_count"] if variant_data else 0
    result_id = save_panel_result(
        results=result_dicts,
        model=model,
        total_usage=total_usage.to_dict(),
        total_cost=total_cost.format_usd(),
        persona_count=len(personas),
        question_count=len(questions),
        variant_count=variant_count,
    )

    result: dict[str, Any] = {
        "result_id": result_id,
        "model": model,
        "persona_count": len(personas),
        "question_count": len(questions),
        "panelist_cost": panelist_cost.format_usd(),
        "synthesis": synthesis_dict,
        "total_cost": total_cost.format_usd(),
        "total_usage": total_usage.to_dict(),
        "rounds": [
            {
                "name": "default",
                "results": result_dicts,
                "synthesis": None,
            }
        ],
        "path": [],
        "warnings": [],
        "metadata": metadata,
    }

    if variant_data:
        result["robustness_scores"] = variant_data["robustness_scores"]
        result["per_persona_robustness"] = variant_data["per_persona_robustness"]
        result["variant_count"] = variant_data["variant_count"]

    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_prompt(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Send a single prompt to an LLM and get a response. No personas required.

    The simplest tool — ask a quick research question without constructing
    personas or running a full panel.

    Args:
        prompt: The question or prompt to send.
        model: LLM model to use. Defaults to haiku.
        temperature: Sampling temperature (0.0-1.0). Controls randomness.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
    """
    model = model or MCP_DEFAULT_MODEL
    logger.info("run_prompt: model=%s prompt_len=%d", model, len(prompt))
    client = _shared_client
    request = CompletionRequest(
        model=model,
        max_tokens=4096,
        messages=[InputMessage(role="user", content=[TextBlock(text=prompt)])],
        temperature=temperature,
        top_p=top_p,
    )
    response = await asyncio.to_thread(client.send, request)
    usage = CostTokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_creation_input_tokens=response.usage.cache_write_tokens,
        cache_read_input_tokens=response.usage.cache_read_tokens,
    )
    pricing, _ = lookup_pricing(model)
    cost = estimate_cost(usage, pricing)
    return json.dumps(
        {
            "response": response.text,
            "model": response.model,
            "usage": usage.to_dict(),
            "cost": cost.format_usd(),
        },
        indent=2,
    )


@mcp.tool()
async def run_panel(
    questions: list[dict[str, Any]] | None = None,
    personas: list[dict[str, Any]] | None = None,
    pack_id: str | None = None,
    instrument: dict[str, Any] | None = None,
    instrument_pack: str | None = None,
    model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    persona_models: dict[str, str] | None = None,
    extract_schema: str | dict[str, Any] | None = None,
    models: list[str] | None = None,
    synthesis_temperature: float | None = None,
    variants: int | None = None,
    ctx: Context = None,
) -> str:
    """Run a full synthetic focus group panel.

    Each persona answers all questions independently in parallel.
    After responses are collected, a synthesis step aggregates findings
    into themes, agreements, disagreements, and recommendations.
    Results are saved and can be retrieved later.

    Three input modes for the question stream:

    1. Inline ``questions`` list — single round, v1-equivalent.
    2. Inline ``instrument`` dict — a v1/v2/v3 instrument body. v3
       instruments with ``route_when`` clauses run as a branching
       multi-round panel.
    3. ``instrument_pack`` name — load an installed instrument pack
       from ``$SYNTH_PANEL_DATA_DIR/packs/instruments/<name>.yaml``.

    Response shape: every successful run returns ``rounds`` (list of
    per-round panelist results + synthesis), ``path`` (the executed
    routing decisions: ``[{round, branch, next}]``), ``warnings``
    (parser + runtime warnings), and ``terminal_round`` (the round
    whose synthesis fed final synthesis). For v1/v2 instruments and
    raw ``questions`` input, ``path`` has length 1 or N (linear) and
    ``warnings`` is empty in the typical case — the shape is uniform
    across versions so callers don't need to special-case.

    Args:
        questions: Flat list of question dicts (v1-equivalent). Each
            should have a ``text`` key. Ignored when ``instrument`` or
            ``instrument_pack`` is provided.
        personas: Inline persona definitions. Each needs at minimum a
            ``name`` key.
        pack_id: ID of a saved persona pack. Merged with inline
            personas (inline first). At least one of ``personas`` or
            ``pack_id`` must be provided.
        instrument: Raw instrument body (the value under the
            top-level ``instrument:`` key in YAML). Takes precedence
            over ``questions``.
        instrument_pack: Name of an installed instrument pack.
            Takes precedence over both ``instrument`` and ``questions``.
        model: LLM model to use. Defaults to haiku.
        response_schema: Optional JSON Schema for structured output. When
            provided, each panelist's responses are extracted as structured
            JSON matching this schema instead of free text.
        synthesis: Whether to run synthesis after collecting responses.
            Defaults to true.
        synthesis_model: Model to use for synthesis. Defaults to panelist model.
        synthesis_prompt: Custom synthesis prompt. Replaces the default.
        temperature: Sampling temperature (0.0-1.0) for panelist responses.
        top_p: Nucleus sampling threshold (0.0-1.0) for panelist responses.
        persona_models: Per-persona model overrides. Maps persona name to
            model alias (e.g. {"Sarah Chen": "sonnet", "Mike": "haiku"}).
        extract_schema: Schema for post-hoc structured extraction from
            free-text responses. Pass a built-in name ("sentiment",
            "themes", "rating") or an inline JSON Schema dict.
        models: List of model names for multi-model ensemble. When
            provided, the panel is run once per model and results are
            compared. Returns per_model_results and cost_breakdown.
            Mutually exclusive with ``model``.
        synthesis_temperature: Sampling temperature for the synthesis step.
            Independent of the panelist temperature.
        variants: Number of persona variants to generate per persona for
            robustness analysis. When > 0, each persona is perturbed K times
            and all variants run through the same questions. Results include
            robustness_scores and per_persona_robustness. Default: no variants.
    """
    model = model or MCP_DEFAULT_MODEL
    variants_k = variants or 0
    if variants_k < 0 or variants_k > 20:
        return json.dumps({"error": "variants must be between 0 and 20."})
    logger.info("run_panel: model=%s synthesis=%s variants=%d", model, synthesis, variants_k)

    # Resolve extract_schema name → dict before threading to orchestrator.
    try:
        resolved_extract_schema = _resolve_extract_schema(extract_schema)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": str(exc)})
    merged = list(personas) if personas else []
    if pack_id is not None:
        pack = _data_get_persona_pack(pack_id)
        merged.extend(pack.get("personas", []))
    if not merged:
        return json.dumps({"error": "No personas provided. Supply personas and/or pack_id."})

    # Validate personas: must be dicts with "name"
    for i, p in enumerate(merged):
        if not isinstance(p, dict):
            return json.dumps({"error": f"Persona at index {i} must be a dict, got {type(p).__name__}."})
        if "name" not in p or not str(p["name"]).strip():
            return json.dumps({"error": f"Persona at index {i} is missing required field 'name'."})

    if len(merged) > MAX_PERSONAS:
        return json.dumps({"error": f"Too many personas ({len(merged)}). Maximum is {MAX_PERSONAS}."})

    # Validate questions if provided directly (not via instrument)
    if questions is not None and instrument is None and instrument_pack is None:
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return json.dumps({"error": f"Question at index {i} must be a dict, got {type(q).__name__}."})
            if "text" not in q or not str(q["text"]).strip():
                return json.dumps({"error": f"Question at index {i} is missing required field 'text'."})
        if len(questions) > MAX_QUESTIONS:
            return json.dumps({"error": f"Too many questions ({len(questions)}). Maximum is {MAX_QUESTIONS}."})

    # ── Ensemble mode: run with each model, compare across models ────────
    if models and len(models) >= 2:
        if not questions and instrument is None and instrument_pack is None:
            return json.dumps({"error": "Ensemble mode requires questions or instrument."})
        ens_questions = questions or []
        if not ens_questions:
            # Instruments: extract flat questions for ensemble (v1/v2 only)
            if instrument is not None:
                raw = instrument.get("instrument", instrument)
                inst = parse_instrument(raw)
                ens_questions = [{"text": q["text"]} for q in inst.questions]
            elif instrument_pack is not None:
                pack_body = _data_load_instrument_pack(instrument_pack)
                raw = pack_body.get("instrument", pack_body)
                inst = parse_instrument(raw)
                ens_questions = [{"text": q["text"]} for q in inst.questions]
        ens_result = await asyncio.to_thread(
            _run_ensemble_sync,
            merged,
            ens_questions,
            models,
            response_schema,
            extract_schema,
            temperature,
            top_p,
        )
        return json.dumps(ens_result, indent=2)

    # Resolve instrument source (pack > inline instrument > questions).
    instrument_obj: Instrument | None = None
    if instrument_pack is not None:
        pack_body = _data_load_instrument_pack(instrument_pack)
        raw = pack_body.get("instrument", pack_body)
        instrument_obj = parse_instrument(raw)
    elif instrument is not None:
        raw = instrument.get("instrument", instrument)
        instrument_obj = parse_instrument(raw)

    if instrument_obj is not None:
        result = await _run_panel_async_instrument(
            merged,
            instrument_obj,
            model,
            ctx,
            response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
            temperature=temperature,
            top_p=top_p,
            persona_models=persona_models,
            extract_schema=resolved_extract_schema,
            synthesis_temperature=synthesis_temperature,
        )
        return json.dumps(result, indent=2)

    if not questions:
        return json.dumps({"error": "No questions or instrument provided."})

    result = await _run_panel_async(
        merged,
        questions,
        model,
        ctx,
        response_schema,
        synthesis=synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
        temperature=temperature,
        top_p=top_p,
        persona_models=persona_models,
        extract_schema=resolved_extract_schema,
        synthesis_temperature=synthesis_temperature,
        variants=variants_k,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_quick_poll(
    question: str,
    personas: list[dict[str, Any]],
    model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    ctx: Context = None,
) -> str:
    """Quick single-question poll across personas.

    A simplified version of run_panel for quick feedback on one question.
    Includes synthesis by default.

    Args:
        question: The question to ask all personas.
        personas: List of persona definitions.
        model: LLM model to use. Defaults to haiku.
        response_schema: Optional JSON Schema for structured output. When
            provided, responses are extracted as structured JSON matching
            this schema instead of free text.
        synthesis: Whether to run synthesis after collecting responses.
            Defaults to true.
        synthesis_model: Model to use for synthesis. Defaults to panelist model.
        synthesis_prompt: Custom synthesis prompt. Replaces the default.
        temperature: Sampling temperature (0.0-1.0). Controls randomness.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
    """
    model = model or MCP_DEFAULT_MODEL
    logger.info("run_quick_poll: model=%s personas=%d", model, len(personas))

    if not question or not question.strip():
        return json.dumps({"error": "Question text must be a non-empty string."})

    # Validate personas: must be dicts with "name"
    for i, p in enumerate(personas):
        if not isinstance(p, dict):
            return json.dumps({"error": f"Persona at index {i} must be a dict, got {type(p).__name__}."})
        if "name" not in p or not str(p["name"]).strip():
            return json.dumps({"error": f"Persona at index {i} is missing required field 'name'."})

    if len(personas) > MAX_PERSONAS:
        return json.dumps({"error": f"Too many personas ({len(personas)}). Maximum is {MAX_PERSONAS}."})

    questions = [{"text": question}]
    result = await _run_panel_async(
        personas,
        questions,
        model,
        ctx,
        response_schema,
        synthesis=synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_persona_packs() -> str:
    """List all saved persona packs.

    Returns metadata for each pack including ID, name, and persona count.
    """
    packs = _data_list_persona_packs()
    return json.dumps(packs, indent=2)


@mcp.tool()
async def get_persona_pack(pack_id: str) -> str:
    """Get a specific persona pack by ID.

    Args:
        pack_id: The ID of the persona pack to retrieve.
    """
    pack = _data_get_persona_pack(pack_id)
    return json.dumps(pack, indent=2)


@mcp.tool()
async def save_persona_pack(
    name: str,
    personas: list[dict[str, Any]],
    pack_id: str | None = None,
) -> str:
    """Save a persona pack for reuse.

    Args:
        name: Human-readable name for the pack.
        personas: List of persona definitions.
        pack_id: Optional ID. Auto-generated if not provided.
    """
    result = _data_save_persona_pack(name, personas, pack_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_instrument_packs() -> str:
    """List installed instrument packs.

    Instrument packs live as single ``<name>.yaml`` files under
    ``$SYNTH_PANEL_DATA_DIR/packs/instruments/`` and carry the four
    shared manifest fields (name, version, description, author) at
    the top level alongside the instrument body.
    """
    return json.dumps(_data_list_instrument_packs(), indent=2)


@mcp.tool()
async def get_instrument_pack(name: str) -> str:
    """Load an installed instrument pack by name.

    Args:
        name: The pack name (filename stem under packs/instruments/).
    """
    return json.dumps(_data_load_instrument_pack(name), indent=2)


@mcp.tool()
async def save_instrument_pack(
    name: str,
    content: dict[str, Any],
) -> str:
    """Install an instrument pack to the local instrument-pack directory.

    The instrument body is validated via the parser before being
    written: a malformed v1/v2/v3 instrument fails fast and is never
    written to disk. Manifest fields (name, version, description,
    author) are expected at the top level of ``content`` alongside
    either ``instrument:`` or the instrument keys directly.

    Args:
        name: Pack name. Becomes ``<name>.yaml`` on disk.
        content: Full pack body — manifest fields plus the
            instrument definition.
    """
    raw = content.get("instrument", content)
    parse_instrument(raw)  # validate before write
    return json.dumps(_data_save_instrument_pack(name, content), indent=2)


@mcp.tool()
async def extend_panel(
    result_id: str,
    questions: list[dict[str, Any]],
    model: str | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    ctx: Context = None,
) -> str:
    """Append a single ad-hoc round to a saved panel result.

    ``extend_panel`` always appends ONE improvised round on top of an
    existing result, reusing each panelist's saved session so the
    follow-up sees full conversational context. It is **not** a way
    to re-enter the authored v3 DAG: the original instrument's
    ``route_when`` clauses are not consulted, no routing decision is
    made, and the result's ``path`` is extended by exactly one entry
    tagged as an extension. If you want branching, run a fresh
    ``run_panel`` with a v3 instrument instead.

    The pre-extend snapshot is preserved alongside the result file
    (``<result_id>.pre-extend.json``) so the operation is reversible.

    Args:
        result_id: ID of a previously saved panel result.
        questions: One or more questions for the ad-hoc round. They
            run as a single round, in order, against the same
            personas as the original run.
        model: LLM model to use for the new round. Defaults to haiku.
        synthesis: Whether to synthesize the new round.
        synthesis_model: Synthesis model. Defaults to panelist model.
        synthesis_prompt: Custom synthesis prompt for the new round.
    """
    model = model or MCP_DEFAULT_MODEL
    logger.info("extend_panel: result_id=%s questions=%d model=%s", result_id, len(questions), model)
    existing = _data_get_panel_result(result_id)

    # Reuse the original personas (recovered from saved sessions if possible).
    sessions = load_panel_sessions(result_id)
    personas: list[dict[str, Any]] = [{"name": name} for name in sessions]
    if not personas:
        return json.dumps({"error": f"No sessions found for result {result_id}"})

    if ctx is not None:
        await ctx.report_progress(0, len(personas))

    def _go() -> tuple[list[PanelistResult], Any]:
        client = _shared_client
        results, _registry, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model=model,
            system_prompt_fn=persona_system_prompt,
            question_prompt_fn=build_question_prompt,
            sessions=sessions,
        )
        synth = None
        if synthesis:
            try:
                synth = synthesize_panel(
                    client,
                    results,
                    questions,
                    model=synthesis_model,
                    panelist_model=model,
                    custom_prompt=synthesis_prompt,
                )
            except Exception:
                logger.error("extend_panel synthesis failed (non-fatal)", exc_info=True)
                synth = None
        return results, synth

    panelist_results, synth = await asyncio.wait_for(
        asyncio.to_thread(_go),
        timeout=PANELIST_TIMEOUT * len(personas),
    )

    if ctx is not None:
        await ctx.report_progress(len(personas), len(personas))

    new_round_results = [_format_panelist_result(pr, model) for pr in panelist_results]

    # Append the ad-hoc round to the existing result and persist.
    rounds = existing.get("rounds") or []
    rounds = [
        *list(rounds),
        {
            "name": f"extension-{len(rounds) + 1}",
            "results": new_round_results,
            "synthesis": synth.to_dict() if synth is not None and hasattr(synth, "to_dict") else None,
            "extension": True,
        },
    ]
    path = list(existing.get("path") or [])
    path.append(
        {
            "round": rounds[-1]["name"],
            "branch": "extension (ad-hoc, not DAG re-entry)",
            "next": "__end__",
        }
    )

    updated = dict(existing)
    updated["rounds"] = rounds
    updated["path"] = path
    updated["results"] = new_round_results  # mirrors latest round (back-compat)
    updated["question_count"] = int(existing.get("question_count", 0)) + len(questions)
    update_panel_result(result_id, updated)

    return json.dumps(
        {
            "result_id": result_id,
            "appended_round": rounds[-1]["name"],
            "results": new_round_results,
            "synthesis": rounds[-1]["synthesis"],
            "path": path,
        },
        indent=2,
    )


@mcp.tool()
async def list_panel_results() -> str:
    """List all saved panel results.

    Returns metadata for each result including ID, date, model, and counts.
    """
    results = _data_list_panel_results()
    return json.dumps(results, indent=2)


@mcp.tool()
async def get_panel_result(result_id: str) -> str:
    """Get a specific panel result by ID.

    Args:
        result_id: The ID of the panel result to retrieve.
    """
    result = _data_get_panel_result(result_id)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("persona-pack://{pack_id}")
async def resource_persona_pack(pack_id: str) -> str:
    """A specific persona pack."""
    pack = _data_get_persona_pack(pack_id)
    return json.dumps(pack, indent=2)


@mcp.resource("persona-pack://")
async def resource_persona_packs_list() -> str:
    """List all persona packs."""
    return json.dumps(_data_list_persona_packs(), indent=2)


@mcp.resource("panel-result://{result_id}")
async def resource_panel_result(result_id: str) -> str:
    """A specific panel result."""
    result = _data_get_panel_result(result_id)
    return json.dumps(result, indent=2)


@mcp.resource("panel-result://")
async def resource_panel_results_list() -> str:
    """List all panel results."""
    return json.dumps(_data_list_panel_results(), indent=2)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


@mcp.prompt()
def focus_group(
    topic: str,
    num_personas: int = 5,
    follow_up: bool = True,
) -> str:
    """Generate a focus group discussion prompt.

    Creates a structured prompt for running a focus group on a given topic.

    Args:
        topic: The topic or product to discuss.
        num_personas: Number of diverse personas to include.
        follow_up: Whether to include follow-up questions.
    """
    follow_up_section = ""
    if follow_up:
        follow_up_section = "\n\nAfter each response, ask one follow-up question to dig deeper into their perspective."

    return (
        f"Run a synthetic focus group with {num_personas} diverse personas "
        f"discussing: {topic}\n\n"
        f"For each persona, ask:\n"
        f"1. What is your initial reaction to this topic?\n"
        f"2. How does this relate to your daily experience?\n"
        f"3. What concerns or opportunities do you see?"
        f"{follow_up_section}\n\n"
        f"Use the run_panel tool with appropriate personas and questions."
    )


@mcp.prompt()
def name_test(
    names: str,
    context: str = "",
) -> str:
    """Test product or feature names with personas.

    Creates a prompt for evaluating name options with diverse perspectives.

    Args:
        names: Comma-separated list of name options to test.
        context: Optional context about what the name is for.
    """
    context_line = f" for {context}" if context else ""
    return (
        f"Test these name options{context_line}: {names}\n\n"
        f"For each persona, ask:\n"
        f"1. What does each name make you think of?\n"
        f"2. Which name do you prefer and why?\n"
        f"3. Does any name confuse you or feel wrong?\n\n"
        f"Use the run_panel tool with diverse personas to get varied feedback."
    )


@mcp.prompt()
def concept_test(
    concept: str,
    target_audience: str = "",
) -> str:
    """Test a concept or idea with personas.

    Creates a prompt for evaluating a concept with targeted personas.

    Args:
        concept: Description of the concept to test.
        target_audience: Optional description of the target audience.
    """
    audience_line = ""
    if target_audience:
        audience_line = f"\n\nTarget the personas toward: {target_audience}"

    return (
        f"Test this concept with synthetic personas:\n\n{concept}"
        f"{audience_line}\n\n"
        f"For each persona, ask:\n"
        f"1. Does this concept solve a problem you have?\n"
        f"2. What would make you try this?\n"
        f"3. What concerns would prevent you from using this?\n"
        f"4. How much would you expect to pay for this?\n\n"
        f"Use the run_panel tool with personas matching the target audience."
    )


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def serve() -> None:
    """Run the MCP server on stdio transport."""
    logger.info("MCP server starting (stdio transport)")
    mcp.run(transport="stdio")
