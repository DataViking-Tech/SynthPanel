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

from synth_panel import __version__ as _synthpanel_version
from synth_panel._runners import (
    EXTRACT_SCHEMA_REGISTRY,
    MAX_PERSONAS,
    MAX_QUESTIONS,
    PANELIST_TIMEOUT,
)
from synth_panel._runners import (
    compute_variant_data as _compute_variant_data,  # re-exported for back-compat
)
from synth_panel._runners import (
    format_panelist_result as _format_panelist_result,
)
from synth_panel._runners import (
    resolve_extract_schema as _resolve_extract_schema,
)
from synth_panel._runners import (
    run_multi_round_sync as _run_multi_round_sync,
)
from synth_panel._runners import (
    run_panel_sync as _run_panel_sync,
)
from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    aggregate_per_model,
    build_cost_fallback_warnings,
    estimate_cost,
    lookup_pricing,
    resolve_cost,
)
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
from synth_panel.mcp.sampling import (
    SAMPLING_MAX_PERSONAS,
    SAMPLING_MAX_QUESTIONS,
    SAMPLING_MAX_TOKENS_DEFAULT,
)
from synth_panel.mcp.sampling import (
    decide_mode as _decide_sampling_mode,
)
from synth_panel.mcp.sampling import (
    sample_text as _sample_text,
)
from synth_panel.metadata import PanelTimer, build_metadata
from synth_panel.orchestrator import (
    MultiRoundResult,
    PanelistResult,
    run_panel_parallel,
)
from synth_panel.prompts import build_question_prompt, persona_system_prompt
from synth_panel.synthesis import synthesize_panel

logger = logging.getLogger(__name__)

# Default model for MCP mode — used as the terminal fallback when no
# provider credentials are present in the environment. Prefer
# :func:`_resolve_mcp_default_model` at call sites so users with a
# non-Anthropic key (OpenRouter, Gemini, xAI, OpenAI) aren't silently
# routed into the Anthropic provider and a misleading missing-key error.
MCP_DEFAULT_MODEL = "haiku"

# Preference chain for the MCP default model. Mirrors the CLI's
# _DEFAULT_MODEL_PREFERENCE and sdk._DEFAULT_MODEL_PREFERENCE, but picks
# the cheap/fast model per provider since MCP is optimised for
# iterative use (whereas the CLI defaults to workhorse models).
_MCP_DEFAULT_MODEL_PREFERENCE: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "haiku"),
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("GEMINI_API_KEY", "gemini-2.5-flash"),
    ("GOOGLE_API_KEY", "gemini-2.5-flash"),
    ("XAI_API_KEY", "grok-3"),
    ("OPENROUTER_API_KEY", "openrouter/auto"),
]


def _resolve_mcp_default_model() -> str:
    """Pick a cheap/fast default model based on available provider creds.

    Walks :data:`_MCP_DEFAULT_MODEL_PREFERENCE` and returns the first
    alias whose credential is available via env OR the on-disk store
    written by ``synthpanel login``. Falls back to
    :data:`MCP_DEFAULT_MODEL` when nothing is set so the LLM client's
    missing-credentials error is the one the user sees.
    """
    from synth_panel.credentials import has_credential

    for env_var, alias in _MCP_DEFAULT_MODEL_PREFERENCE:
        if has_credential(env_var):
            return alias
    return MCP_DEFAULT_MODEL


# Re-export for backward compatibility — callers patch these names.
__all__ = [
    "EXTRACT_SCHEMA_REGISTRY",
    "MAX_PERSONAS",
    "MAX_QUESTIONS",
    "MCP_DEFAULT_MODEL",
    "PANELIST_TIMEOUT",
    "SAMPLING_MAX_PERSONAS",
    "SAMPLING_MAX_QUESTIONS",
    "_compute_variant_data",
    "mcp",
    "serve",
]


mcp = FastMCP(
    "synthpanel",
    instructions=(
        "Synthetic focus group server. Run panels of AI personas to get "
        "structured qualitative feedback on products, concepts, and names."
    ),
)
# FastMCP forwards to an internal low-level Server whose ``version`` falls
# back to ``importlib.metadata.version("mcp")`` when left unset — that
# leaks the MCP SDK version into serverInfo. Pin the synthpanel package
# version so clients see the correct release string on initialize.
mcp._mcp_server.version = _synthpanel_version


# Minimal default persona set for ``run_quick_poll`` — three diverse
# voices so the first-run story works without hand-crafting personas.
# Kept intentionally small: sampling mode caps at SAMPLING_MAX_PERSONAS
# and we want the BYOK path to stay cheap by default too.
DEFAULT_QUICK_POLL_PERSONAS: list[dict[str, Any]] = [
    {
        "name": "Alex Rivera",
        "age": 29,
        "occupation": "Software Engineer",
        "background": "Early-career developer at a mid-sized SaaS company.",
        "personality_traits": ["analytical", "curious", "pragmatic"],
    },
    {
        "name": "Jordan Park",
        "age": 42,
        "occupation": "Small Business Owner",
        "background": "Runs an independent retail shop; values clarity and ROI.",
        "personality_traits": ["practical", "skeptical", "value-driven"],
    },
    {
        "name": "Sam Okafor",
        "age": 35,
        "occupation": "Marketing Manager",
        "background": "Leads growth at a consumer brand; follows trends closely.",
        "personality_traits": ["creative", "social", "brand-aware"],
    },
]

# Shared LLM client — reused across tool calls to avoid rebuilding the
# provider cache on every invocation.  Thread-safe by design (see LLMClient).
# Lazy-initialised so that module import in test/CI contexts doesn't trigger
# provider resolution before patches or env vars are set up.
_shared_client: LLMClient | None = None


def _get_shared_client() -> LLMClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = LLMClient()
    return _shared_client


# ---------------------------------------------------------------------------
# Internal panel runner (bridges threads to async)
# ---------------------------------------------------------------------------


def _server_run_panel_sync(
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
    """Thin shim around :func:`synth_panel._runners.run_panel_sync` using the shared client."""
    return _run_panel_sync(
        _get_shared_client(),
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
    )


def _server_run_multi_round_sync(
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
    """Thin shim around :func:`synth_panel._runners.run_multi_round_sync` using the shared client."""
    return _run_multi_round_sync(
        _get_shared_client(),
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
    )


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
    from synth_panel.ensemble import build_ensemble_output, ensemble_run

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
    return build_ensemble_output(ens, panelist_formatter=_format_panelist_result)


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
            _server_run_multi_round_sync,
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

    timer.stop()

    final_synth_dict = (
        mr.final_synthesis.to_dict()
        if mr.final_synthesis is not None and hasattr(mr.final_synthesis, "to_dict")
        else None
    )

    # sp-atvc: aggregate panelist usage/cost per actual model across all
    # rounds so multi-model instrument runs get accurate per-model cost
    # instead of pricing every bucket at the default model's rate.
    all_panelist_results: list[Any] = [pr for rr in mr.rounds for pr in rr.panelist_results]
    per_model_usage, per_model_cost = aggregate_per_model(all_panelist_results, model)
    multi_model_run = len(per_model_usage) > 1

    panelist_usage = ZERO_USAGE
    for rr in mr.rounds:
        for pr in rr.panelist_results:
            panelist_usage = panelist_usage + pr.usage

    if multi_model_run:
        panelist_cost_est = CostEstimate()
        for _c in per_model_cost.values():
            panelist_cost_est = panelist_cost_est + _c
    else:
        panelist_cost_est = estimate_cost(panelist_usage, pricing)

    synthesis_usage_for_meta: CostTokenUsage | None = None
    synthesis_cost_for_meta = None
    if mr.final_synthesis is not None and hasattr(mr.final_synthesis, "usage"):
        synthesis_usage_for_meta = mr.final_synthesis.usage
        synth_model = getattr(mr.final_synthesis, "model", model)
        synth_pricing, _ = lookup_pricing(synth_model)
        synthesis_cost_for_meta = estimate_cost(synthesis_usage_for_meta, synth_pricing)

    # Derive the run's reported total_cost from the accurate components
    # instead of the single-model rollup that mr.usage assumes.
    total_cost = panelist_cost_est + (synthesis_cost_for_meta or CostEstimate())

    panelist_per_model_meta = (
        {_m: (per_model_usage[_m], per_model_cost[_m]) for _m in per_model_usage} if multi_model_run else None
    )
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
        panelist_per_model=panelist_per_model_meta,
    )

    result_id = save_panel_result(
        results=flat_results,
        model=model,
        total_usage=mr.usage.to_dict(),
        total_cost=total_cost.format_usd(),
        persona_count=len(personas),
        question_count=total_question_count,
    )

    # sp-0h9x: mirror the ensemble path's per-model rollup onto every
    # non-ensemble panel result. The terminal round's panelist list is the
    # canonical "one row per persona" view, so grouping those by model
    # matches the flat ``results`` field that back-compat consumers read.
    from synth_panel.ensemble import build_mixed_model_rollup

    terminal_prs = mr.rounds[-1].panelist_results if mr.rounds else []
    per_model_results, cost_breakdown = build_mixed_model_rollup(
        terminal_prs,
        default_model=model,
        panelist_formatter=lambda pr, m: _format_panelist_result(pr, m),
    )

    # sp-nn8k: warn loudly when any contributing model was priced via
    # DEFAULT_PRICING fallback instead of an explicit tier. Candidates are
    # every model we actually priced above plus the synthesis model.
    synth_model_name = getattr(mr.final_synthesis, "model", None) if mr.final_synthesis else None
    cost_warnings = build_cost_fallback_warnings([*per_model_usage.keys(), synth_model_name])
    merged_warnings = list(mr.warnings) + cost_warnings

    return {
        "result_id": result_id,
        "model": model,
        "persona_count": len(personas),
        "question_count": total_question_count,
        "rounds": rounds_payload,
        "path": mr.path,
        "terminal_round": mr.terminal_round,
        "warnings": merged_warnings,
        "cost_is_estimated": bool(cost_warnings),
        "synthesis": final_synth_dict,
        "total_cost": total_cost.format_usd(),
        "total_usage": mr.usage.to_dict(),
        # Back-compat: ``results`` mirrors the terminal round's flat panelist
        # list so v1/v2 callers see the same shape they did pre-0.5.0.
        "results": flat_results,
        "per_model_results": per_model_results,
        "cost_breakdown": cost_breakdown,
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
        panelist_results_full,
        result_dicts,
        panelist_usage,
        panelist_cost,
        synthesis_dict,
        variant_data,
    ) = await asyncio.wait_for(
        asyncio.to_thread(
            _server_run_panel_sync,
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

    # sp-atvc: re-price panelist usage per actual model when panelists
    # were dispatched across multiple providers (persona_models routing).
    # Without this, total_cost prices every token at the default model's
    # rate and metadata.cost.per_model hides the cheaper/dearer providers.
    per_model_usage, per_model_cost = aggregate_per_model(panelist_results_full, model)
    multi_model_run = len(per_model_usage) > 1
    if multi_model_run:
        panelist_cost = CostEstimate()
        for _c in per_model_cost.values():
            panelist_cost = panelist_cost + _c

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
    panelist_per_model_meta = (
        {_m: (per_model_usage[_m], per_model_cost[_m]) for _m in per_model_usage} if multi_model_run else None
    )
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
        panelist_per_model=panelist_per_model_meta,
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

    # sp-0h9x: emit per_model_results + cost_breakdown so downstream
    # consumers see the same rollup shape as the ensemble path, even on
    # single-model and mixed-model (persona_models) panels.
    from synth_panel.ensemble import build_mixed_model_rollup

    per_model_results, cost_breakdown = build_mixed_model_rollup(
        panelist_results_full,
        default_model=model,
        panelist_formatter=lambda pr, m: _format_panelist_result(pr, m),
    )

    # sp-nn8k: surface DEFAULT_PRICING fallback loudly so estimated totals
    # don't blend into billed ones silently.
    synth_model_name = synthesis_dict.get("model") if synthesis_dict else None
    cost_warnings = build_cost_fallback_warnings([*per_model_usage.keys(), synth_model_name])

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
        "warnings": list(cost_warnings),
        "cost_is_estimated": bool(cost_warnings),
        "per_model_results": per_model_results,
        "cost_breakdown": cost_breakdown,
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
    use_sampling: bool | None = None,
    ctx: Context = None,
) -> str:
    """Send a single prompt to an LLM and get a response. No personas required.

    The simplest tool — ask a quick research question without constructing
    personas or running a full panel.

    Two execution modes:

    * **BYOK** (bring-your-own-key): calls a provider directly using env
      credentials (``ANTHROPIC_API_KEY``, etc.). Supports the full model
      list and per-call cost accounting.
    * **Sampling**: when no creds are set and the invoking MCP client
      (Claude Desktop, Claude Code, Cursor, Windsurf) advertises the
      ``sampling`` capability, synthpanel asks the client to run the
      completion itself. Model is whatever the host agent is using, and
      token cost is charged to the host agent's subscription rather
      than reported here.

    Args:
        prompt: The question or prompt to send.
        model: LLM model to use. Defaults to haiku. Ignored in sampling
            mode (the host agent picks its own model).
        temperature: Sampling temperature (0.0-1.0). Controls randomness.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to
            temperature. Ignored in sampling mode.
        use_sampling: Explicit mode override. ``True`` forces sampling
            (error if unsupported), ``False`` forces BYOK. ``None``
            auto-picks based on creds + client capability.
    """
    model = model or _resolve_mcp_default_model()
    decision = _decide_sampling_mode(ctx, use_sampling=use_sampling)
    logger.info("run_prompt: mode=%s model=%s prompt_len=%d", decision.mode, model, len(prompt))

    if decision.mode == "error":
        return json.dumps({"error": decision.error})

    if decision.mode == "sampling":
        sample = await _sample_text(
            ctx,
            prompt=prompt,
            max_tokens=4096,
            temperature=temperature,
        )
        return json.dumps(
            {
                "response": sample["text"],
                "model": sample["model"],
                "mode": "sampling",
                "usage": None,
                "cost": None,
                "hint": decision.hint,
            },
            indent=2,
        )

    client = _get_shared_client()
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
        provider_reported_cost=response.usage.provider_reported_cost,
        reasoning_tokens=response.usage.reasoning_tokens,
        cached_tokens=response.usage.cached_tokens,
    )
    cost = resolve_cost(usage, model)
    return json.dumps(
        {
            "response": response.text,
            "model": response.model,
            "mode": "byok",
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
    use_sampling: bool | None = None,
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

    Every non-ensemble run also includes ``per_model_results``
    (``{model: {results, cost, usage}}``) and ``cost_breakdown``
    (``{by_model, total}``) — the same shape the ``models=[...]``
    ensemble path returns. Single-model panels produce a one-entry
    dict; ``persona_models`` runs produce one entry per distinct
    model. Consumers can read the rollup unconditionally instead of
    iterating ``rounds[].results[]`` themselves.

    Args:
        questions: Flat list of question dicts (v1-equivalent). Each
            should have a ``text`` key. Ignored when ``instrument`` or
            ``instrument_pack`` is provided.
        personas: Inline persona definitions. Each persona is a JSON
            object with the following recognized fields (additional
            fields are preserved and remain available to custom
            prompt templates):

            * ``name`` (str, **required**) — persona's display name.
            * ``age`` (int, optional) — persona's age.
            * ``occupation`` (str, optional) — job title or role.
            * ``background`` (str, optional) — paragraph-sized bio
              giving context (company, tenure, constraints, etc.).
            * ``personality_traits`` (list[str], optional) — short
              trait adjectives, e.g. ``["analytical", "skeptical"]``.

            Example::

                [
                  {
                    "name": "Sarah Chen",
                    "age": 34,
                    "occupation": "Product Manager",
                    "background": "8 years in tech at a mid-size SaaS company; manages a team of 5.",
                    "personality_traits": ["analytical", "pragmatic", "detail-oriented"]
                  },
                  {
                    "name": "Marcus Johnson",
                    "age": 52,
                    "occupation": "Small Business Owner",
                    "background": "Runs a family restaurant chain; values simplicity over features.",
                    "personality_traits": ["practical", "skeptical of technology"]
                  }
                ]
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
            provided (length ≥ 2), the panel is run once per model and
            results are compared. Mutually exclusive with ``model``. The
            ensemble response replaces the single-model shape with:

            * ``per_model_results`` — ``{model: {results, cost, usage}}``
              where ``results`` is the formatted panelist list for that
              model, ``cost`` is a formatted USD string, and ``usage``
              is the token bucket dict for the model's run.
            * ``cost_breakdown`` — ``{by_model: {model: "$X"}, total: "$Z"}``.
            * ``models`` — the input model list.
            * ``total_usage`` — summed token buckets across all models.
        synthesis_temperature: Sampling temperature for the synthesis step.
            Independent of the panelist temperature.
        variants: Number of persona variants to generate per persona for
            robustness analysis. When > 0, each persona is perturbed K times
            and all variants run through the same questions. Results include
            robustness_scores and per_persona_robustness. Default: no variants.
        use_sampling: Explicit mode override. ``True`` forces sampling
            (error if unsupported or if limits exceeded), ``False`` forces
            BYOK. ``None`` auto-picks based on creds + client capability.
            Sampling mode is capped at :data:`SAMPLING_MAX_PERSONAS`
            personas by :data:`SAMPLING_MAX_QUESTIONS` questions; larger
            panels require BYOK. Ensemble mode (``models``) and v3
            branching are BYOK-only.
    """
    model = model or _resolve_mcp_default_model()
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

    # ── Sampling fallback: route through MCP sampling when no BYOK creds ─
    # Ensemble mode is BYOK-only (sampling host exposes only one model),
    # so we only consult the decision in the non-ensemble branch.
    if not (models and len(models) >= 2):
        decision = _decide_sampling_mode(ctx, use_sampling=use_sampling)
        if decision.mode == "error":
            return json.dumps({"error": decision.error})
        if decision.mode == "sampling":
            # Resolve question stream for sampling — no v3 branching.
            sampling_questions: list[dict[str, Any]]
            if instrument_pack is not None:
                pack_body = _data_load_instrument_pack(instrument_pack)
                raw = pack_body.get("instrument", pack_body)
                inst = parse_instrument(raw)
                if len(inst.rounds) > 1:
                    return json.dumps(
                        {
                            "error": (
                                "Sampling mode does not support v3 branching "
                                "instruments (multiple rounds). Set a provider "
                                "API key (e.g. ANTHROPIC_API_KEY) to run this "
                                "pack under BYOK."
                            )
                        }
                    )
                sampling_questions = [{"text": q["text"]} for q in inst.questions]
            elif instrument is not None:
                raw = instrument.get("instrument", instrument)
                inst = parse_instrument(raw)
                if len(inst.rounds) > 1:
                    return json.dumps(
                        {
                            "error": (
                                "Sampling mode does not support v3 branching "
                                "instruments (multiple rounds). Set a provider "
                                "API key (e.g. ANTHROPIC_API_KEY) to run this "
                                "instrument under BYOK."
                            )
                        }
                    )
                sampling_questions = [{"text": q["text"]} for q in inst.questions]
            elif questions:
                sampling_questions = questions
            else:
                return json.dumps({"error": "No questions or instrument provided."})

            if len(merged) > SAMPLING_MAX_PERSONAS:
                return json.dumps(
                    {
                        "error": (
                            f"Sampling mode is capped at {SAMPLING_MAX_PERSONAS} personas "
                            f"to protect the host agent's context window (got {len(merged)}). "
                            f"Set ANTHROPIC_API_KEY (or another provider key) in your "
                            f"environment to run larger panels via BYOK."
                        )
                    }
                )
            if len(sampling_questions) > SAMPLING_MAX_QUESTIONS:
                return json.dumps(
                    {
                        "error": (
                            f"Sampling mode is capped at {SAMPLING_MAX_QUESTIONS} questions "
                            f"(got {len(sampling_questions)}). Set a provider API key to "
                            f"run larger panels via BYOK."
                        )
                    }
                )
            if variants_k > 0:
                return json.dumps(
                    {
                        "error": (
                            "Sampling mode does not support persona variants. "
                            "Set a provider API key to use robustness analysis."
                        )
                    }
                )

            sampling_result = await _run_panel_sampling(
                ctx,
                personas=merged,
                questions=sampling_questions,
                synthesis=synthesis,
                synthesis_prompt=synthesis_prompt,
                temperature=temperature,
                hint=decision.hint,
            )
            return json.dumps(sampling_result, indent=2)

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
    personas: list[dict[str, Any]] | None = None,
    model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    synthesis_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    use_sampling: bool | None = None,
    ctx: Context = None,
) -> str:
    """Quick single-question poll across personas.

    A simplified version of run_panel for quick feedback on one question.
    Includes synthesis by default.

    When no provider credentials are set and the invoking MCP client
    advertises the ``sampling`` capability, the poll runs in
    **sampling mode**: synthpanel asks the host agent to run each
    persona's completion using its own LLM access, so the user can run
    their first poll with zero configuration. Sampling mode is capped
    at :data:`SAMPLING_MAX_PERSONAS` personas to keep the host agent's
    context footprint small — larger panels require BYOK credentials.

    Args:
        question: The question to ask all personas.
        personas: List of persona definitions. Optional — when omitted,
            a small built-in pack of diverse personas is used so the
            tool works with zero configuration. Each persona is a JSON
            object with the following recognized fields (additional
            fields are preserved and remain available to custom
            prompt templates):

            * ``name`` (str, **required**) — persona's display name.
            * ``age`` (int, optional) — persona's age.
            * ``occupation`` (str, optional) — job title or role.
            * ``background`` (str, optional) — paragraph-sized bio
              giving context (company, tenure, constraints, etc.).
            * ``personality_traits`` (list[str], optional) — short
              trait adjectives, e.g. ``["analytical", "skeptical"]``.

            Example::

                [
                  {
                    "name": "Alex Rivera",
                    "age": 29,
                    "occupation": "Software Engineer",
                    "background": "Early-career developer at a mid-sized SaaS company.",
                    "personality_traits": ["analytical", "curious", "pragmatic"]
                  }
                ]
        model: LLM model to use. Defaults to haiku. Ignored in sampling
            mode (the host agent picks its own model).
        response_schema: Optional JSON Schema for structured output. When
            provided, responses are extracted as structured JSON matching
            this schema instead of free text. Not supported in sampling
            mode — raw text is returned instead.
        synthesis: Whether to run synthesis after collecting responses.
            Defaults to true. In sampling mode synthesis is also
            performed via the host agent.
        synthesis_model: Model to use for synthesis. Defaults to panelist model.
        synthesis_prompt: Custom synthesis prompt. Replaces the default.
        temperature: Sampling temperature (0.0-1.0). Controls randomness.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
        use_sampling: Explicit mode override. ``True`` forces sampling
            (error if unsupported), ``False`` forces BYOK. ``None``
            auto-picks based on creds + client capability.
    """
    model = model or _resolve_mcp_default_model()

    if not question or not question.strip():
        return json.dumps({"error": "Question text must be a non-empty string."})

    # Fall back to the built-in diverse persona set when the caller
    # omits personas — preserves the zero-config first-run story.
    if personas is None or len(personas) == 0:
        personas = [dict(p) for p in DEFAULT_QUICK_POLL_PERSONAS]

    # Validate personas: must be dicts with "name"
    for i, p in enumerate(personas):
        if not isinstance(p, dict):
            return json.dumps({"error": f"Persona at index {i} must be a dict, got {type(p).__name__}."})
        if "name" not in p or not str(p["name"]).strip():
            return json.dumps({"error": f"Persona at index {i} is missing required field 'name'."})

    if len(personas) > MAX_PERSONAS:
        return json.dumps({"error": f"Too many personas ({len(personas)}). Maximum is {MAX_PERSONAS}."})

    decision = _decide_sampling_mode(ctx, use_sampling=use_sampling)
    logger.info("run_quick_poll: mode=%s model=%s personas=%d", decision.mode, model, len(personas))

    if decision.mode == "error":
        return json.dumps({"error": decision.error})

    if decision.mode == "sampling":
        if len(personas) > SAMPLING_MAX_PERSONAS:
            return json.dumps(
                {
                    "error": (
                        f"Sampling mode is capped at {SAMPLING_MAX_PERSONAS} personas "
                        f"to protect the host agent's context window (got "
                        f"{len(personas)}). Set ANTHROPIC_API_KEY (or another "
                        f"provider key) in your environment to run larger panels "
                        f"via BYOK."
                    )
                }
            )
        result = await _run_quick_poll_sampling(
            ctx,
            question=question,
            personas=personas,
            synthesis=synthesis,
            synthesis_prompt=synthesis_prompt,
            temperature=temperature,
            hint=decision.hint,
        )
        return json.dumps(result, indent=2)

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
    result["mode"] = "byok"
    return json.dumps(result, indent=2)


async def _run_panel_sampling(
    ctx: Context,
    *,
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    synthesis: bool,
    synthesis_prompt: str | None,
    temperature: float | None,
    hint: str | None,
) -> dict[str, Any]:
    """Run a full panel via MCP sampling.

    Mirrors :func:`_run_panel_async`'s shape for the fields callers
    depend on (``results``, ``rounds``, ``synthesis``, ``persona_count``,
    ``question_count``, ``path``, ``warnings``) so downstream tooling
    doesn't have to special-case sampling-mode output. BYOK-only fields
    (``usage``, ``cost``, ``metadata``) are ``None`` — the host agent
    absorbs token cost.

    Serial across personas (host agents rate-limit sampling) but each
    persona answers all questions in a single sampling call to keep
    round-trips small.
    """
    from synth_panel.prompts import SYNTHESIS_PROMPT

    await ctx.report_progress(0, len(personas))
    panelist_entries: list[dict[str, Any]] = []
    host_model: str | None = None

    question_texts = [str(q["text"]) for q in questions]
    joined_questions = "\n\n".join(f"Q{i + 1}: {t}" for i, t in enumerate(question_texts))

    for i, persona in enumerate(personas):
        system_prompt = persona_system_prompt(persona)
        user_prompt = (
            "Answer each of the following questions in order. "
            "Label each answer with the matching Q number.\n\n" + joined_questions
        )
        sample = await _sample_text(
            ctx,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=SAMPLING_MAX_TOKENS_DEFAULT,
            temperature=temperature,
        )
        host_model = sample["model"]
        # Surface the full answer string against every question so the
        # output matches BYOK shape: one response entry per question.
        responses = [{"question": q_text, "answer": sample["text"]} for q_text in question_texts]
        panelist_entries.append(
            {
                "persona": persona,
                "responses": responses,
                "model": sample["model"],
                "usage": None,
            }
        )
        await ctx.report_progress(i + 1, len(personas))

    synthesis_block: dict[str, Any] | None = None
    if synthesis and panelist_entries:
        synth_prompt = synthesis_prompt or SYNTHESIS_PROMPT
        rendered_panel = "\n\n".join(
            "Panelist: {name}\n{responses}".format(
                name=entry["persona"].get("name", "anon"),
                responses="\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in entry["responses"]),
            )
            for entry in panelist_entries
        )
        synth = await _sample_text(
            ctx,
            prompt=rendered_panel,
            system_prompt=synth_prompt,
            max_tokens=SAMPLING_MAX_TOKENS_DEFAULT,
            temperature=temperature,
        )
        synthesis_block = {
            "summary": synth["text"],
            "model": synth["model"],
            "usage": None,
        }

    return {
        "mode": "sampling",
        "hint": hint,
        "model": host_model,
        "persona_count": len(personas),
        "question_count": len(questions),
        "results": panelist_entries,
        "rounds": [
            {
                "name": "default",
                "results": panelist_entries,
                "synthesis": synthesis_block,
            }
        ],
        "synthesis": synthesis_block,
        "path": [],
        "warnings": [],
        "usage": None,
        "cost": None,
        "metadata": None,
    }


async def _run_quick_poll_sampling(
    ctx: Context,
    *,
    question: str,
    personas: list[dict[str, Any]],
    synthesis: bool,
    synthesis_prompt: str | None,
    temperature: float | None,
    hint: str | None,
) -> dict[str, Any]:
    """Run a quick poll via MCP sampling.

    One ``create_message`` call per persona (serial — host agents
    generally rate-limit sampling), plus one synthesis call when
    enabled. The result shape deliberately mirrors the BYOK
    :func:`_run_panel_async` output for the fields callers care about
    (``results``, ``synthesis``, ``rounds``, ``persona_count``,
    ``question_count``) so downstream tooling works uniformly across
    modes. Fields that only make sense in BYOK (``usage``, ``cost``,
    ``metadata``) are ``None``.
    """
    from synth_panel.prompts import SYNTHESIS_PROMPT

    await ctx.report_progress(0, len(personas))
    panelist_entries: list[dict[str, Any]] = []
    host_model: str | None = None
    for i, persona in enumerate(personas):
        system_prompt = persona_system_prompt(persona)
        user_prompt = build_question_prompt({"text": question})
        sample = await _sample_text(
            ctx,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=temperature,
        )
        host_model = sample["model"]
        panelist_entries.append(
            {
                "persona": persona,
                "responses": [
                    {
                        "question": question,
                        "answer": sample["text"],
                    }
                ],
                "model": sample["model"],
                "usage": None,
            }
        )
        await ctx.report_progress(i + 1, len(personas))

    synthesis_block: dict[str, Any] | None = None
    if synthesis and panelist_entries:
        synth_prompt = synthesis_prompt or SYNTHESIS_PROMPT
        rendered_panel = "\n\n".join(
            f"Panelist: {entry['persona'].get('name', 'anon')}\nQ: {question}\nA: {entry['responses'][0]['answer']}"
            for entry in panelist_entries
        )
        synth = await _sample_text(
            ctx,
            prompt=rendered_panel,
            system_prompt=synth_prompt,
            max_tokens=2048,
            temperature=temperature,
        )
        synthesis_block = {
            "summary": synth["text"],
            "model": synth["model"],
            "usage": None,
        }

    return {
        "mode": "sampling",
        "hint": hint,
        "model": host_model,
        "persona_count": len(personas),
        "question_count": 1,
        "results": panelist_entries,
        "rounds": [
            {
                "name": "default",
                "results": panelist_entries,
                "synthesis": synthesis_block,
            }
        ],
        "synthesis": synthesis_block,
        "path": [],
        "warnings": [],
        "usage": None,
        "cost": None,
        "metadata": None,
    }


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
    model = model or _resolve_mcp_default_model()
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
        client = _get_shared_client()
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
    """Run the MCP server on stdio transport.

    FastMCP's default ``run_stdio_async`` calls
    ``create_initialization_options()`` with no arguments, so synthpanel
    cannot advertise that it *uses* MCP sampling. We reimplement the
    stdio loop here to advertise sampling in two places on the
    initialize response: at the top level of ``capabilities`` so hosts
    and inspectors that scan top-level keys can discover the dependency
    directly, and nested under ``experimental`` for backwards
    compatibility with clients that only look there. The MCP spec does
    not reserve a ``sampling`` field on ``ServerCapabilities`` (sampling
    is defined as a client capability), but ``ServerCapabilities`` is
    declared ``extra="allow"`` so the top-level key round-trips cleanly.
    """
    import anyio
    from mcp.server.stdio import stdio_server

    logger.info("MCP server starting (stdio transport)")

    async def _run() -> None:
        server = mcp._mcp_server
        init_opts = server.create_initialization_options(
            experimental_capabilities={"sampling": {}},
        )
        # Also surface `sampling` at the top of ServerCapabilities —
        # multiple MCP inspectors and hosts enumerate top-level
        # capability keys and miss the experimental nesting.
        init_opts.capabilities = init_opts.capabilities.model_copy(update={"sampling": {}})
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_opts)

    anyio.run(_run)
