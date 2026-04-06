"""MCP server implementation for synth-panel.

Exposes 11 tools, 4 resource URI patterns, and 3 prompt templates.
Uses stdio transport. Default model is haiku for MCP mode.

Tools:
    run_prompt              - Send a single prompt to an LLM (no personas)
    run_panel               - Run a full synthetic focus group panel
    run_quick_poll          - Quick single-question poll across personas
    list_persona_packs      - List saved persona packs
    get_persona_pack        - Get a specific persona pack
    save_persona_pack       - Save a persona pack
    list_instrument_packs   - List installed instrument packs
    get_instrument_pack     - Get a specific instrument pack
    save_instrument_pack    - Save an instrument pack
    list_panel_results      - List saved panel results
    get_panel_result        - Get a specific panel result

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
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from synth_panel.cost import ZERO_USAGE, TokenUsage as CostTokenUsage, UsageTracker, estimate_cost, lookup_pricing
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock
from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.mcp.data import (
    get_panel_result as _data_get_panel_result,
    get_persona_pack as _data_get_persona_pack,
    list_instrument_packs as _data_list_instrument_packs,
    list_panel_results as _data_list_panel_results,
    list_persona_packs as _data_list_persona_packs,
    load_instrument_pack as _data_load_instrument_pack,
    save_instrument_pack as _data_save_instrument_pack,
    save_panel_result,
    save_persona_pack as _data_save_persona_pack,
)
from synth_panel.orchestrator import (
    MultiRoundResult,
    PanelistResult,
    RoundResult,
    run_multi_round_panel,
    run_panel_parallel,
)
from synth_panel.prompts import build_question_prompt, persona_system_prompt
from synth_panel.synthesis import synthesize_panel

# Default model for MCP mode
MCP_DEFAULT_MODEL = "haiku"

# Per-panelist timeout (seconds)
PANELIST_TIMEOUT = 30

mcp = FastMCP(
    "synth-panel",
    instructions=(
        "Synthetic focus group server. Run panels of AI personas to get "
        "structured qualitative feedback on products, concepts, and names."
    ),
)


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
) -> tuple[list[PanelistResult], list[dict[str, Any]], CostTokenUsage, Any, dict[str, Any] | None]:
    """Run panel synchronously. Returns (results, result_dicts, panelist_usage, panelist_cost, synthesis_dict)."""
    client = LLMClient()
    panelist_results, _registry, _sessions = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        response_schema=response_schema,
    )

    panelist_usage = ZERO_USAGE
    result_dicts: list[dict[str, Any]] = []
    for pr in panelist_results:
        pricing, _ = lookup_pricing(model)
        persona_cost = estimate_cost(pr.usage, pricing)
        result_dicts.append({
            "persona": pr.persona_name,
            "responses": pr.responses,
            "usage": pr.usage.to_dict(),
            "cost": persona_cost.format_usd(),
            "error": pr.error,
        })
        panelist_usage = panelist_usage + pr.usage

    pricing, _ = lookup_pricing(model)
    panelist_cost = estimate_cost(panelist_usage, pricing)

    # Synthesis
    synthesis_dict: dict[str, Any] | None = None
    if synthesis:
        try:
            synthesis_result = synthesize_panel(
                client,
                panelist_results,
                questions,
                model=synthesis_model,
                custom_prompt=synthesis_prompt,
            )
            synthesis_dict = synthesis_result.to_dict()
        except Exception:
            pass  # Synthesis failure is non-fatal for MCP

    return panelist_results, result_dicts, panelist_usage, panelist_cost, synthesis_dict


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
) -> dict[str, Any]:
    """Run panel via asyncio.to_thread with progress notifications."""
    total = len(personas)
    await ctx.report_progress(0, total)

    # Run the blocking panel execution in a thread
    panelist_results, result_dicts, panelist_usage, panelist_cost, synthesis_dict = await asyncio.wait_for(
        asyncio.to_thread(
            _run_panel_sync, personas, questions, model, response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
        ),
        timeout=PANELIST_TIMEOUT * total,
    )

    await ctx.report_progress(total, total)

    # Compute total cost (panelist + synthesis)
    if synthesis_dict:
        synthesis_usage = CostTokenUsage.from_dict(synthesis_dict["usage"])
        synthesis_pricing, _ = lookup_pricing(synthesis_dict.get("model"))
        synthesis_cost_est = estimate_cost(synthesis_usage, synthesis_pricing)
        total_usage = panelist_usage + synthesis_usage
        total_cost = panelist_cost + synthesis_cost_est
    else:
        total_usage = panelist_usage
        total_cost = panelist_cost

    # Save result
    result_id = save_panel_result(
        results=result_dicts,
        model=model,
        total_usage=total_usage.to_dict(),
        total_cost=total_cost.format_usd(),
        persona_count=len(personas),
        question_count=len(questions),
    )

    return {
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
    }


def _run_branching_panel_sync(
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    response_schema: dict[str, Any] | None,
    *,
    synthesis: bool,
    synthesis_model: str | None,
) -> MultiRoundResult:
    """Execute a multi-round / branching panel run synchronously."""
    client = LLMClient()

    def _round_synth(client_, panelist_results, qs, model=None):
        return synthesize_panel(
            client_, panelist_results, qs, model=synthesis_model
        )

    final_fn = _round_synth if synthesis else None
    return run_multi_round_panel(
        client=client,
        personas=personas,
        instrument=instrument,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        synthesize_round_fn=_round_synth,
        synthesize_final_fn=final_fn,
        response_schema=response_schema,
    )


def _round_to_dict(rr: RoundResult, model: str) -> dict[str, Any]:
    pricing, _ = lookup_pricing(model)
    panelist_dicts: list[dict[str, Any]] = []
    for pr in rr.panelist_results:
        panelist_dicts.append({
            "persona": pr.persona_name,
            "responses": pr.responses,
            "usage": pr.usage.to_dict(),
            "cost": estimate_cost(pr.usage, pricing).format_usd(),
            "error": pr.error,
        })
    synthesis_dict = (
        rr.synthesis.to_dict() if rr.synthesis is not None and hasattr(rr.synthesis, "to_dict") else None
    )
    return {
        "name": rr.name,
        "results": panelist_dicts,
        "synthesis": synthesis_dict,
        "usage": rr.usage.to_dict(),
    }


async def _run_branching_panel_async(
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    ctx: Context,
    response_schema: dict[str, Any] | None,
    *,
    synthesis: bool,
    synthesis_model: str | None,
) -> dict[str, Any]:
    """Execute multi-round panel via asyncio.to_thread, build response payload."""
    total = max(1, len(instrument.rounds)) * len(personas)
    await ctx.report_progress(0, total)

    multi: MultiRoundResult = await asyncio.wait_for(
        asyncio.to_thread(
            _run_branching_panel_sync,
            personas,
            instrument,
            model,
            response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
        ),
        timeout=PANELIST_TIMEOUT * total,
    )

    await ctx.report_progress(total, total)

    rounds_payload = [_round_to_dict(rr, model) for rr in multi.rounds]

    pricing, _ = lookup_pricing(model)
    panelist_usage = ZERO_USAGE
    for rr in multi.rounds:
        for pr in rr.panelist_results:
            panelist_usage = panelist_usage + pr.usage
    panelist_cost = estimate_cost(panelist_usage, pricing)

    final_synth_dict: dict[str, Any] | None = None
    if multi.final_synthesis is not None and hasattr(multi.final_synthesis, "to_dict"):
        final_synth_dict = multi.final_synthesis.to_dict()

    total_usage = multi.usage
    total_cost = estimate_cost(total_usage, pricing)

    # Build a flat results list for save_panel_result (for retrievability).
    flat_results: list[dict[str, Any]] = []
    for rp in rounds_payload:
        for r in rp["results"]:
            flat_results.append({**r, "round": rp["name"]})

    result_id = save_panel_result(
        results=flat_results,
        model=model,
        total_usage=total_usage.to_dict(),
        total_cost=total_cost.format_usd(),
        persona_count=len(personas),
        question_count=sum(len(r.questions) for r in instrument.rounds),
    )

    return {
        "result_id": result_id,
        "model": model,
        "persona_count": len(personas),
        "rounds": rounds_payload,
        "path": list(multi.path),
        "warnings": list(multi.warnings),
        "terminal_round": multi.terminal_round,
        "synthesis": final_synth_dict,
        "panelist_cost": panelist_cost.format_usd(),
        "total_cost": total_cost.format_usd(),
        "total_usage": total_usage.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_prompt(
    prompt: str,
    model: str | None = None,
) -> str:
    """Send a single prompt to an LLM and get a response. No personas required.

    The simplest tool — ask a quick research question without constructing
    personas or running a full panel.

    Args:
        prompt: The question or prompt to send.
        model: LLM model to use. Defaults to haiku.
    """
    model = model or MCP_DEFAULT_MODEL
    client = LLMClient()
    request = CompletionRequest(
        model=model,
        max_tokens=4096,
        messages=[InputMessage(role="user", content=[TextBlock(text=prompt)])],
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
    return json.dumps({
        "response": response.text,
        "model": response.model,
        "usage": usage.to_dict(),
        "cost": cost.format_usd(),
    }, indent=2)


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
    ctx: Context = None,
) -> str:
    """Run a full synthetic focus group panel.

    Three input shapes are accepted:

    1. ``questions`` — flat v1 list of question dicts (legacy single round).
    2. ``instrument`` — full instrument body (v1/v2/v3) as a dict. v2 (linear
       multi-round) and v3 (branching with ``route_when``) execute via the
       router-driven multi-round runner.
    3. ``instrument_pack`` — name of an installed instrument pack. Loaded
       via ``load_instrument_pack`` and parsed exactly like ``instrument``.

    The response always includes ``path`` and ``warnings`` keys. For
    single-round runs ``path`` has length 1 and ``warnings`` is the parser
    warning list (may be empty). For multi-round/branching runs ``path``
    contains one entry per executed round describing the routing decision.

    Note on extension: there is no ``extend_panel`` re-entry into the DAG.
    If a future ``extend_panel`` tool is added, it will append a single
    ad-hoc round on top of the executed path — it will *not* re-enter the
    authored DAG, restart routing, or replay branches (architect note 3).

    Args:
        questions: Legacy v1 question list. Each entry should have a 'text'
            key. Mutually exclusive with ``instrument``/``instrument_pack``.
        personas: List of persona definitions. Each should have at minimum
            a 'name' key. Optional: age, occupation, background, personality_traits.
        pack_id: ID of a saved persona pack to use. Personas from the pack are
            merged with any inline personas (inline personas come first).
            At least one of personas or pack_id must be provided.
        instrument: Full instrument body (v1/v2/v3) as a dict. Use this for
            multi-round and branching runs.
        instrument_pack: Name of an installed instrument pack to load.
        model: LLM model to use. Defaults to haiku.
        response_schema: Optional JSON Schema for structured output. When
            provided, each panelist's responses are extracted as structured
            JSON matching this schema instead of free text.
        synthesis: Whether to run synthesis after collecting responses.
            Defaults to true.
        synthesis_model: Model to use for synthesis. Defaults to sonnet.
        synthesis_prompt: Custom synthesis prompt. Replaces the default.
    """
    model = model or MCP_DEFAULT_MODEL
    merged = list(personas) if personas else []
    if pack_id is not None:
        pack = _data_get_persona_pack(pack_id)
        merged.extend(pack.get("personas", []))
    if not merged:
        return json.dumps({"error": "No personas provided. Supply personas and/or pack_id."})

    # Resolve instrument source: instrument dict > instrument_pack > questions
    instrument_obj: Instrument | None = None
    raw_instrument: dict[str, Any] | None = None
    if instrument is not None:
        raw_instrument = dict(instrument)
    elif instrument_pack is not None:
        try:
            data = _data_load_instrument_pack(instrument_pack)
        except FileNotFoundError as exc:
            return json.dumps({"error": str(exc)})
        # Pack body may nest the instrument under 'instrument' or live flat.
        if isinstance(data, dict) and "instrument" in data:
            raw_instrument = dict(data["instrument"])
        else:
            raw_instrument = {
                k: v for k, v in data.items()
                if k not in {"id", "name", "version", "description", "author"}
            }

    if raw_instrument is not None:
        raw_instrument.setdefault("version", 1)
        try:
            instrument_obj = parse_instrument(raw_instrument)
        except InstrumentError as exc:
            return json.dumps({"error": f"instrument validation failed: {exc}"})

    if instrument_obj is None and not questions:
        return json.dumps({
            "error": "No questions or instrument provided. Supply questions, instrument, or instrument_pack."
        })

    if instrument_obj is not None and len(instrument_obj.rounds) > 1:
        # Multi-round / branching path.
        result = await _run_branching_panel_async(
            merged, instrument_obj, model, ctx, response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
        )
        return json.dumps(result, indent=2)

    # Single-round path: prefer instrument's own questions if provided.
    effective_questions: list[dict[str, Any]]
    if instrument_obj is not None:
        effective_questions = list(instrument_obj.questions)
    else:
        effective_questions = list(questions or [])

    result = await _run_panel_async(
        merged, effective_questions, model, ctx, response_schema,
        synthesis=synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
    )
    # Always emit path + warnings keys for shape consistency.
    if instrument_obj is not None:
        round_name = instrument_obj.rounds[0].name if instrument_obj.rounds else "default"
        result["path"] = [{"round": round_name, "branch": "linear", "next": "__end__"}]
        result["warnings"] = list(getattr(instrument_obj, "warnings", []) or [])
    else:
        result["path"] = [{"round": "default", "branch": "linear", "next": "__end__"}]
        result["warnings"] = []
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
        synthesis_model: Model to use for synthesis. Defaults to sonnet.
        synthesis_prompt: Custom synthesis prompt. Replaces the default.
    """
    model = model or MCP_DEFAULT_MODEL
    questions = [{"text": question}]
    result = await _run_panel_async(
        personas, questions, model, ctx, response_schema,
        synthesis=synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
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
    """List all installed instrument packs.

    Instrument packs are single-file YAMLs under
    ``$SYNTH_PANEL_DATA_DIR/packs/instruments/``. Each carries a top-level
    manifest (name, version, description, author) plus a v1/v2/v3
    instrument body. The flywheel companion to persona packs.
    """
    packs = _data_list_instrument_packs()
    return json.dumps(packs, indent=2)


@mcp.tool()
async def get_instrument_pack(name: str) -> str:
    """Get a specific instrument pack by name.

    Returns the full YAML body, including manifest fields and the
    instrument definition. Pass the result's instrument body to
    ``run_panel`` via the ``instrument`` argument, or pass ``name``
    directly via ``instrument_pack``.

    Args:
        name: The pack name (filename stem under packs/instruments/).
    """
    try:
        pack = _data_load_instrument_pack(name)
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(pack, indent=2)


@mcp.tool()
async def save_instrument_pack(
    name: str,
    content: dict[str, Any],
) -> str:
    """Save an instrument pack to disk.

    The content is parsed and validated before being written — bad
    instruments are rejected. The manifest 'name' is forced to match
    the pack id on disk.

    Args:
        name: Pack name (used as filename stem).
        content: Full YAML body — manifest fields at top level, plus an
            ``instrument`` key (or instrument fields directly at top level).
    """
    if not isinstance(content, dict):
        return json.dumps({"error": "content must be a mapping"})

    # Validate before saving: extract instrument body and parse.
    if "instrument" in content and isinstance(content["instrument"], dict):
        raw = dict(content["instrument"])
    else:
        raw = {
            k: v for k, v in content.items()
            if k not in {"id", "name", "version", "description", "author"}
        }
    raw.setdefault("version", 1)
    try:
        parse_instrument(raw)
    except InstrumentError as exc:
        return json.dumps({"error": f"instrument validation failed: {exc}"})

    meta = _data_save_instrument_pack(name, content)
    return json.dumps(meta, indent=2)


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
        follow_up_section = (
            "\n\nAfter each response, ask one follow-up question to dig deeper "
            "into their perspective."
        )

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
        audience_line = (
            f"\n\nTarget the personas toward: {target_audience}"
        )

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
    mcp.run(transport="stdio")
