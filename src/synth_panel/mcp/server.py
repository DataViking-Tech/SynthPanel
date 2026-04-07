"""MCP server implementation for synth-panel.

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
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from synth_panel.cost import ZERO_USAGE, TokenUsage as CostTokenUsage, UsageTracker, estimate_cost, lookup_pricing
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock
from synth_panel.instrument import Instrument, parse_instrument
from synth_panel.mcp.data import (
    get_panel_result as _data_get_panel_result,
    get_persona_pack as _data_get_persona_pack,
    list_instrument_packs as _data_list_instrument_packs,
    list_panel_results as _data_list_panel_results,
    list_persona_packs as _data_list_persona_packs,
    load_instrument_pack as _data_load_instrument_pack,
    load_panel_sessions,
    save_instrument_pack as _data_save_instrument_pack,
    save_panel_result,
    save_persona_pack as _data_save_persona_pack,
    update_panel_result,
)
from synth_panel.orchestrator import (
    MultiRoundResult,
    PanelistResult,
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


def _run_multi_round_sync(
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    response_schema: dict[str, Any] | None,
    *,
    synthesis: bool,
    synthesis_model: str | None,
    synthesis_prompt: str | None,
) -> MultiRoundResult:
    """Drive run_multi_round_panel for v1/v2/v3 instruments."""
    client = LLMClient()

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
            custom_prompt=synthesis_prompt,
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
    )


def _format_panelist_result(pr: PanelistResult, model: str) -> dict[str, Any]:
    pricing, _ = lookup_pricing(model)
    persona_cost = estimate_cost(pr.usage, pricing)
    return {
        "persona": pr.persona_name,
        "responses": pr.responses,
        "usage": pr.usage.to_dict(),
        "cost": persona_cost.format_usd(),
        "error": pr.error,
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
) -> dict[str, Any]:
    """Run a (possibly branching) instrument and return v3-shaped response."""
    total = len(personas)
    await ctx.report_progress(0, total)

    mr: MultiRoundResult = await asyncio.wait_for(
        asyncio.to_thread(
            _run_multi_round_sync,
            personas, instrument, model, response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
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
        questions_for_round = next(
            (r.questions for r in instrument.rounds if r.name == rr.name), []
        )
        total_question_count += len(questions_for_round)
        rounds_payload.append({
            "name": rr.name,
            "results": round_dict_results,
            "synthesis": rr.synthesis.to_dict() if hasattr(rr.synthesis, "to_dict") else None,
            "usage": rr.usage.to_dict(),
        })
        # Flat results for back-compat / persistence: use the *last* round per persona.
        flat_results = round_dict_results

    pricing, _ = lookup_pricing(model)
    total_cost = estimate_cost(mr.usage, pricing)

    final_synth_dict = (
        mr.final_synthesis.to_dict()
        if mr.final_synthesis is not None and hasattr(mr.final_synthesis, "to_dict")
        else None
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
            merged, instrument_obj, model, ctx, response_schema,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            synthesis_prompt=synthesis_prompt,
        )
        return json.dumps(result, indent=2)

    if not questions:
        return json.dumps({"error": "No questions or instrument provided."})

    result = await _run_panel_async(
        merged, questions, model, ctx, response_schema,
        synthesis=synthesis,
        synthesis_model=synthesis_model,
        synthesis_prompt=synthesis_prompt,
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
        synthesis_model: Synthesis model. Defaults to sonnet.
        synthesis_prompt: Custom synthesis prompt for the new round.
    """
    model = model or MCP_DEFAULT_MODEL
    existing = _data_get_panel_result(result_id)

    # Reuse the original personas (recovered from saved sessions if possible).
    sessions = load_panel_sessions(result_id)
    personas: list[dict[str, Any]] = [{"name": name} for name in sessions.keys()]
    if not personas:
        return json.dumps({"error": f"No sessions found for result {result_id}"})

    if ctx is not None:
        await ctx.report_progress(0, len(personas))

    def _go() -> tuple[list[PanelistResult], Any]:
        client = LLMClient()
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
                    client, results, questions,
                    model=synthesis_model, custom_prompt=synthesis_prompt,
                )
            except Exception:
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
    rounds = list(rounds) + [{
        "name": f"extension-{len(rounds) + 1}",
        "results": new_round_results,
        "synthesis": synth.to_dict() if synth is not None and hasattr(synth, "to_dict") else None,
        "extension": True,
    }]
    path = list(existing.get("path") or [])
    path.append({
        "round": rounds[-1]["name"],
        "branch": "extension (ad-hoc, not DAG re-entry)",
        "next": "__end__",
    })

    updated = dict(existing)
    updated["rounds"] = rounds
    updated["path"] = path
    updated["results"] = new_round_results  # mirrors latest round (back-compat)
    updated["question_count"] = int(existing.get("question_count", 0)) + len(questions)
    update_panel_result(result_id, updated)

    return json.dumps({
        "result_id": result_id,
        "appended_round": rounds[-1]["name"],
        "results": new_round_results,
        "synthesis": rounds[-1]["synthesis"],
        "path": path,
    }, indent=2)


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
