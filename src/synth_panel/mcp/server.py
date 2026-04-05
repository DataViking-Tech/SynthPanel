"""MCP server implementation for synth-panel.

Exposes 8 tools, 4 resource URI patterns, and 3 prompt templates.
Uses stdio transport. Default model is haiku for MCP mode.

Tools:
    run_prompt          - Send a single prompt to an LLM (no personas)
    run_panel           - Run a full synthetic focus group panel
    run_quick_poll      - Quick single-question poll across personas
    list_persona_packs  - List saved persona packs
    get_persona_pack    - Get a specific persona pack
    save_persona_pack   - Save a persona pack
    list_panel_results  - List saved panel results
    get_panel_result    - Get a specific panel result

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
from synth_panel.mcp.data import (
    get_panel_result as _data_get_panel_result,
    get_persona_pack as _data_get_persona_pack,
    list_panel_results as _data_list_panel_results,
    list_persona_packs as _data_list_persona_packs,
    save_panel_result,
    save_persona_pack as _data_save_persona_pack,
)
from synth_panel.orchestrator import PanelistResult, run_panel_parallel
from synth_panel.prompts import build_question_prompt, persona_system_prompt

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
) -> tuple[list[PanelistResult], dict[str, Any], str]:
    """Run panel synchronously. Returns (results, total_usage_dict, total_cost)."""
    client = LLMClient()
    panelist_results, _registry = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        response_schema=response_schema,
    )

    total_usage = ZERO_USAGE
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
        total_usage = total_usage + pr.usage

    pricing, _ = lookup_pricing(model)
    total_cost = estimate_cost(total_usage, pricing)

    return panelist_results, result_dicts, total_usage, total_cost


async def _run_panel_async(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str,
    ctx: Context,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run panel via asyncio.to_thread with progress notifications."""
    total = len(personas)
    await ctx.report_progress(0, total)

    # Run the blocking panel execution in a thread
    panelist_results, result_dicts, total_usage, total_cost = await asyncio.wait_for(
        asyncio.to_thread(_run_panel_sync, personas, questions, model, response_schema),
        timeout=PANELIST_TIMEOUT * total,
    )

    await ctx.report_progress(total, total)

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
        "total_cost": total_cost.format_usd(),
        "total_usage": total_usage.to_dict(),
        "results": result_dicts,
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
    questions: list[dict[str, Any]],
    personas: list[dict[str, Any]] | None = None,
    pack_id: str | None = None,
    model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Run a full synthetic focus group panel.

    Each persona answers all questions independently in parallel.
    Results are saved and can be retrieved later.

    Args:
        questions: List of question definitions. Each should have a 'text' key.
            Optional: follow_ups (list of follow-up question strings).
        personas: List of persona definitions. Each should have at minimum
            a 'name' key. Optional: age, occupation, background, personality_traits.
        pack_id: ID of a saved persona pack to use. Personas from the pack are
            merged with any inline personas (inline personas come first).
            At least one of personas or pack_id must be provided.
        model: LLM model to use. Defaults to haiku.
        response_schema: Optional JSON Schema for structured output. When
            provided, each panelist's responses are extracted as structured
            JSON matching this schema instead of free text.
    """
    model = model or MCP_DEFAULT_MODEL
    merged = list(personas) if personas else []
    if pack_id is not None:
        pack = _data_get_persona_pack(pack_id)
        merged.extend(pack.get("personas", []))
    if not merged:
        return json.dumps({"error": "No personas provided. Supply personas and/or pack_id."})
    result = await _run_panel_async(merged, questions, model, ctx, response_schema)
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_quick_poll(
    question: str,
    personas: list[dict[str, Any]],
    model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Quick single-question poll across personas.

    A simplified version of run_panel for quick feedback on one question.

    Args:
        question: The question to ask all personas.
        personas: List of persona definitions.
        model: LLM model to use. Defaults to haiku.
        response_schema: Optional JSON Schema for structured output. When
            provided, responses are extracted as structured JSON matching
            this schema instead of free text.
    """
    model = model or MCP_DEFAULT_MODEL
    questions = [{"text": question}]
    result = await _run_panel_async(personas, questions, model, ctx, response_schema)
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
