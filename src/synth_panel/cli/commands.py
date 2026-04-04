"""Subcommand handlers for synth-panel CLI.

Each handler receives parsed args and output format, returns an exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cost import ZERO_USAGE, UsageTracker, estimate_cost, format_summary, lookup_pricing
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import TextBlock
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.runtime import AgentRuntime


def _resolve_model(args: argparse.Namespace) -> str:
    """Return the model alias from CLI args, defaulting to 'sonnet'."""
    return args.model or "sonnet"


def handle_prompt(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run a single non-interactive prompt and exit."""
    prompt_text = " ".join(args.text)
    model = _resolve_model(args)

    client = LLMClient()
    session = Session()
    runtime = AgentRuntime(
        client=client,
        session=session,
        model=model,
    )

    try:
        summary = runtime.run_turn(prompt_text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Extract assistant response text
    response_text = ""
    for msg in summary.assistant_messages:
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                response_text += block.get("text", "")

    usage_dict = summary.usage.to_dict() if summary.usage else None

    # Show cost summary for text output
    if fmt is OutputFormat.TEXT:
        print(response_text)
        pricing, is_estimated = lookup_pricing(model)
        cost = estimate_cost(summary.usage, pricing)
        print(format_summary("Cost", summary.usage, cost, model=model, is_estimated=is_estimated),
              file=sys.stderr)
    else:
        extra: dict[str, Any] = {}
        pricing, is_estimated = lookup_pricing(model)
        cost = estimate_cost(summary.usage, pricing)
        extra["cost"] = cost.format_usd()
        extra["model"] = model
        emit(fmt, message=response_text, usage=usage_dict, extra=extra)

    return 0


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Any:
    """Load and return parsed YAML from *path*."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_personas(path: str) -> list[dict[str, Any]]:
    """Load personas from a YAML file.

    Expected format::

        personas:
          - name: Alice
            age: 34
            occupation: Teacher
            background: ...
            personality_traits: [curious, empathetic]
    """
    data = _load_yaml(path)
    if isinstance(data, dict) and "personas" in data:
        return data["personas"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Invalid personas file: expected 'personas' key or a list, got {type(data).__name__}")


def _load_instrument(path: str) -> dict[str, Any]:
    """Load an instrument/survey from a YAML file.

    Expected format::

        instrument:
          questions:
            - text: "What do you think about ...?"
              response_schema: {type: text}
              follow_ups: ["Can you elaborate?"]
    """
    data = _load_yaml(path)
    if isinstance(data, dict) and "instrument" in data:
        return data["instrument"]
    if isinstance(data, dict) and "questions" in data:
        return data
    raise ValueError(f"Invalid instrument file: expected 'instrument' or 'questions' key")


def _persona_system_prompt(persona: dict[str, Any]) -> str:
    """Build a system prompt from a persona definition."""
    parts = [f"You are role-playing as {persona.get('name', 'an anonymous person')}."]
    if persona.get("age"):
        parts.append(f"Age: {persona['age']}.")
    if persona.get("occupation"):
        parts.append(f"Occupation: {persona['occupation']}.")
    if persona.get("background"):
        parts.append(f"Background: {persona['background']}.")
    if persona.get("personality_traits"):
        traits = persona["personality_traits"]
        if isinstance(traits, list):
            traits = ", ".join(str(t) for t in traits)
        parts.append(f"Personality traits: {traits}.")
    parts.append(
        "Answer questions in character. Be authentic to this persona's "
        "perspective, experiences, and communication style. "
        "Give concise, direct answers."
    )
    return " ".join(parts)


def _build_question_prompt(question: dict[str, Any]) -> str:
    """Build a user prompt from a question definition."""
    text = question.get("text", question) if isinstance(question, dict) else str(question)
    return str(text)


def handle_panel_run(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run a panel: load personas + instrument, run panelists in parallel."""
    model = _resolve_model(args)

    try:
        personas = _load_personas(args.personas)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading personas: {exc}", file=sys.stderr)
        return 1

    try:
        instrument = _load_instrument(args.instrument)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading instrument: {exc}", file=sys.stderr)
        return 1

    questions = instrument.get("questions", [])
    if not questions:
        print("Error: instrument has no questions", file=sys.stderr)
        return 1

    client = LLMClient()

    # Run all panelists in parallel via the orchestrator
    panelist_results, _registry = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model=model,
        system_prompt_fn=_persona_system_prompt,
        question_prompt_fn=_build_question_prompt,
    )

    # Build output results and aggregate usage
    results: list[dict[str, Any]] = []
    total_usage = ZERO_USAGE

    for pr in panelist_results:
        pricing, is_estimated = lookup_pricing(model)
        persona_cost = estimate_cost(pr.usage, pricing)
        results.append({
            "persona": pr.persona_name,
            "responses": pr.responses,
            "usage": pr.usage.to_dict(),
            "cost": persona_cost.format_usd(),
            "error": pr.error,
        })
        total_usage = total_usage + pr.usage

    # Output results
    if fmt is OutputFormat.TEXT:
        for r in results:
            print(f"\n{'='*60}")
            print(f"Persona: {r['persona']}")
            print(f"{'='*60}")
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
            for resp in r["responses"]:
                prefix = "  [follow-up] " if resp.get("follow_up") else "  "
                print(f"{prefix}Q: {resp['question']}")
                print(f"{prefix}A: {resp['response']}")
                print()
            print(f"  Cost: {r['cost']}")

        # Total cost summary
        pricing, is_estimated = lookup_pricing(model)
        total_cost = estimate_cost(total_usage, pricing)
        print(f"\n{'='*60}")
        print(format_summary("Total", total_usage, total_cost,
                             model=model, is_estimated=is_estimated))
    else:
        pricing, is_estimated = lookup_pricing(model)
        total_cost = estimate_cost(total_usage, pricing)
        emit(fmt, message="Panel complete", extra={
            "results": results,
            "total_usage": total_usage.to_dict(),
            "total_cost": total_cost.format_usd(),
            "model": model,
            "persona_count": len(personas),
            "question_count": len(questions),
        })

    return 0


def handle_mcp_serve(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Start the MCP server on stdio transport."""
    from synth_panel.mcp.server import serve
    serve()
    return 0


def handle_login(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Start an authentication flow."""
    # TODO: implement OAuth flow
    emit(fmt, message="[stub] Login not yet implemented.")
    return 0


def handle_logout(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Clear saved authentication credentials."""
    # TODO: implement credential clearing
    emit(fmt, message="[stub] Logout not yet implemented.")
    return 0
