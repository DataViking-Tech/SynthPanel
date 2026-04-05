"""Subcommand handlers for synth-panel CLI.

Each handler receives parsed args and output format, returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cost import ZERO_USAGE, TokenUsage, UsageTracker, estimate_cost, format_summary, lookup_pricing
from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import TextBlock
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.prompts import build_question_prompt, persona_system_prompt
from synth_panel.runtime import AgentRuntime
from synth_panel.synthesis import synthesize_panel


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


def _load_instrument(path: str) -> Instrument:
    """Load an instrument/survey from a YAML file.

    Supports both v1 (flat questions) and v2 (multi-round) formats.
    Returns a validated :class:`Instrument` with normalized round definitions.
    """
    data = _load_yaml(path)
    if isinstance(data, dict) and "instrument" in data:
        raw = data["instrument"]
    elif isinstance(data, dict) and ("questions" in data or "rounds" in data):
        raw = data
    else:
        raise ValueError("Invalid instrument file: expected 'instrument', 'questions', or 'rounds' key")
    raw.setdefault("version", 1)
    return parse_instrument(raw)


def _load_schema(value: str) -> dict[str, Any]:
    """Load a JSON Schema from a file path or inline JSON string."""
    p = Path(value)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    # Try parsing as inline JSON
    try:
        schema = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Schema is not a valid file path or JSON string: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object, got {type(schema).__name__}")
    return schema


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
    except (FileNotFoundError, ValueError, InstrumentError) as exc:
        print(f"Error loading instrument: {exc}", file=sys.stderr)
        return 1

    questions = instrument.questions
    if not questions:
        print("Error: instrument has no questions", file=sys.stderr)
        return 1

    # Load optional response schema
    response_schema: dict[str, Any] | None = None
    if getattr(args, "schema", None):
        try:
            response_schema = _load_schema(args.schema)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error loading schema: {exc}", file=sys.stderr)
            return 1

    client = LLMClient()

    # Run all panelists in parallel via the orchestrator
    panelist_results, _registry, _sessions = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model=model,
        system_prompt_fn=persona_system_prompt,
        question_prompt_fn=build_question_prompt,
        response_schema=response_schema,
    )

    # Build output results and aggregate panelist usage
    results: list[dict[str, Any]] = []
    panelist_usage = ZERO_USAGE

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
        panelist_usage = panelist_usage + pr.usage

    # Synthesis step (unless --no-synthesis)
    skip_synthesis = getattr(args, "no_synthesis", False)
    synthesis_result = None

    if not skip_synthesis:
        synthesis_model = getattr(args, "synthesis_model", None)
        custom_prompt = getattr(args, "synthesis_prompt", None)
        try:
            synthesis_result = synthesize_panel(
                client,
                panelist_results,
                questions,
                model=synthesis_model,
                custom_prompt=custom_prompt,
            )
        except Exception as exc:
            print(f"Warning: synthesis failed: {exc}", file=sys.stderr)

    # Compute costs
    pricing, is_estimated = lookup_pricing(model)
    panelist_cost_est = estimate_cost(panelist_usage, pricing)

    if synthesis_result:
        total_usage = panelist_usage + synthesis_result.usage
        total_cost_est = panelist_cost_est + synthesis_result.cost
    else:
        total_usage = panelist_usage
        total_cost_est = panelist_cost_est

    synthesis_dict = synthesis_result.to_dict() if synthesis_result else None

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

        # Synthesis output
        if synthesis_dict:
            print(f"\n{'='*60}")
            print("SYNTHESIS")
            print(f"{'='*60}")
            print(f"\n  Summary: {synthesis_dict['summary']}")
            if synthesis_dict.get("themes"):
                print(f"\n  Themes:")
                for t in synthesis_dict["themes"]:
                    print(f"    - {t}")
            if synthesis_dict.get("agreements"):
                print(f"\n  Agreements:")
                for a in synthesis_dict["agreements"]:
                    print(f"    - {a}")
            if synthesis_dict.get("disagreements"):
                print(f"\n  Disagreements:")
                for d in synthesis_dict["disagreements"]:
                    print(f"    - {d}")
            if synthesis_dict.get("surprises"):
                print(f"\n  Surprises:")
                for s in synthesis_dict["surprises"]:
                    print(f"    - {s}")
            print(f"\n  Recommendation: {synthesis_dict['recommendation']}")
            print(f"  Synthesis cost: {synthesis_dict['cost']}")

        # Cost summaries
        print(f"\n{'='*60}")
        print(format_summary("Panelist cost", panelist_usage, panelist_cost_est,
                             model=model, is_estimated=is_estimated))
        if synthesis_result:
            synth_pricing, synth_est = lookup_pricing(synthesis_result.model)
            print(format_summary("Synthesis cost", synthesis_result.usage, synthesis_result.cost,
                                 model=synthesis_result.model, is_estimated=synth_est))
        print(format_summary("Total", total_usage, total_cost_est,
                             model=model, is_estimated=is_estimated))
    else:
        extra: dict[str, Any] = {
            "results": results,
            "panelist_cost": panelist_cost_est.format_usd(),
            "synthesis": synthesis_dict,
            "total_usage": total_usage.to_dict(),
            "total_cost": total_cost_est.format_usd(),
            "model": model,
            "persona_count": len(personas),
            "question_count": len(questions),
        }
        emit(fmt, message="Panel complete", extra=extra)

    return 0


def handle_pack_list(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """List all saved persona packs."""
    from synth_panel.mcp.data import list_persona_packs

    packs = list_persona_packs()

    if fmt is OutputFormat.TEXT:
        if not packs:
            print("No persona packs found.")
        else:
            for p in packs:
                print(f"  {p['id']}  {p['name']}  ({p['persona_count']} personas)")
    else:
        emit(fmt, message="Persona packs", extra={"packs": packs})

    return 0


def handle_pack_import(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Import a persona pack from a YAML file."""
    from synth_panel.mcp.data import PackValidationError, save_persona_pack

    try:
        personas = _load_personas(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading file: {exc}", file=sys.stderr)
        return 1

    # Determine pack name: explicit flag > YAML 'name' key > filename stem
    pack_name = args.name
    if not pack_name:
        data = _load_yaml(args.file)
        if isinstance(data, dict):
            pack_name = data.get("name")
    if not pack_name:
        pack_name = Path(args.file).stem

    try:
        result = save_persona_pack(pack_name, personas, pack_id=args.pack_id)
    except PackValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1

    if fmt is OutputFormat.TEXT:
        print(f"Imported pack '{result['name']}' ({result['persona_count']} personas) as {result['id']}")
    else:
        emit(fmt, message="Pack imported", extra=result)

    return 0


def handle_pack_export(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Export a saved persona pack to stdout or a file."""
    from synth_panel.mcp.data import get_persona_pack

    try:
        pack = get_persona_pack(args.pack_id)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Build clean export data (without internal 'id' field)
    export_data = {k: v for k, v in pack.items() if k != "id"}

    content = yaml.dump(export_data, default_flow_style=False)

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        if fmt is OutputFormat.TEXT:
            print(f"Exported pack '{pack.get('name', args.pack_id)}' to {args.output}")
        else:
            emit(fmt, message="Pack exported", extra={"path": args.output, "pack_id": args.pack_id})
    else:
        if fmt is OutputFormat.TEXT:
            print(content, end="")
        else:
            emit(fmt, message="Pack exported", extra={"pack_id": args.pack_id, "data": export_data})

    return 0


def handle_mcp_serve(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Start the MCP server on stdio transport."""
    from synth_panel.mcp.server import serve
    serve()
    return 0


