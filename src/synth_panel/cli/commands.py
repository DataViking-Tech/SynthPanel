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
from synth_panel.cost import ZERO_USAGE, TokenUsage, estimate_cost, format_summary, lookup_pricing
from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.llm.client import LLMClient
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.prompts import build_question_prompt, persona_system_prompt
from synth_panel.runtime import AgentRuntime
from synth_panel.synthesis import synthesize_panel

# Preference chain for --model default (sp-f4t). Each entry is
# (env var that signals provider availability, model alias or name).
# Walked in order: the first provider with credentials wins. The alias
# side prefers the provider's workhorse model so the user gets a
# sensible default without guessing which canonical ID to type.
_DEFAULT_MODEL_PREFERENCE: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "sonnet"),
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("GEMINI_API_KEY", "gemini-2.5-flash"),
    ("GOOGLE_API_KEY", "gemini-2.5-flash"),
    ("XAI_API_KEY", "grok-3"),
]


def _resolve_default_model() -> tuple[str, str | None]:
    """Pick a default model alias based on which API keys are present.

    Returns ``(model_alias, source_env_var)``. If no provider credentials
    are available the fallback is still ``"sonnet"`` (the previous
    hard-coded default) so the LLM client's own credential check can
    emit its canonical error message.
    """
    import os

    for env_var, alias in _DEFAULT_MODEL_PREFERENCE:
        if os.environ.get(env_var, "").strip():
            return alias, env_var
    return "sonnet", None


def _resolve_model(args: argparse.Namespace) -> str:
    """Return the model alias from CLI args.

    When ``--model`` is not supplied, walk the preference chain in
    :data:`_DEFAULT_MODEL_PREFERENCE` and pick the first provider with
    credentials present in the environment. Falls back to ``"sonnet"``
    if nothing is set so the LLM client's missing-credentials error is
    still the one the user sees.
    """
    if args.model:
        return args.model
    alias, _source = _resolve_default_model()
    return alias


def _announce_default_model(args: argparse.Namespace) -> None:
    """Print the auto-selected default model to stderr.

    sp-f4t: the CLI help claims "Default: best available" but the
    selection was previously silent — an operator running a panel
    against the auto-picked model could not tell *which* provider was
    being used. Surface the selection loudly so the user can ^C and
    re-run with ``--model`` if the pick is wrong.
    """
    if getattr(args, "model", None):
        return
    alias, source = _resolve_default_model()
    if source:
        print(
            f"[synth-panel] --model not specified; using '{alias}' (detected {source}). Override with --model <name>.",
            file=sys.stderr,
        )
    else:
        print(
            f"[synth-panel] --model not specified and no provider API "
            f"keys detected; falling back to '{alias}'. Set one of: "
            f"{', '.join(env for env, _ in _DEFAULT_MODEL_PREFERENCE)}.",
            file=sys.stderr,
        )


def handle_prompt(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run a single non-interactive prompt and exit."""
    prompt_text = " ".join(args.text)
    _announce_default_model(args)
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
        print(format_summary("Cost", summary.usage, cost, model=model, is_estimated=is_estimated), file=sys.stderr)
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


def _load_instrument(path_or_name: str) -> Instrument:
    """Load an instrument from a file path *or* an installed pack name.

    Supports v1 (flat), v2 (linear rounds), v3 (branching). When
    ``path_or_name`` is not an existing file, falls back to
    :func:`load_instrument_pack` which looks under
    ``$SYNTH_PANEL_DATA_DIR/packs/instruments/``. The unified resolver
    is what makes ``panel run --instrument <name>`` accept pack names
    in addition to file paths.
    """
    if Path(path_or_name).exists():
        data = _load_yaml(path_or_name)
    else:
        from synth_panel.mcp.data import load_instrument_pack

        try:
            data = load_instrument_pack(path_or_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Instrument not found as file or installed pack: {path_or_name}") from exc

    if isinstance(data, dict) and "instrument" in data:
        raw = data["instrument"]
    elif isinstance(data, dict) and ("questions" in data or "rounds" in data):
        raw = data
    else:
        raise ValueError("Invalid instrument: expected 'instrument', 'questions', or 'rounds' key")
    raw.setdefault("version", 1)
    return parse_instrument(raw)


def _collect_template_vars(args: argparse.Namespace) -> dict[str, str]:
    """Merge ``--vars-file`` and repeated ``--var KEY=VALUE`` flags into a
    single substitution context.

    CLI ``--var`` entries override matching keys from ``--vars-file``
    so ad-hoc overrides win on conflict. Both sources stringify their
    values because the underlying template engine operates on strings.
    Raises :class:`ValueError` on malformed input so the caller can
    surface a friendly message.
    """
    merged: dict[str, str] = {}

    vars_file = getattr(args, "vars_file", None)
    if vars_file:
        try:
            data = _load_yaml(vars_file)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"--vars-file must be a YAML mapping, got {type(data).__name__}")
        for k, v in data.items():
            merged[str(k)] = _stringify_var(v)

    for entry in getattr(args, "vars", None) or []:
        if "=" not in entry:
            raise ValueError(f"--var expects KEY=VALUE, got: {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--var has empty key: {entry!r}")
        merged[key] = value
    return merged


def _stringify_var(value: Any) -> str:
    """Convert a YAML-loaded var value to a string suitable for substitution.

    Lists and tuples are joined with ``", "`` so ``candidates: [A, B, C]``
    in a vars file renders the way a human would write it inline. Other
    scalars go through ``str()``.
    """
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _apply_vars_to_instrument(instrument: Instrument, template_vars: dict[str, str]) -> None:
    """Substitute ``template_vars`` into every round's question text.

    Mutates the instrument in place: each ``Round.questions`` list is
    replaced with its rendered form. The rendering is routed through
    :func:`synth_panel.templates.render_questions` so we inherit safe-
    failure behavior (unknown keys render as literal ``{placeholder}``
    rather than raising).
    """
    from synth_panel.templates import render_questions

    for rnd in instrument.rounds:
        rnd.questions = render_questions(rnd.questions, template_vars)


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
        raise ValueError(f"Schema is not a valid file path or JSON string: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object, got {type(schema).__name__}")
    return schema


def handle_panel_run(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run a panel: load personas + instrument, run panelists in parallel."""
    _announce_default_model(args)
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

    # Surface parser warnings (e.g. unreachable rounds) to stderr — these
    # are non-fatal but the user should see them before any LLM call fires.
    for w in instrument.warnings:
        print(f"warning: {w}", file=sys.stderr)

    # ── sp-1hb: variable substitution ──────────────────────────────────
    # Bundled instrument packs ship with {candidates}, {theme_0}, and
    # similar placeholders so they can be reused as research templates.
    # Without --var the user had to hand-edit YAML; this resolves the
    # variables at load time so the panelists see substituted text.
    try:
        template_vars = _collect_template_vars(args)
    except ValueError as exc:
        print(f"Error parsing --var / --vars-file: {exc}", file=sys.stderr)
        return 1
    if template_vars:
        _apply_vars_to_instrument(instrument, template_vars)

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
        results.append(
            {
                "persona": pr.persona_name,
                "responses": pr.responses,
                "usage": pr.usage.to_dict(),
                "cost": persona_cost.format_usd(),
                "error": pr.error,
            }
        )
        panelist_usage = panelist_usage + pr.usage

    # ── Failure analysis (sp-2hg) ──────────────────────────────────────
    # Count panelist-question pairs and determine how many errored. A pair
    # is considered errored if its response dict is flagged ``error: True``
    # OR if the panelist wrapper itself failed (no responses recorded).
    failure_stats = _analyze_failures(panelist_results, questions)
    strict = getattr(args, "strict", False)
    threshold = getattr(args, "failure_threshold", 0.5)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.5
    run_invalid = failure_stats["total_pairs"] > 0 and failure_stats["failure_rate"] > threshold
    strict_violation = strict and failure_stats["errored_pairs"] > 0

    # Synthesis step (unless --no-synthesis)
    skip_synthesis = getattr(args, "no_synthesis", False)
    if run_invalid or strict_violation:
        skip_synthesis = True  # don't synthesize garbage
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
    banner = _build_invalid_banner(failure_stats, threshold, strict=strict, strict_violation=strict_violation)

    if fmt is OutputFormat.TEXT:
        if banner:
            print(banner, file=sys.stderr)
        path_line = _format_path(_degenerate_path(instrument))
        if path_line:
            print(f"path: {path_line}")
        for r in results:
            print(f"\n{'=' * 60}")
            print(f"Persona: {r['persona']}")
            print(f"{'=' * 60}")
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
            print(f"\n{'=' * 60}")
            print("SYNTHESIS")
            print(f"{'=' * 60}")
            print(f"\n  Summary: {synthesis_dict['summary']}")
            if synthesis_dict.get("themes"):
                print("\n  Themes:")
                for t in synthesis_dict["themes"]:
                    print(f"    - {t}")
            if synthesis_dict.get("agreements"):
                print("\n  Agreements:")
                for a in synthesis_dict["agreements"]:
                    print(f"    - {a}")
            if synthesis_dict.get("disagreements"):
                print("\n  Disagreements:")
                for d in synthesis_dict["disagreements"]:
                    print(f"    - {d}")
            if synthesis_dict.get("surprises"):
                print("\n  Surprises:")
                for s in synthesis_dict["surprises"]:
                    print(f"    - {s}")
            print(f"\n  Recommendation: {synthesis_dict['recommendation']}")
            print(f"  Synthesis cost: {synthesis_dict['cost']}")

        # Cost summaries
        print(f"\n{'=' * 60}")
        print(
            format_summary("Panelist cost", panelist_usage, panelist_cost_est, model=model, is_estimated=is_estimated)
        )
        if synthesis_result:
            _synth_pricing, synth_est = lookup_pricing(synthesis_result.model)
            print(
                format_summary(
                    "Synthesis cost",
                    synthesis_result.usage,
                    synthesis_result.cost,
                    model=synthesis_result.model,
                    is_estimated=synth_est,
                )
            )
        print(format_summary("Total", total_usage, total_cost_est, model=model, is_estimated=is_estimated))
        if total_cost_est.total_cost == 0 and failure_stats["errored_pairs"] > 0:
            print(
                "  \u26a0\ufe0f  no-cost = no-data (every request errored; "
                "cost line reflects that no tokens were billed)",
                file=sys.stderr,
            )
        if banner:
            # Repeat the banner at the bottom so a scrolled terminal still
            # surfaces it — this is the critical fix for sp-2hg.
            print(banner, file=sys.stderr)
    else:
        legacy = getattr(args, "legacy_output", False)
        if legacy:
            sys.stderr.write(
                "DeprecationWarning: --legacy-output emits the flat "
                "single-round shape and will be removed in 0.6.0. Migrate "
                "consumers to the rounds-shaped output.\n"
            )
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
        else:
            extra = _build_rounds_shape(
                instrument=instrument,
                results=results,
                synthesis_dict=synthesis_dict,
                panelist_cost=panelist_cost_est,
                total_usage=total_usage,
                total_cost=total_cost_est,
                model=model,
                persona_count=len(personas),
                question_count=len(questions),
            )
        # Surface failure stats + run validity in structured output so
        # downstream consumers (MCP, CI) can gate on it without parsing text.
        extra["failure_stats"] = failure_stats
        extra["run_invalid"] = bool(run_invalid or strict_violation)
        if run_invalid or strict_violation:
            extra["message"] = banner.replace("\n", " ").strip() if banner else "PANEL RUN INVALID"
        emit(fmt, message=extra.get("message", "Panel complete"), extra=extra)

    # sp-2hg: exit non-zero when the run is invalid so automation (CI,
    # refinery, wrapper scripts) can detect silent-failure scenarios.
    if strict_violation:
        return 3
    if run_invalid:
        return 2
    return 0


def _analyze_failures(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count errored panelist-question pairs across a run.

    A pair errors if:
      - the panelist-level wrapper threw (``pr.error`` set), in which
        case *every* authored question is counted as an error, even
        those the panelist never reached, OR
      - an individual response dict was flagged ``error: True`` (the
        orchestrator sets this when a per-question call raises).

    Returns a dict with ``total_pairs``, ``errored_pairs``,
    ``failure_rate`` (0-1), ``failed_panelists`` (panelist-level
    failures) and ``errored_personas`` (names of affected personas).
    """
    total_questions = len(questions) if questions else 0
    total_pairs = 0
    errored_pairs = 0
    failed_panelists = 0
    errored_personas: set[str] = set()

    for pr in panelist_results:
        if getattr(pr, "error", None):
            # Whole panelist exploded — every authored question counts as
            # an error for the purposes of the failure-rate calculation.
            failed_panelists += 1
            errored_pairs += total_questions
            total_pairs += total_questions
            errored_personas.add(getattr(pr, "persona_name", "unknown"))
            continue

        pair_count = 0
        err_count = 0
        for resp in getattr(pr, "responses", []) or []:
            if isinstance(resp, dict) and resp.get("follow_up"):
                # Follow-ups are not counted as primary QA pairs — they
                # are second-order and would double-count toward the rate.
                continue
            pair_count += 1
            if isinstance(resp, dict) and resp.get("error"):
                err_count += 1
        # If the panelist never produced any primary responses (e.g. a
        # structured-output path that bailed before recording), treat the
        # shortfall as errored against the authored question count.
        if total_questions and pair_count < total_questions:
            shortfall = total_questions - pair_count
            err_count += shortfall
            pair_count += shortfall
        total_pairs += pair_count
        errored_pairs += err_count
        if err_count > 0:
            errored_personas.add(getattr(pr, "persona_name", "unknown"))

    failure_rate = (errored_pairs / total_pairs) if total_pairs > 0 else 0.0
    return {
        "total_pairs": total_pairs,
        "errored_pairs": errored_pairs,
        "failure_rate": failure_rate,
        "failed_panelists": failed_panelists,
        "errored_personas": sorted(errored_personas),
    }


def _build_invalid_banner(
    stats: dict[str, Any],
    threshold: float,
    *,
    strict: bool,
    strict_violation: bool,
) -> str:
    """Render the fatal banner printed to stderr on an invalid run.

    Returns an empty string when the run is valid. The banner is
    intentionally loud — the entire point of sp-2hg is that a user
    skimming output cannot miss that the panel was not executed
    successfully.
    """
    total = stats["total_pairs"]
    errored = stats["errored_pairs"]
    if total == 0:
        return ""
    rate = stats["failure_rate"]
    over_threshold = rate > threshold
    if not (over_threshold or strict_violation):
        return ""

    bar = "!" * 70
    lines = [bar]
    if strict_violation and not over_threshold:
        lines.append(f"PANEL RUN INVALID (--strict): {errored}/{total} panelist-question pairs errored.")
    else:
        lines.append(
            f"PANEL RUN INVALID: {errored}/{total} panelist-question pairs"
            f" errored ({rate:.0%} > threshold {threshold:.0%})."
        )
    lines.append("No synthesis was performed — the run produced no usable data.")
    if stats.get("failed_panelists"):
        lines.append(f"  {stats['failed_panelists']} panelist(s) failed wholesale (no responses recorded).")
    if stats.get("errored_personas"):
        shown = ", ".join(stats["errored_personas"][:4])
        extra = len(stats["errored_personas"]) - 4
        if extra > 0:
            shown += f", +{extra} more"
        lines.append(f"  Affected personas: {shown}")
    lines.append("Check provider status (e.g. rate-limit / 503), run with a different")
    lines.append("--model, or re-run once the upstream provider is healthy.")
    lines.append(bar)
    return "\n".join(lines)


def _build_rounds_shape(
    *,
    instrument: Instrument,
    results: list[dict[str, Any]],
    synthesis_dict: dict[str, Any] | None,
    panelist_cost: Any,
    total_usage: TokenUsage,
    total_cost: Any,
    model: str,
    persona_count: int,
    question_count: int,
) -> dict[str, Any]:
    """Build the rounds-shaped panel output payload.

    Single-round runs surface as one round entry whose ``name`` is the
    instrument's only round (``"default"`` for v1). Multi-round/branching
    runs use this same shape with one entry per executed round; that
    wiring lives in F3-A. Per-round ``synthesis`` is ``null`` for the
    single-round case — the final synthesis goes at the top level.
    """
    round_name = instrument.rounds[0].name if instrument.rounds else "default"
    return {
        "rounds": [
            {
                "name": round_name,
                "results": results,
                "synthesis": None,
            }
        ],
        "path": [],
        "warnings": list(getattr(instrument, "warnings", []) or []),
        "synthesis": synthesis_dict,
        "panelist_cost": panelist_cost.format_usd(),
        "total_usage": total_usage.to_dict(),
        "total_cost": total_cost.format_usd(),
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
    }


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


def handle_pack_show(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Print a saved persona pack's YAML to stdout.

    sp-oem: API parity alias for ``instruments show``. Internally
    delegates to :func:`handle_pack_export` with ``output=None`` so
    behaviour stays identical to ``pack export <id>`` (no file arg).
    """
    args.output = None
    return handle_pack_export(args, fmt)


def handle_mcp_serve(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Start the MCP server on stdio transport."""
    from synth_panel.mcp.server import serve

    serve()
    return 0


# ---------------------------------------------------------------------------
# Instruments subcommands (sp-xsu)
# ---------------------------------------------------------------------------


def handle_instruments_list(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """List installed instrument packs."""
    from synth_panel.mcp.data import list_instrument_packs

    packs = list_instrument_packs()

    if fmt is OutputFormat.TEXT:
        if not packs:
            print("No instrument packs installed.")
            print("Install one with: synth-panel instruments install <file.yaml>")
        else:
            for p in packs:
                version = f" v{p['version']}" if p.get("version") else ""
                desc = f" — {p['description']}" if p.get("description") else ""
                print(f"  {p['id']}{version}{desc}")
    else:
        emit(fmt, message="Instrument packs", extra={"packs": packs})
    return 0


def handle_instruments_install(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Install an instrument pack from a YAML file (or bundled name)."""
    from synth_panel.mcp.data import save_instrument_pack

    source = args.source
    if not Path(source).exists():
        print(
            f"Error: source file not found: {source}\n"
            f"(bundled instrument packs are not yet shipped — pass a "
            f"YAML file path)",
            file=sys.stderr,
        )
        return 1

    try:
        data = _load_yaml(source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading file: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: instrument pack file must be a YAML mapping", file=sys.stderr)
        return 1

    # Validate by parsing — bad instruments must fail before install.
    raw = data.get("instrument", data)
    raw.setdefault("version", 1)
    try:
        parse_instrument(raw)
    except InstrumentError as exc:
        print(f"Error: instrument validation failed: {exc}", file=sys.stderr)
        return 1

    pack_name = args.name or data.get("name") or Path(source).stem
    meta = save_instrument_pack(pack_name, data)

    if fmt is OutputFormat.TEXT:
        print(f"Installed instrument pack '{meta['id']}' -> {meta['path']}")
    else:
        emit(fmt, message="Instrument pack installed", extra=meta)
    return 0


def handle_instruments_show(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Print an installed instrument pack's contents."""
    from synth_panel.mcp.data import load_instrument_pack

    try:
        data = load_instrument_pack(args.name)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if fmt is OutputFormat.TEXT:
        print(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), end="")
    else:
        emit(fmt, message="Instrument pack", extra={"name": args.name, "data": data})
    return 0


def handle_instruments_graph(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Render the round DAG for an instrument file or installed pack name."""
    try:
        instrument = _load_instrument(args.source)
    except (FileNotFoundError, ValueError, InstrumentError) as exc:
        print(f"Error loading instrument: {exc}", file=sys.stderr)
        return 1

    for w in instrument.warnings:
        print(f"warning: {w}", file=sys.stderr)

    rendered = (
        _render_mermaid(instrument) if getattr(args, "format", "text") == "mermaid" else _render_text_dag(instrument)
    )

    if fmt is OutputFormat.TEXT:
        print(rendered)
    else:
        emit(
            fmt,
            message="Instrument graph",
            extra={
                "rounds": [r.name for r in instrument.rounds],
                "graph": rendered,
                "warnings": instrument.warnings,
            },
        )
    return 0


def _render_text_dag(instrument: Instrument) -> str:
    """Plain-text node + edge listing of an instrument's round DAG."""

    lines: list[str] = []
    lines.append(f"# instrument v{instrument.version}")
    lines.append(f"# {len(instrument.rounds)} round(s)")
    lines.append("")
    for r in instrument.rounds:
        lines.append(f"[{r.name}] ({len(r.questions)} question(s))")
        if r.depends_on:
            lines.append(f"  depends_on: {r.depends_on}")
        if r.route_when:
            for entry in r.route_when:
                if "if" in entry:
                    pred = entry["if"]
                    lines.append(
                        f"  if {pred.get('field')} {pred.get('op')} {pred.get('value')!r} -> {entry.get('goto')}"
                    )
                elif "else" in entry:
                    lines.append(f"  else -> {entry['else']}")
    return "\n".join(lines)


def _render_mermaid(instrument: Instrument) -> str:
    """Mermaid flowchart of an instrument's round DAG."""
    from synth_panel.instrument import END_SENTINEL

    lines = ["flowchart TD"]
    for r in instrument.rounds:
        lines.append(f"    {r.name}([{r.name}])")

    # Collect __end__ targets so we can declare the terminal node once
    # if any route lands there. The terminal sentinel renders as a
    # double-circle stadium so it's visually distinct from regular rounds.
    has_end_target = False

    for r in instrument.rounds:
        if r.depends_on:
            lines.append(f"    {r.depends_on} --> {r.name}")
        if r.route_when:
            for entry in r.route_when:
                if "if" in entry:
                    pred = entry["if"]
                    label = f"{pred.get('field')} {pred.get('op')} {pred.get('value')}"
                    target = entry.get("goto")
                    if target == END_SENTINEL:
                        has_end_target = True
                    lines.append(f"    {r.name} -->|{label}| {target}")
                elif "else" in entry:
                    target = entry["else"]
                    if target == END_SENTINEL:
                        has_end_target = True
                    lines.append(f"    {r.name} -->|else| {target}")

    if has_end_target:
        lines.append(f"    {END_SENTINEL}(((end)))")
    return "\n".join(lines)


def _degenerate_path(instrument: Instrument) -> list[dict[str, Any]]:
    """Synthesize a path log for non-branching runs.

    Single-round and linear v2 runs do not currently flow through
    ``run_multi_round_panel`` (they use the legacy single-round path),
    so they have no real router-emitted path. F3-E still wants the
    path line rendered for those cases — fall back to a linear walk
    over the instrument's declared rounds, ending in ``__end__``.
    """
    from synth_panel.instrument import END_SENTINEL

    rounds = list(instrument.rounds)
    if not rounds:
        return []
    path: list[dict[str, Any]] = []
    for i, r in enumerate(rounds):
        nxt = rounds[i + 1].name if i + 1 < len(rounds) else END_SENTINEL
        path.append({"round": r.name, "branch": "linear", "next": nxt})
    return path


def _format_path(path: list[dict[str, Any]]) -> str:
    """Render an executed branching path as a single human line.

    Example: ``exploration -> probe[themes contains pricing] -> probe_pricing -> validation``
    """
    if not path:
        return ""
    parts: list[str] = []
    for i, entry in enumerate(path):
        if i == 0:
            parts.append(entry["round"])
        branch = entry.get("branch", "")
        nxt = entry.get("next", "")
        if branch and branch not in ("linear",):
            parts.append(f"[{branch}] -> {nxt}")
        else:
            parts.append(f"-> {nxt}")
    return " ".join(parts)
