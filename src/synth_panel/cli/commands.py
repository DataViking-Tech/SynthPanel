"""Subcommand handlers for synthpanel CLI.

Each handler receives parsed args and output format, returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cost import ZERO_USAGE, TokenUsage, estimate_cost, format_summary, lookup_pricing
from synth_panel.credentials import has_credential
from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.llm.client import LLMClient
from synth_panel.metadata import PanelTimer, build_metadata
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.perturbation import generate_panel_variants
from synth_panel.prompts import (
    build_question_prompt,
    load_prompt_template,
    persona_system_prompt,
    persona_system_prompt_from_template,
)
from synth_panel.runtime import AgentRuntime
from synth_panel.synthesis import synthesize_panel

logger = logging.getLogger(__name__)

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
    ("OPENROUTER_API_KEY", "openrouter/auto"),
]


def _resolve_default_model() -> tuple[str, str | None]:
    """Pick a default model alias based on which API keys are present.

    Returns ``(model_alias, source_env_var)``. Consults both the
    environment and the on-disk credential store so ``synthpanel login``
    is enough to get a working default. If nothing is available the
    fallback is still ``"sonnet"`` so the LLM client's own credential
    check emits the canonical error message.
    """
    for env_var, alias in _DEFAULT_MODEL_PREFERENCE:
        if has_credential(env_var):
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


def parse_models_spec(spec: str) -> list[tuple[str, float]]:
    """Parse a --models spec string into [(model, weight)] pairs.

    Accepted formats:

    * **Weighted** (per-persona assignment): ``"haiku:0.5,gemini:0.5"``
    * **Ensemble** (no weights): ``"haiku,sonnet"`` — each model gets weight 1.0

    When no entry contains ``:``, the spec is treated as an ensemble list.
    Weights must be positive and are normalized to sum to 1.0 by the caller.
    """
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    if not entries:
        raise ValueError("Empty --models spec")

    # Detect ensemble vs weighted: ensemble if *no* entry contains ':'
    is_ensemble = all(":" not in e for e in entries)

    pairs: list[tuple[str, float]] = []
    for entry in entries:
        if is_ensemble:
            if not entry:
                raise ValueError("Empty model name in spec")
            pairs.append((entry, 1.0))
        else:
            if ":" not in entry:
                raise ValueError(f"Invalid model spec entry '{entry}': expected 'model:weight' (e.g. 'haiku:0.5')")
            model_part, weight_str = entry.rsplit(":", 1)
            model_part = model_part.strip()
            weight_str = weight_str.strip()
            if not model_part:
                raise ValueError(f"Empty model name in spec entry '{entry}'")
            try:
                weight = float(weight_str)
            except ValueError as exc:
                raise ValueError(f"Invalid weight '{weight_str}' in spec entry '{entry}': must be a number") from exc
            if weight <= 0:
                raise ValueError(f"Weight must be positive, got {weight} for model '{model_part}'")
            pairs.append((model_part, weight))
    return pairs


def is_ensemble_spec(spec: str) -> bool:
    """Return True when *spec* is a weight-free ensemble list (e.g. ``"haiku,sonnet"``)."""
    return all(":" not in e for e in spec.split(",") if e.strip())


def assign_models_to_personas(
    personas: list[dict[str, Any]],
    model_spec: list[tuple[str, float]],
    default_model: str,
) -> dict[str, str]:
    """Assign models to personas based on weighted spec and YAML overrides.

    Resolution order: persona YAML 'model' field > weighted assignment > default.
    Returns a dict mapping persona name → resolved model.
    """
    # Personas with YAML model overrides are pre-assigned
    needs_assignment: list[str] = []
    result: dict[str, str] = {}
    for p in personas:
        name = p.get("name", "Anonymous")
        if p.get("model"):
            result[name] = p["model"]
        else:
            needs_assignment.append(name)

    if not needs_assignment:
        return result

    # Normalize weights
    total_weight = sum(w for _, w in model_spec)
    normalized = [(m, w / total_weight) for m, w in model_spec]

    # Assign proportionally: distribute personas across models by weight
    remaining = len(needs_assignment)
    assigned_idx = 0
    for i, (model, weight) in enumerate(normalized):
        if i == len(normalized) - 1:
            # Last model gets all remaining to avoid rounding gaps
            count = remaining
        else:
            count = max(0, round(weight * len(needs_assignment)))
            remaining -= count
        for _ in range(count):
            if assigned_idx < len(needs_assignment):
                result[needs_assignment[assigned_idx]] = model
                assigned_idx += 1

    return result


def _load_profile(args: argparse.Namespace) -> tuple[Any, int | None]:
    """Load and apply a profile from --profile or --config.

    Returns ``(profile_or_None, error_exit_code_or_None)``. On error,
    the message is already printed to stderr and the caller should return
    the exit code.
    """
    from synth_panel.profiles import (
        Profile,
        apply_profile_to_args,
        load_profile_by_name,
        load_profile_from_path,
    )

    profile_name = getattr(args, "profile", None)
    config_path = getattr(args, "config", None)

    if profile_name and config_path:
        print("Error: --profile and --config are mutually exclusive.", file=sys.stderr)
        return None, 1

    profile: Profile | None = None
    if profile_name:
        try:
            profile = load_profile_by_name(profile_name)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading profile: {exc}", file=sys.stderr)
            return None, 1
    elif config_path:
        try:
            profile = load_profile_from_path(config_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading config: {exc}", file=sys.stderr)
            return None, 1

    if profile is not None:
        apply_profile_to_args(profile, args)

    return profile, None


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
        logger.info(
            "--model not specified; using '%s' (detected %s). Override with --model <name>.",
            alias,
            source,
        )
    else:
        logger.warning(
            "--model not specified and no provider API keys detected; falling back to '%s'. Set one of: %s.",
            alias,
            ", ".join(env for env, _ in _DEFAULT_MODEL_PREFERENCE),
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
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


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


def _merge_persona_lists(base: list[dict[str, Any]], merge_paths: list[str]) -> list[dict[str, Any]]:
    """Append personas from each merge path onto *base*.

    Files are loaded in order and their personas appended. If a later
    persona shares a ``name`` with an earlier one, the later entry
    replaces the earlier one in place (order of first occurrence is
    preserved). Personas without a ``name`` are always appended — we
    cannot safely dedupe them.
    """
    by_name: dict[str, int] = {}
    merged: list[dict[str, Any]] = []
    for p in base:
        name = p.get("name") if isinstance(p, dict) else None
        if isinstance(name, str) and name:
            by_name[name] = len(merged)
        merged.append(p)
    for path in merge_paths:
        for p in _load_personas(path):
            name = p.get("name") if isinstance(p, dict) else None
            if isinstance(name, str) and name and name in by_name:
                merged[by_name[name]] = p
            else:
                if isinstance(name, str) and name:
                    by_name[name] = len(merged)
                merged.append(p)
    return merged


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
    # ── sp-prof: profile loading ──────────────────────────────────────
    # Load profile defaults before anything else so CLI flags can override.
    profile, profile_err = _load_profile(args)
    if profile_err is not None:
        return profile_err

    # Validate mutual exclusivity of --model and --models
    has_model = getattr(args, "model", None)
    has_models = getattr(args, "models", None)
    if has_model and has_models:
        print("Error: --model and --models are mutually exclusive.", file=sys.stderr)
        return 1

    if not has_models:
        _announce_default_model(args)
    model = _resolve_model(args)

    try:
        personas = _load_personas(args.personas)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading personas: {exc}", file=sys.stderr)
        return 1

    # sp-on4: --personas-merge appends personas from additional files onto
    # the base --personas pack. When a persona name collides, the later
    # entry wins so callers can override pack defaults with a follow-on file.
    merge_paths = getattr(args, "personas_merge", None) or []
    if merge_paths:
        try:
            personas = _merge_persona_lists(personas, merge_paths)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading --personas-merge: {exc}", file=sys.stderr)
            return 1

    try:
        instrument = _load_instrument(args.instrument)
    except (FileNotFoundError, ValueError, InstrumentError) as exc:
        print(f"Error loading instrument: {exc}", file=sys.stderr)
        return 1

    # Surface parser warnings (e.g. unreachable rounds) to stderr — these
    # are non-fatal but the user should see them before any LLM call fires.
    for w in instrument.warnings:
        logger.warning("instrument: %s", w)

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

    # Load optional extraction schema (--extract-schema)
    extract_schema: dict[str, Any] | None = None
    if getattr(args, "extract_schema", None):
        try:
            extract_schema = _load_schema(args.extract_schema)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error loading extract-schema: {exc}", file=sys.stderr)
            return 1

    # Load optional prompt template (--prompt-template)
    prompt_template_path = getattr(args, "prompt_template", None)
    prompt_template: str | None = None
    if prompt_template_path:
        try:
            prompt_template = load_prompt_template(prompt_template_path)
        except FileNotFoundError as exc:
            print(f"Error loading prompt template: {exc}", file=sys.stderr)
            return 1

    # Build the system prompt function
    if prompt_template is not None:

        def system_prompt_fn(persona: dict) -> str:
            return persona_system_prompt_from_template(persona, prompt_template)  # type: ignore[arg-type]
    else:
        system_prompt_fn = persona_system_prompt

    # Generation parameters
    temperature: float | None = getattr(args, "temperature", None)
    top_p: float | None = getattr(args, "top_p", None)

    # ── sp-blend: multi-model ensemble ───────────────────────────────────
    # Parse --models spec and build per-persona model assignments.
    # Resolution order: persona YAML 'model' > --models weighted > --model.
    blend_mode = getattr(args, "blend", False)
    persona_models: dict[str, str] | None = None
    model_spec: list[tuple[str, float]] | None = None
    ensemble_mode = has_models and is_ensemble_spec(has_models)

    if has_models:
        try:
            model_spec = parse_models_spec(has_models)
        except ValueError as exc:
            print(f"Error parsing --models: {exc}", file=sys.stderr)
            return 1
        if not ensemble_mode and not blend_mode:
            persona_models = assign_models_to_personas(personas, model_spec, model)
        # Use the first model in the spec as the "primary" for cost fallback
        if not has_model:
            model = model_spec[0][0]
    elif not has_models:
        # Even without --models, respect per-persona YAML model overrides
        yaml_overrides = {p.get("name", "Anonymous"): p["model"] for p in personas if p.get("model")}
        if yaml_overrides:
            persona_models = yaml_overrides

    # Validate --blend requires --models
    if blend_mode and not has_models:
        print("Error: --blend requires --models.", file=sys.stderr)
        return 1

    import uuid as _uuid

    request_id = _uuid.uuid4().hex[:12]
    logger.info("[%s] panel run: model=%s personas=%d questions=%d", request_id, model, len(personas), len(questions))

    client = LLMClient()
    timer = PanelTimer()

    # ── sp-5on.15: variant expansion ─────────────────────────────────
    variants_k = getattr(args, "variants", None)
    if variants_k is not None:
        if variants_k < 1 or variants_k > 20:
            print("Error: --variants must be between 1 and 20.", file=sys.stderr)
            return 1
        orig_count = len(personas)
        total = variants_k * orig_count
        print(
            f"Generating {variants_k} variants for {orig_count} personas ({total} total sessions)...",
            file=sys.stderr,
        )
        variant_sets = generate_panel_variants(
            personas,
            client,
            k=variants_k,
            model=model,
        )
        personas = [v.persona for vs in variant_sets for v in vs.variants]
        print(
            f"Variant expansion complete: {len(personas)} variant personas ready.",
            file=sys.stderr,
        )

    # ── Ensemble mode: run with each model, compare across models ────────
    if ensemble_mode:
        from synth_panel.ensemble import ensemble_run

        ensemble_models = [m for m, _w in model_spec]
        ens_result = ensemble_run(
            personas,
            questions,
            ensemble_models,
            client,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=build_question_prompt,
            response_schema=response_schema,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
        )
        timer.stop()
        output = {
            "per_model_results": {
                mr.model: [{"persona": pr.persona_name, "responses": pr.responses} for pr in mr.panelist_results]
                for mr in ens_result.model_results
            },
            "cost_breakdown": ens_result.per_model_cost,
            "models": ens_result.models,
            "total_usage": ens_result.total_usage.to_dict(),
        }
        emit(fmt, message="Ensemble complete", extra=output)
        return 0

    # Run all panelists in parallel via the orchestrator
    blend_result = None  # populated only when --blend is active
    if blend_mode and model_spec:
        # ── Blend mode: run full panel once per model, then blend ────
        from synth_panel.ensemble import blend_distributions, ensemble_run

        ensemble_models = [m for m, _w in model_spec]
        ensemble = ensemble_run(
            personas,
            questions,
            ensemble_models,
            client,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=build_question_prompt,
            response_schema=response_schema,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
        )
        # Build weights dict from the model spec
        blend_weights = {m: w for m, w in model_spec}
        blend_result = blend_distributions(ensemble, weights=blend_weights, questions=questions)

        # Flatten all panelist results across models for output + synthesis
        panelist_results = [pr for mr in ensemble.model_results for pr in mr.panelist_results]
    else:
        panelist_results, _registry, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model=model,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=build_question_prompt,
            response_schema=response_schema,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
            persona_models=persona_models,
        )

    # Build output results and aggregate panelist usage
    results: list[dict[str, Any]] = []
    panelist_usage = ZERO_USAGE

    for pr in panelist_results:
        # Use per-panelist model for accurate cost tracking
        pr_model = pr.model or model
        pricing, is_estimated = lookup_pricing(pr_model)
        persona_cost = estimate_cost(pr.usage, pricing)
        result_dict: dict[str, Any] = {
            "persona": pr.persona_name,
            "responses": pr.responses,
            "usage": pr.usage.to_dict(),
            "cost": persona_cost.format_usd(),
            "error": pr.error,
        }
        if pr.model:
            result_dict["model"] = pr.model
        results.append(result_dict)
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

    # Compute panelist costs (needed before synthesis for cost estimate)
    pricing, is_estimated = lookup_pricing(model)
    panelist_cost_est = estimate_cost(panelist_usage, pricing)

    # Synthesis step (unless --no-synthesis)
    skip_synthesis = getattr(args, "no_synthesis", False)
    if run_invalid or strict_violation:
        skip_synthesis = True  # don't synthesize garbage
    synthesis_result = None

    if not skip_synthesis:
        synthesis_model = getattr(args, "synthesis_model", None)
        custom_prompt = getattr(args, "synthesis_prompt", None)
        synthesis_temperature = getattr(args, "synthesis_temperature", None)
        # synthesis_temperature overrides panelist temperature for synthesis
        effective_synth_temp = synthesis_temperature if synthesis_temperature is not None else temperature
        try:
            synthesis_result = synthesize_panel(
                client,
                panelist_results,
                questions,
                model=synthesis_model,
                panelist_model=model,
                custom_prompt=custom_prompt,
                panelist_cost=panelist_cost_est,
                temperature=effective_synth_temp,
                top_p=top_p,
            )
        except Exception as exc:
            logger.warning("synthesis failed: %s", exc)

    if synthesis_result:
        total_usage = panelist_usage + synthesis_result.usage
        total_cost_est = panelist_cost_est + synthesis_result.cost
    else:
        total_usage = panelist_usage
        total_cost_est = panelist_cost_est

    timer.stop()
    synthesis_dict = synthesis_result.to_dict() if synthesis_result else None

    metadata = build_metadata(
        panelist_model=model,
        synthesis_model=getattr(args, "synthesis_model", None),
        panelist_usage=panelist_usage,
        panelist_cost=panelist_cost_est,
        synthesis_usage=synthesis_result.usage if synthesis_result else None,
        synthesis_cost=synthesis_result.cost if synthesis_result else None,
        total_usage=total_usage,
        total_cost=total_cost_est,
        persona_count=len(personas),
        question_count=len(questions),
        timer=timer,
    )

    # sp-prof: inject profile info into metadata for synthbench
    if profile is not None:
        metadata["profile"] = profile.name
        metadata["profile_hash"] = profile.config_hash()

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

        # Blend output
        if blend_result:
            print(f"\n{'=' * 60}")
            print("BLENDED DISTRIBUTIONS")
            print(f"{'=' * 60}")
            print(f"  Models: {', '.join(blend_result.models)}")
            print(f"  Weights: {', '.join(f'{m}={w:.2f}' for m, w in blend_result.weights.items())}")
            for bq in blend_result.questions:
                print(f"\n  Q{bq.question_index + 1}: {bq.question_text}")
                print(f"  Responses: {bq.response_count}")
                if bq.distribution:
                    sorted_dist = sorted(bq.distribution.items(), key=lambda x: x[1], reverse=True)
                    for option, prob in sorted_dist:
                        print(f"    {option}: {prob:.1%}")

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
        extra: dict[str, Any] = _build_rounds_shape(
            instrument=instrument,
            results=results,
            synthesis_dict=synthesis_dict,
            panelist_cost=panelist_cost_est,
            panelist_usage=panelist_usage,
            total_usage=total_usage,
            total_cost=total_cost_est,
            model=model,
            persona_count=len(personas),
            question_count=len(questions),
            metadata=metadata,
        )
        extra["parameters"] = _build_params_metadata(args, temperature, top_p)
        # Surface failure stats + run validity in structured output so
        # downstream consumers (MCP, CI) can gate on it without parsing text.
        extra["failure_stats"] = failure_stats
        extra["run_invalid"] = bool(run_invalid or strict_violation)
        # Include blend distributions when --blend is active
        if blend_result:
            extra["blend"] = {
                "models": blend_result.models,
                "weights": blend_result.weights,
                "questions": [
                    {
                        "index": bq.question_index,
                        "question": bq.question_text,
                        "distribution": bq.distribution,
                        "per_model": bq.per_model,
                        "response_count": bq.response_count,
                    }
                    for bq in blend_result.questions
                ],
            }
        if run_invalid or strict_violation:
            extra["message"] = banner.replace("\n", " ").strip() if banner else "PANEL RUN INVALID"
        emit(fmt, message=extra.get("message", "Panel complete"), extra=extra)

    # ── sp-7vp: auto-save results with --save ─────────────────────────
    if getattr(args, "save", False):
        from synth_panel.mcp.data import save_panel_result

        # Determine instrument name (pack name or filename stem)
        inst_name: str | None = None
        inst_arg = getattr(args, "instrument", None)
        if inst_arg:
            inst_path = Path(inst_arg)
            if inst_path.exists():
                inst_name = inst_path.stem
            else:
                inst_name = inst_arg  # pack name

        all_models: list[str] | None = None
        if persona_models:
            all_models = sorted(set(persona_models.values()))
        elif model_spec:
            all_models = [m for m, _w in model_spec]

        result_id = save_panel_result(
            results=results,
            model=model,
            total_usage=total_usage.to_dict(),
            total_cost=total_cost_est.format_usd(),
            persona_count=len(personas),
            question_count=len(questions),
            instrument_name=inst_name,
            models=all_models,
        )
        print(f"Result saved: {result_id}", file=sys.stderr)

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
    panelist_usage: TokenUsage | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the rounds-shaped panel output payload.

    Single-round runs surface as one round entry whose ``name`` is the
    instrument's only round (``"default"`` for v1). Multi-round/branching
    runs use this same shape with one entry per executed round; that
    wiring lives in F3-A. Per-round ``synthesis`` is ``null`` for the
    single-round case — the final synthesis goes at the top level.
    """
    round_name = instrument.rounds[0].name if instrument.rounds else "default"
    result: dict[str, Any] = {
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
        "panelist_usage": (panelist_usage.to_dict() if panelist_usage is not None else None),
        "total_usage": total_usage.to_dict(),
        "total_cost": total_cost.format_usd(),
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
    }
    if metadata is not None:
        result["metadata"] = metadata
    return result


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


def handle_pack_generate(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Generate a persona pack using an LLM."""
    from synth_panel.llm.models import InputMessage, TextBlock
    from synth_panel.mcp.data import PackValidationError, save_persona_pack
    from synth_panel.structured.output import StructuredOutputConfig, StructuredOutputEngine

    count = args.count
    if count < 1 or count > 50:
        print("Error: --count must be between 1 and 50.", file=sys.stderr)
        return 1

    model = _resolve_model(args)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name."},
                        "age": {"type": "integer", "description": "Age in years."},
                        "occupation": {"type": "string", "description": "Job title or role."},
                        "background": {
                            "type": "string",
                            "description": "One-paragraph professional and personal background.",
                        },
                        "personality_traits": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 personality traits relevant to their product usage.",
                        },
                    },
                    "required": ["name", "age", "occupation", "background", "personality_traits"],
                },
            },
        },
        "required": ["personas"],
    }

    prompt = (
        f"Generate exactly {count} diverse, realistic personas for a synthetic focus group.\n\n"
        f"Product/service: {args.product}\n"
        f"Target audience: {args.audience}\n\n"
        "Requirements:\n"
        f"- Generate exactly {count} personas\n"
        "- Each persona needs: name, age, occupation, background (one paragraph), "
        "and 3-5 personality_traits\n"
        "- Personas should be diverse in age, occupation, background, and perspective\n"
        "- Backgrounds should be specific and relevant to the product/audience context\n"
        "- Personality traits should reflect how they'd engage with the product\n"
        "- Names should reflect diverse cultural backgrounds"
    )

    client = LLMClient()
    engine = StructuredOutputEngine(client)
    config = StructuredOutputConfig(
        schema=schema,
        tool_name="generate_personas",
        tool_description="Generate a list of diverse personas for a synthetic focus group.",
    )

    try:
        result = engine.extract(
            model=model,
            max_tokens=4096,
            messages=[InputMessage(role="user", content=[TextBlock(text=prompt)])],
            config=config,
            temperature=1.0,
        )
    except Exception as exc:
        print(f"Error calling LLM: {exc}", file=sys.stderr)
        return 1

    if result.is_fallback:
        print(f"Error: LLM did not produce valid structured output: {result.error}", file=sys.stderr)
        return 1

    personas = result.data.get("personas", [])
    if not personas:
        print("Error: LLM returned no personas.", file=sys.stderr)
        return 1

    pack_name = args.name or f"{args.product[:60]} personas"

    try:
        saved = save_persona_pack(pack_name, personas, pack_id=args.pack_id)
    except PackValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1

    if fmt is OutputFormat.TEXT:
        print(f"Generated pack '{saved['name']}' ({saved['persona_count']} personas) as {saved['id']}")
        print(f"Saved to: {saved['path']}")
    else:
        emit(fmt, message="Pack generated", extra=saved)

    return 0


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
            print("Install one with: synthpanel instruments install <file.yaml>")
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
    from synth_panel.mcp.data import load_instrument_pack, save_instrument_pack

    source = args.source
    data: dict | None = None

    if Path(source).exists():
        try:
            data = _load_yaml(source)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading file: {exc}", file=sys.stderr)
            return 1
    else:
        # Try loading as a bundled pack name.
        try:
            data = load_instrument_pack(source)
        except FileNotFoundError:
            from synth_panel.mcp.data import list_instrument_packs

            bundled = [p["id"] for p in list_instrument_packs() if p.get("source") == "bundled"]
            hint = ", ".join(bundled) if bundled else "(none)"
            print(
                f"Error: '{source}' is not a file path or known bundled pack.\n"
                f"Available bundled packs: {hint}\n"
                f"Or pass a YAML file path.",
                file=sys.stderr,
            )
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
        logger.warning("instrument: %s", w)

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


def _build_params_metadata(
    args: argparse.Namespace,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    """Build a metadata dict of generation parameters for output."""
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    synthesis_temperature = getattr(args, "synthesis_temperature", None)
    if synthesis_temperature is not None:
        params["synthesis_temperature"] = synthesis_temperature
    prompt_template = getattr(args, "prompt_template", None)
    if prompt_template is not None:
        params["prompt_template"] = prompt_template
    return params


def handle_panel_synthesize(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Re-synthesize a saved panel result (sp-5on.5).

    Loads a saved panel result, reconstructs the panelist results and
    question list, invokes :func:`synthesize_panel` with the requested
    model/prompt, and writes a sidecar file
    ``<result_id>.synthesis-<ts>.json`` next to the original result.
    """
    from datetime import datetime, timezone

    from synth_panel.orchestrator import PanelistResult

    result_ref = args.result
    path = Path(result_ref)
    if path.exists() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("id", path.stem)
    else:
        from synth_panel.mcp.data import get_panel_result

        try:
            data = get_panel_result(result_ref)
        except FileNotFoundError:
            print(f"Error: panel result not found: {result_ref}", file=sys.stderr)
            return 1

    saved_results = data.get("results") or []
    if not saved_results:
        print(f"Error: no panelist results found in {result_ref}", file=sys.stderr)
        return 1

    panelist_results: list[PanelistResult] = []
    for r in saved_results:
        usage_dict = r.get("usage") or {}
        usage = TokenUsage.from_dict(usage_dict) if usage_dict else ZERO_USAGE
        panelist_results.append(
            PanelistResult(
                persona_name=r.get("persona", "Anonymous"),
                responses=r.get("responses") or [],
                usage=usage,
                error=r.get("error"),
                model=r.get("model"),
            )
        )

    saved_questions = data.get("questions")
    if saved_questions:
        questions = saved_questions
    else:
        seen: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for pr in panelist_results:
            for resp in pr.responses:
                if resp.get("follow_up"):
                    continue
                q_text = resp.get("question", "")
                if q_text and q_text not in seen_texts:
                    seen.append({"text": q_text})
                    seen_texts.add(q_text)
        questions = seen

    panelist_model = data.get("model")
    client = LLMClient()
    try:
        synth = synthesize_panel(
            client,
            panelist_results,
            questions,
            model=getattr(args, "synthesis_model", None),
            panelist_model=panelist_model,
            custom_prompt=getattr(args, "synthesis_prompt", None),
        )
    except Exception as exc:
        print(f"Error: synthesis failed: {exc}", file=sys.stderr)
        return 1

    synthesis_dict = synth.to_dict()

    from synth_panel.mcp.data import save_panel_synthesis

    source_id = data.get("id") or path.stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sidecar_name = save_panel_synthesis(
        source_id,
        ts,
        {
            "source_result_id": source_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "synthesis_model": synth.model,
            "synthesis_prompt_override": getattr(args, "synthesis_prompt", None) is not None,
            "synthesis": synthesis_dict,
        },
    )

    if fmt is OutputFormat.TEXT:
        print(f"{'=' * 60}")
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
        print(f"\nSaved synthesis: {sidecar_name}", file=sys.stderr)
    else:
        emit(
            fmt,
            message="Panel synthesis complete",
            extra={
                "source_result_id": source_id,
                "synthesis": synthesis_dict,
                "saved_as": sidecar_name,
            },
        )

    return 0


def handle_analyze(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run statistical analysis on a saved panel result."""
    import json as _json

    from synth_panel.analyze import (
        analysis_to_dict,
        analyze_panel_result,
        format_csv,
        format_text,
    )

    result_ref = args.result
    output_mode = getattr(args, "output", "text")

    # Load the panel result — either by ID or file path
    path = Path(result_ref)
    if path.exists() and path.suffix == ".json":
        data = _json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("id", path.stem)
    else:
        from synth_panel.mcp.data import get_panel_result

        try:
            data = get_panel_result(result_ref)
        except FileNotFoundError:
            print(f"Error: panel result not found: {result_ref}", file=sys.stderr)
            return 1

    analysis = analyze_panel_result(data)

    if output_mode == "json":
        print(_json.dumps(analysis_to_dict(analysis), indent=2))
    elif output_mode == "csv":
        print(format_csv(analysis))
    else:
        print(format_text(analysis))

    return 0


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


# ---------------------------------------------------------------------------
# Credential commands (sp-lve)
# ---------------------------------------------------------------------------


_PROVIDER_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _read_api_key_interactive() -> str:
    """Read an API key from stdin.

    Uses :func:`getpass.getpass` when attached to a TTY so the key
    doesn't echo; falls back to a plain ``input`` read when stdin is
    piped (CI, scripts) so ``echo sk-... | synthpanel login`` still
    works.
    """
    import getpass

    if sys.stdin.isatty():
        return getpass.getpass("API key: ").strip()
    return sys.stdin.readline().strip()


def handle_login(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Store an API key for the selected provider.

    Reads the key from ``--api-key`` or stdin (hidden on TTY) and writes
    it to the on-disk credential store. Validates that the provider
    name maps to a recognised env var so typos don't silently persist.
    """
    from synth_panel.credentials import (
        KNOWN_CREDENTIAL_ENV_VARS,
        PROVIDER_LABELS,
        save_credential,
    )

    provider = getattr(args, "provider", None) or "anthropic"
    env_var = _PROVIDER_ENV_VAR.get(provider)
    if env_var is None or env_var not in KNOWN_CREDENTIAL_ENV_VARS:
        msg = f"Unknown provider: {provider!r}"
        if fmt is OutputFormat.TEXT:
            print(msg, file=sys.stderr)
        else:
            emit(fmt, message=msg, extra={"error": "unknown_provider"})
        return 2

    key = (getattr(args, "api_key", None) or "").strip()
    if not key:
        key = _read_api_key_interactive()
    if not key:
        msg = "No API key provided."
        if fmt is OutputFormat.TEXT:
            print(msg, file=sys.stderr)
        else:
            emit(fmt, message=msg, extra={"error": "empty_key"})
        return 2

    path = save_credential(env_var, key)
    label = PROVIDER_LABELS.get(env_var, provider)
    if fmt is OutputFormat.TEXT:
        print(f"Stored {label} key ({env_var}) in {path}")
    else:
        emit(
            fmt,
            message="credential_stored",
            extra={"provider": provider, "env_var": env_var, "path": str(path)},
        )
    return 0


def handle_logout(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Remove one or all stored API keys."""
    from synth_panel.credentials import (
        KNOWN_CREDENTIAL_ENV_VARS,
        delete_credential,
        load_credentials,
    )

    provider = getattr(args, "provider", None) or "anthropic"

    if provider == "all":
        stored = list(load_credentials().keys())
        removed = [env for env in stored if delete_credential(env)]
        if fmt is OutputFormat.TEXT:
            if removed:
                print(f"Removed {len(removed)} stored credential(s): {', '.join(removed)}")
            else:
                print("No stored credentials to remove.")
        else:
            emit(fmt, message="credentials_removed", extra={"removed": removed})
        return 0

    env_var = _PROVIDER_ENV_VAR.get(provider)
    if env_var is None or env_var not in KNOWN_CREDENTIAL_ENV_VARS:
        msg = f"Unknown provider: {provider!r}"
        if fmt is OutputFormat.TEXT:
            print(msg, file=sys.stderr)
        else:
            emit(fmt, message=msg, extra={"error": "unknown_provider"})
        return 2

    removed = delete_credential(env_var)
    if fmt is OutputFormat.TEXT:
        if removed:
            print(f"Removed stored credential: {env_var}")
        else:
            print(f"No stored credential for {env_var}")
    else:
        emit(
            fmt,
            message="credential_removed" if removed else "credential_absent",
            extra={"provider": provider, "env_var": env_var, "removed": removed},
        )
    return 0


def handle_whoami(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Report which providers have credentials available (env or stored)."""
    import os

    from synth_panel.credentials import (
        KNOWN_CREDENTIAL_ENV_VARS,
        PROVIDER_LABELS,
        credentials_path,
        load_credentials,
    )

    stored = load_credentials()
    rows: list[dict[str, Any]] = []
    for env_var in KNOWN_CREDENTIAL_ENV_VARS:
        env_set = bool(os.environ.get(env_var, "").strip())
        stored_set = env_var in stored
        source: str | None = None
        if env_set:
            source = "env"
        elif stored_set:
            source = "stored"
        rows.append(
            {
                "provider": PROVIDER_LABELS.get(env_var, env_var),
                "env_var": env_var,
                "available": env_set or stored_set,
                "source": source,
            }
        )

    if fmt is OutputFormat.TEXT:
        any_available = any(r["available"] for r in rows)
        if not any_available:
            print("No provider credentials found.")
            print("Run `synthpanel login` or export an API key env var.")
        else:
            print(f"Credential store: {credentials_path()}")
            for row in rows:
                if not row["available"]:
                    continue
                print(f"  [{row['source']:<6}] {row['env_var']:<20} — {row['provider']}")
        return 0

    emit(
        fmt,
        message="credentials_status",
        extra={"credentials_path": str(credentials_path()), "providers": rows},
    )
    return 0
