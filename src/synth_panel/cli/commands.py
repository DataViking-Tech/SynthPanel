"""Subcommand handlers for synthpanel CLI.

Each handler receives parsed args and output format, returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from synth_panel._runners import (
    build_synthesis_error_payload,
    detect_synthesis_context_overflow,
    detect_total_failure,
    format_synthesis_overflow_message,
    format_total_failure_message,
)
from synth_panel.cli.output import OutputFormat, emit
from synth_panel.convergence import (
    DEFAULT_CHECK_EVERY,
    DEFAULT_EPSILON,
    DEFAULT_M_CONSECUTIVE,
    DEFAULT_MIN_N,
    ConvergenceTracker,
    SynthbenchUnavailableError,
    identify_tracked_questions,
    load_synthbench_baseline,
)
from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    TokenUsage,
    aggregate_per_model,
    build_cost_fallback_warnings,
    estimate_cost,
    format_summary,
    lookup_pricing,
)
from synth_panel.credentials import has_credential
from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.llm.client import LLMClient
from synth_panel.metadata import PanelTimer, build_metadata
from synth_panel.orchestrator import PanelistResult, run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.perturbation import generate_panel_variants
from synth_panel.prompts import (
    build_question_prompt,
    load_prompt_template,
    persona_system_prompt,
    persona_system_prompt_from_template,
)
from synth_panel.runtime import AgentRuntime
from synth_panel.synthesis import (
    STRATEGY_MAP_REDUCE,
    STRATEGY_SINGLE,
    select_strategy,
    synthesize_panel,
    synthesize_panel_mapreduce,
)
from synth_panel.templates import find_unresolved_in_questions

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
    Weights must be positive. Order is preserved as given (the assignment
    algorithm walks the list left-to-right; see
    :func:`assign_models_to_personas` for the full algorithm).

    Validation:

    * ``weight <= 0`` → ``ValueError``
    * non-numeric weight → ``ValueError``
    * empty spec or empty model name → ``ValueError``
    * mixing weighted and unweighted entries (e.g. ``"haiku,sonnet:0.5"``)
      → ``ValueError``

    Sum-of-weights is **not** enforced here — specs like ``"a:2,b:3"`` parse
    cleanly and are normalized by :func:`assign_models_to_personas`. Callers
    that want to warn on non-unit sums should inspect the parsed pairs.
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
) -> tuple[dict[str, str], list[str]]:
    """Assign models to personas using Hamilton's largest-remainder method.

    Resolution order: persona YAML 'model' field > weighted assignment > default.

    Algorithm (fully deterministic — same inputs produce the same output):

    1. **YAML override pass.** Walk ``personas`` in list order. Any persona
       with a ``model`` field keeps that model and is removed from the
       assignment pool. This override takes precedence over the ``--models``
       spec unconditionally.
    2. **Normalize weights.** Compute ``total = sum(weights)`` and divide each
       weight by it. Absolute weights don't matter, only ratios — so
       ``"a:0.5,b:0.5"``, ``"a:1,b:1"`` and ``"a:50,b:50"`` all behave the
       same.
    3. **Largest-remainder allocation.** Floor each ``weight * N`` quota,
       then hand out leftover seats in descending order of fractional
       remainder. Ties are broken deterministically by position in the spec.
    4. **Min-1 guarantee** (sp-27rz). When ``N >= len(model_spec)``, every
       model in ``model_spec`` is guaranteed at least one persona: any
       zero-count model pulls a seat from the currently largest bucket.
       The prior round-based allocation could silently hand the last model
       0 personas, dropping it from per_model_results without any warning.
       When there are fewer personas than models, affected models still
       end up at zero and a warning is appended to the returned list so
       callers can surface it to the user.

    Edge cases:

    * **Empty pool** (every persona has a YAML ``model`` override):
      ``model_spec`` is effectively ignored and the YAML map is returned.
    * **Weights that don't sum to 1.0** (e.g. ``"a:2,b:3"``): normalized
      internally, so the split still works. The CLI warns about this so
      the user knows the ratio they intended is what got applied.
    * **Zero weight**: rejected upstream by :func:`parse_models_spec`.
    * **``default_model``**: currently unused by the algorithm — YAML
      override and the ``--models`` spec between them cover every persona.
      Kept in the signature so callers can pass the CLI's fallback for
      future use.

    Returns a ``(assignments, warnings)`` tuple.
    """
    warnings: list[str] = []
    needs_assignment: list[str] = []
    result: dict[str, str] = {}
    for p in personas:
        name = p.get("name", "Anonymous")
        if p.get("model"):
            result[name] = p["model"]
        else:
            needs_assignment.append(name)

    if not needs_assignment:
        return result, warnings

    n = len(needs_assignment)
    total_weight = sum(w for _, w in model_spec)
    normalized = [(m, w / total_weight) for m, w in model_spec]

    # Hamilton's method (largest-remainder): floor each quota, then hand out
    # leftover seats in descending order of fractional remainder. Deterministic
    # ties broken by position in the spec.
    quotas = [w * n for _, w in normalized]
    counts = [int(q) for q in quotas]
    remainders = [q - int(q) for q in quotas]
    seats_left = n - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: (-remainders[i], i))
    for i in order[:seats_left]:
        counts[i] += 1

    # Min-1 guarantee: when there are enough personas to cover every model,
    # ensure no model is left at zero by taking one seat from the currently
    # largest bucket. Preserves proportions as closely as possible while
    # eliminating the silent-drop bug.
    if n >= len(model_spec):
        for zi, c in enumerate(counts):
            if c != 0:
                continue
            donor = max(
                range(len(counts)),
                key=lambda i: (counts[i], -i),
            )
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[zi] += 1

    # Any remaining zeros mean personas < models — surface loudly.
    for (model, _), c in zip(normalized, counts):
        if c == 0:
            warnings.append(
                f"model_allocation: '{model}' received 0 of {n} personas "
                f"(fewer personas than weighted models); it will be absent "
                f"from per_model_results"
            )

    idx = 0
    for (model, _), c in zip(normalized, counts):
        for _ in range(c):
            result[needs_assignment[idx]] = model
            idx += 1

    return result, warnings


# sp-zdul: display helpers for --models weighted assignment
_WEIGHT_SUM_TOLERANCE = 0.02


def check_weight_sum(
    model_spec: list[tuple[str, float]],
    tolerance: float = _WEIGHT_SUM_TOLERANCE,
) -> tuple[float, bool]:
    """Return ``(sum_of_weights, is_close_to_one)``.

    Weights are normalized internally by :func:`assign_models_to_personas`,
    but a user who wrote ``"a:0.3,b:0.3,c:0.3"`` probably meant ``0.33`` each
    and is relying on equal split. Warning loudly when the sum deviates
    from 1.0 by more than ``tolerance`` catches these typos.
    """
    total = sum(w for _, w in model_spec)
    return total, abs(total - 1.0) <= tolerance


def format_assignment_breakdown(persona_models: dict[str, str]) -> str:
    """Format a per-persona → model breakdown for pre-run display.

    Example output::

        Model assignment:
          Maya Chen       → haiku
          Derek Washington → gpt-mini
          ...
        Totals: haiku=2, gpt-mini=2, gemini-flash=2
    """
    if not persona_models:
        return ""
    name_width = max(len(n) for n in persona_models)
    lines = ["Model assignment:"]
    for name, mdl in persona_models.items():
        lines.append(f"  {name.ljust(name_width)} → {mdl}")
    counts: dict[str, int] = {}
    for mdl in persona_models.values():
        counts[mdl] = counts.get(mdl, 0) + 1
    totals = ", ".join(f"{m}={c}" for m, c in sorted(counts.items()))
    lines.append(f"Totals: {totals}")
    return "\n".join(lines)


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
    replaced with its rendered form. Callers in ``handle_panel_run``
    invoke :func:`find_unresolved_in_questions` *before* this function
    so any missing key aborts the run (sp-6yi); any ``{placeholder}``
    that survives here is either dynamic (resolved downstream) or
    explicitly opted in via ``--allow-unresolved``.
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


def _emit_dry_run_preview(
    *,
    personas: list[dict[str, Any]],
    instrument: Instrument,
    system_prompt_fn,
    model: str,
    fmt: OutputFormat,
) -> None:
    """Print the fully substituted panel inputs without any LLM call.

    Shows each question as it will appear to the LLM (after --var
    substitution, which has already been applied to the instrument),
    plus persona/question counts and a rough input-token estimate.
    """
    persona_count = len(personas)
    questions = instrument.questions
    question_count = len(questions)

    system_prompt_chars = sum(len(system_prompt_fn(p)) for p in personas)
    question_chars = sum(len(build_question_prompt(q)) for q in questions)
    follow_up_chars = 0
    for q in questions:
        follow_ups = q.get("follow_ups") if isinstance(q, dict) else None
        if not follow_ups:
            continue
        for fu in follow_ups:
            if isinstance(fu, dict):
                follow_up_chars += len(str(fu.get("text", "")))
            else:
                follow_up_chars += len(str(fu))
    total_chars = system_prompt_chars + persona_count * (question_chars + follow_up_chars)
    estimated_input_tokens = max(1, total_chars // 4)

    if fmt is OutputFormat.TEXT:
        print("DRY RUN — no LLM calls will be made", file=sys.stderr)
        print(f"Model: {model}", file=sys.stderr)
        print(f"Personas: {persona_count}", file=sys.stderr)
        if instrument.is_multi_round:
            print(f"Questions: {question_count} across {len(instrument.rounds)} rounds", file=sys.stderr)
        else:
            print(f"Questions: {question_count}", file=sys.stderr)
        print("", file=sys.stderr)

        for round_ in instrument.rounds:
            if instrument.is_multi_round:
                print(f"Round: {round_.name}", file=sys.stderr)
            for idx, q in enumerate(round_.questions, start=1):
                prompt_text = build_question_prompt(q)
                prefix = f"  {idx}." if instrument.is_multi_round else f"{idx}."
                print(f"{prefix} {prompt_text}", file=sys.stderr)
                follow_ups = q.get("follow_ups") if isinstance(q, dict) else None
                if follow_ups:
                    for fu in follow_ups:
                        fu_text = fu.get("text", "") if isinstance(fu, dict) else str(fu)
                        indent = "     " if instrument.is_multi_round else "   "
                        print(f"{indent}↳ (follow-up) {fu_text}", file=sys.stderr)
            if instrument.is_multi_round:
                print("", file=sys.stderr)

        print(
            f"Estimated input tokens: ~{estimated_input_tokens:,} "
            f"({persona_count} personas x {question_count} questions, ~4 chars/token)",
            file=sys.stderr,
        )
        return

    preview: dict[str, Any] = {
        "dry_run": True,
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
        "estimated_input_tokens": estimated_input_tokens,
        "rounds": [
            {
                "name": r.name,
                "questions": [build_question_prompt(q) for q in r.questions],
            }
            for r in instrument.rounds
        ],
    }
    emit(fmt, message="Dry-run preview", extra=preview)


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

    # ── sp-6yi: fail fast on unsubstituted placeholders ────────────────
    # If any question still contains a literal {placeholder} whose key is
    # not in template_vars, abort before burning LLM calls on a panel the
    # personas will correctly refuse to answer. --allow-unresolved opts
    # out for the rare case where the braces are genuinely intentional.
    unresolved = find_unresolved_in_questions(
        [q for rnd in instrument.rounds for q in rnd.questions],
        template_vars,
    )
    if unresolved:
        keys = ", ".join(unresolved)
        example = unresolved[0]
        if getattr(args, "allow_unresolved", False):
            print(
                f"Warning: instrument has unsubstituted placeholders ({keys}). "
                f"Proceeding because --allow-unresolved was passed; personas "
                f"will see the literal braces.",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: instrument has unsubstituted placeholders: {keys}. "
                f"Pass a value for each (e.g. --var {example}=...) or "
                f"--allow-unresolved to proceed anyway.",
                file=sys.stderr,
            )
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
    # Resolved before --dry-run so the preview shows the same breakdown
    # the real run would.
    blend_mode = getattr(args, "blend", False)
    persona_models: dict[str, str] | None = None
    model_spec: list[tuple[str, float]] | None = None
    ensemble_mode = has_models and is_ensemble_spec(has_models)
    assignment_warnings: list[str] = []

    if has_models:
        try:
            model_spec = parse_models_spec(has_models)
        except ValueError as exc:
            print(f"Error parsing --models: {exc}", file=sys.stderr)
            return 1
        # sp-zdul: warn if weighted sum is far from 1.0 — the user likely
        # intended an exact-ratio split and a typo (e.g. "0.3,0.3,0.3")
        # silently reweights their panel.
        if not ensemble_mode:
            weight_sum, is_close = check_weight_sum(model_spec)
            if not is_close:
                print(
                    f"Warning: --models weights sum to {weight_sum:.3f}, not 1.0. "
                    f"Weights will be normalized, preserving the ratio.",
                    file=sys.stderr,
                )
        if not ensemble_mode and not blend_mode:
            persona_models, assignment_warnings = assign_models_to_personas(personas, model_spec, model)
            for w in assignment_warnings:
                logger.warning("model allocation: %s", w)
                print(f"Warning: {w}", file=sys.stderr)
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

    # sp-zdul: surface the deterministic persona→model assignment before
    # any LLM calls happen. An operator can ^C if the split looks wrong
    # (e.g. unbalanced due to rounding on small panels, or YAML overrides
    # they forgot about).
    if persona_models:
        print(format_assignment_breakdown(persona_models), file=sys.stderr)

    # ── sp-x8g: --dry-run preview ────────────────────────────────────────
    # Short-circuit before any LLM-invoking code (variant expansion,
    # ensemble, blend, orchestrator). Shows the user what each question
    # will look like after --var substitution + prompt templating, plus
    # a rough input-token estimate, without spending any tokens.
    if getattr(args, "dry_run", False):
        _emit_dry_run_preview(
            personas=personas,
            instrument=instrument,
            system_prompt_fn=system_prompt_fn,
            model=model,
            fmt=fmt,
        )
        return 0

    import uuid as _uuid

    request_id = _uuid.uuid4().hex[:12]
    logger.info("[%s] panel run: model=%s personas=%d questions=%d", request_id, model, len(personas), len(questions))

    max_concurrent = getattr(args, "max_concurrent", None)
    rate_limit_rps = getattr(args, "rate_limit_rps", None)
    client = LLMClient(
        max_concurrent=max_concurrent,
        rate_limit_rps=rate_limit_rps,
    )
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
        from synth_panel.ensemble import build_ensemble_output, ensemble_run

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

        # sp-efip: fail loud if every panelist of every model failed.
        # Without this, ensemble runs with a knowingly-bad model name
        # silently returned exit 0 + an empty-but-well-shaped result.
        ensemble_panelists: list[PanelistResult] = []
        for mr in ens_result.model_results:
            ensemble_panelists.extend(mr.panelist_results)
        ensemble_failure = detect_total_failure(ensemble_panelists)
        if ensemble_failure is not None:
            msg = format_total_failure_message(ensemble_failure)
            print(_build_total_failure_banner(ensemble_failure), file=sys.stderr)
            if fmt is not OutputFormat.TEXT:
                emit(
                    fmt,
                    message=msg,
                    extra={
                        "run_invalid": True,
                        "total_failure": ensemble_failure,
                        "error": msg,
                    },
                )
            return 2

        # Per-persona usage/cost attribution (sp-hwe): pass the richer
        # ``format_panelist_result`` formatter so each result row in
        # ``per_model_results[*].results`` carries its own token counts
        # + priced cost, letting consumers answer "which model/persona
        # burned the most tokens?" without re-running the panel.
        from synth_panel._runners import format_panelist_result as _fmt_panelist

        output = build_ensemble_output(ens_result, panelist_formatter=_fmt_panelist)
        # sp-atvc: attach a metadata bundle whose cost.per_model covers
        # every ensemble model, not just the first in the spec.
        ens_per_model_meta = {mr.model: (mr.usage, mr.cost) for mr in ens_result.model_results}
        output["metadata"] = build_metadata(
            panelist_model=ensemble_models[0],
            panelist_usage=ens_result.total_usage,
            panelist_cost=ens_result.total_cost,
            total_usage=ens_result.total_usage,
            total_cost=ens_result.total_cost,
            persona_count=ens_result.persona_count,
            question_count=ens_result.question_count,
            timer=timer,
            template_vars=template_vars or None,
            panelist_per_model=ens_per_model_meta,
        )
        emit(fmt, message="Ensemble complete", extra=output)
        return 0

    # ── sp-yaru: convergence telemetry ───────────────────────────────────
    # Build the tracker when any convergence flag opts in. We always
    # respect --auto-stop / --convergence-baseline even if the user
    # forgot --convergence-check-every, by falling back to the default
    # cadence so those flags never silently no-op.
    convergence_tracker: ConvergenceTracker | None = None
    convergence_baseline_payload: dict[str, Any] | None = None
    convergence_baseline_error: str | None = None
    wants_convergence = any(
        [
            getattr(args, "convergence_check_every", None) is not None,
            getattr(args, "auto_stop", False),
            getattr(args, "convergence_log", None) is not None,
            getattr(args, "convergence_baseline", None) is not None,
            getattr(args, "convergence_eps", None) is not None,
            getattr(args, "convergence_min_n", None) is not None,
            getattr(args, "convergence_m", None) is not None,
        ]
    )
    if wants_convergence:
        tracked = identify_tracked_questions(questions)
        if not tracked:
            print(
                "Warning: --convergence-* flags set but the instrument has no "
                "bounded (Likert / yes-no / pick-one / enum) questions; "
                "convergence tracking disabled.",
                file=sys.stderr,
            )
        else:
            # Resolve baseline before kicking off the run so a missing
            # synthbench dep fails fast — we do not want to burn LLM
            # spend on a run whose report we cannot assemble.
            baseline_spec = getattr(args, "convergence_baseline", None)
            if baseline_spec:
                try:
                    convergence_baseline_payload = load_synthbench_baseline(baseline_spec)
                except SynthbenchUnavailableError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                except (ValueError, RuntimeError) as exc:
                    convergence_baseline_error = str(exc)
                    print(
                        f"Warning: could not load convergence baseline: {exc}",
                        file=sys.stderr,
                    )
            convergence_tracker = ConvergenceTracker(
                tracked,
                check_every=getattr(args, "convergence_check_every", None) or DEFAULT_CHECK_EVERY,
                epsilon=getattr(args, "convergence_eps", None) or DEFAULT_EPSILON,
                min_n=(
                    getattr(args, "convergence_min_n", None)
                    if getattr(args, "convergence_min_n", None) is not None
                    else DEFAULT_MIN_N
                ),
                m_consecutive=getattr(args, "convergence_m", None) or DEFAULT_M_CONSECUTIVE,
                auto_stop=bool(getattr(args, "auto_stop", False)),
                log_path=getattr(args, "convergence_log", None),
            )

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
<<<<<<< HEAD
            max_workers=max_concurrent,
=======
            convergence_tracker=convergence_tracker,
>>>>>>> 27d1fa6 (feat(convergence): live JSD telemetry + auto-stop for panel runs (sp-yaru))
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

    # sp-atvc: bucket panelist usage/cost by actual model so multi-model
    # runs (persona_models routing, --blend, ensemble) price each token
    # at the rate its provider charged, instead of applying the primary
    # model's rate to every bucket.
    panelist_per_model_usage, panelist_per_model_cost = aggregate_per_model(panelist_results, model)
    multi_model_run = len(panelist_per_model_usage) > 1

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

    # sp-efip: separate total-failure short-circuit. _analyze_failures
    # can report total_pairs==0 when every panelist bailed before
    # recording any response (e.g. runtime init threw) — in that case
    # the failure_rate math falls to 0 and run_invalid stays False.
    # Regardless of the pair-rate, if no panelist produced usable data
    # the run is invalid and the banner must name the failing model(s)
    # and first upstream error.
    total_failure = detect_total_failure(panelist_results)
    if total_failure is not None:
        run_invalid = True

    # sp-bjt4: detect the "polite refusal" failure mode — panelists returned
    # clean responses but uniformly flagged the question as unanswerable
    # because the source material never made it into the prompt. Without
    # this check, ``run_invalid`` stays False and the failure is only
    # narrated in the synthesis ``surprises`` field, so CI gates (which
    # key on ``run_invalid``) miss it silently.
    missing_input_stats = _detect_missing_input_refusals(panelist_results)
    missing_input_invalid = (
        missing_input_stats["considered"] > 0 and missing_input_stats["refusal_rate"] >= _MISSING_INPUT_THRESHOLD
    )
    if missing_input_invalid:
        run_invalid = True

    # Compute panelist costs (needed before synthesis for cost estimate).
    # For multi-model runs, sum the per-model costs so the total reflects
    # each provider's real rate; single-model runs keep the legacy path.
    pricing, is_estimated = lookup_pricing(model)
    if multi_model_run:
        panelist_cost_est = CostEstimate()
        for _m, _c in panelist_per_model_cost.items():
            panelist_cost_est = panelist_cost_est + _c
    else:
        panelist_cost_est = estimate_cost(panelist_usage, pricing)

    # Synthesis step (unless --no-synthesis)
    skip_synthesis = getattr(args, "no_synthesis", False)
    if run_invalid or strict_violation:
        skip_synthesis = True  # don't synthesize garbage
    synthesis_result = None
    synthesis_error_payload: dict[str, Any] | None = None

    if not skip_synthesis:
        synthesis_model = getattr(args, "synthesis_model", None)
        custom_prompt = getattr(args, "synthesis_prompt", None)
        synthesis_temperature = getattr(args, "synthesis_temperature", None)
        requested_strategy = getattr(args, "synthesis_strategy", "auto") or "auto"
        # sp-kkzz: --synthesis-prompt only applies to the single-pass call.
        # Map/reduce prompts are not overridable (yet) — warn and force single.
        if custom_prompt is not None and requested_strategy == STRATEGY_MAP_REDUCE:
            print(
                "warning: --synthesis-prompt is incompatible with "
                "--synthesis-strategy=map-reduce; forcing strategy=single. "
                "Future work: --synthesis-map-prompt / --synthesis-reduce-prompt.",
                file=sys.stderr,
            )
            requested_strategy = STRATEGY_SINGLE
        elif custom_prompt is not None and requested_strategy == "auto":
            requested_strategy = STRATEGY_SINGLE
        # synthesis_temperature overrides panelist temperature for synthesis
        effective_synth_temp = synthesis_temperature if synthesis_temperature is not None else temperature
        effective_synth_model = synthesis_model or model
        # sp-avmm: pre-flight size check — refuse to make the API call when
        # the estimated prompt exceeds the synthesis model's context window.
        # Still applies under map-reduce because each map-phase call also
        # has to fit; select_strategy will additionally pick map-reduce when
        # single-call overflow is the only way to fit.
        overflow = detect_synthesis_context_overflow(
            panelist_results,
            questions,
            synthesis_model=effective_synth_model,
            custom_prompt=custom_prompt,
        )
        if overflow is not None:
            actionable = format_synthesis_overflow_message(overflow)
            print(
                f"Error: synthesis pre-flight rejected: {actionable}",
                file=sys.stderr,
            )
            synthesis_error_payload = build_synthesis_error_payload(
                None,
                error_type="synthesis_context_overflow",
                message=actionable,
                suggested_fix=(
                    "Rerun with --synthesis-model gemini-2.5-flash-lite (1M context) "
                    "or gemini-2.5-pro (1M context), --synthesis-strategy=map-reduce, "
                    "or reduce panel size."
                ),
                diagnostic=overflow,
            )
            run_invalid = True
        else:
            resolved_strategy = select_strategy(
                requested_strategy,
                effective_synth_model,
                panelist_results,
                questions,
                prompt=custom_prompt,
            )
            try:
                if resolved_strategy == STRATEGY_MAP_REDUCE:
                    synthesis_result = synthesize_panel_mapreduce(
                        client,
                        panelist_results,
                        questions,
                        model=synthesis_model,
                        panelist_model=model,
                        panelist_cost=panelist_cost_est,
                        temperature=effective_synth_temp,
                        top_p=top_p,
                        personas=personas,
                    )
                else:
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
                # sp-avmm: fail loud — previously this was a WARN log and the
                # run exited 0 with synthesis=null. That silent-skip made
                # downstream consumers believe the run succeeded.
                from synth_panel._runners import _sanitize_api_error

                sanitized = _sanitize_api_error(exc)
                logger.error("synthesis failed: %s", sanitized)
                print(f"Error: synthesis call failed: {sanitized}", file=sys.stderr)
                synthesis_error_payload = build_synthesis_error_payload(
                    exc,
                    error_type="synthesis_api_error",
                    message=f"Synthesis call failed: {sanitized}",
                    suggested_fix=(
                        "Check provider credentials and model availability;"
                        " if context-related, rerun with a larger-context synthesis model"
                        " (e.g. gemini-2.5-flash-lite or gemini-2.5-pro)"
                        " or --synthesis-strategy=map-reduce."
                    ),
                )
                run_invalid = True

    if synthesis_result:
        total_usage = panelist_usage + synthesis_result.usage
        total_cost_est = panelist_cost_est + synthesis_result.cost
    else:
        total_usage = panelist_usage
        total_cost_est = panelist_cost_est

    timer.stop()
    synthesis_dict = synthesis_result.to_dict() if synthesis_result else None

    panelist_per_model_meta: dict[str, tuple[TokenUsage, CostEstimate]] | None = None
    if multi_model_run:
        panelist_per_model_meta = {
            _m: (panelist_per_model_usage[_m], panelist_per_model_cost[_m]) for _m in panelist_per_model_usage
        }

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
        template_vars=template_vars or None,
        panelist_per_model=panelist_per_model_meta,
    )

    # sp-kkzz: when map-reduce ran, expose the per-call cost breakdown
    # under metadata.cost.synthesis so downstream consumers can attribute
    # spend to the map vs reduce phases without re-deriving it.
    if synthesis_result is not None and synthesis_result.strategy == STRATEGY_MAP_REDUCE:
        cost_section = metadata.setdefault("cost", {})
        cost_section["synthesis"] = {
            "strategy": STRATEGY_MAP_REDUCE,
            "map_calls": synthesis_result.map_cost_breakdown or [],
            "reduce_call": synthesis_result.reduce_cost_breakdown or {},
            "total_cost_usd": round(synthesis_result.cost.total_cost, 6),
        }

    # sp-prof: inject profile info into metadata for synthbench
    if profile is not None:
        metadata["profile"] = profile.name
        metadata["profile_hash"] = profile.config_hash()

    # Output results
    banner = _build_invalid_banner(
        failure_stats,
        threshold,
        strict=strict,
        strict_violation=strict_violation,
        missing_input_stats=missing_input_stats,
        missing_input_invalid=missing_input_invalid,
        total_failure=total_failure,
    )

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
        # sp-yaru: even in TEXT mode, render a compact convergence summary
        # so operators watching stdout can see "converged at n=473" without
        # re-running with --output-format json.
        if convergence_tracker is not None:
            report = convergence_tracker.build_report(baseline=convergence_baseline_payload)
            convergence_tracker.close()
            print("\n" + "=" * 60)
            print("CONVERGENCE")
            print("=" * 60)
            overall = report.get("overall_converged_at")
            if overall is None:
                print(f"  overall: not yet converged (final n={report['final_n']})")
            else:
                print(f"  overall converged_at: n={overall}")
            if report.get("auto_stopped"):
                print(f"  auto-stopped at n={report['final_n']}")
            for qkey, qdata in report.get("per_question", {}).items():
                ca = qdata.get("converged_at")
                ca_str = f"n={ca}" if ca else "pending"
                print(f"  {qkey}: converged_at={ca_str}, support={qdata.get('support_size', 0)}")
            if convergence_baseline_payload:
                hb_n = convergence_baseline_payload.get("converged_at")
                if hb_n is not None:
                    print(f"  human baseline: converged_at=n={hb_n}")
    else:
        # sp-efip: mirror the TEXT-mode behaviour and print the fatal
        # banner to stderr even when JSON/NDJSON is the primary output.
        # Operators grep stderr for "PANEL RUN INVALID" in CI and the
        # banner is where the failing model + upstream error surface.
        if banner:
            print(banner, file=sys.stderr)

        # sp-0h9x: always populate per_model_results + cost_breakdown, even
        # for single-model panels. mayor's ensemble audits use persona_models
        # (mixed-model mode) rather than the ``models=[...]`` ensemble path,
        # so these fields were silently None in 0.9.5 despite sp-gl9 claiming
        # to ship them. A single-entry rollup still eliminates the
        # None-vs-dict branch for consumers.
        from synth_panel.ensemble import build_mixed_model_rollup

        def _cli_panelist_formatter(pr: PanelistResult, panel_model: str) -> dict[str, Any]:
            pr_pricing, _ = lookup_pricing(pr.model or panel_model)
            pr_cost = estimate_cost(pr.usage, pr_pricing)
            out: dict[str, Any] = {
                "persona": pr.persona_name,
                "responses": pr.responses,
                "usage": pr.usage.to_dict(),
                "cost": pr_cost.format_usd(),
                "error": pr.error,
            }
            if pr.model:
                out["model"] = pr.model
            return out

        per_model_results, cost_breakdown = build_mixed_model_rollup(
            panelist_results,
            default_model=model,
            panelist_formatter=_cli_panelist_formatter,
        )

        # sp-nn8k: surface DEFAULT_PRICING fallbacks loudly. Contributing
        # models are the per-panelist buckets we priced above plus the
        # synthesis model when present.
        synth_model_for_warning = synthesis_result.model if synthesis_result else None
        cost_fallback_warnings = build_cost_fallback_warnings(
            [*panelist_per_model_usage.keys(), synth_model_for_warning]
        )

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
            per_model_results=per_model_results,
            cost_breakdown=cost_breakdown,
            cost_fallback_warnings=cost_fallback_warnings,
        )
        extra["parameters"] = _build_params_metadata(args, temperature, top_p)
        # Surface failure stats + run validity in structured output so
        # downstream consumers (MCP, CI) can gate on it without parsing text.
        extra["failure_stats"] = failure_stats
        extra["missing_input_stats"] = missing_input_stats
        extra["run_invalid"] = bool(run_invalid or strict_violation)
        if total_failure is not None:
            # sp-efip: carry the structured total-failure diagnostic so
            # CI/MCP consumers can detect the "every panelist failed"
            # case without parsing banner text.
            extra["total_failure"] = total_failure
        if synthesis_error_payload is not None:
            # sp-avmm: top-level synthesis_error so JSON/NDJSON consumers
            # see the failure without walking into the synthesis dict.
            extra["synthesis_error"] = synthesis_error_payload
        if missing_input_invalid:
            # sp-bjt4: surface as a top-level warning so MCP / CI consumers
            # see the condition even if they don't special-case
            # missing_input_stats.
            warnings_list = extra.setdefault("warnings", [])
            if isinstance(warnings_list, list):
                warnings_list.append(
                    "missing_input_refusals: "
                    f"{missing_input_stats['refusing']}/{missing_input_stats['considered']}"
                    f" panelists reported missing or unavailable input"
                    f" ({missing_input_stats['refusal_rate']:.0%})"
                )
        if assignment_warnings:
            warnings_list = extra.setdefault("warnings", [])
            if isinstance(warnings_list, list):
                warnings_list.extend(assignment_warnings)
        # sp-zdul: surface the deterministic persona→model assignment so
        # JSON consumers (dashboards, analyze pipelines) can record which
        # model answered which persona without re-deriving from the spec.
        if persona_models:
            extra["model_assignment"] = dict(persona_models)
        # sp-yaru: surface convergence telemetry for large runs.
        if convergence_tracker is not None:
            convergence_report = convergence_tracker.build_report(
                baseline=convergence_baseline_payload,
            )
            if convergence_baseline_error:
                convergence_report["human_baseline_error"] = convergence_baseline_error
            extra["convergence"] = convergence_report
            convergence_tracker.close()
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


# sp-bjt4: patterns that signal a panelist is refusing because the prompt
# arrived with empty/malformed source material (e.g. a landing-page audit
# rendered with a blank {page_html} section). These fire *after* a clean
# panelist run, so they cannot be caught by _analyze_failures — the model
# replied politely with "I don't see…" and no error flag was set. Kept as
# ordered tuples of (pattern, description) to keep the list auditable.
_MISSING_INPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:don't|do not|can't|cannot|can not|don't actually|do not actually)\s+(?:see|find|have)\s+"
        r"(?:the|any|a|an)\s+(?:actual\s+)?"
        r"(?:content|text|page|document|article|example|examples|material|information|"
        r"details|data|image|screenshot|section|excerpt|passage|transcript|copy)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:actual\s+)?"
        r"(?:content|text|input|page|document|material|information|article|copy|examples?|excerpt|passage|transcript)\s+"
        r"(?:was|has\s+been|is|to)\s+"
        r"(?:provided|shared|given|included|attached|review|analyze|evaluate)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:haven't|have not|hasn't|has not|wasn't|was not)\s+(?:been\s+|actually\s+)?"
        r"(?:shared|provided|given|included|attached|pasted)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnothing\s+(?:here\s+)?to\s+"
        r"(?:review|analyze|evaluate|assess|comment\s+on|respond\s+to|discuss|work\s+with|go\s+on)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:content|text|page|document|material|copy|excerpt|passage)\s+"
        r"(?:you|that|which)\s+"
        r"(?:mentioned|referenced|described|referred\s+to|linked|pointed\s+to)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:missing|absent|empty|blank)\s+"
        r"(?:content|input|text|page|document|material|information|body|section)",
        re.IGNORECASE,
    ),
)

# Threshold is a module constant rather than a CLI flag — the bead specifies
# ≥50% as the heuristic and adding a flag would invite misuse (set to 100%
# and the signal disappears). Tune here if real-world data demands it.
_MISSING_INPUT_THRESHOLD: float = 0.5


def _response_flags_missing_input(text: str) -> bool:
    """Return True iff *text* matches any missing-input refusal pattern."""
    if not text:
        return False
    return any(p.search(text) for p in _MISSING_INPUT_PATTERNS)


def _detect_missing_input_refusals(
    panelist_results: list[Any],
) -> dict[str, Any]:
    """Count panelists whose primary responses flag missing/unavailable input.

    A panelist is counted as *refusing* if at least one primary (non-follow-up)
    response matches a missing-input pattern. Wholesale-errored panelists are
    excluded from the denominator — their failure is already captured by
    ``_analyze_failures`` and double-counting would muddy the rate.

    Returns a dict with ``considered`` (panelists included in denominator),
    ``refusing`` (panelists with at least one flagged response),
    ``refusal_rate`` (0-1), and ``refusing_personas`` (sorted names).
    """
    considered = 0
    refusing = 0
    refusing_personas: set[str] = set()

    for pr in panelist_results:
        if getattr(pr, "error", None):
            continue
        responses = getattr(pr, "responses", []) or []
        if not responses:
            continue
        considered += 1
        for resp in responses:
            if not isinstance(resp, dict) or resp.get("follow_up"):
                continue
            answer = resp.get("response", "")
            if isinstance(answer, dict):
                answer = json.dumps(answer)
            if _response_flags_missing_input(str(answer)):
                refusing += 1
                refusing_personas.add(getattr(pr, "persona_name", "unknown"))
                break

    rate = (refusing / considered) if considered > 0 else 0.0
    return {
        "considered": considered,
        "refusing": refusing,
        "refusal_rate": rate,
        "refusing_personas": sorted(refusing_personas),
    }


def _build_total_failure_banner(
    total_failure: dict[str, Any],
    stats: dict[str, Any] | None = None,
) -> str:
    """Render the sp-efip banner for a wholesale "every panelist failed" run.

    Names the model(s) exercised and the first sample error so the user
    sees the upstream failure (e.g. "OpenRouter API error 400: ...") at
    the top of stderr — not buried inside a JSON blob or an aggregate
    failure-rate line. Still prints the errored/total pair count when
    available, so the aggregate sp-2hg diagnostic stays intact.
    """
    bar = "!" * 70
    lines = [bar]
    headline = (
        f"PANEL RUN INVALID: every panelist failed"
        f" ({total_failure.get('panelists', 0)} panelist(s), no usable Q/A pairs)"
    )
    if stats and stats.get("total_pairs"):
        headline += f" — {stats['errored_pairs']}/{stats['total_pairs']} pairs errored"
    lines.append(headline + ".")
    models = total_failure.get("models") or []
    if models:
        lines.append(f"  Failing model(s): {', '.join(models)}")
    sample_errors = total_failure.get("sample_errors") or []
    for persona, err in sample_errors:
        snippet = err.strip().splitlines()[0] if err else ""
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"  {persona}: {snippet}")
    lines.append("No synthesis was performed — the run produced no usable data.")
    lines.append("Check the model name, provider credentials, and upstream status.")
    lines.append(bar)
    return "\n".join(lines)


def _build_invalid_banner(
    stats: dict[str, Any],
    threshold: float,
    *,
    strict: bool,
    strict_violation: bool,
    missing_input_stats: dict[str, Any] | None = None,
    missing_input_invalid: bool = False,
    total_failure: dict[str, Any] | None = None,
) -> str:
    """Render the fatal banner printed to stderr on an invalid run.

    Returns an empty string when the run is valid. The banner is
    intentionally loud — the entire point of sp-2hg is that a user
    skimming output cannot miss that the panel was not executed
    successfully.
    """
    total = stats["total_pairs"]
    errored = stats["errored_pairs"]
    rate = stats["failure_rate"] if total > 0 else 0.0
    over_threshold = total > 0 and rate > threshold
    if not (over_threshold or strict_violation or missing_input_invalid or total_failure):
        return ""

    # sp-efip: total-wipeout banner takes precedence so the user sees
    # the failing model and first upstream error at the top, not buried
    # under aggregate failure-rate language.
    if total_failure is not None:
        return _build_total_failure_banner(total_failure, stats)

    bar = "!" * 70
    lines = [bar]
    if over_threshold:
        lines.append(
            f"PANEL RUN INVALID: {errored}/{total} panelist-question pairs"
            f" errored ({rate:.0%} > threshold {threshold:.0%})."
        )
    elif strict_violation:
        lines.append(f"PANEL RUN INVALID (--strict): {errored}/{total} panelist-question pairs errored.")
    elif missing_input_invalid and missing_input_stats:
        considered = missing_input_stats["considered"]
        refusing = missing_input_stats["refusing"]
        mi_rate = missing_input_stats["refusal_rate"]
        lines.append(
            f"PANEL RUN INVALID: {refusing}/{considered} panelist(s) reported"
            f" missing or unavailable input ({mi_rate:.0%} >= threshold"
            f" {_MISSING_INPUT_THRESHOLD:.0%})."
        )
        lines.append(
            "The prompt likely arrived with an empty or malformed section (e.g. an unrendered template fragment)."
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
    if missing_input_invalid and missing_input_stats and missing_input_stats.get("refusing_personas"):
        shown = ", ".join(missing_input_stats["refusing_personas"][:4])
        extra = len(missing_input_stats["refusing_personas"]) - 4
        if extra > 0:
            shown += f", +{extra} more"
        lines.append(f"  Personas reporting missing input: {shown}")
    if missing_input_invalid and not (over_threshold or strict_violation):
        lines.append("Check that every --var / {placeholder} rendered with real content,")
        lines.append("then re-run once the prompt is whole.")
    else:
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
    per_model_results: dict[str, Any] | None = None,
    cost_breakdown: dict[str, Any] | None = None,
    cost_fallback_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the rounds-shaped panel output payload.

    Single-round runs surface as one round entry whose ``name`` is the
    instrument's only round (``"default"`` for v1). Multi-round/branching
    runs use this same shape with one entry per executed round; that
    wiring lives in F3-A. Per-round ``synthesis`` is ``null`` for the
    single-round case — the final synthesis goes at the top level.

    ``per_model_results`` and ``cost_breakdown``, when provided, surface
    the sp-0h9x rollup shape alongside the primary ``model`` field — a
    superset of the mixed-model artifact that mayor's ensemble audits
    previously had to reconstruct by iterating ``rounds[].results[]``.

    ``cost_fallback_warnings`` (sp-nn8k) is appended to ``warnings`` for
    every model that was priced via ``DEFAULT_PRICING`` fallback; the
    top-level ``cost_is_estimated`` boolean mirrors ``bool(...)`` of that
    list so programmatic consumers can gate on it without string-matching
    the warning text.
    """
    round_name = instrument.rounds[0].name if instrument.rounds else "default"
    warnings = list(getattr(instrument, "warnings", []) or [])
    fallback_warnings = list(cost_fallback_warnings or [])
    warnings.extend(fallback_warnings)
    result: dict[str, Any] = {
        "rounds": [
            {
                "name": round_name,
                "results": results,
                "synthesis": None,
            }
        ],
        "path": [],
        "warnings": warnings,
        "cost_is_estimated": bool(fallback_warnings),
        "synthesis": synthesis_dict,
        "panelist_cost": panelist_cost.format_usd(),
        "panelist_usage": (panelist_usage.to_dict() if panelist_usage is not None else None),
        "total_usage": total_usage.to_dict(),
        "total_cost": total_cost.format_usd(),
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
        "per_model_results": per_model_results,
        "cost_breakdown": cost_breakdown,
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


def handle_panel_inspect(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Inspect a saved panel result: metadata, per-model rollup, failures (sp-76gm).

    No LLM calls. Loads a result by ID or path (same resolution as
    ``panel synthesize``) and prints a compact summary. JSON/NDJSON
    modes emit the full structured :class:`InspectReport` dict under
    the ``extra`` key.
    """
    from synth_panel.analysis.inspect import build_inspect_report, format_inspect_text

    result_ref = args.result
    path = Path(result_ref)
    if path.exists() and path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Error: not valid JSON: {result_ref}: {exc}"
            if fmt is OutputFormat.TEXT:
                print(msg, file=sys.stderr)
            else:
                emit(fmt, message=msg, extra={"error": "invalid_json"})
            return 1
        data.setdefault("id", path.stem)
    else:
        from synth_panel.mcp.data import get_panel_result

        try:
            data = get_panel_result(result_ref)
        except FileNotFoundError:
            msg = f"Error: panel result not found: {result_ref}"
            if fmt is OutputFormat.TEXT:
                print(msg, file=sys.stderr)
            else:
                emit(fmt, message=msg, extra={"error": "not_found"})
            return 1

    if not isinstance(data, dict):
        msg = "Error: panel result is not a JSON object"
        if fmt is OutputFormat.TEXT:
            print(msg, file=sys.stderr)
        else:
            emit(fmt, message=msg, extra={"error": "invalid_shape"})
        return 1

    report = build_inspect_report(data)

    if fmt is OutputFormat.TEXT:
        print(format_inspect_text(report))
    else:
        emit(fmt, message="Panel inspect", extra={"inspect": report.to_dict()})

    return 0


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
    # sp-avmm: pre-flight size check for the re-synthesize path too, so a
    # saved panel cannot silently overflow a smaller-context synthesis model
    # picked via --synthesis-model on the rerun.
    synth_model_for_check = getattr(args, "synthesis_model", None) or panelist_model
    custom_prompt = getattr(args, "synthesis_prompt", None)
    overflow = detect_synthesis_context_overflow(
        panelist_results,
        questions,
        synthesis_model=synth_model_for_check,
        custom_prompt=custom_prompt,
    )
    if overflow is not None:
        actionable = format_synthesis_overflow_message(overflow)
        payload = build_synthesis_error_payload(
            None,
            error_type="synthesis_context_overflow",
            message=actionable,
            suggested_fix=(
                "Rerun with --synthesis-model gemini-2.5-flash-lite (1M context) "
                "or gemini-2.5-pro (1M context), or reduce panel size."
            ),
            diagnostic=overflow,
        )
        print(f"Error: synthesis pre-flight rejected: {actionable}", file=sys.stderr)
        if fmt is not OutputFormat.TEXT:
            emit(
                fmt,
                message=actionable,
                extra={"run_invalid": True, "synthesis_error": payload},
            )
        return 2
    try:
        synth = synthesize_panel(
            client,
            panelist_results,
            questions,
            model=getattr(args, "synthesis_model", None),
            panelist_model=panelist_model,
            custom_prompt=custom_prompt,
        )
    except Exception as exc:
        # sp-avmm: fail loud with a structured payload so MCP / CI consumers
        # of `panel synthesize` see the same envelope as `panel run`.
        from synth_panel._runners import _sanitize_api_error

        sanitized = _sanitize_api_error(exc)
        payload = build_synthesis_error_payload(
            exc,
            error_type="synthesis_api_error",
            message=f"Synthesis call failed: {sanitized}",
            suggested_fix=(
                "Check provider credentials and model availability;"
                " if context-related, rerun with a larger-context synthesis model."
            ),
        )
        print(f"Error: synthesis failed: {sanitized}", file=sys.stderr)
        if fmt is not OutputFormat.TEXT:
            emit(
                fmt,
                message=f"Synthesis call failed: {sanitized}",
                extra={"run_invalid": True, "synthesis_error": payload},
            )
        return 2

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
