"""Multi-model ensemble runner.

Runs the same panel N times (once per model) using existing
run_panel_parallel(), then aggregates per-model costs via
CostEstimate/TokenUsage.  Optionally blends response distributions
across models for multiple-choice / structured-option questions.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    TokenUsage,
    build_cost_fallback_warnings,
    resolve_cost,
)
from synth_panel.llm.client import LLMClient
from synth_panel.orchestrator import PanelistResult, run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.prompts import build_question_prompt, persona_system_prompt

logger = logging.getLogger(__name__)

# Mirrors CLI `_RATE_LIMIT_ABORT_MARKERS` (sp-56pb): substring hints that an
# error likely came from provider rate-limit / retry exhaustion. Used to flag
# ensemble incidents without importing the CLI stack.
_RATE_LIMIT_HINT_MARKERS = (
    "retries_exhausted",
    "retries exhausted",
    "rate_limit",
    "rate limit",
    "429",
)


def _rate_limit_likely(message: str | None) -> bool:
    if not message:
        return False
    lower = message.lower()
    return any(marker in lower for marker in _RATE_LIMIT_HINT_MARKERS)


def collect_ensemble_incidents(ens: EnsembleResult) -> list[dict[str, Any]]:
    """Collect structured records for partial ensemble failures (GH #312).

    Walks each model's panelists and records panelist-level failures,
    per-question ``error`` responses, and shortfalls vs ``question_count``.
    Downstream consumers use this for provenance; :func:`build_ensemble_output`
    merges derived warnings into the public ``warnings`` list.
    """
    incidents: list[dict[str, Any]] = []
    expected_primary = ens.question_count

    for mr in ens.model_results:
        model = mr.model
        for pr in mr.panelist_results:
            persona = pr.persona_name
            if pr.error:
                incidents.append(
                    {
                        "model": model,
                        "persona": persona,
                        "kind": "panelist_failure",
                        "question_index": None,
                        "question": None,
                        "detail": pr.error,
                        "rate_limit_suspected": _rate_limit_likely(pr.error),
                    }
                )
                continue

            primary: list[dict[str, Any]] = []
            for resp in pr.responses or []:
                if isinstance(resp, dict) and resp.get("follow_up"):
                    continue
                if isinstance(resp, dict):
                    primary.append(resp)

            for qi, resp in enumerate(primary):
                if not isinstance(resp, dict):
                    continue
                if resp.get("error"):
                    qtext = resp.get("question")
                    detail = str(resp.get("response", ""))
                    incidents.append(
                        {
                            "model": model,
                            "persona": persona,
                            "kind": "question_failure",
                            "question_index": qi,
                            "question": qtext,
                            "detail": detail,
                            "rate_limit_suspected": _rate_limit_likely(detail),
                        }
                    )

            if expected_primary > len(primary):
                for qi in range(len(primary), expected_primary):
                    incidents.append(
                        {
                            "model": model,
                            "persona": persona,
                            "kind": "missing_response",
                            "question_index": qi,
                            "question": None,
                            "detail": "panelist produced fewer primary answers than questions in the instrument",
                            "rate_limit_suspected": False,
                        }
                    )

    return incidents


def build_ensemble_incident_warnings(incidents: list[dict[str, Any]]) -> list[str]:
    """Human-readable summary lines for partial ensemble runs."""
    if not incidents:
        return []

    by_model: dict[str, int] = {}
    rate_limit_hits = 0
    for inc in incidents:
        model = str(inc.get("model") or "unknown")
        by_model[model] = by_model.get(model, 0) + 1
        if inc.get("rate_limit_suspected"):
            rate_limit_hits += 1

    model_parts = [f"{m} ({by_model[m]})" for m in sorted(by_model)]
    summary = (
        f"Ensemble partial failure: {len(incidents)} incident(s) "
        f"across models [{', '.join(model_parts)}]. "
        "Some provider-persona-question tuples did not complete successfully — "
        "cross-model comparisons may be skewed."
    )
    out = [summary]
    if rate_limit_hits:
        out.append(
            f"{rate_limit_hits} incident(s) look rate-limit-related "
            "(429 / retries exhausted / rate_limit markers in errors)."
        )
    return out


@dataclass
class ModelRunResult:
    """Result from a single model's panel run."""

    model: str
    panelist_results: list[PanelistResult]
    usage: TokenUsage
    cost: CostEstimate
    sessions: dict[str, Session]


@dataclass
class EnsembleResult:
    """Aggregated result from running the panel across multiple models."""

    model_results: list[ModelRunResult]
    models: list[str]
    total_usage: TokenUsage
    total_cost: CostEstimate
    per_model_cost: dict[str, str]  # model -> formatted USD
    per_model_usage: dict[str, dict[str, int]]  # model -> usage dict
    persona_count: int
    question_count: int


def ensemble_run(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    models: list[str],
    client: LLMClient,
    *,
    system_prompt_fn: Callable[[dict[str, Any]], str] | None = None,
    question_prompt_fn: Callable[[dict[str, Any]], str] | None = None,
    response_schema: dict[str, Any] | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> EnsembleResult:
    """Run a panel once per model and aggregate results.

    Args:
        personas: List of persona dicts.
        questions: List of question dicts (each with a "text" key).
        models: List of model aliases to run. Panel is run once per model.
        client: Shared LLM client.
        system_prompt_fn: Builds system prompt from persona. Default:
            built-in persona_system_prompt.
        question_prompt_fn: Builds question text from question dict.
            Default: built-in build_question_prompt.
        response_schema: Optional JSON Schema for structured output.
        extract_schema: Optional JSON Schema for post-hoc extraction.
        temperature: Sampling temperature for panelist responses.
        top_p: Nucleus sampling threshold.

    Returns:
        EnsembleResult with per-model and aggregated data.

    Raises:
        ValueError: If models is empty.
    """
    if not models:
        raise ValueError("models list must not be empty")

    sys_fn = system_prompt_fn or persona_system_prompt
    q_fn = question_prompt_fn or build_question_prompt

    model_results: list[ModelRunResult] = []
    total_usage = ZERO_USAGE
    total_cost = CostEstimate()
    per_model_cost: dict[str, str] = {}
    per_model_usage: dict[str, dict[str, int]] = {}

    for model in models:
        logger.info(
            "Ensemble: running panel with model=%s (%d personas, %d questions)", model, len(personas), len(questions)
        )

        panelist_results, _registry, sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model=model,
            system_prompt_fn=sys_fn,
            question_prompt_fn=q_fn,
            response_schema=response_schema,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        # Aggregate usage for this model
        model_usage = ZERO_USAGE
        for pr in panelist_results:
            model_usage = model_usage + pr.usage

        # sp-kvpx: route through resolve_cost so per-model cost honors
        # sp-j3vk's precedence (provider-reported → local fallback). The
        # prior ``estimate_cost(model_usage, pricing)`` ignored the
        # summed ``usage.provider_reported_cost``, so ensemble
        # ``cost_breakdown`` / ``per_model_cost`` drifted from the
        # authoritative top-level total for every model whose local
        # pricing entry diverged from the real OpenRouter bill.
        model_cost = resolve_cost(model_usage, model)

        # Tag each result with its model
        for pr in panelist_results:
            pr.model = model

        model_results.append(
            ModelRunResult(
                model=model,
                panelist_results=panelist_results,
                usage=model_usage,
                cost=model_cost,
                sessions=sessions,
            )
        )

        total_usage = total_usage + model_usage
        total_cost = total_cost + model_cost
        per_model_cost[model] = model_cost.format_usd()
        per_model_usage[model] = model_usage.to_dict()

    # sp-27rz: defensive sanity check — every input model must have produced
    # a bucket. Silent drops at this layer (mis-iteration, stray early return)
    # would reintroduce the "absent model" bug the weighted-assign fix closes,
    # so assert the invariant explicitly rather than trusting the loop.
    produced = {mr.model for mr in model_results}
    expected = set(models)
    if produced != expected:
        missing = expected - produced
        raise RuntimeError(
            f"ensemble_run: per_model_results is missing {sorted(missing)} "
            f"(expected {sorted(expected)}, produced {sorted(produced)})"
        )

    return EnsembleResult(
        model_results=model_results,
        models=models,
        total_usage=total_usage,
        total_cost=total_cost,
        per_model_cost=per_model_cost,
        per_model_usage=per_model_usage,
        persona_count=len(personas),
        question_count=len(questions),
    )


def _default_panelist_formatter(pr: PanelistResult, model: str) -> dict[str, Any]:
    """Minimal panelist dict shared by ensemble + mixed-model rollups."""
    out: dict[str, Any] = {
        "persona": pr.persona_name,
        "responses": pr.responses,
    }
    if pr.model or model:
        out["model"] = pr.model or model
    if pr.error:
        out["error"] = pr.error
    return out


def build_ensemble_output(
    ens: EnsembleResult,
    *,
    panelist_formatter: Callable[[PanelistResult, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape an :class:`EnsembleResult` into the public JSON output.

    Produces the shape documented on the ``run_panel`` MCP tool and the CLI
    ``panel run`` ensemble path:

    ``per_model_results`` is keyed by model and each value is a dict with
    ``results`` (list of formatted panelist dicts), ``cost`` (formatted
    USD string), and ``usage`` (token bucket dict). ``cost_breakdown``
    exposes both the per-model USD breakdown (``by_model``) and the total
    (``total``).

    ``metadata`` carries the synthbench-shaped ``cost.per_model`` bundle
    populated from every model in the ensemble (sp-atvc). Without this,
    downstream audits that read ``metadata.cost.per_model`` only saw the
    first model and undercounted multi-model ensemble spend.

    ``panelist_formatter`` customises how each :class:`PanelistResult` is
    rendered; when omitted, a minimal ``{persona, responses, model}`` dict
    is produced so callers without the full runner context still get a
    useful payload.

    ``ensemble_incidents`` lists structured partial-failure records (GH #312):
    panelist-level failures, per-question errors, and shortfalls versus
    ``question_count``. Non-empty incidents append loud warnings alongside
    any cost-tier fallback warnings.
    """
    from synth_panel.metadata import build_metadata

    fmt = panelist_formatter or _default_panelist_formatter

    per_model_results: dict[str, dict[str, Any]] = {}
    for mr in ens.model_results:
        per_model_results[mr.model] = {
            "results": [fmt(pr, mr.model) for pr in mr.panelist_results],
            "cost": mr.cost.format_usd(),
            "usage": mr.usage.to_dict(),
        }

    # sp-atvc: build a metadata bundle whose cost.per_model covers every
    # ensemble model so downstream audits see real per-provider spend.
    primary_model = ens.models[0] if ens.models else ""
    panelist_per_model = {mr.model: (mr.usage, mr.cost) for mr in ens.model_results}
    ens_metadata = build_metadata(
        panelist_model=primary_model,
        panelist_usage=ens.total_usage,
        panelist_cost=ens.total_cost,
        total_usage=ens.total_usage,
        total_cost=ens.total_cost,
        persona_count=ens.persona_count,
        question_count=ens.question_count,
        panelist_per_model=panelist_per_model,
    )

    # sp-nn8k: flag models priced via DEFAULT_PRICING fallback so the
    # ensemble payload exposes estimated spend the same way the
    # single-model + mixed-model rollups do.
    cost_warnings = build_cost_fallback_warnings(ens.models)

    # GH #312 / sp-4y5.5: surface partial ensemble failures (rate limits,
    # per-question errors) so callers are not fooled into treating N models
    # as fully represented when some tuples dropped on the floor.
    ensemble_incidents = collect_ensemble_incidents(ens)
    incident_warnings = build_ensemble_incident_warnings(ensemble_incidents)
    merged_warnings = list(cost_warnings) + incident_warnings

    return {
        "per_model_results": per_model_results,
        "cost_breakdown": {
            "by_model": dict(ens.per_model_cost),
            "total": ens.total_cost.format_usd(),
        },
        "models": list(ens.models),
        "total_usage": ens.total_usage.to_dict(),
        "warnings": merged_warnings,
        "cost_is_estimated": bool(cost_warnings),
        "ensemble_incidents": ensemble_incidents,
        "metadata": ens_metadata,
    }


def build_mixed_model_rollup(
    panelist_results: list[PanelistResult],
    default_model: str,
    *,
    panelist_formatter: Callable[[PanelistResult, str], dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Group panelist results by model and produce ``per_model_results`` + ``cost_breakdown``.

    Unlike :func:`build_ensemble_output`, this operates on the output of a
    single :func:`run_panel_parallel` call where panelists may have run on
    different models via ``persona_models``. The returned shape matches the
    ensemble path so downstream consumers (dashboards, CI gates, cost
    comparators) see the same keys regardless of how the mix arose.

    Single-model panels still produce a one-entry ``per_model_results`` dict
    rather than ``None`` — "only one model ran" is a valid rollup, and
    keeping the field populated eliminates a None-vs-dict branch for
    consumers.

    Args:
        panelist_results: Flat list of :class:`PanelistResult` objects from
            a single panel run. Each result's ``model`` attribute is used
            to key the rollup; untagged results fall back to ``default_model``.
        default_model: Model to use for results whose ``model`` attribute
            is ``None`` (e.g. pre-multi-model panels).
        panelist_formatter: Callable returning the per-panelist dict shape.
            Defaults to a minimal ``{persona, responses, model}`` dict; CLI
            callers override this to emit the full ``cost`` + ``usage``
            shape already used in ``results[]``.

    Returns:
        ``(per_model_results, cost_breakdown)`` where:

        * ``per_model_results`` is ``{model: {results, cost, usage}}``
        * ``cost_breakdown`` is ``{by_model: {model: "$X"}, total: "$Y"}``

        Both are empty when ``panelist_results`` is empty.
    """
    fmt = panelist_formatter or _default_panelist_formatter

    by_model: dict[str, list[PanelistResult]] = {}
    for pr in panelist_results:
        key = pr.model or default_model
        by_model.setdefault(key, []).append(pr)

    per_model_results: dict[str, dict[str, Any]] = {}
    by_model_cost: dict[str, str] = {}
    total_cost = CostEstimate()

    for model_name, prs in by_model.items():
        model_usage = ZERO_USAGE
        for pr in prs:
            model_usage = model_usage + pr.usage
        # sp-kvpx: resolve_cost for per-model mixed-model rollup too.
        model_cost = resolve_cost(model_usage, model_name)
        per_model_results[model_name] = {
            "results": [fmt(pr, model_name) for pr in prs],
            "cost": model_cost.format_usd(),
            "usage": model_usage.to_dict(),
        }
        by_model_cost[model_name] = model_cost.format_usd()
        total_cost = total_cost + model_cost

    cost_breakdown: dict[str, Any] = {
        "by_model": by_model_cost,
        "total": total_cost.format_usd(),
    }
    return per_model_results, cost_breakdown


# ---------------------------------------------------------------------------
# Distribution blending
# ---------------------------------------------------------------------------


@dataclass
class BlendedQuestion:
    """Blended distribution for a single question across models."""

    question_index: int
    question_text: str
    distribution: dict[str, float]  # option -> blended probability
    per_model: dict[str, dict[str, float]]  # model -> {option -> probability}
    response_count: int  # total panelist responses that contributed


@dataclass
class BlendedResult:
    """Complete blended distribution set from an ensemble run."""

    questions: list[BlendedQuestion]
    models: list[str]
    weights: dict[str, float]  # model -> normalized weight


def _extract_response_value(response: dict[str, Any]) -> str:
    """Extract a comparable response value from a panelist response dict.

    Handles both free-text and structured responses. For structured
    responses, extracts the primary value (first string field or the
    ``response`` key). For free-text, returns the raw response string.
    """
    val = response.get("response", "")
    if isinstance(val, dict):
        # Structured response — try common keys, then first string value
        for key in ("answer", "choice", "selection", "value", "response"):
            if key in val and isinstance(val[key], str):
                return val[key].strip()
        # Fallback: first string value in the dict
        for v in val.values():
            if isinstance(v, str):
                return v.strip()
        return str(val)
    return str(val).strip()


def _match_to_option(value: str, options: list[str]) -> str:
    """Match a response value to the closest option from a defined list.

    Matching strategy (first match wins):
    1. Exact match (case-insensitive)
    2. Option is a substring of the response (case-insensitive),
       preferring the longest matching option
    3. Response is a substring of an option (case-insensitive),
       preferring the longest matching option

    Returns the original option string (preserving case) on match,
    or the original value if no match is found.
    """
    val_lower = value.lower()

    # 1. Exact match
    for opt in options:
        if val_lower == opt.lower():
            return opt

    # 2. Option contained in response (longest wins)
    contained: list[str] = []
    for opt in options:
        if opt.lower() in val_lower:
            contained.append(opt)
    if contained:
        return max(contained, key=len)

    # 3. Response contained in option (longest matching option wins)
    reverse_contained: list[str] = []
    for opt in options:
        if val_lower in opt.lower():
            reverse_contained.append(opt)
    if reverse_contained:
        return max(reverse_contained, key=len)

    return value


def _build_distribution(responses: list[str]) -> dict[str, float]:
    """Build a probability distribution from a list of response strings.

    Each unique response gets a probability equal to its frequency.
    Returns a dict mapping response -> probability (sums to 1.0).
    """
    if not responses:
        return {}
    counts = Counter(responses)
    total = len(responses)
    return {option: count / total for option, count in counts.items()}


def blend_distributions(
    ensemble_result: EnsembleResult,
    *,
    weights: dict[str, float] | None = None,
    questions: list[dict[str, Any]] | None = None,
) -> BlendedResult:
    """Blend response distributions across models in an ensemble result.

    For each question, collects all panelist responses from each model,
    computes per-model response distributions (option frequencies), and
    produces a weighted average across models.

    When *questions* is provided and a question defines an ``options``
    list, each response value is matched to the closest option before
    aggregation.  This collapses free-text variations into canonical
    option names (e.g. "I'd go with hybrid" → "Hybrid 3 days").

    Args:
        ensemble_result: Result from :func:`ensemble_run` containing
            per-model panelist results.
        weights: Optional model -> weight mapping. When provided, weights
            are normalized to sum to 1.0. When ``None``, all models get
            equal weight.
        questions: Optional list of question dicts from the instrument.
            When a question dict contains an ``options`` key (a list of
            strings), responses are matched to those options before
            distribution calculation.

    Returns:
        :class:`BlendedResult` with per-question blended distributions.

    Raises:
        ValueError: If ensemble_result has no model results.
    """
    if not ensemble_result.model_results:
        raise ValueError("ensemble_result has no model results")

    models = ensemble_result.models

    # Resolve and normalize weights
    if weights:
        raw_weights = {m: weights.get(m, 0.0) for m in models}
    else:
        raw_weights = {m: 1.0 for m in models}

    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        raise ValueError("model weights must sum to a positive value")
    norm_weights = {m: w / weight_sum for m, w in raw_weights.items()}

    # Determine the number of questions from the first model's results.
    # All models ran the same questions, so we use the max response count
    # across panelists as the question count.
    n_questions = 0
    for mr in ensemble_result.model_results:
        for pr in mr.panelist_results:
            n_questions = max(n_questions, len(pr.responses))

    blended_questions: list[BlendedQuestion] = []

    for qi in range(n_questions):
        per_model_dist: dict[str, dict[str, float]] = {}
        question_text = ""
        total_responses = 0

        # Resolve options list for this question index (if provided)
        q_options: list[str] | None = None
        if questions and qi < len(questions):
            opts = questions[qi].get("options")
            if isinstance(opts, list) and opts:
                q_options = [str(o) for o in opts]

        for mr in ensemble_result.model_results:
            model_responses: list[str] = []
            for pr in mr.panelist_results:
                if qi < len(pr.responses):
                    resp = pr.responses[qi]
                    if not question_text:
                        question_text = resp.get("question", f"Q{qi + 1}")
                    if not resp.get("error"):
                        val = _extract_response_value(resp)
                        if q_options:
                            val = _match_to_option(val, q_options)
                        model_responses.append(val)

            total_responses += len(model_responses)
            per_model_dist[mr.model] = _build_distribution(model_responses)

        # Weighted average across models: collect all options, then for
        # each option compute sum(weight_m * prob_m(option)).
        all_options: set[str] = set()
        for dist in per_model_dist.values():
            all_options.update(dist.keys())

        blended: dict[str, float] = {}
        for option in all_options:
            blended[option] = sum(norm_weights.get(m, 0.0) * per_model_dist.get(m, {}).get(option, 0.0) for m in models)

        blended_questions.append(
            BlendedQuestion(
                question_index=qi,
                question_text=question_text,
                distribution=blended,
                per_model=per_model_dist,
                response_count=total_responses,
            )
        )

    return BlendedResult(
        questions=blended_questions,
        models=models,
        weights=norm_weights,
    )
