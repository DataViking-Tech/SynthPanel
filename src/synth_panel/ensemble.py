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
    estimate_cost,
    lookup_pricing,
)
from synth_panel.llm.client import LLMClient
from synth_panel.orchestrator import PanelistResult, run_panel_parallel
from synth_panel.persistence import Session
from synth_panel.prompts import build_question_prompt, persona_system_prompt

logger = logging.getLogger(__name__)


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
        )

        # Aggregate usage for this model
        model_usage = ZERO_USAGE
        for pr in panelist_results:
            model_usage = model_usage + pr.usage

        pricing, _ = lookup_pricing(model)
        model_cost = estimate_cost(model_usage, pricing)

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
