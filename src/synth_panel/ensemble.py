"""Multi-model ensemble runner.

Runs the same panel N times (once per model) using existing
run_panel_parallel(), then aggregates per-model costs via
CostEstimate/TokenUsage.
"""

from __future__ import annotations

import logging
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
