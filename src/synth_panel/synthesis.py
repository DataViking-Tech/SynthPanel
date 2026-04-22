"""Panel synthesis: aggregate panelist responses into research findings.

Takes individual panelist responses from a panel run and synthesizes them
into a structured SynthesisResult via an LLM call using tool-use forcing
(StructuredOutputEngine).

Two strategies are supported:

* ``single`` (default): one synthesis call concatenating every panelist
  response. Simple and cheap for small panels.
* ``map-reduce`` (sp-kkzz): one map call per question in parallel,
  summarizing that question's responses, followed by one reduce call
  that combines the question-level summaries into the final synthesis.
  Lets panels scale to n=50-500 without hitting synthesis-model context
  limits.
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    TokenUsage,
    estimate_cost,
    lookup_pricing,
)
from synth_panel.llm.aliases import resolve_alias
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.prompts import SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_VERSION
from synth_panel.structured import StructuredOutputConfig, StructuredOutputEngine

logger = logging.getLogger(__name__)

# Synthesis strategy identifiers.
STRATEGY_SINGLE = "single"
STRATEGY_MAP_REDUCE = "map-reduce"
STRATEGY_AUTO = "auto"
SYNTHESIS_STRATEGIES = (STRATEGY_SINGLE, STRATEGY_MAP_REDUCE, STRATEGY_AUTO)


def _convert_llm_usage(llm_usage: LLMTokenUsage) -> TokenUsage:
    """Convert LLM-layer TokenUsage to cost-layer TokenUsage."""
    return TokenUsage(
        input_tokens=llm_usage.input_tokens,
        output_tokens=llm_usage.output_tokens,
        cache_creation_input_tokens=llm_usage.cache_write_tokens,
        cache_read_input_tokens=llm_usage.cache_read_tokens,
        provider_reported_cost=llm_usage.provider_reported_cost,
        reasoning_tokens=llm_usage.reasoning_tokens,
        cached_tokens=llm_usage.cached_tokens,
    )


# JSON Schema for the synthesis result, used by StructuredOutputEngine.
_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A concise summary of the panel findings (2-4 sentences).",
        },
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key themes that emerged across panelist responses.",
        },
        "agreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where panelists broadly agreed.",
        },
        "disagreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where panelists disagreed or diverged.",
        },
        "surprises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unexpected or notable findings from the panel.",
        },
        "recommendation": {
            "type": "string",
            "description": "A brief actionable recommendation based on the panel findings.",
        },
    },
    "required": ["summary", "themes", "agreements", "disagreements", "surprises", "recommendation"],
}

_DEFAULT_MODEL = "sonnet"
_MAX_TOKENS = 4096


@dataclass
class SynthesisResult:
    """Aggregated synthesis of a panel run."""

    summary: str
    themes: list[str]
    agreements: list[str]
    disagreements: list[str]
    surprises: list[str]
    recommendation: str
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)
    cost: CostEstimate = field(default_factory=CostEstimate)
    model: str = _DEFAULT_MODEL
    synthesis_prompt_version: int = SYNTHESIS_PROMPT_VERSION
    is_fallback: bool = False
    error: str | None = None
    # sp-kkzz: map-reduce synthesis metadata. These stay ``None`` for
    # the default ``single`` strategy so the serialized shape is unchanged
    # for backwards-compat callers.
    strategy: str = STRATEGY_SINGLE
    per_question_synthesis: dict[int, str] | None = None
    map_cost_breakdown: list[dict[str, Any]] | None = None
    reduce_cost_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d: dict[str, Any] = {
            "summary": self.summary,
            "themes": self.themes,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "surprises": self.surprises,
            "recommendation": self.recommendation,
            "usage": self.usage.to_dict(),
            "cost": self.cost.format_usd(),
            "model": self.model,
            "prompt_version": self.synthesis_prompt_version,
        }
        if self.strategy != STRATEGY_SINGLE:
            d["strategy"] = self.strategy
        if self.per_question_synthesis is not None:
            # JSON requires string keys; keep indices as strings in the
            # serialized shape so MCP / file persistence round-trips cleanly.
            d["per_question_synthesis"] = {str(k): v for k, v in self.per_question_synthesis.items()}
        return d


def _format_panelist_data(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
) -> str:
    """Format panelist results and questions into a structured text block for the LLM."""
    parts: list[str] = []

    # Questions asked
    parts.append("## Questions Asked")
    for i, q in enumerate(questions, 1):
        text = q.get("text", q) if isinstance(q, dict) else str(q)
        parts.append(f"{i}. {text}")

    parts.append("")
    parts.append("## Panelist Responses")

    for result in panelist_results:
        name = result.persona_name
        parts.append(f"\n### {name}")
        for resp in result.responses:
            q_text = resp.get("question", "")
            answer = resp.get("response", "")
            is_follow_up = resp.get("follow_up", False)
            prefix = "  Follow-up" if is_follow_up else "  Q"
            if isinstance(answer, dict):
                answer = json.dumps(answer, indent=2)
            parts.append(f"{prefix}: {q_text}")
            parts.append(f"  A: {answer}")

    return "\n".join(parts)


def _print_cost_estimate(
    model: str,
    user_content: str,
    panelist_cost: CostEstimate | None,
) -> None:
    """Print a pre-synthesis cost estimate to stderr."""
    pricing, _ = lookup_pricing(model)
    # Rough token estimate: ~4 chars per token for English text
    est_input_tokens = len(user_content) // 4
    est_output_tokens = 1000  # typical synthesis output
    est_usage = TokenUsage(input_tokens=est_input_tokens, output_tokens=est_output_tokens)
    est = estimate_cost(est_usage, pricing)
    parts = [f"Synthesis will cost ~{est.format_usd()} using {model}"]
    if panelist_cost is not None:
        parts.append(f"(panelist cost was {panelist_cost.format_usd()})")
    print(" ".join(parts), file=sys.stderr)


def synthesize_panel(
    client: LLMClient,
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    panelist_model: str | None = None,
    custom_prompt: str | None = None,
    panelist_cost: CostEstimate | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> SynthesisResult:
    """Synthesize panelist responses into a structured research finding.

    Args:
        client: LLM client for the synthesis call.
        panelist_results: List of PanelistResult from run_panel_parallel.
        questions: The questions that were asked.
        model: Explicit model for synthesis (e.g. --synthesis-model).
        panelist_model: Model used for panelists; used as fallback when
            *model* is not set so synthesis matches panelist cost tier.
        custom_prompt: Override the default synthesis prompt.
        panelist_cost: Panelist cost estimate, shown in the pre-synthesis
            cost estimate printed to stderr.
        temperature: Sampling temperature for the synthesis call.
        top_p: Nucleus sampling cutoff for the synthesis call.

    Returns:
        SynthesisResult with structured findings and cost tracking.
    """
    resolved_model = model or panelist_model or _DEFAULT_MODEL
    prompt_text = custom_prompt or SYNTHESIS_PROMPT

    # Format the panelist data into a user message
    panelist_data = _format_panelist_data(panelist_results, questions)
    user_content = f"{prompt_text}\n\n{panelist_data}"

    messages = [
        InputMessage(role="user", content=[TextBlock(text=user_content)]),
    ]

    config = StructuredOutputConfig(
        schema=_SYNTHESIS_SCHEMA,
        tool_name="synthesize",
        tool_description="Provide structured synthesis of the panel responses.",
    )

    # Pre-synthesis cost estimate (printed to stderr)
    _print_cost_estimate(resolved_model, user_content, panelist_cost)

    engine = StructuredOutputEngine(client)
    result = engine.extract(
        model=resolved_model,
        max_tokens=_MAX_TOKENS,
        messages=messages,
        config=config,
        temperature=temperature,
        top_p=top_p,
    )

    # Convert usage
    usage = _convert_llm_usage(result.response.usage)
    pricing, _ = lookup_pricing(resolved_model)
    cost = estimate_cost(usage, pricing)

    if result.is_fallback:
        return SynthesisResult(
            summary="Synthesis failed — see error field.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            usage=usage,
            cost=cost,
            model=resolved_model,
            is_fallback=True,
            error=result.error,
        )

    data = result.data
    return SynthesisResult(
        summary=data.get("summary", ""),
        themes=data.get("themes", []),
        agreements=data.get("agreements", []),
        disagreements=data.get("disagreements", []),
        surprises=data.get("surprises", []),
        recommendation=data.get("recommendation", ""),
        usage=usage,
        cost=cost,
        model=resolved_model,
    )


# ---------------------------------------------------------------------------
# sp-kkzz: per-question map-reduce synthesis
# ---------------------------------------------------------------------------


_MAP_PROMPT_TEMPLATE = (
    "You are summarizing responses from a focus group panel to ONE specific "
    "question. Produce a structured synthesis focused on just this question.\n\n"
    "- summary: A concise summary of what panelists said to this question "
    "(2-4 sentences). When panelists cluster into identifiable subgroups "
    "(e.g. by occupation, background, or personality), surface this — "
    '"developers emphasized X while designers pushed back on Y".\n'
    "- themes: The main themes in the responses to this question.\n"
    "- agreements / disagreements / surprises: per the usual synthesis rubric, "
    "but scoped to this single question.\n"
    "- recommendation: A tactical suggestion based on the responses to this "
    "question. Leave empty if nothing actionable.\n\n"
    "Cite panelist names when useful. Be specific and substantive."
)

_REDUCE_PROMPT_TEMPLATE = (
    "You are producing the final cross-question synthesis of a focus group "
    "panel. Per-question summaries have already been produced by an earlier "
    "pass. Your input below is the list of those per-question summaries, "
    "presented as if each question summary were a 'panelist' answering the "
    "prompt 'summarize responses to this question'. Your job is to combine "
    "them into one coherent narrative that SURFACES CROSS-QUESTION PATTERNS — "
    "themes, tensions, and surprises that don't emerge from any single "
    "question alone.\n\n"
    "Produce a structured synthesis with:\n"
    "- summary: Overall findings across all questions (2-4 sentences).\n"
    "- themes: Cross-question themes, not question-specific ones. If the "
    "panel keeps circling the same underlying concern across different "
    "prompts, that's a cross-question theme.\n"
    "- agreements: Where the panel broadly agreed across questions.\n"
    "- disagreements: Recurring divisions or tensions visible across "
    "questions.\n"
    "- surprises: Patterns a single-question view would have missed.\n"
    "- recommendation: A single actionable recommendation rooted in the "
    "cross-question pattern.\n\n"
    "Do not re-summarize each question individually — that level of detail "
    "already exists in the per-question summaries. Synthesize ACROSS them."
)


def _char_to_token(s: str) -> int:
    """Rough token estimate: ~4 chars/token for English text."""
    return max(1, len(s) // 4)


_CONTEXT_WINDOWS: dict[str, int] = {
    # prefix match against the resolved canonical model id (lowercased)
    "claude-haiku": 200_000,
    "claude-sonnet": 200_000,
    "claude-opus": 200_000,
    "claude": 200_000,
    "gemini": 1_000_000,
    "grok": 128_000,
    "gpt-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4": 128_000,
    "qwen": 131_000,
    "deepseek": 128_000,
    "mistral": 128_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000
_CONTEXT_HEADROOM = 8_000


def resolve_context_window(model: str) -> int:
    """Return the context-window size (in tokens) for *model*.

    Prefix-matches against the resolved canonical model id. Unknown models
    fall through to a conservative 128k default. Intentionally a local
    lookup rather than the richer one introduced by sp-avmm so the two
    branches do not collide mid-flight — once sp-avmm lands we can
    consolidate on that helper.
    """
    canonical = resolve_alias(model).lower()
    # Longest-prefix-first so "claude-haiku" beats "claude"
    for prefix in sorted(_CONTEXT_WINDOWS.keys(), key=len, reverse=True):
        if canonical.startswith(prefix) or prefix in canonical:
            return _CONTEXT_WINDOWS[prefix]
    return _DEFAULT_CONTEXT_WINDOW


def estimate_single_pass_tokens(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    prompt: str | None = None,
) -> int:
    """Pessimistic pre-call token estimate for a ``single``-strategy call."""
    prompt_text = prompt or SYNTHESIS_PROMPT
    body = _format_panelist_data(panelist_results, questions)
    # scaffold overhead (tool definitions, role wrappers, schema) — give
    # 2k tokens of headroom so small panels that flirt with the boundary
    # prefer map-reduce.
    scaffold = 2_000
    return _char_to_token(prompt_text) + _char_to_token(body) + scaffold


def select_strategy(
    strategy: str,
    model: str,
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    prompt: str | None = None,
) -> str:
    """Resolve ``auto`` into ``single`` or ``map-reduce`` based on context fit.

    ``single`` and ``map-reduce`` pass through unchanged. For ``auto``,
    the single-pass token estimate is compared against the synthesis
    model's context window (less headroom); ``single`` when it fits,
    ``map-reduce`` otherwise.
    """
    if strategy not in SYNTHESIS_STRATEGIES:
        raise ValueError(f"invalid synthesis strategy: {strategy!r}. Expected one of {SYNTHESIS_STRATEGIES}.")
    if strategy in (STRATEGY_SINGLE, STRATEGY_MAP_REDUCE):
        return strategy
    est = estimate_single_pass_tokens(panelist_results, questions, prompt)
    limit = resolve_context_window(model) - _CONTEXT_HEADROOM
    return STRATEGY_SINGLE if est <= limit else STRATEGY_MAP_REDUCE


def _question_text(q: dict[str, Any] | str) -> str:
    return q.get("text", str(q)) if isinstance(q, dict) else str(q)


def _filter_panelist_to_question(
    pr: Any,
    question_text: str,
) -> Any:
    """Return a shallow copy of *pr* containing only responses for this question.

    Matches on question text; includes follow-ups that reference the same
    primary question text (they carry their own question string). Preserves
    persona_name, usage (zeroed — irrelevant to synthesis input), error.
    """
    # Local import to avoid an import cycle at module load time: orchestrator
    # imports from persistence which imports from synthesis in some paths.
    from synth_panel.orchestrator import PanelistResult

    kept: list[dict[str, Any]] = []
    for resp in pr.responses:
        q = resp.get("question", "")
        if q == question_text:
            kept.append(resp)
    return PanelistResult(
        persona_name=pr.persona_name,
        responses=kept,
        usage=ZERO_USAGE,
        error=pr.error,
        model=pr.model,
    )


def _format_persona_metadata(personas: list[dict[str, Any]] | None) -> str:
    """Produce a compact cluster-aware metadata block for the map prompt.

    Leaves the block empty when no personas are supplied — callers should
    skip appending it in that case.
    """
    if not personas:
        return ""
    lines: list[str] = ["## Panelist Backgrounds (for cluster-aware summary)"]
    for p in personas:
        name = p.get("name", "Anonymous")
        bits: list[str] = []
        if p.get("occupation"):
            bits.append(str(p["occupation"]))
        traits = p.get("personality_traits")
        if isinstance(traits, list):
            bits.append(", ".join(str(t) for t in traits))
        elif traits:
            bits.append(str(traits))
        if bits:
            lines.append(f"- {name}: {'; '.join(bits)}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _build_synthetic_reduce_panelists(
    map_results: list[SynthesisResult],
    questions: list[dict[str, Any]],
) -> list[Any]:
    """Wrap per-question map summaries as synthetic PanelistResults.

    Each question becomes one synthetic 'panelist' whose single response is
    the map-phase summary for that question. This reuses the existing
    ``_format_panelist_data`` + ``synthesize_panel`` pipeline for the reduce
    call without having to introduce a second content-formatting path.
    """
    from synth_panel.orchestrator import PanelistResult

    synthetic: list[Any] = []
    for i, (res, q) in enumerate(zip(map_results, questions)):
        q_text = _question_text(q)
        synthetic.append(
            PanelistResult(
                persona_name=f"Question {i + 1} summary",
                responses=[
                    {
                        "question": q_text,
                        "response": res.summary or "(no summary produced)",
                    }
                ],
                usage=ZERO_USAGE,
            )
        )
    return synthetic


def synthesize_panel_mapreduce(
    client: LLMClient,
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    panelist_model: str | None = None,
    panelist_cost: CostEstimate | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    personas: list[dict[str, Any]] | None = None,
    max_workers: int | None = None,
) -> SynthesisResult:
    """Map-reduce synthesis for large panels (sp-kkzz).

    Map phase: one :func:`synthesize_panel` call per question, run in
    parallel via :class:`ThreadPoolExecutor`. Each map call receives only
    the responses to its question (plus optional persona metadata for
    cluster-aware summaries). Reduce phase: a single
    :func:`synthesize_panel` call that combines the per-question summaries
    into the final cross-question synthesis.

    Returns a :class:`SynthesisResult` whose top-level fields hold the
    reduce-phase output (so downstream callers keep their current shape)
    and whose ``per_question_synthesis`` / ``map_cost_breakdown`` /
    ``reduce_cost_breakdown`` fields carry the map-reduce details.

    ``personas`` is optional; when supplied, background metadata is
    appended to every map prompt so summaries can surface cluster
    patterns ("developers said X, designers pushed back on Y").

    Custom map / reduce prompts are intentionally NOT accepted here —
    ``--synthesis-prompt`` forces ``strategy=single`` upstream. Adding
    map/reduce prompt overrides is tracked as future work.
    """
    resolved_model = model or panelist_model or _DEFAULT_MODEL
    if not questions:
        raise ValueError("synthesize_panel_mapreduce requires at least one question")

    # Pre-call cost estimate — we print a single line that covers all
    # map calls + reduce, so the operator sees one number rather than n+1.
    est_body = _format_panelist_data(panelist_results, questions)
    _print_cost_estimate(resolved_model, f"{_MAP_PROMPT_TEMPLATE}\n\n{est_body}", panelist_cost)

    persona_block = _format_persona_metadata(personas)
    map_prompt = _MAP_PROMPT_TEMPLATE
    if persona_block:
        map_prompt = f"{_MAP_PROMPT_TEMPLATE}\n\n{persona_block}"

    n = len(questions)
    workers = max_workers if max_workers is not None else min(max(n, 1), 8)

    def _run_one_map(idx: int) -> tuple[int, SynthesisResult]:
        q = questions[idx]
        q_text = _question_text(q)
        filtered = [_filter_panelist_to_question(pr, q_text) for pr in panelist_results]
        res = synthesize_panel(
            client,
            filtered,
            [q],
            model=resolved_model,
            panelist_model=panelist_model,
            custom_prompt=map_prompt,
            panelist_cost=None,
            temperature=temperature,
            top_p=top_p,
        )
        return idx, res

    map_results: list[SynthesisResult | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one_map, i) for i in range(n)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            map_results[idx] = res

    # Narrow to non-None — ThreadPoolExecutor either fills every slot or
    # raises. Keep the assert so mypy/readers see the invariant.
    assert all(r is not None for r in map_results)
    completed_maps: list[SynthesisResult] = [r for r in map_results if r is not None]

    # Reduce phase
    synthetic_panelists = _build_synthetic_reduce_panelists(completed_maps, questions)
    reduce_result = synthesize_panel(
        client,
        synthetic_panelists,
        questions,
        model=resolved_model,
        panelist_model=panelist_model,
        custom_prompt=_REDUCE_PROMPT_TEMPLATE,
        panelist_cost=None,
        temperature=temperature,
        top_p=top_p,
    )

    # Cost rollup
    total_usage: TokenUsage = ZERO_USAGE
    total_cost: CostEstimate = CostEstimate()
    map_breakdown: list[dict[str, Any]] = []
    for i, res in enumerate(completed_maps):
        total_usage = total_usage + res.usage
        total_cost = total_cost + res.cost
        map_breakdown.append(
            {
                "question_index": i,
                "tokens": res.usage.total_tokens,
                "cost_usd": round(res.cost.total_cost, 6),
                "is_fallback": res.is_fallback,
            }
        )
    total_usage = total_usage + reduce_result.usage
    total_cost = total_cost + reduce_result.cost
    reduce_breakdown: dict[str, Any] = {
        "tokens": reduce_result.usage.total_tokens,
        "cost_usd": round(reduce_result.cost.total_cost, 6),
        "is_fallback": reduce_result.is_fallback,
    }

    per_question: dict[int, str] = {i: res.summary for i, res in enumerate(completed_maps)}

    # If the reduce call itself failed, propagate the fallback flag so
    # downstream run_invalid gates still trigger. Map fallbacks leave the
    # reduce call able to recover, so we don't treat partial map failure
    # as total failure here.
    return SynthesisResult(
        summary=reduce_result.summary,
        themes=reduce_result.themes,
        agreements=reduce_result.agreements,
        disagreements=reduce_result.disagreements,
        surprises=reduce_result.surprises,
        recommendation=reduce_result.recommendation,
        usage=total_usage,
        cost=total_cost,
        model=resolved_model,
        is_fallback=reduce_result.is_fallback,
        error=reduce_result.error,
        strategy=STRATEGY_MAP_REDUCE,
        per_question_synthesis=per_question,
        map_cost_breakdown=map_breakdown,
        reduce_cost_breakdown=reduce_breakdown,
    )
