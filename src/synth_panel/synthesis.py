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
import re
import sys
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic import ValidationError as _PydanticValidationError

from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    TokenUsage,
    actual_cost_usd,
    estimate_cost,
    local_estimate_usd,
    lookup_pricing,
)
from synth_panel.llm.aliases import resolve_alias
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.prompts import SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_VERSION
from synth_panel.structured import StructuredOutputConfig, StructuredOutputEngine
from synth_panel.structured.models import PartialSummary

logger = logging.getLogger(__name__)

# Synthesis strategy identifiers.
STRATEGY_SINGLE = "single"
STRATEGY_MAP_REDUCE = "map-reduce"
STRATEGY_AUTO = "auto"
SYNTHESIS_STRATEGIES = (STRATEGY_SINGLE, STRATEGY_MAP_REDUCE, STRATEGY_AUTO)


# ---------------------------------------------------------------------------
# sy-huo: pyodide_safe_mode — async LLM DI for Cloudflare Python Workers.
# ---------------------------------------------------------------------------
#
# Synthpanel's default LLMClient uses threading.Lock + Semaphore + a
# token-bucket. None of those run under pyodide (Cloudflare's Python Worker
# runtime). To let boardroom-style consumers reuse their own async OpenRouter
# / Anthropic client end-to-end, ``synthesize_panel`` accepts an injected
# async client conforming to :class:`AsyncLLMClient` via ``llm_client=``.
#
# The protocol is intentionally narrow — one method, two required kwargs,
# a single text response — so any consumer can satisfy it with a 10-line
# adapter around their existing async stack.


@dataclass
class AsyncCompletion:
    """Result returned by :meth:`AsyncLLMClient.complete`.

    Attributes:
        text: The model's raw text output. Synthpanel will attempt to
            parse a JSON object matching the synthesis schema; on parse
            failure a :class:`SynthesisResult` with ``is_fallback=True``
            is returned.
        usage: Optional token accounting. When ``None``, :data:`ZERO_USAGE`
            is recorded — cost figures will read $0.00 unless the consumer
            populates usage itself.
    """

    text: str
    usage: TokenUsage | None = None


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Minimal async LLM client protocol for :func:`synthesize_panel` DI.

    Consumers implementing this protocol can be injected via the
    ``llm_client=`` kwarg on :func:`synthesize_panel`. When provided
    (and ``judge_enabled=True``), synthpanel skips its own provider
    resolution and routes the judge call through the consumer's async
    client — critical for Cloudflare Python Workers (pyodide) where
    synth_panel's threading primitives don't run.

    The protocol is intentionally narrow:

    * ``complete`` is an async method.
    * It receives the full prompt and resolved model alias as keyword
      arguments. ``max_tokens`` is forwarded for clients that honor it;
      clients that don't may ignore it.
    * It returns an :class:`AsyncCompletion` carrying the raw text
      output and optional token usage.

    Synthpanel asks the model to emit a single JSON object matching the
    internal synthesis schema; the consumer's client does NOT need to
    implement tool-use forcing. Synthpanel handles JSON parsing and
    falls back to an ``is_fallback=True`` :class:`SynthesisResult` if
    parsing fails.
    """

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
    ) -> AsyncCompletion: ...


def _typed_or_dict(extr: Any, attr: str) -> Any:
    """Read *attr* from *extr* whether it's a Pydantic model or a dict.

    v1.0.3 P2 lets ``extract_schema=`` resolve a Pydantic ``BaseModel``
    subclass; when it does, the orchestrator may attach a typed instance
    on ``responses[i]["extraction"]`` instead of the raw dict the
    pre-typed code path produced. Callers that consume extraction
    payloads should route every read through this helper so they accept
    both shapes — the typed form (new) and the dict form (legacy /
    fallback when no model was registered for the schema name).

    Returns ``None`` when *extr* is neither a ``BaseModel`` nor a
    ``dict``, or when the named attribute is absent. Callers that need
    a hard failure on a missing field should explicitly check for
    ``None``.
    """
    if isinstance(extr, BaseModel):
        return getattr(extr, attr, None)
    if isinstance(extr, dict):
        return extr.get(attr)
    return None


class MapPhaseFailure(RuntimeError):
    """A map-phase partial in :func:`synthesize_panel_mapreduce` is unusable.

    Raised at the map boundary (before the reduce call fans out) when a
    per-question map result is either an explicit fallback
    (``is_fallback=True``) or fails typed validation against
    :class:`synth_panel.structured.models.PartialSummary`. Surfacing the
    failure here turns a previously-silent "reduce got empty themes"
    flow into a loud, actionable error naming the offending question.
    """

    def __init__(
        self,
        message: str,
        *,
        question_index: int,
        is_fallback: bool = False,
        validation_error: _PydanticValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.question_index = question_index
        self.is_fallback = is_fallback
        self.validation_error = validation_error


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

# sp-g59o: heuristic threshold for detecting "schema honored in form, not in
# spirit" output — when every list field is empty but the recommendation
# field carries a long prose dump, the synthesizer most likely ignored the
# structure and consolidated everything into the unstructured slot. The
# 600-char cutoff sits comfortably between the largest healthy
# recommendation observed in the v0.12 variance probe (~507) and the
# smallest degenerate one (~1078).
_UNSTRUCTURED_RECOMMENDATION_CHAR_THRESHOLD = 600
_UNSTRUCTURED_OUTPUT_WARNING = (
    "synthesis structured output appears unstructured — model may have "
    "ignored schema (themes/agreements/disagreements/surprises all empty "
    "while recommendation carries long prose). Consider rerunning or "
    "switching synthesis model."
)


def _detect_unstructured_output(
    themes: list[str],
    agreements: list[str],
    disagreements: list[str],
    surprises: list[str],
    recommendation: str,
) -> bool:
    """Heuristic for the gemini-flash-lite-style schema-adherence flake.

    Returns True when every list field is empty AND the recommendation is
    longer than the threshold. The information content is similar in that
    case — the model just packed the whole synthesis into the one
    unstructured slot instead of partitioning across the schema.
    """
    if themes or agreements or disagreements or surprises:
        return False
    return len(recommendation) > _UNSTRUCTURED_RECOMMENDATION_CHAR_THRESHOLD


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
    # sp-g59o: post-parse heuristic warnings (e.g. degenerate structured
    # output where every list field is empty but the recommendation field
    # carries long prose). Empty by default; included in to_dict() only
    # when non-empty so the serialized shape is unchanged for healthy runs.
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_estimated_usd(self) -> float:
        """Local-pricing-table estimate for this synthesis's judge call(s) (sy-ye1).

        Always computed from ``self.usage`` against the local pricing
        table. For map-reduce strategies, ``self.usage`` already sums
        every map call + the reduce call, so this returns the combined
        synthesis-tier estimate. Pair with :pyattr:`cost_actual_usd` to
        compare estimate vs. real bill.
        """
        return local_estimate_usd(self.usage, self.model)

    @property
    def cost_actual_usd(self) -> float | None:
        """Provider-reported actual USD for this synthesis's judge call(s) (sy-ye1).

        Sum of ``usage.provider_reported_cost`` across map+reduce calls.
        ``None`` when the judge model is a direct provider (Anthropic,
        OpenAI, Google) that does not return per-call cost — distinct
        from $0.00.
        """
        return actual_cost_usd(self.usage)

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
            # sy-ye1: explicit estimate-vs-actual surface so downstream
            # budget reconciliation does not have to infer which kind of
            # number ``cost`` is. ``cost_estimated_usd`` is always the
            # local-pricing-table estimate; ``cost_actual_usd`` is the
            # provider-reported number, or null when the upstream call
            # did not return a cost.
            "cost_estimated_usd": round(self.cost_estimated_usd, 6),
            "cost_actual_usd": (round(self.cost_actual_usd, 6) if self.cost_actual_usd is not None else None),
            "model": self.model,
            "prompt_version": self.synthesis_prompt_version,
        }
        if self.strategy != STRATEGY_SINGLE:
            d["strategy"] = self.strategy
        if self.per_question_synthesis is not None:
            # JSON requires string keys; keep indices as strings in the
            # serialized shape so MCP / file persistence round-trips cleanly.
            d["per_question_synthesis"] = {str(k): v for k, v in self.per_question_synthesis.items()}
        if self.warnings:
            d["warnings"] = list(self.warnings)
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
            if resp.get("skipped_by_condition") or resp.get("skipped_by_budget"):
                continue
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
    client: LLMClient | None,
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    panelist_model: str | None = None,
    custom_prompt: str | None = None,
    panelist_cost: CostEstimate | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    pyodide_safe_mode: bool = False,
    llm_client: AsyncLLMClient | None = None,
    judge_enabled: bool = True,
) -> SynthesisResult | Coroutine[Any, Any, SynthesisResult]:
    """Synthesize panelist responses into a structured research finding.

    Args:
        client: LLM client for the synthesis call. May be ``None`` when
            ``judge_enabled=False`` or when ``llm_client`` is provided.
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
        pyodide_safe_mode: When ``True``, refuse code paths that spawn
            threads or import C-extension fetch deps. Requires either
            ``judge_enabled=False`` or ``llm_client=<AsyncLLMClient>``.
            Added in v1.2.0 (sy-huo) for Cloudflare Python Workers
            (pyodide) compatibility.
        llm_client: Optional async client conforming to
            :class:`AsyncLLMClient`. When provided AND
            ``judge_enabled=True``, synthpanel skips its own provider
            resolution + threading-based client and routes the judge
            call through the injected async client. **The function then
            returns an awaitable** — callers must ``await`` the result.
        judge_enabled: When ``False``, skip the judge LLM call entirely
            and return a degenerate :class:`SynthesisResult` with empty
            LLM-derived fields. Cost stays at zero. Default ``True``
            preserves v1.1.0 behavior.

    Returns:
        :class:`SynthesisResult` (sync) for the default and
        ``judge_enabled=False`` paths. When ``llm_client`` is provided
        and ``judge_enabled=True``, returns a coroutine that resolves
        to a :class:`SynthesisResult` — callers must ``await`` it.
    """
    # sy-huo: degenerate path — skip the judge LLM call entirely. No
    # client.send(), no thread spawn, no fetch deps imported. Returns
    # synchronously so callers in pyodide / CF Workers can use a single
    # code path without bridging to asyncio just to disable synthesis.
    if not judge_enabled:
        return _synthesize_panel_no_judge(
            panelist_results,
            questions,
            model=model,
            panelist_model=panelist_model,
        )

    # sy-huo: async DI path — consumer injected an async client (their
    # own OpenRouter / Anthropic / etc. stack). Skip internal provider
    # resolution and return a coroutine so the consumer's existing event
    # loop drives the call. Required for pyodide where threading.Lock /
    # ThreadPoolExecutor / Semaphore do not run.
    if llm_client is not None:
        return _synthesize_panel_async(
            panelist_results,
            questions,
            llm_client=llm_client,
            model=model,
            panelist_model=panelist_model,
            custom_prompt=custom_prompt,
        )

    # sy-huo: pyodide_safe_mode guard — once we get here, the call would
    # use the internal threading-based client. Refuse loudly rather than
    # silently spawning a thread the runtime can't actually run.
    if pyodide_safe_mode:
        raise ValueError(
            "synthesize_panel(..., pyodide_safe_mode=True) requires either "
            "judge_enabled=False or llm_client=<AsyncLLMClient>. The internal "
            "synth_panel LLMClient uses threading.Lock / Semaphore which "
            "do not run under pyodide / Cloudflare Python Workers."
        )

    if client is None:
        raise ValueError("synthesize_panel: client is required when judge_enabled=True and llm_client is not provided.")

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
        seed=seed,
    )

    # Convert usage
    usage = _convert_llm_usage(result.total_usage)
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
    required_keys = _SYNTHESIS_SCHEMA["required"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        error_msg = f"synthesizer returned partial schema: missing keys {missing}"
        logger.warning(
            "%s (model=%s, present_keys=%s)",
            error_msg,
            resolved_model,
            sorted(data.keys()),
        )
        return SynthesisResult(
            summary=data.get("summary", "") or "Synthesis failed — see error field.",
            themes=data.get("themes", []) if isinstance(data.get("themes"), list) else [],
            agreements=data.get("agreements", []) if isinstance(data.get("agreements"), list) else [],
            disagreements=data.get("disagreements", []) if isinstance(data.get("disagreements"), list) else [],
            surprises=data.get("surprises", []) if isinstance(data.get("surprises"), list) else [],
            recommendation=data.get("recommendation", ""),
            usage=usage,
            cost=cost,
            model=resolved_model,
            is_fallback=True,
            error=error_msg,
        )

    themes = data.get("themes", [])
    agreements = data.get("agreements", [])
    disagreements = data.get("disagreements", [])
    surprises = data.get("surprises", [])
    recommendation = data.get("recommendation", "")
    warnings: list[str] = []
    # sp-g59o: detect-and-warn for schema-adherence flake (gemini-flash-lite
    # was the trigger but this is provider-agnostic). The synthesis call
    # technically returned every required key, but every list is empty and
    # the recommendation slot carries a long prose dump — the model
    # ignored the structure. Surface a warning rather than silently
    # accepting degenerate output. Cheap, no retry, no extra cost.
    if _detect_unstructured_output(themes, agreements, disagreements, surprises, recommendation):
        logger.warning(
            "synthesis output appears unstructured (model=%s, recommendation_chars=%d): %s",
            resolved_model,
            len(recommendation),
            _UNSTRUCTURED_OUTPUT_WARNING,
        )
        warnings.append(_UNSTRUCTURED_OUTPUT_WARNING)

    return SynthesisResult(
        summary=data.get("summary", ""),
        themes=themes,
        agreements=agreements,
        disagreements=disagreements,
        surprises=surprises,
        recommendation=recommendation,
        usage=usage,
        cost=cost,
        model=resolved_model,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# sy-huo: pyodide-safe helpers — degenerate (no judge) + async DI paths.
# ---------------------------------------------------------------------------


def _synthesize_panel_no_judge(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    panelist_model: str | None = None,
) -> SynthesisResult:
    """Return a degenerate :class:`SynthesisResult` without calling an LLM.

    Used when ``judge_enabled=False``. Records the panelist + question
    counts in the summary for traceability but leaves every LLM-derived
    field empty so downstream consumers don't mistake an unsynthesized
    panel for one with empty findings.

    Pyodide-safe: no thread spawn, no fetch deps imported, no LLM call.
    """
    resolved_model = model or panelist_model or _DEFAULT_MODEL
    n_panelists = len(panelist_results)
    n_questions = len(questions)
    summary = (
        f"Judge disabled (judge_enabled=False). {n_panelists} panelist(s) "
        f"answered {n_questions} question(s); raw responses preserved in the "
        "underlying PanelistResult objects."
    )
    return SynthesisResult(
        summary=summary,
        themes=[],
        agreements=[],
        disagreements=[],
        surprises=[],
        recommendation="",
        usage=ZERO_USAGE,
        cost=CostEstimate(),
        model=resolved_model,
    )


# Schema-hint suffix appended to the synthesis prompt when running through
# an injected async client. Tool-use forcing isn't part of the injected
# protocol — consumers point synthpanel at any text-completion endpoint —
# so we explicitly request a JSON object matching the synthesis schema.
_ASYNC_JSON_SUFFIX = (
    "\n\n---\n"
    "Return ONLY a valid JSON object matching this exact schema, with no "
    "prose, no commentary, and no markdown fences:\n\n"
    "{\n"
    '  "summary": "<2-4 sentence summary>",\n'
    '  "themes": ["<theme>", ...],\n'
    '  "agreements": ["<agreement>", ...],\n'
    '  "disagreements": ["<disagreement>", ...],\n'
    '  "surprises": ["<surprise>", ...],\n'
    '  "recommendation": "<actionable recommendation>"\n'
    "}"
)


def _parse_async_synthesis_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an async client's text response.

    Tolerates a leading ```json fence (some models still emit them despite
    instructions), trailing prose, or a JSON object embedded in the middle
    of text. Returns ``None`` when no parseable JSON object is found.
    """
    cleaned = text.strip()

    # Strip code fences if present.
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Try direct parse first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to scanning for the first balanced {...} block.
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


async def _synthesize_panel_async(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    llm_client: AsyncLLMClient,
    model: str | None = None,
    panelist_model: str | None = None,
    custom_prompt: str | None = None,
) -> SynthesisResult:
    """Async synthesis path that drives a consumer-supplied async LLM client.

    Skips ``StructuredOutputEngine`` (which relies on the threading-based
    :class:`LLMClient`) and asks the injected client to emit a JSON object
    matching the synthesis schema. Returns a fallback result on parse
    failure rather than raising, so a flaky async client doesn't bubble
    out of synthesis into the consumer's request handler.

    Pyodide-safe: no threading, no Semaphore, no internal provider stack.
    """
    resolved_model = model or panelist_model or _DEFAULT_MODEL
    prompt_text = custom_prompt or SYNTHESIS_PROMPT
    panelist_data = _format_panelist_data(panelist_results, questions)
    full_prompt = f"{prompt_text}\n\n{panelist_data}{_ASYNC_JSON_SUFFIX}"

    completion = await llm_client.complete(
        prompt=full_prompt,
        model=resolved_model,
        max_tokens=_MAX_TOKENS,
    )

    text = completion.text or ""
    usage = completion.usage if completion.usage is not None else ZERO_USAGE
    pricing, _ = lookup_pricing(resolved_model)
    cost = estimate_cost(usage, pricing)

    data = _parse_async_synthesis_json(text)
    if data is None:
        return SynthesisResult(
            summary="Synthesis failed — injected async client returned non-JSON output.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            usage=usage,
            cost=cost,
            model=resolved_model,
            is_fallback=True,
            error="async_client_returned_unparseable_output",
        )

    def _list_field(key: str) -> list[str]:
        val = data.get(key)
        return val if isinstance(val, list) else []

    return SynthesisResult(
        summary=str(data.get("summary", "")),
        themes=_list_field("themes"),
        agreements=_list_field("agreements"),
        disagreements=_list_field("disagreements"),
        surprises=_list_field("surprises"),
        recommendation=str(data.get("recommendation", "")),
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

# sp-4g6a: inner reduce used when a single question's responses overflow the
# synthesis model's context and we partition the panelists into sub-batches.
# Each batch's map-phase summary becomes the "panelist response" here, and
# this reduce combines them into ONE question-level summary. The outer
# cross-question reduce then consumes that summary as usual.
_INNER_REDUCE_PROMPT_TEMPLATE = (
    "You are combining partial summaries of panelist responses to ONE "
    "specific question. An earlier pass split the panel into batches — "
    "each batch was summarized separately. Your job is to merge those "
    "batch summaries into a single coherent per-question synthesis, "
    "preserving the usual rubric (summary / themes / agreements / "
    "disagreements / surprises / recommendation) but scoped to this one "
    "question.\n\n"
    "Do not re-summarize each batch individually; combine them. When a "
    "theme or tension recurs across batches, surface it. When batches "
    "disagree, flag the disagreement. Leave recommendation empty if "
    "nothing actionable emerges."
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

# sp-4g6a: target for --synthesis-auto-escalate. Chosen because it is the
# cheapest 1M-context model available through the default provider set.
# Operators can still pick gemini-2.5-pro manually via --synthesis-model.
_ESCALATION_MODEL = "gemini-2.5-flash-lite"


class MapChunkOverflowError(RuntimeError):
    """A single map-phase chunk would not fit in the synthesis model's context.

    Raised from :func:`synthesize_panel_mapreduce` when any per-question
    map call's pessimistic token estimate exceeds the resolved model's
    context window (less headroom). Carries a diagnostic dict that names
    the offending question so callers can report it.
    """

    def __init__(self, message: str, *, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


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
        if resp.get("skipped_by_condition") or resp.get("skipped_by_budget"):
            continue
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


def _partition_panelists_for_context(
    panelist_results: list[Any],
    question: dict[str, Any] | str,
    map_prompt: str,
    context_limit: int,
) -> list[list[Any]] | None:
    """Greedy partition of *panelist_results* into batches fitting *context_limit*.

    Walks panelists in order and grows the current batch while its estimated
    single-pass token count stays below the limit. Starts a new batch when
    adding the next panelist would overflow. Returns ``None`` when a single
    panelist's response already exceeds the limit — sub-chunking cannot help
    in that case and the caller must escalate the model or error.
    """
    q_dict: dict[str, Any] = {"text": question} if isinstance(question, str) else question
    batches: list[list[Any]] = []
    current: list[Any] = []
    for pr in panelist_results:
        candidate = [*current, pr]
        est = estimate_single_pass_tokens(candidate, [q_dict], prompt=map_prompt)
        if est <= context_limit:
            current = candidate
            continue
        if not current:
            # A single panelist's response does not fit — sub-chunking is
            # structurally unable to resolve this. Caller handles.
            return None
        batches.append(current)
        current = [pr]
    if current:
        batches.append(current)
    return batches


def _sub_chunk_question_synthesis(
    client: LLMClient,
    filtered_panelists: list[Any],
    question: dict[str, Any] | str,
    map_prompt: str,
    *,
    model: str,
    panelist_model: str | None,
    context_limit: int,
    temperature: float | None,
    top_p: float | None,
    seed: int | None = None,
    question_index: int | None = None,
) -> tuple[SynthesisResult, int]:
    """Synthesize one overflowing question by partitioning panelists (sp-4g6a).

    Splits the filtered panelists into batches that each fit within
    *context_limit*, runs one :func:`synthesize_panel` per batch with the
    supplied *map_prompt*, then performs an inner reduce that combines the
    batch summaries into a single per-question :class:`SynthesisResult`.
    The returned result's ``usage`` / ``cost`` / ``is_fallback`` aggregate
    every underlying call (batch maps + inner reduce) so outer roll-ups
    stay accurate.

    Returns ``(result, batch_count)``. Raises :class:`MapChunkOverflowError`
    when partitioning fails — i.e. when a single panelist's response
    cannot fit the context limit even alone.
    """
    # Normalize the question to a dict once so every downstream call sees a
    # consistent shape (estimate_single_pass_tokens + synthesize_panel both
    # expect list[dict[str, Any]]).
    q_dict: dict[str, Any] = {"text": question} if isinstance(question, str) else question
    batches = _partition_panelists_for_context(filtered_panelists, q_dict, map_prompt, context_limit)
    if batches is None:
        q_text = _question_text(q_dict)
        # Estimate tokens for a single panelist so the diagnostic can report
        # what the sub-chunker saw as unsplittable.
        worst_est = 0
        for pr in filtered_panelists:
            est = estimate_single_pass_tokens([pr], [q_dict], prompt=map_prompt)
            if est > worst_est:
                worst_est = est
        diag: dict[str, Any] = {
            "question_text": q_text,
            "estimated_tokens": worst_est,
            "effective_limit": context_limit,
            "synthesis_model": model,
            "reason": "single_panelist_overflow",
        }
        if question_index is not None:
            diag["question_index"] = question_index
        raise MapChunkOverflowError(
            (
                f"map-reduce sub-chunk failed for question '{q_text[:60]}': "
                f"a single panelist's response (~{worst_est} tokens) exceeds "
                f"the effective limit ({context_limit} tokens). Rerun with a "
                "larger-context synthesis model (e.g. --synthesis-auto-escalate "
                "or --synthesis-model gemini-2.5-flash-lite)."
            ),
            diagnostic=diag,
        )

    batch_results: list[SynthesisResult] = []
    for batch in batches:
        res = synthesize_panel(
            client,
            batch,
            [q_dict],
            model=model,
            panelist_model=panelist_model,
            custom_prompt=map_prompt,
            panelist_cost=None,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        batch_results.append(res)

    # Inner reduce: wrap each batch summary as a synthetic panelist and run
    # the standard synthesis pipeline with a reduce-style prompt.
    synthetic = _build_synthetic_reduce_panelists(batch_results, [q_dict] * len(batch_results))
    inner = synthesize_panel(
        client,
        synthetic,
        [q_dict],
        model=model,
        panelist_model=panelist_model,
        custom_prompt=_INNER_REDUCE_PROMPT_TEMPLATE,
        panelist_cost=None,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    # Aggregate usage / cost across every batch call + the inner reduce
    # so the outer map_breakdown entry for this question reflects the
    # full work done.
    total_usage: TokenUsage = inner.usage
    total_cost: CostEstimate = inner.cost
    for br in batch_results:
        total_usage = total_usage + br.usage
        total_cost = total_cost + br.cost
    any_fallback = inner.is_fallback or any(br.is_fallback for br in batch_results)

    aggregated = SynthesisResult(
        summary=inner.summary,
        themes=inner.themes,
        agreements=inner.agreements,
        disagreements=inner.disagreements,
        surprises=inner.surprises,
        recommendation=inner.recommendation,
        usage=total_usage,
        cost=total_cost,
        model=model,
        is_fallback=any_fallback,
        error=inner.error,
    )
    return aggregated, len(batches)


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
    seed: int | None = None,
    personas: list[dict[str, Any]] | None = None,
    max_workers: int | None = None,
    auto_escalate: bool = False,
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

    Per-chunk overflow handling (sp-4g6a):

    * By default, when a single question's responses exceed the model's
      context window, the panelists for that question are partitioned
      into sub-batches that each fit, every batch is summarized, and an
      inner reduce combines the batch summaries into a single per-question
      summary. This preserves single-model semantics and keeps the
      overall pipeline shape identical.
    * When ``auto_escalate=True`` and the resolved synthesis model has a
      smaller context window than the escalation target, an overflowing
      question's map call is instead retried on a larger-context model
      (default: ``gemini-2.5-flash-lite``, 1M ctx) with a visible warning.
      Sub-chunking still runs if the escalated window is also insufficient.
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

    # sp-exu6: per-map-call overflow pre-flight. Each map call only sees
    # responses for its own question, so the budget is much smaller than
    # the full-panel synthesis. Still, a single question with very long
    # responses (or a large cluster-metadata block) could overflow.
    #
    # sp-4g6a: when it does, we no longer raise immediately. Build a
    # per-question plan:
    #
    # * "direct"     → fits; run one map call on ``resolved_model``.
    # * "escalate"   → does not fit ``resolved_model`` but fits a larger-
    #                  context escalation target, and the caller opted in
    #                  via ``auto_escalate=True``. Run one map call on the
    #                  escalated model (with a visible warning).
    # * "sub_chunk"  → doesn't fit (even after any escalation). Partition
    #                  panelists into batches that each fit, and run an
    #                  inner reduce to produce one per-question summary.
    #
    # The plan is resolved up front so we can surface escalation warnings
    # in deterministic question order before any API calls fan out.
    base_limit = resolve_context_window(resolved_model) - _CONTEXT_HEADROOM
    escalated_limit = resolve_context_window(_ESCALATION_MODEL) - _CONTEXT_HEADROOM
    escalation_available = auto_escalate and escalated_limit > base_limit

    plans: list[dict[str, Any]] = []
    for idx in range(n):
        q = questions[idx]
        q_text = _question_text(q)
        filtered = [_filter_panelist_to_question(pr, q_text) for pr in panelist_results]
        est = estimate_single_pass_tokens(filtered, [q], prompt=map_prompt)
        if est <= base_limit:
            plans.append({"mode": "direct", "model": resolved_model, "filtered": filtered, "est": est})
            continue
        if escalation_available and est <= escalated_limit:
            print(
                f"warning: auto-escalated synthesis from {resolved_model} "
                f"to {_ESCALATION_MODEL} for question {idx} "
                f"(~{est} tokens > {base_limit} effective limit).",
                file=sys.stderr,
            )
            plans.append(
                {
                    "mode": "escalate",
                    "model": _ESCALATION_MODEL,
                    "filtered": filtered,
                    "est": est,
                    "original_model": resolved_model,
                }
            )
            continue
        # Sub-chunk. Use the escalated model's limit when available so we
        # can pack larger batches; otherwise stay on the resolved model.
        sub_model = _ESCALATION_MODEL if escalation_available else resolved_model
        sub_limit = escalated_limit if escalation_available else base_limit
        plans.append(
            {
                "mode": "sub_chunk",
                "model": sub_model,
                "filtered": filtered,
                "est": est,
                "context_limit": sub_limit,
            }
        )

    def _run_one_map(idx: int) -> tuple[int, SynthesisResult, dict[str, Any]]:
        plan = plans[idx]
        q = questions[idx]
        filtered = plan["filtered"]
        meta: dict[str, Any] = {"mode": plan["mode"], "model": plan["model"]}
        if plan["mode"] == "sub_chunk":
            res, batch_count = _sub_chunk_question_synthesis(
                client,
                filtered,
                q,
                map_prompt,
                model=plan["model"],
                panelist_model=panelist_model,
                context_limit=plan["context_limit"],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                question_index=idx,
            )
            meta["batch_count"] = batch_count
            return idx, res, meta
        res = synthesize_panel(
            client,
            filtered,
            [q],
            model=plan["model"],
            panelist_model=panelist_model,
            custom_prompt=map_prompt,
            panelist_cost=None,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return idx, res, meta

    # sy-2wa: lazy concurrent.futures import. Keeps `from synth_panel.ensemble
    # import synthesize_panel` ThreadPoolExecutor-free for pyodide consumers
    # (CF Python Workers), where `.submit()` silently hangs. Map-reduce
    # is opt-in via STRATEGY_MAP_REDUCE and never reached on pyodide.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    map_results: list[SynthesisResult | None] = [None] * n
    map_meta: list[dict[str, Any] | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one_map, i) for i in range(n)]
        for fut in as_completed(futures):
            idx, res, meta = fut.result()
            map_results[idx] = res
            map_meta[idx] = meta

    # Narrow to non-None — ThreadPoolExecutor either fills every slot or
    # raises. Keep the assert so mypy/readers see the invariant.
    assert all(r is not None for r in map_results)
    assert all(m is not None for m in map_meta)
    completed_maps: list[SynthesisResult] = [r for r in map_results if r is not None]
    completed_meta: list[dict[str, Any]] = [m for m in map_meta if m is not None]

    # v1.0.3 P2: typed validation at the map boundary. Every partial is
    # routed through :class:`PartialSummary.model_validate` so schema
    # drift in a single map call surfaces as a clear ValidationError
    # instead of silently feeding empty themes to reduce. An explicit
    # ``is_fallback=True`` map (the synthesizer already detected its
    # own failure) raises the same error class so the caller's
    # ``synthesis_error`` envelope can name the offending question.
    typed_partials: list[PartialSummary] = []
    for i, res in enumerate(completed_maps):
        if res.is_fallback:
            raise MapPhaseFailure(
                f"map-reduce question {i} map call returned a fallback "
                f"(error={res.error!r}); reduce stage cannot proceed with a "
                "degenerate partial. Rerun with a different synthesis model "
                "or reduce panel size for this question.",
                question_index=i,
                is_fallback=True,
            )
        try:
            typed_partials.append(
                PartialSummary.model_validate(
                    {
                        "summary": res.summary,
                        "themes": res.themes,
                        "agreements": res.agreements,
                        "disagreements": res.disagreements,
                        "surprises": res.surprises,
                        "recommendation": res.recommendation,
                    }
                )
            )
        except _PydanticValidationError as ve:
            raise MapPhaseFailure(
                f"map-reduce question {i} partial failed typed validation: {ve}",
                question_index=i,
                validation_error=ve,
            ) from ve

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
        seed=seed,
    )

    # Cost rollup
    total_usage: TokenUsage = ZERO_USAGE
    total_cost: CostEstimate = CostEstimate()
    map_breakdown: list[dict[str, Any]] = []
    for i, (res, meta) in enumerate(zip(completed_maps, completed_meta)):
        total_usage = total_usage + res.usage
        total_cost = total_cost + res.cost
        # sy-ye1: surface estimate + actual side-by-side per map call so
        # audits can spot the (rare) case of a single question whose
        # provider bill diverges from the local estimate.
        map_actual = res.cost_actual_usd
        entry: dict[str, Any] = {
            "question_index": i,
            "tokens": res.usage.total_tokens,
            "cost_usd": round(res.cost.total_cost, 6),
            "cost_estimated_usd": round(res.cost_estimated_usd, 6),
            "cost_actual_usd": (round(map_actual, 6) if map_actual is not None else None),
            "is_fallback": res.is_fallback,
        }
        # sp-4g6a: surface per-question overflow handling in the breakdown
        # so operators can tell at a glance which questions needed a
        # different treatment and why.
        if meta["mode"] == "sub_chunk":
            entry["sub_chunked"] = True
            entry["batch_count"] = meta.get("batch_count")
            entry["model"] = meta["model"]
        elif meta["mode"] == "escalate":
            entry["escalated_model"] = meta["model"]
            entry["model"] = meta["model"]
        map_breakdown.append(entry)
    total_usage = total_usage + reduce_result.usage
    total_cost = total_cost + reduce_result.cost
    reduce_actual = reduce_result.cost_actual_usd
    reduce_breakdown: dict[str, Any] = {
        "tokens": reduce_result.usage.total_tokens,
        "cost_usd": round(reduce_result.cost.total_cost, 6),
        # sy-ye1: split the reduce-phase cost the same way as map entries
        # so consumers see both numbers across the whole map-reduce chain.
        "cost_estimated_usd": round(reduce_result.cost_estimated_usd, 6),
        "cost_actual_usd": (round(reduce_actual, 6) if reduce_actual is not None else None),
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
