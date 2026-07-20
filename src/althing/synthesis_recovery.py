"""Synthesis failure recovery ladder (sp-rcvr).

Motivated by a thrice-reproduced production failure: a 20-persona panel
whose questions embed ~14k-char page dumps produced a synthesis prompt
that PASSED the local pre-flight (the estimate fit the model's documented
context window) yet deterministically failed with::

    OpenRouter (downstream: Azure) API error 400: Provider returned error

OpenRouter had routed the call to a downstream deployment with a smaller
effective context window than the model's documented one, so the
oversized prompt 400'd — and Althing classified the 400 as a fatal,
non-retryable API error. The map-reduce fallback that exists precisely
for context overflow (``--synthesis-strategy=auto``) never fired because
it is only triggered by the PRE-FLIGHT estimate, never by a downstream
rejection.

This module wraps the synthesis stage (both the in-run call and
``panel synthesize``) in a bounded recovery ladder:

1. **Classify** the failure: overflow / transient / fatal, using the
   existing token-estimate machinery plus error-text heuristics.
2. **Retry once** (jittered) for transient-class errors (429 / 5xx /
   timeout / retries-exhausted-on-retryable).
3. **Overflow path**: route into the existing map-reduce synthesis even
   when the overflow was only revealed by a downstream 4xx.
4. **Reroute rung** (OpenRouter only): when the error names the
   downstream provider that rejected the call, retry once with
   ``provider: {"ignore": [<that provider>]}`` routing preferences.
5. **Last resort**: raise :class:`SynthesisRecoveryError`; callers build
   the loud ``synthesis_error`` envelope, whose message now names a
   concrete ``panel synthesize --synthesis-model <suggestion>`` recovery
   command.

Each rung logs exactly one stderr line. Every rung runs at most once, so
the ladder is strictly bounded: primary + transient retry + reroute +
one map-reduce pass, worst case.

Panelist calls are NOT routed through this ladder — they already have
their own retry semantics in ``LLMClient`` / ``RetryPolicy``.
"""

from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from althing.llm.client import LLMClient
from althing.llm.errors import LLMError, LLMErrorCategory
from althing.synthesis import (
    SynthesisResult,
    estimate_single_pass_tokens,
    resolve_context_window,
    synthesize_panel,
    synthesize_panel_mapreduce,
)

# Headroom mirrored from the pre-flight machinery (_runners /
# synthesis._CONTEXT_HEADROOM) so the post-hoc classification agrees with
# the pre-flight boundary.
_CLASSIFY_HEADROOM_TOKENS = 8_000

# A downstream 4xx on a prompt at least this large is treated as a
# *suspected* downstream context overflow after the reroute rung has also
# failed: aggregator downstreams (Azure, Bedrock regional deployments)
# routinely expose smaller effective windows than the model's documented
# one, and the pre-flight can only check the documented number. 32k is
# comfortably above any healthy synthesis prompt that a 400 could reject
# for size-unrelated reasons, and below every documented modern window.
_SUSPECTED_DOWNSTREAM_OVERFLOW_TOKENS = 32_000

# Error-text heuristics for context/length failures. Providers phrase
# overflow rejections inconsistently; these are the recurring fragments.
_OVERFLOW_PATTERNS = re.compile(
    r"context[ _-]?(?:window|length|limit)"
    r"|maximum (?:context|prompt|input)"
    r"|prompt is too long"
    r"|input is too long"
    r"|too many (?:input )?tokens"
    r"|token limit"
    r"|exceeds? the (?:maximum|token|context)"
    r"|max(?:imum)?_? ?(?:input_)?tokens? (?:exceeded|limit)"
    r"|content too large"
    r"|request too large"
    r"|payload too large",
    re.IGNORECASE,
)

_TRANSIENT_CATEGORIES = frozenset(
    {
        LLMErrorCategory.TRANSPORT,
        LLMErrorCategory.RATE_LIMIT,
        LLMErrorCategory.SERVER_ERROR,
        LLMErrorCategory.RETRIES_EXHAUSTED,
    }
)


class SynthesisFailureClass(Enum):
    """Classification of a synthesis-stage provider failure."""

    OVERFLOW = "overflow"
    TRANSIENT = "transient"
    FATAL = "fatal"


def _underlying_llm_error(exc: BaseException) -> LLMError | None:
    """Walk the ``__cause__`` chain and return the deepest ``LLMError``.

    ``RetryPolicy`` wraps exhausted retryables in a ``RETRIES_EXHAUSTED``
    ``LLMError`` whose cause is the real provider error; classification
    and reroute both want the innermost one.
    """
    found: LLMError | None = None
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, LLMError):
            found = node
        node = node.__cause__
    return found


def downstream_provider_of(exc: BaseException) -> str | None:
    """Extract the downstream provider named by an OpenRouter error.

    Prefers the structured ``LLMError.downstream_provider`` attribute;
    falls back to matching the ``(downstream: <name>)`` fragment that
    :mod:`althing.llm.providers.openrouter` embeds in the message.
    """
    node: BaseException | None = exc
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        downstream = getattr(node, "downstream_provider", None)
        if isinstance(downstream, str) and downstream:
            return downstream
        node = node.__cause__
    m = re.search(r"downstream:\s*([A-Za-z0-9][A-Za-z0-9 ._/-]*?)\)", str(exc))
    if m:
        return m.group(1).strip()
    return None


def provider_slug(name: str) -> str:
    """Normalize a provider display name into an OpenRouter routing slug.

    OpenRouter's ``provider.ignore`` field takes lowercase slugs
    ("azure", "google-vertex", "amazon-bedrock"); error metadata may
    carry display forms ("Azure", "Amazon Bedrock").
    """
    return re.sub(r"\s+", "-", name.strip().lower())


def classify_synthesis_failure(
    exc: BaseException,
    *,
    estimated_tokens: int,
    context_window: int,
    headroom_tokens: int = _CLASSIFY_HEADROOM_TOKENS,
) -> SynthesisFailureClass:
    """Classify a synthesis-call failure as overflow / transient / fatal.

    Overflow when either (a) the prompt's token estimate exceeds the
    model's documented window less headroom — the pre-flight boundary —
    or (b) the error text carries context/length/token keywords.
    Transient for 429 / 5xx / transport / retries-exhausted. Everything
    else (plain 4xx, auth, deserialization, non-LLM exceptions) is fatal.
    """
    text = str(exc)
    llm_err = _underlying_llm_error(exc)
    if llm_err is not None and llm_err.__cause__ is not None:
        text = f"{text} {llm_err.__cause__}"

    if estimated_tokens > max(0, context_window - headroom_tokens):
        return SynthesisFailureClass.OVERFLOW
    if _OVERFLOW_PATTERNS.search(text):
        return SynthesisFailureClass.OVERFLOW

    if llm_err is not None:
        category = llm_err.category
        if category == LLMErrorCategory.RETRIES_EXHAUSTED:
            inner = llm_err.__cause__
            if isinstance(inner, LLMError):
                category = inner.category
        if category in _TRANSIENT_CATEGORIES:
            return SynthesisFailureClass.TRANSIENT
        return SynthesisFailureClass.FATAL

    # Non-LLMError (unexpected bug, JSON error, ...) — not safe to retry.
    return SynthesisFailureClass.FATAL


def suggest_recovery_model(model: str | None) -> str:
    """Concrete large-context synthesis model for the guided fallback.

    Stays in the credential family the failing model implies: an
    ``openrouter/``-routed model gets an OpenRouter-routed suggestion so
    the recovery command works with the key the operator already has.
    """
    if model and model.startswith("openrouter/"):
        return "openrouter/google/gemini-2.5-flash"
    if model and "gemini" in model.lower():
        return "gemini-2.5-pro"
    return "gemini-2.5-flash-lite"


def format_recovery_command(result_id: str | None, model: str | None) -> str:
    """Render the exact ``panel synthesize`` command for the fallback message."""
    ref = result_id or "<result-id>"
    return f"althing panel synthesize {ref} --synthesis-model {suggest_recovery_model(model)}"


class SynthesisRecoveryError(RuntimeError):
    """Every rung of the synthesis recovery ladder failed (or was skipped).

    Carries the final underlying exception, the rung log, and the
    concrete guided-recovery command so callers can build the loud
    ``synthesis_error`` envelope without re-deriving anything.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException,
        rungs: list[str],
        suggested_command: str,
        map_reduce_blocked_by_strategy: bool = False,
    ) -> None:
        super().__init__(message)
        self.__cause__ = cause
        self.rungs = rungs
        self.suggested_command = suggested_command
        self.map_reduce_blocked_by_strategy = map_reduce_blocked_by_strategy


@dataclass
class RecoveryContext:
    """Inputs shared by every rung of the recovery ladder."""

    client: LLMClient
    panelist_results: list[Any]
    questions: list[dict[str, Any]]
    synthesis_model: str | None = None
    panelist_model: str | None = None
    custom_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    personas: list[dict[str, Any]] | None = None
    # False when the operator explicitly chose --synthesis-strategy=single
    # (or a custom synthesis prompt forces single); the overflow rung is
    # then skipped and the final error says so.
    allow_map_reduce: bool = True
    auto_escalate: bool = False
    # Saved result id, when known (panel synthesize path) — used to render
    # a copy-pasteable guided-recovery command.
    result_id: str | None = None
    rungs: list[str] = field(default_factory=list)

    @property
    def effective_model(self) -> str:
        return self.synthesis_model or self.panelist_model or "sonnet"


def _log_rung(ctx: RecoveryContext, line: str) -> None:
    ctx.rungs.append(line)
    print(f"synthesis-recovery: {line}", file=sys.stderr)


def _single_call(ctx: RecoveryContext, provider_routing: dict[str, Any] | None = None) -> SynthesisResult:
    result = synthesize_panel(
        ctx.client,
        ctx.panelist_results,
        ctx.questions,
        model=ctx.synthesis_model,
        panelist_model=ctx.panelist_model,
        custom_prompt=ctx.custom_prompt,
        panelist_cost=None,
        temperature=ctx.temperature,
        top_p=ctx.top_p,
        seed=ctx.seed,
        provider_routing=provider_routing,
    )
    assert isinstance(result, SynthesisResult)  # sync path (no llm_client DI)
    return result


def _map_reduce_call(ctx: RecoveryContext, context_limit_override: int | None = None) -> SynthesisResult:
    return synthesize_panel_mapreduce(
        ctx.client,
        ctx.panelist_results,
        ctx.questions,
        model=ctx.synthesis_model,
        panelist_model=ctx.panelist_model,
        panelist_cost=None,
        temperature=ctx.temperature,
        top_p=ctx.top_p,
        seed=ctx.seed,
        personas=ctx.personas,
        auto_escalate=ctx.auto_escalate,
        context_limit_override=context_limit_override,
    )


def _downstream_revealed_limit(estimated_tokens: int, documented_window: int) -> int | None:
    """Effective per-call token cap for a downstream-revealed overflow.

    When a provider rejects a prompt whose estimate FITS the documented
    window, the serving deployment's effective context (or our ~4-chars/
    token estimate) is smaller than documented. Re-running map-reduce
    against the documented number would just re-send the same oversized
    chunk, so cap each call at half the rejected estimate (floored at
    the suspected-overflow threshold) to force real splitting. Returns
    ``None`` for true documented-window overflow — normal map-reduce
    planning already splits those.
    """
    if estimated_tokens > documented_window - _CLASSIFY_HEADROOM_TOKENS:
        return None
    return max(_SUSPECTED_DOWNSTREAM_OVERFLOW_TOKENS, estimated_tokens // 2)


def recover_synthesis_failure(exc: BaseException, ctx: RecoveryContext) -> SynthesisResult:
    """Run the recovery ladder after a failed single-strategy synthesis call.

    *exc* is the exception raised by the primary call. Returns the first
    successful :class:`SynthesisResult`; raises
    :class:`SynthesisRecoveryError` when the ladder is exhausted. Bounded:
    at most one transient retry, one reroute attempt, and one map-reduce
    pass — no rung ever repeats.
    """
    estimated = estimate_single_pass_tokens(ctx.panelist_results, ctx.questions, ctx.custom_prompt)
    window = resolve_context_window(ctx.effective_model)
    last_exc = exc
    map_reduce_blocked = False
    tried_map_reduce = False
    tried_reroute = False
    tried_transient_retry = False

    def classify(e: BaseException) -> SynthesisFailureClass:
        return classify_synthesis_failure(e, estimated_tokens=estimated, context_window=window)

    failure_class = classify(exc)
    _log_rung(
        ctx,
        f"synthesis call failed (class={failure_class.value}, ~{estimated} tokens vs "
        f"{window}-token window for {ctx.effective_model}): {str(exc)[:200]}",
    )

    # Rung 2: one bounded, jittered retry for transient-class errors.
    if failure_class is SynthesisFailureClass.TRANSIENT:
        tried_transient_retry = True
        delay = random.uniform(0.5, 2.0)
        _log_rung(ctx, f"rung=retry transient error; retrying once in {delay:.1f}s")
        time.sleep(delay)
        try:
            return _single_call(ctx)
        except Exception as exc2:
            last_exc = exc2
            failure_class = classify(exc2)
            _log_rung(ctx, f"rung=retry failed (class={failure_class.value}): {str(exc2)[:200]}")

    # Rung 3: overflow → the existing map-reduce machinery, even when the
    # overflow was only revealed by a downstream rejection.
    if failure_class is SynthesisFailureClass.OVERFLOW:
        if ctx.allow_map_reduce:
            tried_map_reduce = True
            override = _downstream_revealed_limit(estimated, window)
            _log_rung(
                ctx,
                "rung=map-reduce classified as context overflow; falling back to map-reduce synthesis"
                + (f" (effective per-call limit capped at ~{override} tokens)" if override is not None else ""),
            )
            try:
                return _map_reduce_call(ctx, context_limit_override=override)
            except Exception as exc3:
                last_exc = exc3
                _log_rung(ctx, f"rung=map-reduce failed: {str(exc3)[:200]}")
        else:
            map_reduce_blocked = True
            _log_rung(
                ctx,
                "rung=map-reduce skipped: --synthesis-strategy=single (or a custom "
                "synthesis prompt) disables the map-reduce fallback",
            )

    # Rung 4: OpenRouter reroute — the error names the downstream provider
    # that rejected the call; retry once excluding it.
    downstream = downstream_provider_of(last_exc)
    if downstream is not None and not tried_reroute and not tried_map_reduce:
        tried_reroute = True
        slug = provider_slug(downstream)
        _log_rung(
            ctx,
            f"rung=reroute downstream provider '{downstream}' rejected the call; "
            f"retrying once via OpenRouter with provider ignore=[{slug!r}]",
        )
        try:
            return _single_call(ctx, provider_routing={"ignore": [slug]})
        except Exception as exc4:
            last_exc = exc4
            _log_rung(ctx, f"rung=reroute failed: {str(exc4)[:200]}")

        # Suspected downstream overflow: the documented window passed
        # pre-flight, but a large prompt keeps 4xx-ing across routing —
        # consistent with a smaller *effective* downstream window. Give
        # the overflow rung one shot before giving up.
        llm_err = _underlying_llm_error(last_exc)
        is_provider_4xx = llm_err is not None and llm_err.status_code is not None and 400 <= llm_err.status_code < 500
        if (
            not tried_map_reduce
            and ctx.allow_map_reduce
            and is_provider_4xx
            and estimated >= _SUSPECTED_DOWNSTREAM_OVERFLOW_TOKENS
        ):
            tried_map_reduce = True
            override = _downstream_revealed_limit(estimated, window)
            _log_rung(
                ctx,
                f"rung=map-reduce suspected downstream context overflow "
                f"(~{estimated}-token prompt, repeated provider 4xx); "
                "falling back to map-reduce synthesis"
                + (f" (effective per-call limit capped at ~{override} tokens)" if override is not None else ""),
            )
            try:
                return _map_reduce_call(ctx, context_limit_override=override)
            except Exception as exc5:
                last_exc = exc5
                _log_rung(ctx, f"rung=map-reduce failed: {str(exc5)[:200]}")

    # Rung 5: ladder exhausted — hand the caller everything it needs for
    # the loud, guided synthesis_error envelope.
    command = format_recovery_command(ctx.result_id, ctx.effective_model)
    attempted = []
    if tried_transient_retry:
        attempted.append("transient retry")
    if tried_map_reduce:
        attempted.append("map-reduce fallback")
    if tried_reroute:
        attempted.append("OpenRouter reroute")
    summary = ", ".join(attempted) if attempted else "none applicable"
    message = (
        f"synthesis recovery ladder exhausted (rungs attempted: {summary}). "
        f"Last error: {str(last_exc)[:300]}. "
        f"Recover the saved panel without re-running panelists: `{command}`"
    )
    if map_reduce_blocked:
        message += (
            " (note: the failure classified as context overflow, but "
            "--synthesis-strategy=single disabled the map-reduce fallback; "
            "rerun with --synthesis-strategy=auto to allow it)"
        )
    _log_rung(ctx, f"rung=fallback {message}")
    raise SynthesisRecoveryError(
        message,
        cause=last_exc,
        rungs=list(ctx.rungs),
        suggested_command=command,
        map_reduce_blocked_by_strategy=map_reduce_blocked,
    )
