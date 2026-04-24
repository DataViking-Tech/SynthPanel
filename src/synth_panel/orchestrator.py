"""Multi-Agent Orchestration (SPEC.md §4).

Thread-safe worker registry and parallel panelist execution coordinator.
Manages lifecycle of independent agent sessions running concurrently.
"""

from __future__ import annotations

import logging
import threading
import time as _time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from synth_panel.conditions import evaluate_condition, normalize_follow_up
from synth_panel.convergence import ConvergenceTracker, extract_categorical_responses
from synth_panel.cost import ZERO_USAGE, CostGate, TokenUsage, UsageTracker, resolve_cost
from synth_panel.instrument import END_SENTINEL, Instrument, Round
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.persistence import Session
from synth_panel.routing import route_round
from synth_panel.runtime import AgentRuntime, TurnSummary
from synth_panel.structured.output import StructuredOutputConfig, StructuredOutputEngine

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Worker status lifecycle
# ---------------------------------------------------------------------------


class WorkerStatus(Enum):
    """Worker state machine per SPEC.md §4."""

    SPAWNING = "spawning"
    READY_FOR_PROMPT = "ready_for_prompt"
    PROMPT_ACCEPTED = "prompt_accepted"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class FailureKind(Enum):
    """Types of worker failure."""

    TRUST_GATE = "trust_gate"
    PROMPT_DELIVERY = "prompt_delivery"
    PROTOCOL = "protocol"


# Valid state transitions
_VALID_TRANSITIONS: dict[WorkerStatus, set[WorkerStatus]] = {
    WorkerStatus.SPAWNING: {WorkerStatus.READY_FOR_PROMPT, WorkerStatus.FAILED},
    WorkerStatus.READY_FOR_PROMPT: {WorkerStatus.PROMPT_ACCEPTED, WorkerStatus.FAILED},
    WorkerStatus.PROMPT_ACCEPTED: {WorkerStatus.RUNNING, WorkerStatus.READY_FOR_PROMPT, WorkerStatus.FAILED},
    WorkerStatus.RUNNING: {WorkerStatus.FINISHED, WorkerStatus.FAILED},
    WorkerStatus.FINISHED: set(),
    WorkerStatus.FAILED: {WorkerStatus.SPAWNING},  # restart
}


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerEvent:
    """A state transition record."""

    timestamp: datetime
    from_status: WorkerStatus | None
    to_status: WorkerStatus
    detail: str = ""


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class Worker:
    """State and metadata for a single orchestrated agent."""

    id: str
    name: str
    status: WorkerStatus = WorkerStatus.SPAWNING
    error: tuple[FailureKind, str] | None = None
    events: list[WorkerEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)

    def __post_init__(self) -> None:
        self.events.append(
            WorkerEvent(
                timestamp=datetime.now(timezone.utc),
                from_status=None,
                to_status=WorkerStatus.SPAWNING,
                detail="created",
            )
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkerNotFoundError(Exception):
    def __init__(self, worker_id: str) -> None:
        super().__init__(f"Worker not found: {worker_id}")
        self.worker_id = worker_id


class InvalidTransitionError(Exception):
    def __init__(self, worker_id: str, from_status: WorkerStatus, to_status: WorkerStatus) -> None:
        super().__init__(f"Invalid transition for worker {worker_id}: {from_status.value} -> {to_status.value}")
        self.worker_id = worker_id
        self.from_status = from_status
        self.to_status = to_status


# ---------------------------------------------------------------------------
# Worker Registry (thread-safe)
# ---------------------------------------------------------------------------


class WorkerRegistry:
    """Thread-safe registry tracking the state of spawned workers.

    All mutations are protected by a reentrant lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[str, Worker] = {}

    def create_worker(self, name: str) -> str:
        """Create a new worker entry and return its ID."""
        worker_id = f"w-{uuid.uuid4().hex[:8]}"
        worker = Worker(id=worker_id, name=name)
        with self._lock:
            self._workers[worker_id] = worker
        return worker_id

    def get_worker(self, worker_id: str) -> Worker:
        """Return a worker by ID. Raises WorkerNotFoundError if missing."""
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotFoundError(worker_id)
            return self._workers[worker_id]

    def transition(
        self,
        worker_id: str,
        to_status: WorkerStatus,
        detail: str = "",
    ) -> None:
        """Advance a worker to a new status. Validates the transition."""
        with self._lock:
            worker = self.get_worker(worker_id)
            if to_status not in _VALID_TRANSITIONS[worker.status]:
                raise InvalidTransitionError(worker_id, worker.status, to_status)
            event = WorkerEvent(
                timestamp=datetime.now(timezone.utc),
                from_status=worker.status,
                to_status=to_status,
                detail=detail,
            )
            worker.events.append(event)
            worker.status = to_status
            if to_status == WorkerStatus.FAILED:
                worker.error = (FailureKind.PROTOCOL, detail)

    def set_result(self, worker_id: str, result: dict[str, Any], usage: TokenUsage) -> None:
        """Store the result and usage for a finished worker."""
        with self._lock:
            worker = self.get_worker(worker_id)
            worker.result = result
            worker.usage = usage

    def set_error(self, worker_id: str, kind: FailureKind, message: str) -> None:
        """Record an error on a worker."""
        with self._lock:
            worker = self.get_worker(worker_id)
            worker.error = (kind, message)

    def list_workers(self) -> list[Worker]:
        """Return a snapshot of all workers."""
        with self._lock:
            return list(self._workers.values())

    def all_finished(self) -> bool:
        """True if every worker is in a terminal state (finished or failed)."""
        with self._lock:
            return all(w.status in (WorkerStatus.FINISHED, WorkerStatus.FAILED) for w in self._workers.values())

    def terminate(self, worker_id: str) -> None:
        """Mark a worker as finished."""
        self.transition(worker_id, WorkerStatus.FINISHED, "terminated")

    def restart(self, worker_id: str) -> None:
        """Reset a failed worker to spawning state."""
        with self._lock:
            worker = self.get_worker(worker_id)
            if worker.status != WorkerStatus.FAILED:
                raise InvalidTransitionError(worker_id, worker.status, WorkerStatus.SPAWNING)
            event = WorkerEvent(
                timestamp=datetime.now(timezone.utc),
                from_status=worker.status,
                to_status=WorkerStatus.SPAWNING,
                detail="restarted",
            )
            worker.events.append(event)
            worker.status = WorkerStatus.SPAWNING
            worker.error = None


# ---------------------------------------------------------------------------
# Panel result types
# ---------------------------------------------------------------------------


@dataclass
class PanelistResult:
    """Result from running one panelist through all questions."""

    persona_name: str
    responses: list[dict[str, Any]]
    usage: TokenUsage
    error: str | None = None
    model: str | None = None


@dataclass
class RoundResult:
    """Per-round panelist + synthesis bundle for a multi-round run."""

    name: str
    panelist_results: list[PanelistResult]
    synthesis: Any  # SynthesisResult; typed as Any to avoid synthesis import cycle
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)


@dataclass
class MultiRoundResult:
    """Result of a branching multi-round panel run.

    ``rounds`` contains only the rounds that actually executed (in order).
    ``path`` records each routing decision: ``{round, branch, next}``.
    ``terminal_round`` is the last round whose synthesis fed final synthesis.
    ``warnings`` carries parser warnings (e.g. unreachable rounds) plus any
    runtime issues observed during the loop.
    """

    rounds: list[RoundResult]
    path: list[dict[str, Any]] = field(default_factory=list)
    terminal_round: str | None = None
    final_synthesis: Any = None  # SynthesisResult
    warnings: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)


# ---------------------------------------------------------------------------
# Parallel panel runner
# ---------------------------------------------------------------------------


def _extract_text(summary: TurnSummary) -> str:
    """Extract response text from a TurnSummary."""
    parts: list[str] = []
    for msg in summary.assistant_messages:
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "".join(parts)


def _run_panelist(
    registry: WorkerRegistry,
    worker_id: str,
    client: LLMClient,
    persona: dict[str, Any],
    questions: list[dict[str, Any]],
    model: str,
    system_prompt_fn: Callable[[dict[str, Any]], str],
    question_prompt_fn: Callable[[dict[str, Any]], str],
    response_schema: dict[str, Any] | None = None,
    session: Session | None = None,
    sentiment_cache: dict[str, str] | None = None,
    sentiment_cache_lock: threading.Lock | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> tuple[PanelistResult, Session]:
    """Execute a single panelist's full interview. Runs in a worker thread.

    Manages the worker lifecycle: spawning → ready → running → finished/failed.
    """
    name = persona.get("name", "Anonymous")
    tracker = UsageTracker()
    responses: list[dict[str, Any]] = []
    t0 = _time.monotonic()
    logger.info("panelist %s starting (model=%s, questions=%d)", name, model, len(questions))

    try:
        # Transition: spawning → ready_for_prompt
        registry.transition(worker_id, WorkerStatus.READY_FOR_PROMPT, "initialized")
        system_prompt = system_prompt_fn(persona)

        # Transition: ready → prompt_accepted → running
        registry.transition(worker_id, WorkerStatus.PROMPT_ACCEPTED, "prompt received")
        registry.transition(worker_id, WorkerStatus.RUNNING, "executing questions")

        if session is None:
            session = Session()
        runtime = AgentRuntime(
            client=client,
            session=session,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        # Set up structured output engine if schema provided
        structured_engine: StructuredOutputEngine | None = None
        structured_config: StructuredOutputConfig | None = None
        if response_schema:
            structured_engine = StructuredOutputEngine(client)
            structured_config = StructuredOutputConfig(schema=response_schema)

        # Set up extraction engine for post-hoc structured extraction
        extract_engine: StructuredOutputEngine | None = None
        extract_config: StructuredOutputConfig | None = None
        if extract_schema:
            extract_engine = StructuredOutputEngine(client)
            extract_config = StructuredOutputConfig(schema=extract_schema)

        for question in questions:
            question_text = question_prompt_fn(question)

            try:
                if structured_engine and structured_config:
                    # Use structured output: run turn for conversation context,
                    # then extract structured response
                    summary = runtime.run_turn(question_text)
                    tracker.record_turn(summary.usage)

                    # Build messages from session history for structured extraction
                    messages = [InputMessage(role="user", content=[TextBlock(text=question_text)])]
                    result = structured_engine.extract(
                        model=model,
                        max_tokens=4096,
                        messages=messages,
                        config=structured_config,
                        system=system_prompt,
                    )
                    tracker.record_turn(_convert_llm_usage(result.response.usage))
                    responses.append(
                        {
                            "question": question_text,
                            "response": result.data,
                            "structured": True,
                            "is_fallback": result.is_fallback,
                        }
                    )
                else:
                    summary = runtime.run_turn(question_text)
                    response_text = _extract_text(summary)
                    resp_dict: dict[str, Any] = {
                        "question": question_text,
                        "response": response_text,
                    }
                    tracker.record_turn(summary.usage)

                    # Extraction pass: extract structured data from the
                    # free-text response (--extract-schema).
                    if extract_engine and extract_config:
                        try:
                            extract_messages = [
                                InputMessage(
                                    role="user",
                                    content=[TextBlock(text=question_text)],
                                ),
                                InputMessage(
                                    role="assistant",
                                    content=[TextBlock(text=response_text)],
                                ),
                            ]
                            extract_result = extract_engine.extract(
                                model=model,
                                max_tokens=4096,
                                messages=extract_messages,
                                config=extract_config,
                                system=system_prompt,
                            )
                            tracker.record_turn(_convert_llm_usage(extract_result.response.usage))
                            resp_dict["extraction"] = extract_result.data
                            resp_dict["extraction_is_fallback"] = extract_result.is_fallback
                        except Exception as extract_exc:
                            resp_dict["extraction"] = None
                            resp_dict["extraction_error"] = str(extract_exc)

                    responses.append(resp_dict)
            except Exception as exc:
                responses.append(
                    {
                        "question": question_text,
                        "response": f"[error: {exc}]",
                        "error": True,
                    }
                )

            # Handle conditional follow-ups (text mode only)
            raw_follow_ups = question.get("follow_ups", []) if isinstance(question, dict) else []
            # Get the last main-question response text for condition eval
            last_response = responses[-1].get("response", "") if responses else ""
            for raw_fu in raw_follow_ups:
                fu = normalize_follow_up(raw_fu)
                condition = fu.get("condition", "always")
                if not evaluate_condition(
                    condition,
                    last_response,
                    client=client,
                    sentiment_cache=sentiment_cache,
                    sentiment_cache_lock=sentiment_cache_lock,
                ):
                    continue
                try:
                    fu_summary = runtime.run_turn(fu["text"])
                    fu_text = _extract_text(fu_summary)
                    responses.append(
                        {
                            "question": fu["text"],
                            "response": fu_text,
                            "follow_up": True,
                        }
                    )
                    tracker.record_turn(fu_summary.usage)
                except Exception as exc:
                    logger.warning(
                        "panelist %s follow-up failed: %s: %s",
                        name,
                        type(exc).__name__,
                        exc,
                    )
                    responses.append(
                        {
                            "question": fu["text"],
                            "response": f"[error: {exc}]",
                            "error": True,
                            "follow_up": True,
                        }
                    )

        # Transition: running → finished
        registry.set_result(worker_id, {"responses": responses}, tracker.cumulative_usage)
        registry.transition(worker_id, WorkerStatus.FINISHED, "all questions complete")

        elapsed = _time.monotonic() - t0
        logger.info(
            "panelist %s completed in %.2fs (tokens=%d)",
            name,
            elapsed,
            tracker.cumulative_usage.total_tokens,
        )

        # sp-2xy: silent usage-capture failure produces $0 cost for the whole
        # panel. If we successfully produced responses but tokens are 0, the
        # upstream provider almost certainly returned an empty ``usage`` block —
        # warn loudly so this doesn't slip through to JSON again.
        if tracker.cumulative_usage.total_tokens == 0 and responses and not all(r.get("error") for r in responses):
            logger.warning(
                "panelist %s (model=%s) produced %d responses but usage=0 — "
                "provider likely omitted the usage block; cost will be $0",
                name,
                model,
                len(responses),
            )

        result = PanelistResult(
            persona_name=name,
            responses=responses,
            usage=tracker.cumulative_usage,
            model=model,
        )
        return result, session

    except Exception as exc:
        elapsed = _time.monotonic() - t0
        logger.error("panelist %s failed after %.2fs: %s", name, elapsed, exc)
        # Transition to failed
        try:
            registry.set_error(worker_id, FailureKind.PROTOCOL, str(exc))
            registry.transition(worker_id, WorkerStatus.FAILED, str(exc))
        except InvalidTransitionError:
            pass  # Already in terminal state

        return PanelistResult(
            persona_name=name,
            responses=responses,
            usage=tracker.cumulative_usage,
            error=str(exc),
            model=model,
        ), session


def run_panel_parallel(
    client: LLMClient,
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str,
    system_prompt_fn: Callable[[dict[str, Any]], str],
    question_prompt_fn: Callable[[dict[str, Any]], str],
    max_workers: int | None = None,
    response_schema: dict[str, Any] | None = None,
    sessions: dict[str, Session] | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    persona_models: dict[str, str] | None = None,
    convergence_tracker: ConvergenceTracker | None = None,
    cost_gate: CostGate | None = None,
) -> tuple[list[PanelistResult], WorkerRegistry, dict[str, Session]]:
    """Run all panelists in parallel and return ordered results.

    Args:
        client: Shared LLM client (must be thread-safe for concurrent sends).
        personas: List of persona definitions.
        questions: List of question definitions from the instrument.
        model: Default model alias (used when no per-persona override exists).
        system_prompt_fn: Builds system prompt from a persona dict.
        question_prompt_fn: Builds question text from a question dict.
        max_workers: Max concurrent threads. Defaults to number of personas.
        response_schema: Optional JSON Schema for structured output. When
            provided, responses are extracted as structured data via tool-use
            forcing instead of free text.
        sessions: Optional mapping of persona names to existing sessions.
            When provided, panelists reuse their session (conversation history
            preserved). When None, each panelist gets a fresh session.
        extract_schema: Optional JSON Schema for post-hoc extraction from
            free-text responses. When provided (and response_schema is not),
            each text response is followed by a second LLM call that extracts
            structured data matching this schema. The result is stored under
            an ``extraction`` key alongside the raw ``response``.
        persona_models: Optional mapping of persona name → model override.
            Resolution order: persona_models[name] > model (global default).
        convergence_tracker: Optional :class:`ConvergenceTracker`. When
            supplied, each completing panelist's categorical responses are
            recorded; if the tracker signals auto-stop, pending futures are
            cancelled and ``run_panel_parallel`` returns only the panelists
            that had already finished. Errored panelists are still surfaced
            — they never contribute to the running distributions.
        cost_gate: Optional :class:`CostGate`. Each completing panelist's
            priced cost is recorded against the gate; if the projected run
            total exceeds the gate's ceiling, pending futures are cancelled
            and only finished panelists are returned. The caller is expected
            to inspect ``cost_gate.should_halt()`` on return and surface a
            partial, ``run_invalid`` result.

    Returns:
        Tuple of (ordered results matching persona order, registry,
        sessions dict mapping persona names to their sessions).
    """
    registry = WorkerRegistry()
    effective_workers = max_workers or len(personas)
    sentiment_cache: dict[str, str] = {}
    sentiment_cache_lock = threading.Lock()
    request_id = uuid.uuid4().hex[:12]
    logger.info(
        "[%s] panel starting: %d personas, %d questions, model=%s, workers=%d",
        request_id,
        len(personas),
        len(questions),
        model,
        effective_workers,
    )

    # Create workers and map to personas (preserves order)
    worker_ids: list[str] = []
    for persona in personas:
        name = persona.get("name", "Anonymous")
        wid = registry.create_worker(name)
        worker_ids.append(wid)

    results: list[PanelistResult | None] = [None] * len(personas)
    out_sessions: dict[str, Session] = {}
    session_lock = threading.Lock()

    # sp-yaru: build a per-run list of the bounded questions so the tracker
    # and the orchestrator agree on indices. Done outside the executor
    # so no worker pays the inspection cost.
    tracked_questions: list[tuple[int, str, dict[str, Any]]] = []
    if convergence_tracker is not None:
        tracked_questions = [
            (i, k, q) for i, k, q in _inspect_bounded_questions(questions) if k in set(convergence_tracker.tracked_keys)
        ]

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index = {}
        for idx, (persona, worker_id) in enumerate(zip(personas, worker_ids)):
            name = persona.get("name", "Anonymous")
            existing_session = (sessions or {}).get(name)
            # Resolve per-persona model: persona_models mapping > global default
            effective_model = (persona_models or {}).get(name, model)
            future = executor.submit(
                _run_panelist,
                registry,
                worker_id,
                client,
                persona,
                questions,
                effective_model,
                system_prompt_fn,
                question_prompt_fn,
                response_schema,
                existing_session,
                sentiment_cache,
                sentiment_cache_lock,
                extract_schema,
                temperature,
                top_p,
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result, sess = future.result()
                results[idx] = result
                with session_lock:
                    out_sessions[result.persona_name] = sess
            except Exception as exc:
                name = personas[idx].get("name", "Anonymous")
                results[idx] = PanelistResult(
                    persona_name=name,
                    responses=[],
                    usage=ZERO_USAGE,
                    error=str(exc),
                    model=model,
                )

            if convergence_tracker is not None and results[idx] is not None:
                completed_result = results[idx]
                assert completed_result is not None  # help mypy; checked above
                if completed_result.error is None:
                    categorical = extract_categorical_responses(completed_result, tracked_questions)
                    try:
                        should_stop = convergence_tracker.record(categorical)
                    except Exception as track_exc:  # pragma: no cover - defensive
                        logger.warning("convergence tracker record failed: %s", track_exc)
                        should_stop = False
                    if should_stop:
                        logger.info(
                            "auto-stop: convergence reached at n=%d; cancelling pending futures",
                            convergence_tracker.overall_converged_at or 0,
                        )
                        for pending_future in future_to_index:
                            if not pending_future.done():
                                pending_future.cancel()
                        break

            # sp-utnk: cost-gate check. Price the completed panelist using
            # its per-panelist model (falls back to the run-level default)
            # and record against the gate. If the projected run total
            # exceeds the gate, cancel pending futures and return what we
            # have — the caller synthesizes a partial, run_invalid result.
            if cost_gate is not None and results[idx] is not None:
                completed_result = results[idx]
                assert completed_result is not None
                pr_model = completed_result.model or model
                priced = resolve_cost(completed_result.usage, pr_model)
                halted = cost_gate.record(priced.total_cost)
                if halted:
                    logger.info(
                        "cost gate tripped after %d/%d panelists; cancelling pending futures",
                        cost_gate.completed,
                        len(personas),
                    )
                    for pending_future in future_to_index:
                        if not pending_future.done():
                            pending_future.cancel()
                    break

    # All slots should be filled (unless auto-stop cancelled some); drop Nones
    return [r for r in results if r is not None], registry, out_sessions


def _inspect_bounded_questions(
    questions: list[dict[str, Any]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Local wrapper so the orchestrator can share tracker tagging."""
    from synth_panel.convergence import identify_tracked_questions

    return identify_tracked_questions(questions)


# ---------------------------------------------------------------------------
# Multi-round branching runner (v3 instruments)
# ---------------------------------------------------------------------------


def _round_lookup(instrument: Instrument) -> dict[str, Round]:
    return {r.name: r for r in instrument.rounds}


def _next_via_depends_on(instrument: Instrument, current: str) -> str:
    """Linear-chain fallback for v2 rounds without route_when.

    Returns the next round whose ``depends_on`` is ``current``, or
    ``__end__`` if there is no successor.
    """
    for r in instrument.rounds:
        if r.depends_on == current:
            return r.name
    return END_SENTINEL


def run_multi_round_panel(
    *,
    client: LLMClient,
    personas: list[dict[str, Any]],
    instrument: Instrument,
    model: str,
    system_prompt_fn: Callable[[dict[str, Any]], str],
    question_prompt_fn: Callable[[dict[str, Any]], str],
    synthesize_round_fn: Callable[..., Any],
    synthesize_final_fn: Callable[..., Any] | None = None,
    response_schema: dict[str, Any] | None = None,
    max_workers: int | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    persona_models: dict[str, str] | None = None,
) -> MultiRoundResult:
    """Execute a (possibly branching) multi-round panel run.

    The loop is router-driven: starting from the first round in the
    instrument, each round runs all panelists in parallel, synthesizes
    the round's responses, then asks ``routing.route_round`` for the
    next target. v2 instruments without ``route_when`` fall through to
    ``depends_on``-based linear chaining; v1 single-round instruments
    are a degenerate case that runs once and stops.

    ``synthesize_round_fn`` is called as
    ``synthesize_round_fn(client, panelist_results, questions, model=...)``
    and must return a ``SynthesisResult``-shaped object whose ``to_dict``
    output contains the fields the routing predicates reference.

    ``synthesize_final_fn``, if provided, receives only the executed
    rounds and is used to tag the *terminal* round of the path rather
    than the syntactic last round in the file (architect Q6).
    """
    if not instrument.rounds:
        return MultiRoundResult(rounds=[], warnings=list(instrument.warnings))

    request_id = uuid.uuid4().hex[:12]
    logger.info(
        "[%s] multi-round panel starting: %d personas, %d rounds", request_id, len(personas), len(instrument.rounds)
    )

    by_name = _round_lookup(instrument)
    sessions: dict[str, Session] = {}
    executed: list[RoundResult] = []
    path: list[dict[str, Any]] = []
    warnings: list[str] = list(instrument.warnings)
    cumulative = UsageTracker()

    next_round: str | None = instrument.rounds[0].name
    visited: set[str] = set()

    while next_round and next_round != END_SENTINEL:
        if next_round in visited:
            # Belt-and-suspenders: parser already topo-sorts, but a runtime
            # cycle would loop forever. Stop and warn.
            warnings.append(f"runtime cycle: round '{next_round}' revisited; halting")
            break
        if next_round not in by_name:
            warnings.append(f"router target '{next_round}' is not a defined round; halting")
            break

        current = by_name[next_round]
        visited.add(current.name)
        logger.info("executing round '%s' (%d/%d visited)", current.name, len(visited), len(by_name))

        panelist_results, _registry, sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=current.questions,
            model=model,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=question_prompt_fn,
            max_workers=max_workers,
            response_schema=response_schema,
            sessions=sessions,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
            persona_models=persona_models,
        )

        synthesis = synthesize_round_fn(client, panelist_results, current.questions, model=model)

        round_usage = ZERO_USAGE
        for pr in panelist_results:
            round_usage = round_usage + pr.usage
        if hasattr(synthesis, "usage"):
            round_usage = round_usage + synthesis.usage
        cumulative.record_turn(round_usage)

        executed.append(
            RoundResult(
                name=current.name,
                panelist_results=panelist_results,
                synthesis=synthesis,
                usage=round_usage,
            )
        )

        # ── Router decision ──
        if current.route_when:
            context = synthesis.to_dict() if hasattr(synthesis, "to_dict") else {}
            try:
                target = route_round(current.route_when, context)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"routing failed for '{current.name}': {exc}; halting")
                target = END_SENTINEL
            # Render a human-readable branch description for the path log.
            branch_desc = _describe_branch(current.route_when, context, target)
        else:
            target = _next_via_depends_on(instrument, current.name)
            branch_desc = "linear"

        path.append({"round": current.name, "branch": branch_desc, "next": target})
        logger.debug("route decision: round=%s branch=%s next=%s", current.name, branch_desc, target)
        next_round = target

    terminal = executed[-1].name if executed else None

    final_synthesis = None
    if synthesize_final_fn is not None and executed:
        # Pass only executed rounds to the final synthesis (architect Q6).
        # Flatten panelist results across executed rounds in order.
        merged_results = _merge_panelist_results(executed)
        merged_questions = [q for rr in executed for q in by_name[rr.name].questions]
        final_synthesis = synthesize_final_fn(client, merged_results, merged_questions, model=model)
        if hasattr(final_synthesis, "usage"):
            cumulative.record_turn(final_synthesis.usage)

    return MultiRoundResult(
        rounds=executed,
        path=path,
        terminal_round=terminal,
        final_synthesis=final_synthesis,
        warnings=warnings,
        usage=cumulative.cumulative_usage,
    )


def _describe_branch(
    route_when: list[dict[str, Any]],
    context: dict[str, Any],
    chosen_target: str,
) -> str:
    """Render which clause fired, for the path log entry."""
    from synth_panel.routing import evaluate_predicate

    for clause in route_when:
        if "if" in clause:
            try:
                if evaluate_predicate(clause["if"], context):
                    pred = clause["if"]
                    return f"{pred.get('field')} {pred.get('op')} {pred.get('value')!r} -> {chosen_target}"
            except Exception:
                continue
        elif "else" in clause:
            return f"else -> {chosen_target}"
    return f"-> {chosen_target}"


def _merge_panelist_results(
    executed: list[RoundResult],
) -> list[PanelistResult]:
    """Merge per-round panelist results into one list per persona, in order.

    Each persona ends up with a single ``PanelistResult`` whose
    ``responses`` are the concatenation of their responses across the
    executed rounds.
    """
    by_name: dict[str, PanelistResult] = {}
    order: list[str] = []
    for rr in executed:
        for pr in rr.panelist_results:
            if pr.persona_name not in by_name:
                by_name[pr.persona_name] = PanelistResult(
                    persona_name=pr.persona_name,
                    responses=list(pr.responses),
                    usage=pr.usage,
                    error=pr.error,
                )
                order.append(pr.persona_name)
            else:
                merged = by_name[pr.persona_name]
                merged.responses.extend(pr.responses)
                merged.usage = merged.usage + pr.usage
                if pr.error and not merged.error:
                    merged.error = pr.error
    return [by_name[n] for n in order]


# ---------------------------------------------------------------------------
# Multi-model ensemble
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Result of running the same panel across multiple models."""

    per_model_results: dict[str, list[PanelistResult]]
    convergent_findings: list[dict[str, Any]]
    divergent_findings: list[dict[str, Any]]
    cost_breakdown: dict[str, dict[str, Any]]
    models: list[str]
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)


def ensemble_run(
    *,
    client: LLMClient,
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    models: list[str],
    system_prompt_fn: Callable[[dict[str, Any]], str],
    question_prompt_fn: Callable[[dict[str, Any]], str],
    response_schema: dict[str, Any] | None = None,
    extract_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> EnsembleResult:
    """Run the same panel with each model and compare results.

    Runs ``run_panel_parallel`` once per model, then computes cross-model
    convergence for questions that produced categorical responses.

    Returns an :class:`EnsembleResult` with per-model results and
    convergent/divergent findings.
    """
    per_model: dict[str, list[PanelistResult]] = {}
    cost_breakdown: dict[str, dict[str, Any]] = {}
    total_usage = ZERO_USAGE

    for model_name in models:
        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model=model_name,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=question_prompt_fn,
            response_schema=response_schema,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
        )
        per_model[model_name] = results
        model_usage = ZERO_USAGE
        for pr in results:
            model_usage = model_usage + pr.usage
        total_usage = total_usage + model_usage
        cost_breakdown[model_name] = {"usage": model_usage.to_dict()}

    # Attempt convergence analysis when there are ≥2 models and structured responses
    convergent: list[dict[str, Any]] = []
    divergent: list[dict[str, Any]] = []

    if len(models) >= 2 and questions:
        question_texts = [q.get("text", str(q)) for q in questions]
        try:
            multi_model_responses = _extract_categorical_responses(per_model, len(questions))
            if multi_model_responses:
                from synth_panel.stats import convergence_report

                report = convergence_report(multi_model_responses, question_texts)
                for f in report.findings:
                    entry = {
                        "question_index": f.question_index,
                        "question": f.question_text,
                        "alpha": round(f.alpha, 3),
                        "level": f.level.value,
                        "interpretation": f.interpretation,
                    }
                    if f.alpha >= 0.60:
                        convergent.append(entry)
                    elif f.alpha < 0.40:
                        divergent.append(entry)
        except (ValueError, KeyError):
            pass  # Convergence analysis not applicable (e.g. free-text responses)

    return EnsembleResult(
        per_model_results=per_model,
        convergent_findings=convergent,
        divergent_findings=divergent,
        cost_breakdown=cost_breakdown,
        models=models,
        usage=total_usage,
    )


def _extract_categorical_responses(
    per_model: dict[str, list[PanelistResult]],
    n_questions: int,
) -> dict[str, list[list[str]]] | None:
    """Extract categorical response strings for convergence analysis.

    Returns model_name -> [[response_per_question] per persona], or None
    if responses aren't categorical (e.g. free-text with no structured data).
    """
    result: dict[str, list[list[str]]] = {}
    for model_name, panelist_results in per_model.items():
        personas_data: list[list[str]] = []
        for pr in panelist_results:
            q_responses: list[str] = []
            for resp in pr.responses[:n_questions]:
                # Prefer structured data for categorical comparison
                if isinstance(resp.get("response"), dict):
                    q_responses.append(str(sorted(resp["response"].items())))
                elif isinstance(resp.get("response"), str):
                    # Free-text: truncate to first 200 chars for rough comparison
                    q_responses.append(resp["response"][:200])
                else:
                    return None
            if len(q_responses) != n_questions:
                return None
            personas_data.append(q_responses)
        result[model_name] = personas_data
    return result
