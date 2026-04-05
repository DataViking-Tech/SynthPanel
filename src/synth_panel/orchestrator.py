"""Multi-Agent Orchestration (SPEC.md §4).

Thread-safe worker registry and parallel panelist execution coordinator.
Manages lifecycle of independent agent sessions running concurrently.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from synth_panel.cost import ZERO_USAGE, TokenUsage, UsageTracker
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock, TokenUsage as LLMTokenUsage
from synth_panel.persistence import Session
from synth_panel.runtime import AgentRuntime, TurnSummary
from synth_panel.structured.output import StructuredOutputConfig, StructuredOutputEngine


def _convert_llm_usage(llm_usage: LLMTokenUsage) -> TokenUsage:
    """Convert LLM-layer TokenUsage to cost-layer TokenUsage."""
    return TokenUsage(
        input_tokens=llm_usage.input_tokens,
        output_tokens=llm_usage.output_tokens,
        cache_creation_input_tokens=llm_usage.cache_write_tokens,
        cache_read_input_tokens=llm_usage.cache_read_tokens,
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
        self.events.append(WorkerEvent(
            timestamp=datetime.now(timezone.utc),
            from_status=None,
            to_status=WorkerStatus.SPAWNING,
            detail="created",
        ))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class WorkerNotFoundError(Exception):
    def __init__(self, worker_id: str) -> None:
        super().__init__(f"Worker not found: {worker_id}")
        self.worker_id = worker_id


class InvalidTransitionError(Exception):
    def __init__(self, worker_id: str, from_status: WorkerStatus, to_status: WorkerStatus) -> None:
        super().__init__(
            f"Invalid transition for worker {worker_id}: "
            f"{from_status.value} -> {to_status.value}"
        )
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
            return all(
                w.status in (WorkerStatus.FINISHED, WorkerStatus.FAILED)
                for w in self._workers.values()
            )

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
) -> PanelistResult:
    """Execute a single panelist's full interview. Runs in a worker thread.

    Manages the worker lifecycle: spawning → ready → running → finished/failed.
    """
    name = persona.get("name", "Anonymous")
    tracker = UsageTracker()
    responses: list[dict[str, Any]] = []

    try:
        # Transition: spawning → ready_for_prompt
        registry.transition(worker_id, WorkerStatus.READY_FOR_PROMPT, "initialized")
        system_prompt = system_prompt_fn(persona)

        # Transition: ready → prompt_accepted → running
        registry.transition(worker_id, WorkerStatus.PROMPT_ACCEPTED, "prompt received")
        registry.transition(worker_id, WorkerStatus.RUNNING, "executing questions")

        session = Session()
        runtime = AgentRuntime(
            client=client,
            session=session,
            system_prompt=system_prompt,
            model=model,
        )

        # Set up structured output engine if schema provided
        structured_engine: StructuredOutputEngine | None = None
        structured_config: StructuredOutputConfig | None = None
        if response_schema:
            structured_engine = StructuredOutputEngine(client)
            structured_config = StructuredOutputConfig(schema=response_schema)

        for question in questions:
            question_text = question_prompt_fn(question)

            try:
                if structured_engine and structured_config:
                    # Use structured output: run turn for conversation context,
                    # then extract structured response
                    summary = runtime.run_turn(question_text)
                    tracker.record_turn(summary.usage)

                    # Build messages from session history for structured extraction
                    messages = [
                        InputMessage(role="user", content=[TextBlock(text=question_text)])
                    ]
                    result = structured_engine.extract(
                        model=model,
                        max_tokens=4096,
                        messages=messages,
                        config=structured_config,
                        system=system_prompt,
                    )
                    tracker.record_turn(_convert_llm_usage(result.response.usage))
                    responses.append({
                        "question": question_text,
                        "response": result.data,
                        "structured": True,
                        "is_fallback": result.is_fallback,
                    })
                else:
                    summary = runtime.run_turn(question_text)
                    response_text = _extract_text(summary)
                    responses.append({
                        "question": question_text,
                        "response": response_text,
                    })
                    tracker.record_turn(summary.usage)
            except Exception as exc:
                responses.append({
                    "question": question_text,
                    "response": f"[error: {exc}]",
                    "error": True,
                })

            # Handle follow-ups (always text mode — structured output applies to main questions)
            follow_ups = question.get("follow_ups", []) if isinstance(question, dict) else []
            for follow_up in follow_ups:
                try:
                    fu_summary = runtime.run_turn(follow_up)
                    fu_text = _extract_text(fu_summary)
                    responses.append({
                        "question": follow_up,
                        "response": fu_text,
                        "follow_up": True,
                    })
                    tracker.record_turn(fu_summary.usage)
                except Exception:
                    continue

        # Transition: running → finished
        registry.set_result(worker_id, {"responses": responses}, tracker.cumulative_usage)
        registry.transition(worker_id, WorkerStatus.FINISHED, "all questions complete")

        return PanelistResult(
            persona_name=name,
            responses=responses,
            usage=tracker.cumulative_usage,
        )

    except Exception as exc:
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
        )


def run_panel_parallel(
    client: LLMClient,
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model: str,
    system_prompt_fn: Callable[[dict[str, Any]], str],
    question_prompt_fn: Callable[[dict[str, Any]], str],
    max_workers: int | None = None,
    response_schema: dict[str, Any] | None = None,
) -> tuple[list[PanelistResult], WorkerRegistry]:
    """Run all panelists in parallel and return ordered results.

    Args:
        client: Shared LLM client (must be thread-safe for concurrent sends).
        personas: List of persona definitions.
        questions: List of question definitions from the instrument.
        model: Model alias to use for all panelists.
        system_prompt_fn: Builds system prompt from a persona dict.
        question_prompt_fn: Builds question text from a question dict.
        max_workers: Max concurrent threads. Defaults to number of personas.
        response_schema: Optional JSON Schema for structured output. When
            provided, responses are extracted as structured data via tool-use
            forcing instead of free text.

    Returns:
        Tuple of (ordered results matching persona order, registry).
    """
    registry = WorkerRegistry()
    effective_workers = max_workers or len(personas)

    # Create workers and map to personas (preserves order)
    worker_ids: list[str] = []
    for persona in personas:
        name = persona.get("name", "Anonymous")
        wid = registry.create_worker(name)
        worker_ids.append(wid)

    results: list[PanelistResult | None] = [None] * len(personas)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index = {}
        for idx, (persona, worker_id) in enumerate(zip(personas, worker_ids)):
            future = executor.submit(
                _run_panelist,
                registry,
                worker_id,
                client,
                persona,
                questions,
                model,
                system_prompt_fn,
                question_prompt_fn,
                response_schema,
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                name = personas[idx].get("name", "Anonymous")
                results[idx] = PanelistResult(
                    persona_name=name,
                    responses=[],
                    usage=ZERO_USAGE,
                    error=str(exc),
                )

    # All slots should be filled; cast away None
    return [r for r in results if r is not None], registry
