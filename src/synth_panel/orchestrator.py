"""Multi-Agent Orchestration (SPEC.md §4).

Thread-safe worker registry and parallel panelist execution coordinator.
Manages lifecycle of independent agent sessions running concurrently.
"""

from __future__ import annotations

import hashlib
import logging
import time as _time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # sy-2wa: type-only import; lifted to a guarded block so the real
    # ``threading`` module never lands in synth_panel.ensemble's load
    # chain. The annotation below is stringified at runtime via
    # ``from __future__ import annotations``.
    import threading

from synth_panel.attachments import filter_attachments
from synth_panel.attachments.filter import count_strata
from synth_panel.conditions import evaluate_condition, normalize_follow_up
from synth_panel.convergence import ConvergenceTracker, extract_categorical_responses
from synth_panel.cost import ZERO_USAGE, CostGate, TokenUsage, UsageTracker, resolve_cost
from synth_panel.fetch.cache import CacheL1, UrlCache
from synth_panel.fetch.lower import lower_url_blocks
from synth_panel.instrument import END_SENTINEL, Instrument, Round
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import ContentBlock, InputMessage, TextBlock, URLBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.persistence import Session
from synth_panel.prompts import build_question_blocks
from synth_panel.question_budget import QuestionFailureBudget
from synth_panel.routing import route_round
from synth_panel.runtime import AgentRuntime, TurnSummary
from synth_panel.structured.output import StructuredOutputConfig, StructuredOutputEngine

try:  # pydantic is a hard dep at v1.0.3; guarded for the migration window
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import ValidationError as _PydanticValidationError
except ImportError:  # pragma: no cover - exercised only pre-install
    _PydanticBaseModel = None  # type: ignore[assignment,misc]
    _PydanticValidationError = Exception  # type: ignore[assignment,misc]


def _unpack_extract_schema(
    value: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, type[Any] | None]:
    """Split an ``extract_schema`` parameter into ``(json_schema, pydantic_model)``.

    Accepts both the v1.0.3 resolved envelope produced by
    :func:`synth_panel._runners.resolve_extract_schema`
    (``{"schema": {...}, "model": Class | None}``) and the legacy raw
    JSON Schema dict that pre-resolver call sites still pass directly.
    The resolved envelope is identified by a top-level ``"schema"`` key
    pointing at a dict — no JSON Schema in this codebase exposes that
    name as a property, so the discriminator is unambiguous.
    """
    if value is None:
        return None, None
    if isinstance(value, dict) and isinstance(value.get("schema"), dict):
        return value["schema"], value.get("model")
    return value, None


logger = logging.getLogger(__name__)


class PanelPlanningError(Exception):
    """Raised at the frame stage of :func:`run_panel_parallel` when the
    planned panel violates a structural caching invariant (hq-0pbp).

    Currently the only check is the **K≤5 stratification cap**: if any
    question's per-persona attachment filters partition the panel into
    more than 5 strata, the planner refuses the run before any LLM call.
    Above K=5 the cached economics collapse toward 1.5x the K=1 floor
    (per D-phase hq-cxth §5/§7), defeating the whole point of
    stratification. The fix is on the instrument author's side — coarsen
    the predicates so K stays ≤ 5 (the methodological target is K=3:
    desktop / mobile / tablet, etc.).
    """


_STRATA_CAP = 5


def _resolve_question_attachment_refs(
    questions: list[dict[str, Any]],
    bank: dict[str, dict[str, Any]] | None,
    *,
    exclude_refs: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve bank-ref strings in ``question.attachments`` to inline dicts.

    The hq-xzsm data-model design supports two attachment-reference shapes
    on a question:

    * **Bank-ref string** — ``{"attachments": ["hero_creative_v3"]}`` — looks
      up the named attachment in ``Instrument.attachments`` (the panel-shared
      bank).
    * **Inline dict** — ``{"attachments": [{"type": "image", ...}]}`` — the
      block payload directly on the question.

    The frame-stage filter at ``orchestrator.py:879-883`` (line numbers as of
    v1.0.0) only retains dict-form refs:

        dict_attachments = [a for a in raw_attachments if isinstance(a, dict)]

    so bank-ref strings silently fall out before reaching the multimodal
    block emitter. This helper resolves them up-front, replacing strings with
    a copy of the bank entry so the rest of the orchestrator path treats every
    attachment uniformly.

    Unresolved refs raise ``ValueError`` — the parser already enforces
    reachability at parse time, so reaching this state implies an internal
    inconsistency between parse and runtime, not a user error.

    ``exclude_refs`` (hq-ovxl / G2) drops any bank-ref string in that set
    from per-question attachment lists before resolution. The G2 lift moves
    cross-question shared bank entries onto ``panel_shared_attachments``;
    excluding them here prevents the same payload from being emitted twice
    (once shared, once per-question) by :func:`build_question_blocks`.
    Inline dicts are never excluded — only bank-ref strings.
    """
    if not bank:
        # No bank to resolve against; pass through unchanged. Inline dicts
        # remain valid; bare strings would fall through to the existing
        # filter and produce empty per-persona attachments. That's the
        # legacy v0.12.0 behaviour and we preserve it.
        return list(questions)

    excluded: set[str] = exclude_refs or set()

    resolved: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            resolved.append(q)
            continue
        refs = q.get("attachments")
        if not refs:
            resolved.append(q)
            continue
        # Build a new question dict with bank-strings expanded.
        new_refs: list[dict[str, Any]] = []
        for ref in refs:
            if isinstance(ref, str):
                if ref in excluded:
                    # Lifted to panel_shared_attachments; skip per-question
                    # emission to avoid double-blocks.
                    continue
                if ref not in bank:
                    raise ValueError(
                        f"attachment ref {ref!r} does not resolve to a bank entry (bank keys: {sorted(bank.keys())!r})"
                    )
                # Copy so downstream mutations don't bleed across questions.
                new_refs.append(dict(bank[ref]))
            elif isinstance(ref, dict):
                new_refs.append(ref)
            else:
                raise ValueError(f"attachment ref must be a string or mapping, got {type(ref).__name__}")
        new_q = dict(q)
        new_q["attachments"] = new_refs
        resolved.append(new_q)
    return resolved


def _compute_panel_shared(
    questions: list[dict[str, Any]],
    bank: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Identify bank entries shared by ≥2 questions in this round (hq-ovxl).

    Returns ``(panel_shared_attachments, shared_ref_ids)``:

    * ``panel_shared_attachments`` — list of deep-copied bank entries to
      emit once before the cache_control marker (canonical block order
      hq-0pbp), so the prefix is byte-identical across every question that
      references them and Anthropic's prefix cache hits on the second
      panelist + question pair.
    * ``shared_ref_ids`` — the set of bank ref-ids that were lifted, used
      by :func:`_resolve_question_attachment_refs`'s ``exclude_refs``
      parameter to strip them from per-question lists (otherwise the same
      payload emits twice — once shared, once per-question).

    Heuristic (per the v1.0.2 plan): an entry is panel-shared iff it is
    referenced as a **bank-ref string** by ≥2 questions in this round.
    Inline dicts (``{"type": "image", ...}``) are *never* lifted — they
    have no stable identity to dedupe against and the author chose
    per-question placement deliberately. Single-use bank entries also
    stay per-question; lifting them would cost a wasted cache write.

    The explicit ``shared: true`` flag (heuristic (b) in the design) is
    deferred to v1.1.0; this counts-based rule is the right zero-config
    default for the dogfood instruments shipped today.
    """
    if not bank:
        return [], set()

    ref_counts: dict[str, int] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        seen_in_q: set[str] = set()
        for ref in q.get("attachments", []) or []:
            if isinstance(ref, str) and ref in bank and ref not in seen_in_q:
                # Count each bank-ref at most once per question — duplicate
                # references inside a single question don't make it shared.
                ref_counts[ref] = ref_counts.get(ref, 0) + 1
                seen_in_q.add(ref)

    shared_ids = {rid for rid, count in ref_counts.items() if count >= 2}
    # Preserve insertion order (Python dict ordering) so the shared-block
    # list is deterministic across runs — fingerprints stay stable.
    panel_shared = [dict(bank[rid]) for rid in ref_counts if rid in shared_ids]
    return panel_shared, shared_ids


def _enforce_strata_cap(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    max_k: int = _STRATA_CAP,
) -> None:
    """Frame-stage gate: refuse panels whose stratification produces K > ``max_k``.

    Walks every question with attachments and computes
    :func:`count_strata` for that question's attachment list. The first
    question to exceed the cap raises :class:`PanelPlanningError` with
    its question index, the offending K, and the hard cap so the
    instrument author can fix the predicates.

    No-attachment questions and questions whose attachments lack
    ``filter`` clauses always partition into K=1, so they never trip
    the gate. The check runs per-question rather than aggregate-across-
    questions because the cache prefix is per-question — exceeding the
    cap on any one question collapses caching for THAT question alone.
    """
    if not personas:
        return
    for q_idx, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        atts = q.get("attachments") or []
        if not isinstance(atts, list) or not atts:
            continue
        # Skip if attachments are bank-ref strings — strata count requires
        # dict-form attachments with optional filter predicates.
        dict_atts = [a for a in atts if isinstance(a, dict)]
        if not dict_atts:
            continue
        k = count_strata(personas, dict_atts)
        if k > max_k:
            raise PanelPlanningError(
                f"stratification produces {k} strata at question[{q_idx}]; "
                f"cap is {max_k}. Coarsen attachment filter predicates so "
                f"the panel partitions into at most {max_k} attachment-sets "
                f"(hq-0pbp / hq-cxth §5)."
            )


def _min_stratum_population(
    personas: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> int:
    """Smallest stratum population across all attachment-bearing questions.

    Returns ``len(personas)`` when no question has dict-form
    attachments — equivalent to "the whole panel is one stratum."
    Otherwise walks each question, partitions personas by their matched
    attachment set (same key as :func:`count_strata`), and returns the
    minimum group size observed across all questions. The orchestrator
    feeds this into the per-question cache-marker predicate so a single
    skewed-filter question can't trip the marker for the whole run.
    """
    if not personas:
        return 0
    min_pop = len(personas)
    saw_any_attachments = False
    for q in questions:
        if not isinstance(q, dict):
            continue
        atts = q.get("attachments") or []
        if not isinstance(atts, list):
            continue
        dict_atts = [a for a in atts if isinstance(a, dict)]
        if not dict_atts:
            continue
        saw_any_attachments = True
        groups: dict[frozenset[int], int] = {}
        for p in personas:
            if not isinstance(p, dict):
                continue
            matched = filter_attachments(dict_atts, p)
            key = frozenset(id(a) for a in matched)
            groups[key] = groups.get(key, 0) + 1
        if groups:
            min_pop = min(min_pop, min(groups.values()))
    return min_pop if saw_any_attachments else len(personas)


# ---------------------------------------------------------------------------
# Caching parameters (hq-0pbp / D-phase hq-cxth)
# ---------------------------------------------------------------------------

# Anthropic's documented minimum cacheable prefix length. Below this the
# API silently skips caching — we mirror their floor to keep our predicate
# honest. Approximated via chars/4 since synthpanel ships no tokenizer.
_MIN_CACHEABLE_TOKENS = 1024
_CHARS_PER_TOKEN = 4
_MIN_CACHEABLE_CHARS = _MIN_CACHEABLE_TOKENS * _CHARS_PER_TOKEN

# Minimum stratum population for per-question attachment caching to break
# even (1.25x write + 0.1xN reads vs Nx1.0x uncached). Below this we don't
# emit per-question cache markers regardless of prefix length.
_MIN_STRATUM_POP_FOR_CACHE = 2

CacheTier = Literal["5m", "1h"]


def _approx_prefix_chars(system_prompt: str, blocks: list[ContentBlock]) -> int:
    """Cheap upper-bound estimate of the cacheable prefix size in chars.

    Counts the system prompt plus every text-bearing block in ``blocks``;
    image/document attachments contribute a synthetic 1500-token estimate
    each (≈ 6000 chars) since we have no way to size base64 payloads as
    tokens without invoking the provider tokenizer. Conservative on the
    high side: a slight over-count of prefix tokens just causes us to
    cache a slightly smaller-than-floor prefix, which the API silently
    drops anyway.
    """
    total = len(system_prompt or "")
    for b in blocks:
        if isinstance(b, TextBlock):
            total += len(b.text)
        else:
            # Image/document/html/url blocks are typically much larger than
            # text per token. 6000-char estimate per multimodal block keeps
            # the predicate safely above the 4096-char floor when even one
            # attachment is present.
            total += 6000
    return total


def _stratum_fingerprint(
    *,
    model: str,
    system_prompt: str,
    panel_shared_attachments: list[dict[str, Any]] | None,
    question_attachments: list[dict[str, Any]],
    question_text: str,
) -> str:
    """SHA256-derived 16-char fingerprint of the cacheable prefix (hq-0pbp).

    Logged for cache-hit telemetry. Hashes model id + persona system
    prompt + per-attachment SHA256 hashes (when supplied as
    pre-computed bank refs) or attachment payload data + the question
    text. Two panelists in the same stratum produce byte-identical
    fingerprints; any divergence in the fingerprint prefix indicates a
    cache miss is unavoidable.

    The 16-char truncation is enough to disambiguate within a single
    panel run (collision probability << 1 at K ≤ 5) while keeping log
    lines short. Per-attachment SHA256 reuses the digest from the CAS
    layer (hq-cqt5) when callers stash it under ``att["sha256"]``;
    otherwise we fall back to hashing the source dict so the
    fingerprint stays computable for runs without persisted CAS refs.
    """

    def _att_digest(att: dict[str, Any]) -> str:
        cached = att.get("sha256")
        if isinstance(cached, str) and cached:
            return cached
        payload = att.get("source") or att.get("text") or att.get("url") or att
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(hashlib.sha256((system_prompt or "").encode("utf-8")).digest())
    h.update(b"\x00")
    for att in panel_shared_attachments or []:
        h.update(_att_digest(att).encode("utf-8"))
        h.update(b"|")
    h.update(b"\x00")
    for att in question_attachments:
        h.update(_att_digest(att).encode("utf-8"))
        h.update(b"|")
    h.update(b"\x00")
    h.update(question_text.encode("utf-8"))
    return h.hexdigest()[:16]


class RunAbortedError(Exception):
    """Raised by :func:`run_panel_parallel` when the run aborts mid-flight (sp-56pb).

    Carries the partial panelist results (the prefix 0..k that completed
    before the abort) plus the registry and session dicts, so the caller
    can still surface a valid partial JSON with ``run_invalid: true`` and
    a specific ``abort_reason`` instead of losing the work entirely.

    Currently the only trigger is ``KeyboardInterrupt`` (SIGINT). The cost
    gate and convergence auto-stop paths do not raise — they return
    normally with ``results`` truncated to the completed prefix because
    the caller needs to inspect gate/tracker state to classify the halt.
    """

    def __init__(
        self,
        reason: str,
        results: list[PanelistResult],
        registry: WorkerRegistry,
        sessions: dict[str, Session],
    ) -> None:
        super().__init__(f"panel run aborted: {reason}")
        self.reason = reason
        self.results = results
        self.registry = registry
        self.sessions = sessions


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
# Per-persona LLM overrides (sp-4loufu)
# ---------------------------------------------------------------------------


# Keys recognised inside a persona's ``llm_overrides`` block. ``model`` is
# resolved separately (it routes through ``persona_models``); the remaining
# three flow into ``CompletionRequest`` for that persona's calls.
_LLM_OVERRIDE_KEYS: tuple[str, ...] = ("temperature", "top_p", "max_tokens", "model")


def validate_llm_overrides(overrides: dict[str, Any], *, persona_name: str = "") -> None:
    """Validate a single persona's ``llm_overrides`` block.

    Raises :class:`ValueError` with a persona-tagged message on any
    obviously-wrong value (e.g. ``temperature: 5.0``). Unknown keys are
    rejected so a YAML typo (``temperatur:``) doesn't silently fall
    back to the run-level default.
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"persona {persona_name!r} llm_overrides must be a mapping, got {type(overrides).__name__}")
    for key in overrides:
        if key not in _LLM_OVERRIDE_KEYS:
            allowed = ", ".join(_LLM_OVERRIDE_KEYS)
            raise ValueError(f"persona {persona_name!r} llm_overrides has unknown key {key!r}; allowed: {allowed}")
    if "temperature" in overrides:
        t = overrides["temperature"]
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            raise ValueError(f"persona {persona_name!r} llm_overrides.temperature must be a number, got {t!r}")
        if not 0.0 <= float(t) <= 2.0:
            raise ValueError(f"persona {persona_name!r} llm_overrides.temperature {t} is out of range [0.0, 2.0]")
    if "top_p" in overrides:
        p = overrides["top_p"]
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise ValueError(f"persona {persona_name!r} llm_overrides.top_p must be a number, got {p!r}")
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"persona {persona_name!r} llm_overrides.top_p {p} is out of range [0.0, 1.0]")
    if "max_tokens" in overrides:
        m = overrides["max_tokens"]
        if not isinstance(m, int) or isinstance(m, bool) or m < 1:
            raise ValueError(f"persona {persona_name!r} llm_overrides.max_tokens must be a positive integer, got {m!r}")
    if "model" in overrides:
        mdl = overrides["model"]
        if not isinstance(mdl, str) or not mdl.strip():
            raise ValueError(f"persona {persona_name!r} llm_overrides.model must be a non-empty string, got {mdl!r}")


def get_persona_llm_overrides(persona: dict[str, Any]) -> dict[str, Any]:
    """Return the ``llm_overrides`` block for *persona* (validated, possibly empty).

    Centralises the lookup so callers don't disagree on the key name or
    on what counts as a valid override value.
    """
    raw = persona.get("llm_overrides") or {}
    if not raw:
        return {}
    validate_llm_overrides(raw, persona_name=persona.get("name", "Anonymous"))
    return dict(raw)


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
        # sy-2wa: lazy threading import keeps synth_panel.ensemble loadable
        # under pyodide (CF Python Workers). Bare `threading` is a no-op
        # stub there; we still avoid binding it at module level so the
        # whole load chain stays threadpool-free until a real run.
        import threading

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


# ---------------------------------------------------------------------------
# v1.0.0 contract flags (sy-ac-5)
# ---------------------------------------------------------------------------


# The 7 enum members from schemas/v1.0.0.json#/flags_enum. Frozen at
# the contract level; new codes ship as a parallel schema version, not
# in-place edits. Kept as a tuple here (not re-read from the JSON) so
# the import stays free of disk I/O — :func:`Flag.__post_init__` is on
# the hot path of every panel run.
FLAG_CODES: tuple[str, ...] = (
    "low_convergence",
    "demographic_skew",
    "small_n",
    "persona_collision",
    "out_of_distribution",
    "refusal_or_degenerate",
    "schema_drift",
)

SEVERITIES: tuple[str, ...] = ("info", "warn", "block")


@dataclass(frozen=True)
class Flag:
    """An enum-bound quality flag from the v1.0.0 contract.

    Lands on ``panel_verdict.flags[]``. ``code`` must be one of
    :data:`FLAG_CODES`; non-enum signals belong on :class:`FlagExtension`.
    """

    code: str
    severity: str

    def __post_init__(self) -> None:
        if self.code not in FLAG_CODES:
            raise ValueError(
                f"Flag.code must be one of {FLAG_CODES}, got {self.code!r}; use FlagExtension for non-enum codes"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(f"Flag.severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass(frozen=True)
class FlagExtension:
    """A non-enum signal carried on ``panel_verdict.extension[]``.

    Lets callers surface bespoke quality concerns (e.g. a domain-specific
    cohort warning) without touching the frozen v1.0.0 ``flags_enum``.
    """

    code: str
    message: str
    severity: str

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("FlagExtension.code must be non-empty")
        if self.code in FLAG_CODES:
            raise ValueError(f"FlagExtension.code {self.code!r} collides with an enum member; use Flag instead")
        if self.severity not in SEVERITIES:
            raise ValueError(f"FlagExtension.severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class PanelState:
    """Post-synthesis snapshot the flag-raiser inspects.

    Built by the orchestrator after a run finishes (or aborts — the
    raiser is robust to partial state). The verdict assembler (AC-6)
    consumes the same shape, so additions here are append-only.
    """

    panelist_results: list[PanelistResult] = field(default_factory=list)
    personas: list[dict[str, Any]] = field(default_factory=list)
    convergence: float | None = None
    schema_drift: bool = False
    expected_categories: list[str] | None = None
    observed_categories: list[str] | None = None
    extensions: list[FlagExtension] = field(default_factory=list)


# Threshold knobs. Tuned in build-plan.md, not part of the wire contract;
# treat as safe to nudge without bumping the schema version. The raiser
# trips at the boundary inclusive on the lower side (n < BLOCK → block,
# BLOCK ≤ n < WARN → warn, n ≥ WARN → no flag).
_SMALL_N_BLOCK = 4
_SMALL_N_WARN = 8
_LOW_CONVERGENCE_BLOCK = 0.30
_LOW_CONVERGENCE_WARN = 0.50
_REFUSAL_BLOCK_FRAC = 0.50
_REFUSAL_WARN_FRAC = 0.25

# Demographic keys we check for uniformity. Limited to common persona
# fields so the heuristic doesn't flag on incidental shared metadata
# (e.g. every persona happening to have the same ``llm_overrides``).
_DEMOGRAPHIC_KEYS: tuple[str, ...] = ("occupation", "age", "gender", "region", "country")


def _is_degenerate(pr: PanelistResult) -> bool:
    """A panelist counts as degenerate if their session errored or
    every recorded response was an error / skipped / empty.

    Mirrors how the synthesizer treats ``error`` / ``skipped_by_budget``
    rows when building convergence inputs — keep the two in sync.
    """
    if pr.error:
        return True
    if not pr.responses:
        return True
    bad = 0
    for r in pr.responses:
        if r.get("error") or r.get("skipped_by_budget"):
            bad += 1
            continue
        resp = r.get("response")
        if resp is None or (isinstance(resp, str) and not resp.strip()):
            bad += 1
    return bad == len(pr.responses)


def _detect_demographic_skew(personas: list[dict[str, Any]]) -> bool:
    """All personas share an identical value on at least one demographic key."""
    if len(personas) < 2:
        return False
    for k in _DEMOGRAPHIC_KEYS:
        values = [p.get(k) for p in personas if isinstance(p, dict) and k in p]
        if len(values) == len(personas) and len(set(values)) == 1:
            return True
    return False


def _detect_persona_collision(personas: list[dict[str, Any]], results: list[PanelistResult]) -> bool:
    """Two personas (or two results) carry the same name."""
    pnames = [p.get("name") for p in personas if isinstance(p, dict) and p.get("name")]
    if len(pnames) != len(set(pnames)):
        return True
    rnames = [r.persona_name for r in results]
    return len(rnames) != len(set(rnames))


def _raise_flags(panel_state: PanelState) -> list[Flag]:
    """Inspect ``panel_state`` post-synthesis and emit enum-bound flags.

    Each returned :class:`Flag` maps to one of the 7 ``flags_enum``
    members in ``schemas/v1.0.0.json``. Severity is one of
    :data:`SEVERITIES`. The function never raises on a malformed
    ``panel_state`` — missing fields skip the relevant check so the
    raiser stays callable on partial / aborted runs (sp-56pb).

    Non-enum signals belong on ``panel_state.extensions``; the verdict
    assembler (AC-6) surfaces them under ``panel_verdict.extension[]``
    rather than ``flags[]``. Mixing the two would silently shadow a
    real enum member, hence :class:`FlagExtension` rejects enum codes
    at construction time.
    """
    flags: list[Flag] = []
    n = len(panel_state.panelist_results)

    if 0 < n < _SMALL_N_BLOCK:
        flags.append(Flag(code="small_n", severity="block"))
    elif _SMALL_N_BLOCK <= n < _SMALL_N_WARN:
        flags.append(Flag(code="small_n", severity="warn"))

    if panel_state.convergence is not None:
        c = panel_state.convergence
        if c < _LOW_CONVERGENCE_BLOCK:
            flags.append(Flag(code="low_convergence", severity="block"))
        elif c < _LOW_CONVERGENCE_WARN:
            flags.append(Flag(code="low_convergence", severity="warn"))

    if _detect_demographic_skew(panel_state.personas):
        flags.append(Flag(code="demographic_skew", severity="warn"))

    if _detect_persona_collision(panel_state.personas, panel_state.panelist_results):
        flags.append(Flag(code="persona_collision", severity="warn"))

    if panel_state.expected_categories is not None and panel_state.observed_categories is not None:
        expected = set(panel_state.expected_categories)
        if any(c not in expected for c in panel_state.observed_categories):
            flags.append(Flag(code="out_of_distribution", severity="warn"))

    if n > 0:
        bad_count = sum(1 for pr in panel_state.panelist_results if _is_degenerate(pr))
        frac = bad_count / n
        if frac >= _REFUSAL_BLOCK_FRAC:
            flags.append(Flag(code="refusal_or_degenerate", severity="block"))
        elif frac >= _REFUSAL_WARN_FRAC:
            flags.append(Flag(code="refusal_or_degenerate", severity="warn"))

    if panel_state.schema_drift:
        flags.append(Flag(code="schema_drift", severity="warn"))

    return flags


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
    seed: int | None = None,
    question_budget: QuestionFailureBudget | None = None,
    cache_enabled: bool = True,
    cache_tier: CacheTier = "5m",
    panel_shared_attachments: list[dict[str, Any]] | None = None,
    stratum_population: int = 1,
    request_id: str | None = None,
    url_cache_l1: CacheL1 | None = None,
    url_cache_disk: UrlCache | None = None,
) -> tuple[PanelistResult, Session]:
    """Execute a single panelist's full interview. Runs in a worker thread.

    Manages the worker lifecycle: spawning → ready → running → finished/failed.

    Per-persona ``llm_overrides`` (sp-4loufu) take precedence over the
    run-level ``temperature``/``top_p`` arguments and bump the default
    ``max_tokens`` away from the runtime's 4096. They've already been
    validated by :func:`get_persona_llm_overrides` in the caller, but
    we re-read the dict here so a single thread sees a coherent view.
    """
    name = persona.get("name", "Anonymous")
    overrides = persona.get("llm_overrides") or {}
    eff_temperature = overrides.get("temperature", temperature)
    eff_top_p = overrides.get("top_p", top_p)
    eff_max_tokens = overrides.get("max_tokens", 4096)
    tracker = UsageTracker()
    responses: list[dict[str, Any]] = []
    t0 = _time.monotonic()
    logger.info(
        "panelist %s starting (model=%s, questions=%d, temperature=%s, top_p=%s, max_tokens=%s)",
        name,
        model,
        len(questions),
        eff_temperature,
        eff_top_p,
        eff_max_tokens,
    )

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
            max_tokens=eff_max_tokens,
            temperature=eff_temperature,
            top_p=eff_top_p,
            seed=seed,
            cache_enabled=cache_enabled,
        )

        # Set up structured output engine if schema provided
        structured_engine: StructuredOutputEngine | None = None
        structured_config: StructuredOutputConfig | None = None
        if response_schema:
            structured_engine = StructuredOutputEngine(client)
            structured_config = StructuredOutputConfig(schema=response_schema)

        # Set up extraction engine for post-hoc structured extraction.
        # ``extract_schema`` is either the v1.0.3 resolved envelope or a
        # legacy raw JSON Schema dict — _unpack_extract_schema normalises
        # both forms so the wire-level structured-output engine sees a
        # plain JSON Schema and the optional Pydantic class is held aside
        # for the post-extraction typed validation pass below.
        extract_engine: StructuredOutputEngine | None = None
        extract_config: StructuredOutputConfig | None = None
        extract_json_schema, extract_pydantic_model = _unpack_extract_schema(extract_schema)
        if extract_json_schema:
            extract_engine = StructuredOutputEngine(client)
            extract_config = StructuredOutputConfig(schema=extract_json_schema)

        for qi, question in enumerate(questions):
            question_text = question_prompt_fn(question)

            # hq-iczd: per-persona attachment stratification. Compute the
            # filtered attachment list before the budget gate so it travels
            # alongside the question regardless of skip path. Multimodal
            # block emission off this list is hq-0pbp's responsibility.
            raw_attachments = question.get("attachments", []) if isinstance(question, dict) else []
            # Only treat dict-form attachments as filterable; bank-ref strings
            # pass through as-is and produce no per-persona divergence.
            dict_attachments = [a for a in raw_attachments if isinstance(a, dict)]
            attachments_for_persona = filter_attachments(dict_attachments, persona) if dict_attachments else []

            # hq-0pbp: build the multimodal user-message blocks for this
            # question. When the question has no attachments and no panel-
            # shared blocks, this collapses to a single TextBlock — the
            # legacy text-only path. Per-question cache marker fires only
            # when the run-level cache is enabled AND this stratum is
            # large enough to amortize the cache write AND the cacheable
            # prefix clears Anthropic's 1024-token floor.
            user_blocks = build_question_blocks(
                question,
                attachments=attachments_for_persona,
                panel_shared_attachments=panel_shared_attachments,
            )
            prefix_chars = _approx_prefix_chars(system_prompt, user_blocks)
            has_any_attachment_block = bool(attachments_for_persona) or bool(panel_shared_attachments)
            should_mark_cache = (
                cache_enabled
                and stratum_population >= _MIN_STRATUM_POP_FOR_CACHE
                and prefix_chars >= _MIN_CACHEABLE_CHARS
                and has_any_attachment_block
            )
            if should_mark_cache:
                user_blocks = build_question_blocks(
                    question,
                    attachments=attachments_for_persona,
                    panel_shared_attachments=panel_shared_attachments,
                    cache_marker=True,
                )

            # hq-0pbp: stratum fingerprint for cache-hit telemetry. Logged
            # at INFO so operators can confirm two panelists in the same
            # stratum produce identical fingerprints (and thus a cache hit
            # on the second one). Computed regardless of whether caching
            # is currently enabled — observability cost is negligible and
            # it lets us debug "why didn't this cache?" post hoc.
            fingerprint = _stratum_fingerprint(
                model=model,
                system_prompt=system_prompt,
                panel_shared_attachments=panel_shared_attachments,
                question_attachments=attachments_for_persona,
                question_text=question_text,
            )
            logger.info(
                "[%s] stratum_fp=%s persona=%s q=%d cache=%s tier=%s P=%d prefix_chars=%d",
                request_id or "-",
                fingerprint,
                name,
                qi,
                "on" if should_mark_cache else "off",
                cache_tier,
                stratum_population,
                prefix_chars,
            )

            # sp-xw2z6o: per-question failure budget. If a prior panelist
            # tripped this question's budget, skip it cheaply instead of
            # re-failing. We still record an entry so per-panelist response
            # arrays stay aligned with the authored question list.
            if question_budget is not None and question_budget.is_disabled(qi):
                responses.append(
                    {
                        "question": question_text,
                        "response": None,
                        "skipped_by_budget": True,
                        "question_index": qi,
                    }
                )
                continue

            # hq-0pbp: turn input is multimodal blocks when this question
            # carries any attachments (panel-shared or per-persona).
            # Otherwise the legacy single-text path is used so non-attachment
            # call sites stay untouched.
            has_attachments = bool(attachments_for_persona) or bool(panel_shared_attachments)

            # hq-8iz3: frame-stage URLBlock lowering. Resolve any pre-fetch
            # URL stubs to concrete TextBlock / ImageBlock entries via the
            # hq-gmju content ladder. Runs after the cache fingerprint is
            # computed (so two panelists pointing at the same URL hit the
            # same stratum) and before serialization (so URLBlock never
            # reaches the wire).
            if has_attachments and any(isinstance(b, URLBlock) for b in user_blocks):
                user_blocks = lower_url_blocks(
                    user_blocks,
                    l1=url_cache_l1,
                    cache=url_cache_disk,
                )

            turn_input: str | list[ContentBlock] = user_blocks if has_attachments else question_text

            try:
                if structured_engine and structured_config:
                    # Use structured output: run turn for conversation context,
                    # then extract structured response
                    summary = runtime.run_turn(turn_input)
                    tracker.record_turn(summary.usage)

                    # Build messages from session history for structured extraction
                    extract_user_content: list[ContentBlock] = (
                        list(user_blocks) if has_attachments else [TextBlock(text=question_text)]
                    )
                    messages = [InputMessage(role="user", content=extract_user_content)]
                    result = structured_engine.extract(
                        model=model,
                        max_tokens=eff_max_tokens,
                        messages=messages,
                        config=structured_config,
                        system=system_prompt,
                        temperature=eff_temperature,
                        top_p=eff_top_p,
                        seed=seed,
                    )
                    tracker.record_turn(_convert_llm_usage(result.total_usage))
                    responses.append(
                        {
                            "question": question_text,
                            "response": result.data,
                            "structured": True,
                            "is_fallback": result.is_fallback,
                        }
                    )
                else:
                    summary = runtime.run_turn(turn_input)
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
                            extract_user_content = (
                                list(user_blocks) if has_attachments else [TextBlock(text=question_text)]
                            )
                            extract_messages = [
                                InputMessage(
                                    role="user",
                                    content=extract_user_content,
                                ),
                                InputMessage(
                                    role="assistant",
                                    content=[TextBlock(text=response_text)],
                                ),
                            ]
                            extract_result = extract_engine.extract(
                                model=model,
                                max_tokens=eff_max_tokens,
                                messages=extract_messages,
                                config=extract_config,
                                system=system_prompt,
                                temperature=eff_temperature,
                                top_p=eff_top_p,
                                seed=seed,
                            )
                            tracker.record_turn(_convert_llm_usage(extract_result.total_usage))
                            resp_dict["extraction"] = extract_result.data
                            resp_dict["extraction_is_fallback"] = extract_result.is_fallback
                            # v1.0.3 P1: typed Pydantic validation pass.
                            # When the caller supplied a BaseModel subclass
                            # (or a registered name with a model in
                            # MODEL_REGISTRY), validate the extracted dict
                            # via ``model_validate`` so a usable
                            # field-path error surfaces when the LLM
                            # produced wire-valid JSON that still violates
                            # the typed contract (e.g. ``rating: 7`` for
                            # the 1..5 Likert).
                            if (
                                extract_pydantic_model is not None
                                and not extract_result.is_fallback
                                and isinstance(extract_result.data, dict)
                            ):
                                try:
                                    # extract_pydantic_model is annotated as type[Any] for
                                    # the import-fallback path; in practice it's always a
                                    # Pydantic BaseModel subclass when non-None.
                                    extract_pydantic_model.model_validate(  # type: ignore[attr-defined]
                                        extract_result.data
                                    )
                                except _PydanticValidationError as ve:
                                    resp_dict["extraction_validation_error"] = str(ve)
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
                # sp-xw2z6o: record this failure against the per-question
                # budget so subsequent panelists short-circuit once the
                # threshold is crossed.
                if question_budget is not None:
                    try:
                        question_budget.record_failure(qi, question_text=question_text)
                    except Exception:  # pragma: no cover - defensive
                        logger.warning("question_budget.record_failure raised; ignoring", exc_info=True)

            # hq-iczd: persist the per-persona filtered attachment list on
            # the just-appended main-question response so downstream
            # consumers (hq-0pbp caching/multimodal emission, persistence
            # readback) can recover the stratification decision. Only stamp
            # when the question carried attachments at all.
            if raw_attachments and responses and responses[-1].get("question") == question_text:
                responses[-1]["attachments"] = attachments_for_persona

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
                    responses.append(
                        {
                            "question": fu["text"],
                            "response": None,
                            "follow_up": True,
                            "skipped_by_condition": True,
                            "condition": condition,
                        }
                    )
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
    seed: int | None = None,
    persona_models: dict[str, str] | None = None,
    convergence_tracker: ConvergenceTracker | None = None,
    on_panelist_complete: Callable[[PanelistResult], None] | None = None,
    cost_gate: CostGate | None = None,
    question_budget: QuestionFailureBudget | None = None,
    panel_shared_attachments: list[dict[str, Any]] | None = None,
    attachment_bank: dict[str, dict[str, Any]] | None = None,
    cache_tier: CacheTier = "5m",
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
        on_panelist_complete: Optional callback invoked once per panelist the
            moment their future resolves (sp-hsk3). Used by
            :mod:`synth_panel.checkpoint` to snapshot progress every K
            completions so a crashed or SIGINT'd run can resume. Exceptions
            raised by the callback are logged and suppressed — a broken
            checkpoint writer must never kill a live run.
        cost_gate: Optional :class:`CostGate`. Each completing panelist's
            priced cost is recorded against the gate; if the projected run
            total exceeds the gate's ceiling, pending futures are cancelled
            and only finished panelists are returned. The caller is expected
            to inspect ``cost_gate.should_halt()`` on return and surface a
            partial, ``run_invalid`` result.
        question_budget: Optional :class:`QuestionFailureBudget` (sp-xw2z6o).
            Tracks per-question failure counts; when a question's failure
            count crosses the configured threshold, subsequent panelists
            skip that question (each emits a ``skipped_by_budget`` response
            entry) instead of re-failing it. The run still completes —
            only the offending question is short-circuited.

    Returns:
        Tuple of (ordered results matching persona order, registry,
        sessions dict mapping persona names to their sessions).
    """
    # sp-4loufu: validate per-persona ``llm_overrides`` before any worker
    # spawns so a malformed YAML block fails the run loudly rather than
    # silently dropping the override mid-execution.
    for p in personas:
        if isinstance(p, dict) and p.get("llm_overrides"):
            validate_llm_overrides(p["llm_overrides"], persona_name=p.get("name", "Anonymous"))

    # hq-ilke: legacy single-round path resolves bank-ref strings here when
    # the caller supplies the instrument's attachment bank. Without this,
    # ``question.attachments = ["hero_creative_v3"]`` silently fell out at the
    # downstream dict-only filter and panelists responded as if no attachment
    # had been provided — silent data loss with no log line. Multi-round
    # callers (run_multi_round_panel) pre-compute panel_shared_attachments
    # and pass that instead, so the two parameters are mutually exclusive.
    if attachment_bank is not None:
        if panel_shared_attachments is not None:
            raise ValueError(
                "run_panel_parallel: attachment_bank and panel_shared_attachments are mutually exclusive "
                "(multi-round callers pre-compute the shared list; legacy single-round callers pass the bank)"
            )
        panel_shared_computed, shared_ref_ids = _compute_panel_shared(questions, attachment_bank)
        questions = _resolve_question_attachment_refs(
            questions,
            attachment_bank,
            exclude_refs=shared_ref_ids,
        )
        panel_shared_attachments = panel_shared_computed or None

    # hq-0pbp: K≤5 frame-stage gate. Refuse panels whose attachment
    # stratification produces more than 5 distinct strata before any
    # worker spawns or LLM call lands. Above K=5 the cached economics
    # collapse — see D-phase hq-cxth §5/§7. The check runs per-question;
    # the first violator raises with question index + offending K + cap.
    _enforce_strata_cap(personas, questions)

    # hq-0pbp: per-question stratum population for cache-marker decisions.
    # Min stratum size across all attachment-bearing questions is the
    # safe lower bound — if any question splits the panel into singletons
    # the cache predicate refuses the marker for that question. Empty /
    # no-attachment panels fall back to len(personas) so the legacy
    # always-on prefix caching still applies.
    min_stratum_pop = _min_stratum_population(personas, questions)

    # sy-2wa: lazy threading + concurrent.futures imports. Bound here so
    # `from synth_panel.ensemble import synthesize_panel` never pulls
    # ThreadPoolExecutor into the module's namespace under pyodide
    # (CF Python Workers), where `.submit()` silently hangs.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    registry = WorkerRegistry()
    effective_workers = max_workers or len(personas)
    sentiment_cache: dict[str, str] = {}
    sentiment_cache_lock = threading.Lock()
    request_id = uuid.uuid4().hex[:12]

    # hq-8iz3: per-run URL fetch cache. The L1 is in-memory and shared
    # across all panelists in this run so the same URL is fetched at
    # most once regardless of stratum. The disk-backed UrlCache layers
    # cross-run dedup on top (default root: ~/.synthpanel/cache/url).
    url_cache_l1 = CacheL1()
    url_cache_disk = UrlCache()
    # hq-0pbp: P=1 panels skip caching entirely (D-phase hq-cxth §6 bypass).
    cache_enabled_for_run = len(personas) >= _MIN_STRATUM_POP_FOR_CACHE
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

    sigint_aborted = False
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index = {}
        for idx, (persona, worker_id) in enumerate(zip(personas, worker_ids)):
            name = persona.get("name", "Anonymous")
            existing_session = (sessions or {}).get(name)
            # Resolve per-persona model: persona_models mapping > global default.
            # ``llm_overrides.model`` (sp-4loufu) is folded into
            # ``persona_models`` by the CLI/SDK before this point — the
            # orchestrator does not read it directly so behaviour stays
            # consistent with the legacy top-level ``model:`` field
            # (which is also CLI/SDK-extracted, not orchestrator-resolved).
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
                seed,
                question_budget,
                cache_enabled_for_run,
                cache_tier,
                panel_shared_attachments,
                min_stratum_pop,
                request_id,
                url_cache_l1,
                url_cache_disk,
            )
            future_to_index[future] = idx

        try:
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

                # sp-hsk3: invoke the checkpoint callback for every resolved
                # panelist, success or failure. Errored panelists still count
                # as "done" for resume purposes — otherwise a persistent
                # upstream outage would stall the cadence forever.
                if on_panelist_complete is not None and results[idx] is not None:
                    try:
                        on_panelist_complete(results[idx])  # type: ignore[arg-type]
                    except Exception as cb_exc:  # pragma: no cover - defensive
                        logger.warning(
                            "on_panelist_complete callback failed: %s: %s",
                            type(cb_exc).__name__,
                            cb_exc,
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
        except KeyboardInterrupt:
            # sp-56pb: surface partial results instead of losing them. SIGINT
            # is one of the four abort paths that must produce a valid,
            # partial JSON result. We cancel pending futures so the executor
            # does not block on work the user has already given up on, then
            # raise RunAbortedError so the caller can classify the halt and
            # emit the standard partial-JSON envelope.
            logger.warning(
                "panel run interrupted by SIGINT after %d panelist(s)", sum(1 for r in results if r is not None)
            )
            sigint_aborted = True
            for pending_future in future_to_index:
                if not pending_future.done():
                    pending_future.cancel()

    # All slots should be filled (unless auto-stop cancelled some); drop Nones
    final_results = [r for r in results if r is not None]
    if sigint_aborted:
        raise RunAbortedError(
            reason="sigint",
            results=final_results,
            registry=registry,
            sessions=out_sessions,
        )
    return final_results, registry, out_sessions


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
    """Linear-chain fallback for rounds without route_when.

    Resolution order, mirroring parser reachability semantics
    (``_reachability_warnings`` in ``instrument.py``):

    1. A round whose ``depends_on`` names ``current`` (v2 explicit chaining).
    2. The next positional round in ``instrument.rounds`` (v3 linear
       fallthrough — the implicit linear edge the parser already honors).
    3. ``__end__`` when ``current`` is the last round.
    """
    for r in instrument.rounds:
        if r.depends_on == current:
            return r.name
    for i, r in enumerate(instrument.rounds):
        if r.name == current and i + 1 < len(instrument.rounds):
            return instrument.rounds[i + 1].name
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
    seed: int | None = None,
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

        # G3 fix: resolve bank-ref strings in question.attachments to inline
        # dict-form before run_panel_parallel filters out non-dict refs.
        # Without this, bank-keyed attachments (the canonical pattern per the
        # hq-xzsm data-model design) silently dropped — every persona received
        # only the question text. The bank lives on ``instrument.attachments``;
        # this is the natural resolution site since we have the Instrument
        # object in scope here.
        #
        # G2 (hq-ovxl): lift bank entries referenced by ≥2 questions in this
        # round onto ``panel_shared_attachments`` so they emit once before
        # the cache_control marker (hq-0pbp canonical order). Lifted refs are
        # then excluded from per-question resolution to avoid double-emission.
        panel_shared, shared_ref_ids = _compute_panel_shared(current.questions, instrument.attachments)
        resolved_questions = _resolve_question_attachment_refs(
            current.questions,
            instrument.attachments,
            exclude_refs=shared_ref_ids,
        )

        panelist_results, _registry, sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=resolved_questions,
            model=model,
            system_prompt_fn=system_prompt_fn,
            question_prompt_fn=question_prompt_fn,
            max_workers=max_workers,
            response_schema=response_schema,
            sessions=sessions,
            extract_schema=extract_schema,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            persona_models=persona_models,
            panel_shared_attachments=panel_shared or None,
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
    seed: int | None = None,
    attachment_bank: dict[str, dict[str, Any]] | None = None,
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
            seed=seed,
            attachment_bank=attachment_bank,
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
