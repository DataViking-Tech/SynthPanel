"""Live convergence telemetry for panel runs (sp-yaru).

At competitive-parity scales (n=500..10k) a narrative synthesis is less useful
than watching the per-question response distribution stabilize. This module
provides a :class:`ConvergenceTracker` that:

1. Identifies which instrument questions have a *bounded* response space
   (Likert / yes-no / pick-one / enum). Free-text questions are ignored.
2. Records each completing panelist's categorical response and computes
   Jensen-Shannon divergence between the distribution-so-far and the
   last-``K``-batch after every K panelists.
3. Optionally signals auto-stop once the rolling JSD stays below an
   epsilon threshold for M consecutive checks (with a minimum-n floor).
4. Produces a post-run ``convergence`` report describing per-question
   curves, the smallest n at which convergence held, and — when the
   caller supplied one — a human-baseline comparison.

The tracker is **pure** — callers (the orchestrator) drive it and decide
what to do with its signals. This keeps it trivially unit-testable
without mocking the LLM stack.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from synth_panel.structured.schemas import PICK_ONE_SCHEMA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_CHECK_EVERY = 20
DEFAULT_EPSILON = 0.02
DEFAULT_MIN_N = 50
DEFAULT_M_CONSECUTIVE = 3
DEFAULT_CURVE_POINTS = 20

# Structured-response key heuristics. The bundled schemas
# (src/synth_panel/structured/schemas.py) each reserve exactly one field
# that carries the categorical answer; everything else is free-text
# reasoning. Ordered most-specific → least-specific so a combined schema
# is still keyed deterministically.
_BOUNDED_KEYS: tuple[str, ...] = ("rating", "answer", "choice", "value", "label")


# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------


def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Return JSD(P||Q) in base-2 so the result is bounded in ``[0, 1]``.

    ``p`` and ``q`` are discrete distributions expressed as ``category → mass``.
    They are normalized internally; empty distributions (or any pair whose
    supports fail to normalize) return ``0.0`` — there is no meaningful
    divergence to report on zero samples, and callers treat that as
    "converged by default" only when enough panelists have arrived
    (``min_n`` handles that separately).
    """
    if not p or not q:
        return 0.0
    p_total = sum(p.values())
    q_total = sum(q.values())
    if p_total <= 0 or q_total <= 0:
        return 0.0
    p_norm = {k: v / p_total for k, v in p.items()}
    q_norm = {k: v / q_total for k, v in q.items()}
    support = set(p_norm) | set(q_norm)

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k in support:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak <= 0.0 or bk <= 0.0:
                continue
            total += ak * math.log2(ak / bk)
        return total

    m = {k: 0.5 * (p_norm.get(k, 0.0) + q_norm.get(k, 0.0)) for k in support}
    jsd = 0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m)
    # Clamp floating-point drift so downstream eps comparisons stay stable.
    return max(0.0, min(1.0, jsd))


# ---------------------------------------------------------------------------
# Question inspection
# ---------------------------------------------------------------------------


_KNOWN_BOUNDED_SCHEMA_NAMES: frozenset[str] = frozenset({"likert", "yes_no", "pick_one", "ranking"})


def _question_key(question: dict[str, Any], index: int) -> str:
    """Return a stable, human-meaningful key for a question."""
    if not isinstance(question, dict):
        return f"q{index}"
    for candidate in ("key", "id", "name"):
        value = question.get(candidate)
        if isinstance(value, str) and value:
            return value
    text = question.get("text")
    if isinstance(text, str) and text:
        # Use the first 40 chars as a readable fallback key.
        return text.strip()[:40] or f"q{index}"
    return f"q{index}"


def _schema_is_bounded(schema: Any) -> bool:
    """True when a JSON Schema dict describes a categorical / enum output."""
    if isinstance(schema, str):
        return schema in _KNOWN_BOUNDED_SCHEMA_NAMES
    if not isinstance(schema, dict):
        return False
    # Top-level enum field
    if isinstance(schema.get("enum"), list):
        return True
    props = schema.get("properties")
    if isinstance(props, dict):
        for prop in props.values():
            if not isinstance(prop, dict):
                continue
            if isinstance(prop.get("enum"), list):
                return True
            # Known scalar types that imply a small support. We
            # deliberately do NOT treat plain ``"type": "string"`` as
            # bounded — that would pull every free-text field into the
            # tracker and pollute the report.
            if prop.get("type") == "boolean":
                return True
    return False


def _is_tracked(question: dict[str, Any]) -> bool:
    """True when a question's response space is bounded enough to track."""
    if not isinstance(question, dict):
        return False
    extraction = question.get("extraction_schema")
    if _schema_is_bounded(extraction):
        return True
    response = question.get("response_schema")
    if isinstance(response, dict):
        if response.get("type") == "scaled":
            return True
        if response.get("type") == "enum":
            return True
        if isinstance(response.get("options"), list):
            return True
    return bool(_schema_is_bounded(response))


def identify_tracked_questions(
    questions: list[dict[str, Any]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Return ``[(index, key, question_dict)]`` for every bounded question.

    Questions whose response space is unconstrained (free text) are
    dropped — JSD on free-text responses is ill-defined and would just
    measure paraphrase noise.
    """
    tracked: list[tuple[int, str, dict[str, Any]]] = []
    for i, q in enumerate(questions):
        if _is_tracked(q):
            tracked.append((i, _question_key(q, i), q))
    return tracked


# ---------------------------------------------------------------------------
# Response → category extraction
# ---------------------------------------------------------------------------


def _coerce_category(value: Any) -> str | None:
    """Normalize a structured response value into a hashable category label."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        # For multi-pick or ranking responses use a stable joined form.
        parts = [_coerce_category(v) for v in value]
        parts_clean = [p for p in parts if p is not None]
        return "|".join(parts_clean) if parts_clean else None
    if isinstance(value, dict):
        # Pick the most-specific known field, otherwise sort-join keys so
        # the category is stable across runs.
        for key in _BOUNDED_KEYS:
            if key in value:
                return _coerce_category(value[key])
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return None
    return None


def extract_category(response_entry: dict[str, Any]) -> str | None:
    """Pull a bounded category from one panelist response dict.

    The orchestrator stores each response as
    ``{"question": str, "response": text|dict, "structured"?: bool, "extraction"?: dict}``.
    We prefer structured payloads (extraction or structured response dicts)
    because their support is intentional; plain strings fall through only
    if the whole value parses to one of the bounded key fields.
    """
    if not isinstance(response_entry, dict):
        return None
    if response_entry.get("error"):
        return None
    # Prefer an explicit extraction payload.
    extraction = response_entry.get("extraction")
    if isinstance(extraction, dict):
        for key in _BOUNDED_KEYS:
            if key in extraction:
                return _coerce_category(extraction[key])
        # Fallback: extraction dict with a single key.
        if len(extraction) == 1:
            only_key = next(iter(extraction))
            return _coerce_category(extraction[only_key])
    # Then structured response bodies.
    response = response_entry.get("response")
    if isinstance(response, dict):
        return _coerce_category(response)
    # Plain text responses: only yield a category when the raw value is
    # itself a short bounded token (e.g. ``"yes"``). Otherwise return
    # None so tracking stays opt-in per question.
    if isinstance(response, str):
        text = response.strip().lower()
        if text in {"yes", "no", "true", "false"}:
            return text
    return None


def extract_categorical_responses(
    panelist_result: Any,
    tracked: list[tuple[int, str, dict[str, Any]]],
) -> dict[str, str]:
    """Return ``{question_key: category_label}`` for one panelist.

    Missing responses (failures, un-extractable free text) are simply
    omitted so the running distributions stay honest.
    """
    responses = getattr(panelist_result, "responses", None)
    if not isinstance(responses, list):
        return {}
    # The orchestrator may inject follow-up turns, so we step through
    # main-question responses in order, skipping follow-up-flagged rows.
    main_iter = (r for r in responses if isinstance(r, dict) and not r.get("follow_up"))
    main_list = list(main_iter)
    out: dict[str, str] = {}
    for q_idx, q_key, _q in tracked:
        if q_idx >= len(main_list):
            continue
        category = extract_category(main_list[q_idx])
        if category is not None:
            out[q_key] = category
    return out


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


@dataclass
class _QuestionState:
    """Per-question running state inside the tracker."""

    key: str
    cumulative: Counter = field(default_factory=Counter)
    last_batch: Counter = field(default_factory=Counter)
    rolling_jsd: list[float] = field(default_factory=list)
    per_n_jsd: list[tuple[int, float]] = field(default_factory=list)
    # Smallest n at which *rolling_jsd* stayed below epsilon for M checks
    converged_at: int | None = None


@dataclass
class ConvergenceCheck:
    """One instant-in-time convergence reading, emitted per K panelists."""

    n_completed: int
    per_question: dict[str, dict[str, float]]


class ConvergenceTracker:
    """Running convergence metrics for a panel run.

    The tracker is thread-compatible: orchestrator threads may call
    :meth:`record` concurrently; a single internal lock serializes state
    updates. Callers should drive :meth:`should_stop` from a single
    thread after each record, and prepare the final report via
    :meth:`build_report` once the run ends.
    """

    def __init__(
        self,
        tracked: list[tuple[int, str, dict[str, Any]]],
        *,
        check_every: int = DEFAULT_CHECK_EVERY,
        epsilon: float = DEFAULT_EPSILON,
        min_n: int = DEFAULT_MIN_N,
        m_consecutive: int = DEFAULT_M_CONSECUTIVE,
        auto_stop: bool = False,
        log_path: str | None = None,
    ) -> None:
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        if epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        if min_n < 0:
            raise ValueError("min_n must be >= 0")
        if m_consecutive < 1:
            raise ValueError("m_consecutive must be >= 1")

        self._tracked = tracked
        self._keys = [k for _i, k, _q in tracked]
        self._check_every = check_every
        self._epsilon = epsilon
        self._min_n = min_n
        self._m = m_consecutive
        self._auto_stop = auto_stop
        self._log_path = log_path
        self._states: dict[str, _QuestionState] = {k: _QuestionState(key=k) for k in self._keys}
        self._n = 0
        self._last_check_n = 0
        self._auto_stopped = False
        self._checks: list[ConvergenceCheck] = []
        self._lock = threading.Lock()
        self._log_fh: Any = None
        if log_path:
            try:
                self._log_fh = open(log_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
            except OSError as exc:
                logger.warning("could not open convergence log %s: %s", log_path, exc)
                self._log_fh = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tracked_keys(self) -> list[str]:
        return list(self._keys)

    @property
    def auto_stopped(self) -> bool:
        return self._auto_stopped

    @property
    def overall_converged_at(self) -> int | None:
        with self._lock:
            return self._overall_converged_at_locked()

    def record(self, categorical: dict[str, str]) -> bool:
        """Add one panelist's categorical responses.

        Returns ``True`` when the batch just crossed a check boundary
        *and* the run should stop (auto_stop satisfied). Callers that
        do not want auto-stop can ignore the return value — the stop
        signal is only raised when :attr:`auto_stop` is enabled.
        """
        with self._lock:
            self._n += 1
            for key, value in categorical.items():
                state = self._states.get(key)
                if state is None:
                    continue
                state.cumulative[value] += 1
                state.last_batch[value] += 1
            if self._n - self._last_check_n >= self._check_every:
                check = self._run_check_locked()
                self._emit_log(check)
                self._last_check_n = self._n
                # Reset last-batch buckets after each check.
                for state in self._states.values():
                    state.last_batch = Counter()
                if self._auto_stop and self._is_converged_locked() and self._n >= self._min_n:
                    self._auto_stopped = True
                    return True
        return False

    def close(self) -> None:
        """Flush and close the optional log file."""
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    def build_report(
        self,
        *,
        baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the post-run ``convergence`` section.

        ``baseline`` — when provided — is spliced in verbatim under
        ``human_baseline``; the caller owns the lookup against SynthBench
        so this module stays free of optional-dep imports.
        """
        with self._lock:
            per_question: dict[str, Any] = {}
            for key, state in self._states.items():
                curve = _downsample_curve(state.per_n_jsd, DEFAULT_CURVE_POINTS)
                per_question[key] = {
                    "final_n": self._n,
                    "converged_at": state.converged_at,
                    "curve": [{"n": n, "jsd": round(jsd, 6)} for n, jsd in curve],
                    "support_size": len(state.cumulative),
                }
            overall = self._overall_converged_at_locked()
            report: dict[str, Any] = {
                "final_n": self._n,
                "check_every": self._check_every,
                "epsilon": self._epsilon,
                "min_n": self._min_n,
                "m_consecutive": self._m,
                "auto_stopped": self._auto_stopped,
                "overall_converged_at": overall,
                "tracked_questions": list(self._keys),
                "per_question": per_question,
                "human_baseline": baseline,
            }
            return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_check_locked(self) -> ConvergenceCheck:
        per_q: dict[str, dict[str, float]] = {}
        for key, state in self._states.items():
            if sum(state.last_batch.values()) == 0:
                jsd = 0.0
            else:
                # Compare the post-batch cumulative distribution against
                # just the last K observations. ``cumulative`` contains
                # the batch itself, which is the definition from the
                # acceptance criteria.
                p = {k: float(v) for k, v in state.cumulative.items()}
                q = {k: float(v) for k, v in state.last_batch.items()}
                jsd = jensen_shannon_divergence(p, q)
            state.per_n_jsd.append((self._n, jsd))
            state.rolling_jsd.append(jsd)
            rolling = _rolling_average(state.rolling_jsd, self._m)
            # First n at which the rolling mean crosses below epsilon and
            # stays there for M consecutive checks is our converged_at.
            if (
                state.converged_at is None
                and len(state.rolling_jsd) >= self._m
                and all(j < self._epsilon for j in state.rolling_jsd[-self._m :])
                and self._n >= self._min_n
            ):
                state.converged_at = self._n
            per_q[key] = {
                "jsd_last_batch": round(jsd, 6),
                "jsd_rolling_avg_3": round(rolling, 6),
            }
        check = ConvergenceCheck(n_completed=self._n, per_question=per_q)
        self._checks.append(check)
        return check

    def _is_converged_locked(self) -> bool:
        # All tracked questions must have a converged_at by now.
        if not self._states:
            return False
        return all(state.converged_at is not None for state in self._states.values())

    def _overall_converged_at_locked(self) -> int | None:
        values = [s.converged_at for s in self._states.values() if s.converged_at is not None]
        if not values:
            return None
        if len(values) != len(self._states):
            # Not every question converged — don't report an overall value.
            return None
        return max(values)

    def _emit_log(self, check: ConvergenceCheck) -> None:
        payload = {
            "n_completed": check.n_completed,
            "per_question": check.per_question,
        }
        line = json.dumps(payload, sort_keys=True, default=str)
        if self._log_fh is not None:
            try:
                self._log_fh.write(line + "\n")
            except OSError as exc:
                logger.warning("convergence log write failed: %s", exc)
        else:
            print(line, file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_average(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    tail = values[-window:]
    return sum(tail) / len(tail)


def _downsample_curve(points: list[tuple[int, float]], target: int) -> list[tuple[int, float]]:
    """Return at most ``target`` evenly-spaced samples, keeping endpoints."""
    if target <= 0 or not points:
        return []
    if len(points) <= target:
        return list(points)
    step = (len(points) - 1) / (target - 1)
    selected: list[tuple[int, float]] = []
    seen: set[int] = set()
    for i in range(target):
        idx = round(i * step)
        idx = max(0, min(len(points) - 1, idx))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(points[idx])
    return selected


# ---------------------------------------------------------------------------
# Synthbench baseline (soft dependency)
# ---------------------------------------------------------------------------


class SynthbenchUnavailableError(RuntimeError):
    """Raised when --convergence-baseline is requested but synthbench is missing."""


def load_synthbench_baseline(spec: str) -> dict[str, Any]:
    """Resolve a ``dataset:question_key`` baseline against synthbench.

    The bead lists synthbench as a *soft* dependency — we install it via
    the ``synthpanel[convergence]`` extras and import lazily so the core
    path never breaks when users haven't opted in. When the dependency
    is missing we raise :class:`SynthbenchUnavailableError` with an
    actionable install hint so the CLI can surface it verbatim.

    ``spec`` accepts either ``"dataset:question_key"`` (preferred) or
    ``"dataset"`` when the dataset exposes a single default question.
    """
    try:
        import synthbench  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via patched import
        raise SynthbenchUnavailableError(
            "synthbench is required for --convergence-baseline. Install it with: pip install 'synthpanel[convergence]'"
        ) from exc

    dataset, _, question_key = spec.partition(":")
    dataset = dataset.strip()
    question_key = question_key.strip() or None
    if not dataset:
        raise ValueError(f"invalid baseline spec: {spec!r}")

    # Try the documented interface first; fall back to any attribute
    # synthbench chooses to expose so sb-ygp7 can rename without
    # breaking this call site.
    loader = None
    for candidate in ("load_convergence_baseline", "load_baseline", "convergence_baseline"):
        loader = getattr(synthbench, candidate, None)
        if loader is None and hasattr(synthbench, "convergence"):
            loader = getattr(synthbench.convergence, candidate, None)
        if loader is not None:
            break
    if loader is None:
        raise SynthbenchUnavailableError(
            "synthbench is installed but does not expose a convergence baseline loader. "
            "Upgrade synthbench to a version that ships sb-ygp7 (convergence bootstrap)."
        )

    try:
        data = loader(dataset=dataset, question_key=question_key) if question_key else loader(dataset=dataset)
    except TypeError:
        # Older loader signatures may accept positional args only.
        data = loader(dataset, question_key) if question_key else loader(dataset)

    if not isinstance(data, dict):
        raise SynthbenchUnavailableError(f"synthbench baseline loader returned {type(data).__name__}, expected dict")
    result = dict(data)
    result.setdefault("dataset", dataset)
    if question_key is not None:
        result.setdefault("question_key", question_key)
    return result


def _looks_numeric(value: Any) -> bool:
    """True when *value* is an int/float or a string that coerces to one.

    Booleans are excluded — Python treats them as ints, but ``True``/``False``
    are semantically enum-like for our purposes.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def derive_pick_one_schema_from_baseline(
    baseline_payload: dict[str, Any],
    *,
    max_options: int = 5,
) -> dict[str, Any] | None:
    """Derive a pick_one extraction schema from a SynthBench baseline.

    Reads ``baseline_payload['human_distribution']`` and, when its keys look
    like a small enumerated option set, returns a deep-copied
    :data:`PICK_ONE_SCHEMA` with ``properties.choice.enum`` pinned to the
    sorted baseline keys (verbatim — no normalization, per S-gate OQ1).

    Returns ``None`` when the distribution is missing or empty, when it has
    more than ``max_options`` keys, or when any key is numeric-coercible
    (a Likert-style scale rather than a free enum). The caller is expected
    to fall back to the author-declared schema in those cases.

    Pure: no I/O, no LLM calls.
    """
    if not isinstance(baseline_payload, dict):
        return None
    distribution = baseline_payload.get("human_distribution")
    if not isinstance(distribution, dict) or not distribution:
        return None

    keys = list(distribution.keys())
    if len(keys) > max_options:
        return None
    if any(_looks_numeric(k) for k in keys):
        return None

    schema = copy.deepcopy(PICK_ONE_SCHEMA)
    schema["properties"]["choice"]["enum"] = sorted(keys)
    return schema
