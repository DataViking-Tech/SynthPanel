"""Per-question failure budget for short-circuiting bad questions (sp-xw2z6o).

A single bad question — model hates the prompt format, schema has a typo —
can poison an entire panel by failing for every panelist while burning
tokens on retries. ``QuestionFailureBudget`` tracks per-question failure
counts and marks a question disabled once a threshold is crossed; the
orchestrator then skips that question for subsequent panelists instead of
re-failing it.

The budget is provided as either:

* an integer count (>= 1) — ``2`` means "disable after 2 failures", OR
* a fraction (0 < x < 1) — ``0.25`` means "disable when >= 25% of
  panelists have failed this question".

Both forms compare against ``failure_count`` and ``total_panelists`` at
record time, so the trigger fires the moment the threshold is crossed
mid-run rather than waiting for all panelists to finish.

Designed to mirror :class:`synth_panel.cost.CostGate`: thread-safe (the
orchestrator runs panelists in parallel), idempotent (a disabled question
stays disabled), and cheap to call from per-question completion paths.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)


class QuestionFailureBudget:
    """Per-question failure budget with mid-run disable + skip semantics."""

    def __init__(self, budget: int | float, total_panelists: int) -> None:
        if total_panelists <= 0:
            raise ValueError(f"total_panelists must be > 0, got {total_panelists!r}")

        if isinstance(budget, bool):
            raise TypeError("budget must be a number, not bool")

        if isinstance(budget, int):
            if budget < 1:
                raise ValueError(f"integer budget must be >= 1, got {budget!r}")
            self._count_threshold = int(budget)
            self._fraction_threshold: float | None = None
        elif isinstance(budget, float):
            if not math.isfinite(budget) or budget <= 0 or budget >= 1:
                raise ValueError(
                    f"fractional budget must be in (0, 1), got {budget!r}; pass an int >= 1 for an absolute count"
                )
            self._fraction_threshold = float(budget)
            self._count_threshold = max(1, math.ceil(self._fraction_threshold * total_panelists))
        else:
            raise TypeError(f"budget must be int or float, got {type(budget).__name__}")

        self.total_panelists = int(total_panelists)
        self._lock = threading.Lock()
        self._failures: dict[int, int] = {}
        self._disabled: dict[int, dict[str, Any]] = {}

    @property
    def threshold_count(self) -> int:
        """Effective failure-count threshold (resolved from int or float input)."""
        return self._count_threshold

    @property
    def fraction(self) -> float | None:
        """Original fractional threshold, or ``None`` if budget is an absolute count."""
        return self._fraction_threshold

    def record_failure(
        self,
        question_index: int,
        *,
        question_text: str | None = None,
    ) -> bool:
        """Record one failure for *question_index*; return True if now disabled."""
        with self._lock:
            if question_index in self._disabled:
                return True
            count = self._failures.get(question_index, 0) + 1
            self._failures[question_index] = count
            if count >= self._count_threshold:
                self._disabled[question_index] = {
                    "question_index": question_index,
                    "question_text": question_text,
                    "failures_at_disable": count,
                    "threshold_count": self._count_threshold,
                    "threshold_fraction": self._fraction_threshold,
                    "total_panelists": self.total_panelists,
                }
                logger.warning(
                    "question budget tripped: Q%d disabled after %d/%d failures (threshold=%s)",
                    question_index,
                    count,
                    self.total_panelists,
                    self._format_threshold(),
                )
                return True
            return False

    def is_disabled(self, question_index: int) -> bool:
        with self._lock:
            return question_index in self._disabled

    def disabled_questions(self) -> list[int]:
        """Return sorted list of disabled question indices."""
        with self._lock:
            return sorted(self._disabled)

    def disabled_details(self) -> list[dict[str, Any]]:
        """Return per-disabled-question diagnostic dicts (sorted by index)."""
        with self._lock:
            return [dict(self._disabled[qi]) for qi in sorted(self._disabled)]

    def failure_counts(self) -> dict[int, int]:
        """Snapshot of question_index -> failure count."""
        with self._lock:
            return dict(self._failures)

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable status for structured output."""
        with self._lock:
            return {
                "threshold_count": self._count_threshold,
                "threshold_fraction": self._fraction_threshold,
                "total_panelists": self.total_panelists,
                "disabled_count": len(self._disabled),
                "disabled": [dict(self._disabled[qi]) for qi in sorted(self._disabled)],
                "failure_counts": dict(self._failures),
            }

    def _format_threshold(self) -> str:
        if self._fraction_threshold is not None:
            return f"{self._fraction_threshold:.2%} (>= {self._count_threshold})"
        return str(self._count_threshold)
