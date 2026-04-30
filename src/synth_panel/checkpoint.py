"""Panelist-level checkpointing for scaled panel runs (sp-hsk3).

Slice (b) of sp-i2ub. Persists run state every K completed panelists so a
crashed, SIGINT'd, or resource-exhausted run can resume without reprocessing
panelists that already succeeded. A companion resume loader reconstitutes
config + completed results so the orchestrator only picks up the remaining
personas.

Layout
------
    <root>/<run-id>/state.json

The checkpoint is a single JSON document written atomically via temp-file
rename. Each write fully rewrites the file; there is no append-log. This is
fine for the target scale (10k panelists at ~KB each = ~10 MB, written
every 25 panelists = ~400 rewrites).

Resume flow
-----------
Callers load a checkpoint, compare its config fingerprint to the current
invocation's config (rejecting drift), and feed ``completed`` back into
the output pipeline while re-running only ``remaining`` personas.

Signal handling
---------------
``CheckpointWriter.install_signal_handlers`` traps SIGINT and SIGTERM to
flush the latest state before the previously-installed handler fires. The
trap is idempotent and always restores the prior handler in
``remove_signal_handlers``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from synth_panel.cost import coerce_provider_reported_cost

logger = logging.getLogger(__name__)


DEFAULT_CHECKPOINT_EVERY = 25
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base class for checkpoint-related errors."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a resume target doesn't exist on disk."""


class CheckpointFormatError(CheckpointError):
    """Raised when a checkpoint file is missing required fields or malformed."""


class CheckpointDriftError(CheckpointError):
    """Raised when the resume config doesn't match the original run's config.

    Resuming under a changed instrument/persona/model mix would silently
    produce a mixed-config dataset — not what "resume" promises. Refuse
    rather than paper over it.
    """


# ---------------------------------------------------------------------------
# Paths and run ids
# ---------------------------------------------------------------------------


def default_checkpoint_root() -> Path:
    """Default root: ``$SYNTHPANEL_CHECKPOINT_ROOT`` or ``~/.synthpanel/checkpoints``."""
    env = os.environ.get("SYNTHPANEL_CHECKPOINT_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".synthpanel" / "checkpoints"


def new_run_id() -> str:
    """Build a fresh UTC-timestamped run id with a short suffix for uniqueness."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"run-{stamp}-{suffix}"


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id):
        raise CheckpointError(f"invalid run id {run_id!r}: only alphanumerics, dot, underscore, dash (max 128 chars)")


def checkpoint_dir_for(run_id: str, root: Path | None = None) -> Path:
    """Return the directory holding checkpoint state for ``run_id``."""
    _validate_run_id(run_id)
    r = Path(root) if root is not None else default_checkpoint_root()
    return r / run_id


def _state_path(directory: Path) -> Path:
    return Path(directory) / "state.json"


# ---------------------------------------------------------------------------
# Config fingerprint
# ---------------------------------------------------------------------------


def fingerprint_config(config: dict[str, Any]) -> str:
    """Deterministic fingerprint so resume can reject config drift."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint data model
# ---------------------------------------------------------------------------


@dataclass
class PanelCheckpoint:
    """Snapshot of a panel run sufficient to resume without reprocessing."""

    run_id: str
    created_at: str
    updated_at: str
    config_fingerprint: str
    config: dict[str, Any]
    # One dict per completed panelist. Shape mirrors the CLI's result dict
    # so the caller can feed ``completed`` straight back into the output
    # pipeline without re-deriving anything.
    completed: list[dict[str, Any]] = field(default_factory=list)
    # Persona names still to run, in authored order.
    remaining: list[str] = field(default_factory=list)
    # Accumulated usage (TokenUsage.to_dict() shape) across completed panelists.
    usage: dict[str, Any] = field(default_factory=dict)
    # Populated on signal-triggered flush so consumers can distinguish a
    # mid-run snapshot from a voluntary abort.
    abort_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            "completed": self.completed,
            "remaining": self.remaining,
            "usage": self.usage,
            "abort_reason": self.abort_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelCheckpoint:
        for key in (
            "run_id",
            "created_at",
            "updated_at",
            "config_fingerprint",
            "config",
        ):
            if key not in data:
                raise CheckpointFormatError(f"checkpoint missing required field: {key!r}")
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            config_fingerprint=data["config_fingerprint"],
            config=data["config"],
            completed=list(data.get("completed", [])),
            remaining=list(data.get("remaining", [])),
            usage=dict(data.get("usage", {})),
            abort_reason=data.get("abort_reason"),
        )


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".ckpt-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), default=str)
        shutil.move(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_checkpoint(checkpoint: PanelCheckpoint, directory: Path | str) -> None:
    """Write ``checkpoint`` to ``<directory>/state.json`` atomically."""
    d = Path(directory)
    checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(_state_path(d), checkpoint.to_dict())


def load_checkpoint(run_id: str, root: Path | None = None) -> PanelCheckpoint:
    """Load the checkpoint for ``run_id`` from ``root`` (default ~/.synthpanel/checkpoints)."""
    directory = checkpoint_dir_for(run_id, root)
    path = _state_path(directory)
    if not path.exists():
        raise CheckpointNotFoundError(f"no checkpoint at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(f"checkpoint at {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointFormatError(f"checkpoint at {path} is not a JSON object")
    return PanelCheckpoint.from_dict(data)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class CheckpointWriter:
    """Thread-safe writer that snapshots run state every K completed panelists.

    Usage pattern::

        writer = CheckpointWriter(
            run_id=run_id,
            directory=checkpoint_dir_for(run_id),
            config=config_dict,
            all_personas=[p["name"] for p in personas],
            every=25,
        )
        writer.install_signal_handlers()
        try:
            for result in run_panel(...):
                writer.record_completed(result_dict, usage_increment)
        finally:
            writer.flush()
            writer.remove_signal_handlers()

    The writer is resume-aware: pre-populated with ``preloaded_completed``
    and ``preloaded_usage`` on resume, new completions extend those. The
    resulting checkpoint always contains the full set of finished panelists,
    not just the current run's slice.
    """

    def __init__(
        self,
        *,
        run_id: str,
        directory: Path | str,
        config: dict[str, Any],
        all_personas: list[str],
        every: int = DEFAULT_CHECKPOINT_EVERY,
        preloaded_completed: list[dict[str, Any]] | None = None,
        preloaded_usage: dict[str, Any] | None = None,
    ) -> None:
        _validate_run_id(run_id)
        if every < 1:
            raise ValueError(f"checkpoint every must be >= 1 (got {every})")
        self.run_id = run_id
        self.directory = Path(directory)
        self.config = dict(config)
        self.config_fingerprint = fingerprint_config(self.config)
        self.all_personas = list(all_personas)
        self.every = every
        self.completed: list[dict[str, Any]] = list(preloaded_completed or [])
        self._completed_names: set[str] = {_persona_name_of(r) for r in self.completed if _persona_name_of(r)}
        self.usage: dict[str, Any] = dict(preloaded_usage or {})
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._lock = threading.Lock()
        self._since_flush = 0
        self._abort_reason: str | None = None
        self._prev_sigint: Any = None
        self._prev_sigterm: Any = None
        self._handlers_installed = False

    # -- Signal handling -----------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Trap SIGINT/SIGTERM to flush the latest state before propagating.

        Idempotent. Only installs handlers on the main thread; on other
        threads ``signal.signal`` raises ValueError — we log and skip.
        """
        if self._handlers_installed:
            return
        try:
            self._prev_sigint = signal.signal(signal.SIGINT, self._signal_handler)
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
            self._handlers_installed = True
        except ValueError:
            # Not on the main thread — caller will get uninstrumented signals.
            # We still work correctly, just lose SIGINT flush. This happens
            # in pytest when fixtures run checkpointing from worker threads.
            logger.debug(
                "checkpoint: signal handlers not installable (not on main thread); relying on explicit flush()",
            )

    def remove_signal_handlers(self) -> None:
        if not self._handlers_installed:
            return
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, self._prev_sigint)
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        self._handlers_installed = False

    def _signal_handler(self, signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name if signum in (signal.SIGINT, signal.SIGTERM) else str(signum)
        logger.warning("checkpoint: %s received — flushing state to %s", name, self.directory)
        self._abort_reason = f"signal:{name}"
        try:
            self.flush(force=True)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.error("checkpoint: flush during %s failed: %s", name, exc)
        # Re-raise via the previously-installed handler so callers still see
        # KeyboardInterrupt / terminate. If the previous handler is not a
        # callable (SIG_DFL / SIG_IGN), restore it and re-send.
        prev = self._prev_sigint if signum == signal.SIGINT else self._prev_sigterm
        if callable(prev):
            prev(signum, frame)
        else:
            with contextlib.suppress(ValueError):
                signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    # -- Recording / flushing -----------------------------------------------

    def record_completed(
        self,
        result: dict[str, Any],
        usage_increment: dict[str, Any] | None = None,
    ) -> None:
        """Record one finished panelist and flush if the cadence fires.

        ``result`` must be the CLI-shaped dict (``persona``, ``responses``,
        ``usage``, ``cost``, ``error``, optional ``model``). ``usage_increment``
        is the delta to add to the running ``usage`` rollup — usually
        ``result['usage']`` but callers may supply a pre-summed variant.
        """
        name = _persona_name_of(result)
        with self._lock:
            if name and name in self._completed_names:
                # Idempotent on duplicate records — orchestrator shouldn't
                # emit them but a resumed run could race with a stale write.
                return
            self.completed.append(result)
            if name:
                self._completed_names.add(name)
            if usage_increment:
                self.usage = _merge_usage(self.usage, usage_increment)
            self._since_flush += 1
            should_flush = self._since_flush >= self.every

        if should_flush:
            self.flush()

    def mark_aborted(self, reason: str) -> None:
        """Tag the next (and final) flush with a human-readable abort reason."""
        with self._lock:
            self._abort_reason = reason

    def flush(self, force: bool = False) -> None:
        """Persist the current state. No-op if nothing has been recorded yet.

        ``force=True`` writes even when the cadence counter is zero — used
        by the signal handler and the orchestrator's final flush.
        """
        with self._lock:
            if not force and self._since_flush == 0:
                return
            ckpt = self._build_checkpoint_locked()
            self._since_flush = 0
        save_checkpoint(ckpt, self.directory)

    def build_checkpoint(self) -> PanelCheckpoint:
        """Return the current checkpoint without writing."""
        with self._lock:
            return self._build_checkpoint_locked()

    def _build_checkpoint_locked(self) -> PanelCheckpoint:
        return PanelCheckpoint(
            run_id=self.run_id,
            created_at=self._created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            config_fingerprint=self.config_fingerprint,
            config=self.config,
            completed=list(self.completed),
            remaining=[n for n in self.all_personas if n not in self._completed_names],
            usage=dict(self.usage),
            abort_reason=self._abort_reason,
        )

    # -- Accessors ----------------------------------------------------------

    def remaining(self) -> list[str]:
        with self._lock:
            return [n for n in self.all_personas if n not in self._completed_names]

    def completed_names(self) -> set[str]:
        with self._lock:
            return set(self._completed_names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona_name_of(result: dict[str, Any]) -> str:
    value = result.get("persona") if isinstance(result, dict) else None
    return value if isinstance(value, str) else ""


_USAGE_INT_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_tokens",
    "cached_tokens",
)


def _merge_usage(
    current: dict[str, Any],
    increment: dict[str, Any],
) -> dict[str, Any]:
    """Add two TokenUsage.to_dict() payloads without losing provider cost."""
    merged = dict(current)
    for key in _USAGE_INT_KEYS:
        if key in increment:
            merged[key] = int(merged.get(key, 0)) + int(increment[key])
    if "provider_reported_cost" in increment:
        prev = merged.get("provider_reported_cost")
        add = increment["provider_reported_cost"]
        if prev is None and add is None:
            merged.pop("provider_reported_cost", None)
        else:
            p = coerce_provider_reported_cost(prev) if prev is not None else Decimal(0)
            a = coerce_provider_reported_cost(add) if add is not None else Decimal(0)
            merged["provider_reported_cost"] = float(p + a)
    return merged


def ensure_config_matches(
    checkpoint: PanelCheckpoint,
    current_config: dict[str, Any],
) -> None:
    """Raise :class:`CheckpointDriftError` if the fingerprints disagree."""
    current_fp = fingerprint_config(current_config)
    if current_fp != checkpoint.config_fingerprint:
        raise CheckpointDriftError(
            f"resume config drift: checkpoint fingerprint "
            f"{checkpoint.config_fingerprint[:12]}... does not match current "
            f"fingerprint {current_fp[:12]}... — rerun without --resume or "
            f"restore the original config to continue"
        )
