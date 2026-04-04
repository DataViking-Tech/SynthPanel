"""Session persistence for synth-panel.

Implements SPEC.md Section 6: save/load in JSON and JSONL formats,
incremental append, session forking, atomic writes, and file rotation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from synth_panel.cost import ZERO_USAGE, TokenUsage


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConversationMessage:
    """A single message in a session conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: list[dict[str, Any]]  # Content blocks (text, tool_use, etc.)
    usage: Optional[TokenUsage] = None  # Present on assistant messages

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.usage is not None:
            d["usage"] = self.usage.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ConversationMessage:
        usage = None
        if "usage" in data:
            usage = TokenUsage.from_dict(data["usage"])
        return cls(role=data["role"], content=data["content"], usage=usage)


@dataclass
class CompactionMeta:
    """Metadata about conversation compaction."""

    compaction_count: int = 0
    messages_removed: int = 0
    summary_text: str = ""

    def to_dict(self) -> dict:
        return {
            "compaction_count": self.compaction_count,
            "messages_removed": self.messages_removed,
            "summary_text": self.summary_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompactionMeta:
        return cls(
            compaction_count=data.get("compaction_count", 0),
            messages_removed=data.get("messages_removed", 0),
            summary_text=data.get("summary_text", ""),
        )


@dataclass
class ForkMeta:
    """Metadata about a forked session."""

    parent_session_id: str = ""
    branch_name: str = ""

    def to_dict(self) -> dict:
        return {
            "parent_session_id": self.parent_session_id,
            "branch_name": self.branch_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForkMeta:
        return cls(
            parent_session_id=data.get("parent_session_id", ""),
            branch_name=data.get("branch_name", ""),
        )


@dataclass
class Session:
    """Full state of one agent's conversation."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    messages: list[ConversationMessage] = field(default_factory=list)
    compaction: Optional[CompactionMeta] = None
    fork: Optional[ForkMeta] = None

    # --- Mutation helpers --------------------------------------------------

    def push_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def compact(self, summary: str, keep_last: int = 2) -> None:
        """Replace older messages with *summary*, keeping the last *keep_last*."""
        if len(self.messages) <= keep_last:
            return
        removed = len(self.messages) - keep_last
        kept = self.messages[-keep_last:]
        summary_msg = ConversationMessage(
            role="system",
            content=[{"type": "text", "text": summary}],
        )
        self.messages = [summary_msg] + kept
        if self.compaction is None:
            self.compaction = CompactionMeta()
        self.compaction.compaction_count += 1
        self.compaction.messages_removed += removed
        self.compaction.summary_text = summary
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def fork_session(self, branch_name: str = "") -> Session:
        """Create a new session inheriting all messages from this one."""
        new = Session(
            messages=[
                ConversationMessage(
                    role=m.role, content=list(m.content), usage=m.usage
                )
                for m in self.messages
            ],
            fork=ForkMeta(
                parent_session_id=self.session_id,
                branch_name=branch_name,
            ),
        )
        return new

    # --- Token usage helpers ----------------------------------------------

    def iter_usages(self) -> list[TokenUsage]:
        """Extract per-turn token usages from assistant messages."""
        return [
            m.usage
            for m in self.messages
            if m.role == "assistant" and m.usage is not None
        ]

    # --- Serialisation: full JSON -----------------------------------------

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.compaction is not None:
            d["compaction"] = self.compaction.to_dict()
        if self.fork is not None:
            d["fork"] = self.fork.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        compaction = None
        if "compaction" in data:
            compaction = CompactionMeta.from_dict(data["compaction"])
        fork = None
        if "fork" in data:
            fork = ForkMeta.from_dict(data["fork"])
        return cls(
            session_id=data["session_id"],
            version=data.get("version", 1),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=[
                ConversationMessage.from_dict(m) for m in data["messages"]
            ],
            compaction=compaction,
            fork=fork,
        )

    # --- Serialisation: JSONL ---------------------------------------------

    def to_jsonl(self) -> str:
        """Serialise as newline-delimited JSON records."""
        lines: list[str] = []
        meta: dict[str, Any] = {
            "type": "session_meta",
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.fork is not None:
            meta["fork"] = self.fork.to_dict()
        lines.append(json.dumps(meta, separators=(",", ":")))

        for msg in self.messages:
            record = {"type": "message", **msg.to_dict()}
            lines.append(json.dumps(record, separators=(",", ":")))

        if self.compaction is not None:
            record = {"type": "compaction", **self.compaction.to_dict()}
            lines.append(json.dumps(record, separators=(",", ":")))

        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> Session:
        """Deserialise from newline-delimited JSON records."""
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            raise SessionFormatError("Empty JSONL input")

        session_id = ""
        version = 1
        created_at = ""
        updated_at = ""
        messages: list[ConversationMessage] = []
        compaction: Optional[CompactionMeta] = None
        fork: Optional[ForkMeta] = None

        for i, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionFormatError(
                    f"Malformed JSON at line {i}: {exc}"
                ) from exc

            rtype = record.get("type")
            if rtype == "session_meta":
                session_id = record["session_id"]
                version = record.get("version", 1)
                created_at = record["created_at"]
                updated_at = record["updated_at"]
                if "fork" in record:
                    fork = ForkMeta.from_dict(record["fork"])
            elif rtype == "message":
                messages.append(ConversationMessage.from_dict(record))
            elif rtype == "compaction":
                compaction = CompactionMeta.from_dict(record)
            else:
                raise SessionFormatError(
                    f"Unknown record type {rtype!r} at line {i}"
                )

        if not session_id:
            raise SessionFormatError("Missing session_meta record")

        return cls(
            session_id=session_id,
            version=version,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
            compaction=compaction,
            fork=fork,
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SessionError(Exception):
    """Base error for session persistence operations."""


class SessionFormatError(SessionError):
    """Raised when a persisted session has an invalid format."""


class SessionIOError(SessionError):
    """Raised on I/O failures during save/load."""


# ---------------------------------------------------------------------------
# Atomic file writing
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: str) -> None:
    """Write *data* to *path* atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".sp-"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        shutil.move(tmp, str(path))
    except Exception:
        # Clean up temp file on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# File rotation
# ---------------------------------------------------------------------------

def _rotate(
    path: Path,
    max_rotations: int = 3,
) -> None:
    """Rotate *path* to ``path.1``, shifting existing rotations up."""
    # Delete oldest if at max.
    oldest = path.with_suffix(f"{path.suffix}.{max_rotations}")
    if oldest.exists():
        oldest.unlink()

    # Shift existing rotations.
    for i in range(max_rotations - 1, 0, -1):
        src = path.with_suffix(f"{path.suffix}.{i}")
        dst = path.with_suffix(f"{path.suffix}.{i + 1}")
        if src.exists():
            src.rename(dst)

    # Move current to .1
    if path.exists():
        path.rename(path.with_suffix(f"{path.suffix}.1"))


# ---------------------------------------------------------------------------
# Public persistence API
# ---------------------------------------------------------------------------

DEFAULT_ROTATION_SIZE = 256 * 1024  # 256 KB
DEFAULT_MAX_ROTATIONS = 3


def save_session(
    session: Session,
    path: str | Path,
    *,
    fmt: str = "json",
    rotation_size: int = DEFAULT_ROTATION_SIZE,
    max_rotations: int = DEFAULT_MAX_ROTATIONS,
) -> None:
    """Serialise *session* to *path*.

    Parameters
    ----------
    fmt:
        ``"json"`` for a single JSON object, ``"jsonl"`` for newline-delimited.
    rotation_size:
        Rotate the existing file if it exceeds this byte threshold.
    max_rotations:
        Maximum number of rotated copies to keep.
    """
    p = Path(path)
    try:
        # Rotate if existing file exceeds threshold.
        if p.exists() and p.stat().st_size > rotation_size:
            _rotate(p, max_rotations)

        if fmt == "jsonl":
            _atomic_write(p, session.to_jsonl())
        else:
            _atomic_write(p, json.dumps(session.to_dict(), indent=2) + "\n")
    except OSError as exc:
        raise SessionIOError(f"Failed to save session to {path}: {exc}") from exc


def load_session(path: str | Path) -> Session:
    """Load a session from *path*, auto-detecting JSON vs JSONL format."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SessionIOError(
            f"Failed to load session from {path}: {exc}"
        ) from exc

    if not text.strip():
        raise SessionFormatError(f"Empty session file: {path}")

    # Try JSON first, then JSONL.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return Session.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        pass

    return Session.from_jsonl(text)


def append_message(
    path: str | Path,
    message: ConversationMessage,
) -> None:
    """Append a single message record to a JSONL session file."""
    p = Path(path)
    record = {"type": "message", **message.to_dict()}
    line = json.dumps(record, separators=(",", ":")) + "\n"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        raise SessionIOError(
            f"Failed to append message to {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Session store (simplified variant per SPEC.md §6)
# ---------------------------------------------------------------------------

class SessionStore:
    """Lightweight session store backed by a directory of JSON files.

    Each session is a single JSON file: ``<directory>/<session_id>.json``
    containing the session ID, message texts, and cumulative token counts.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, session: Session) -> None:
        data = {
            "session_id": session.session_id,
            "messages": [
                _extract_text(m) for m in session.messages
            ],
            "input_tokens": sum(
                (u.input_tokens for u in session.iter_usages()), 0
            ),
            "output_tokens": sum(
                (u.output_tokens for u in session.iter_usages()), 0
            ),
        }
        _atomic_write(
            self._path_for(session.session_id),
            json.dumps(data, indent=2) + "\n",
        )

    def load(self, session_id: str) -> dict:
        p = self._path_for(session_id)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionIOError(
                f"Session {session_id} not found in store: {exc}"
            ) from exc
        return json.loads(text)

    def list_sessions(self) -> list[str]:
        return sorted(
            p.stem for p in self.directory.glob("*.json")
        )


def _extract_text(message: ConversationMessage) -> str:
    """Extract plain text from a message's content blocks."""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
