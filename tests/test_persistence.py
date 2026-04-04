"""Tests for synth_panel.persistence — SPEC.md Section 6."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pytest

from synth_panel.cost import TokenUsage
from synth_panel.persistence import (
    ConversationMessage,
    CompactionMeta,
    ForkMeta,
    Session,
    SessionFormatError,
    SessionIOError,
    SessionStore,
    append_message,
    load_session,
    save_session,
)


# --- Helpers ---------------------------------------------------------------

def _text_msg(role: str, text: str, usage: Optional[TokenUsage] = None):
    return ConversationMessage(
        role=role,
        content=[{"type": "text", "text": text}],
        usage=usage,
    )


# --- ConversationMessage ---------------------------------------------------


class TestConversationMessage:
    def test_roundtrip(self):
        m = _text_msg("user", "hello")
        assert ConversationMessage.from_dict(m.to_dict()).content == m.content

    def test_with_usage(self):
        u = TokenUsage(10, 20, 0, 0)
        m = _text_msg("assistant", "reply", usage=u)
        d = m.to_dict()
        assert "usage" in d
        restored = ConversationMessage.from_dict(d)
        assert restored.usage == u

    def test_no_usage(self):
        m = _text_msg("user", "hi")
        d = m.to_dict()
        assert "usage" not in d


# --- Session basics -------------------------------------------------------


class TestSession:
    def test_push_message_updates_timestamp(self):
        s = Session()
        old_ts = s.updated_at
        s.push_message(_text_msg("user", "hello"))
        assert s.updated_at >= old_ts
        assert len(s.messages) == 1

    def test_fork(self):
        s = Session()
        s.push_message(_text_msg("user", "a"))
        s.push_message(_text_msg("assistant", "b"))
        child = s.fork_session("experiment-1")
        assert child.session_id != s.session_id
        assert child.fork is not None
        assert child.fork.parent_session_id == s.session_id
        assert child.fork.branch_name == "experiment-1"
        assert len(child.messages) == 2
        # Verify deep copy (mutating child doesn't affect parent).
        child.push_message(_text_msg("user", "c"))
        assert len(s.messages) == 2

    def test_compact(self):
        s = Session()
        for i in range(5):
            s.push_message(_text_msg("user", f"msg-{i}"))
        s.compact("Summary of messages 0-2", keep_last=2)
        assert len(s.messages) == 3  # summary + 2 kept
        assert s.messages[0].role == "system"
        assert s.compaction is not None
        assert s.compaction.compaction_count == 1
        assert s.compaction.messages_removed == 3

    def test_compact_noop_when_short(self):
        s = Session()
        s.push_message(_text_msg("user", "only one"))
        s.compact("irrelevant", keep_last=2)
        assert len(s.messages) == 1
        assert s.compaction is None

    def test_iter_usages(self):
        s = Session()
        s.push_message(_text_msg("user", "q"))
        s.push_message(
            _text_msg("assistant", "a", usage=TokenUsage(10, 20, 0, 0))
        )
        s.push_message(_text_msg("user", "q2"))
        s.push_message(
            _text_msg("assistant", "a2", usage=TokenUsage(5, 15, 0, 0))
        )
        usages = s.iter_usages()
        assert len(usages) == 2
        assert usages[0].input_tokens == 10


# --- JSON serialisation ---------------------------------------------------


class TestJsonSerialisation:
    def test_roundtrip(self, tmp_path):
        s = Session()
        s.push_message(_text_msg("user", "hello"))
        s.push_message(
            _text_msg("assistant", "hi", usage=TokenUsage(5, 10, 0, 0))
        )
        path = tmp_path / "session.json"
        save_session(s, path, fmt="json")
        loaded = load_session(path)
        assert loaded.session_id == s.session_id
        assert len(loaded.messages) == 2
        assert loaded.messages[1].usage == TokenUsage(5, 10, 0, 0)

    def test_with_compaction(self, tmp_path):
        s = Session()
        for i in range(5):
            s.push_message(_text_msg("user", f"msg-{i}"))
        s.compact("summary")
        path = tmp_path / "s.json"
        save_session(s, path)
        loaded = load_session(path)
        assert loaded.compaction is not None
        assert loaded.compaction.compaction_count == 1

    def test_with_fork(self, tmp_path):
        parent = Session()
        child = parent.fork_session("branch-x")
        path = tmp_path / "child.json"
        save_session(child, path)
        loaded = load_session(path)
        assert loaded.fork is not None
        assert loaded.fork.parent_session_id == parent.session_id


# --- JSONL serialisation --------------------------------------------------


class TestJsonlSerialisation:
    def test_roundtrip(self, tmp_path):
        s = Session()
        s.push_message(_text_msg("user", "q"))
        s.push_message(
            _text_msg("assistant", "a", usage=TokenUsage(1, 2, 3, 4))
        )
        path = tmp_path / "session.jsonl"
        save_session(s, path, fmt="jsonl")
        loaded = load_session(path)
        assert loaded.session_id == s.session_id
        assert len(loaded.messages) == 2

    def test_with_compaction(self, tmp_path):
        s = Session()
        for i in range(5):
            s.push_message(_text_msg("user", f"m{i}"))
        s.compact("summ")
        path = tmp_path / "s.jsonl"
        save_session(s, path, fmt="jsonl")
        loaded = load_session(path)
        assert loaded.compaction is not None

    def test_malformed_line(self):
        bad = '{"type":"session_meta","session_id":"x","created_at":"t","updated_at":"t"}\n{bad json\n'
        with pytest.raises(SessionFormatError, match="line 2"):
            Session.from_jsonl(bad)

    def test_empty_input(self):
        with pytest.raises(SessionFormatError, match="Empty"):
            Session.from_jsonl("")

    def test_missing_meta(self):
        line = json.dumps(
            {"type": "message", "role": "user", "content": []}
        )
        with pytest.raises(SessionFormatError, match="Missing session_meta"):
            Session.from_jsonl(line)

    def test_unknown_record_type(self):
        meta = json.dumps({
            "type": "session_meta",
            "session_id": "x",
            "created_at": "t",
            "updated_at": "t",
        })
        bad = json.dumps({"type": "alien"})
        with pytest.raises(SessionFormatError, match="Unknown record type"):
            Session.from_jsonl(meta + "\n" + bad)


# --- Auto-detection -------------------------------------------------------


class TestAutoDetection:
    def test_json_detected(self, tmp_path):
        s = Session()
        s.push_message(_text_msg("user", "x"))
        path = tmp_path / "s.json"
        save_session(s, path, fmt="json")
        loaded = load_session(path)
        assert loaded.session_id == s.session_id

    def test_jsonl_detected(self, tmp_path):
        s = Session()
        s.push_message(_text_msg("user", "x"))
        path = tmp_path / "s.jsonl"
        save_session(s, path, fmt="jsonl")
        loaded = load_session(path)
        assert loaded.session_id == s.session_id


# --- Append ---------------------------------------------------------------


class TestAppendMessage:
    def test_append_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "append.jsonl"
        msg = _text_msg("user", "hello")
        append_message(path, msg)
        assert path.exists()
        record = json.loads(path.read_text().strip())
        assert record["type"] == "message"
        assert record["role"] == "user"

    def test_append_adds_line(self, tmp_path):
        path = tmp_path / "multi.jsonl"
        append_message(path, _text_msg("user", "a"))
        append_message(path, _text_msg("assistant", "b"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2


# --- File rotation --------------------------------------------------------


class TestFileRotation:
    def test_rotation_on_large_file(self, tmp_path):
        path = tmp_path / "big.json"
        s = Session()
        s.push_message(_text_msg("user", "x" * 300))
        # Write a large initial file (> threshold).
        save_session(s, path, fmt="json", rotation_size=100)
        # Save again — should trigger rotation of the first.
        s2 = Session()
        s2.push_message(_text_msg("user", "y" * 300))
        save_session(s2, path, fmt="json", rotation_size=100)
        assert path.exists()
        rotated = path.with_suffix(".json.1")
        assert rotated.exists()
        # Verify the current file is s2.
        loaded = load_session(path)
        assert loaded.session_id == s2.session_id

    def test_max_rotations(self, tmp_path):
        path = tmp_path / "rot.json"
        sessions = []
        for i in range(5):
            s = Session()
            s.push_message(_text_msg("user", f"{'z' * 300}"))
            sessions.append(s)
            save_session(s, path, fmt="json", rotation_size=100, max_rotations=2)
        # Current + 2 rotated = 3 files max.
        assert path.exists()
        assert path.with_suffix(".json.1").exists()
        assert path.with_suffix(".json.2").exists()
        assert not path.with_suffix(".json.3").exists()


# --- Atomic write safety --------------------------------------------------


class TestAtomicWrite:
    def test_previous_survives_failure(self, tmp_path):
        path = tmp_path / "safe.json"
        s = Session()
        s.push_message(_text_msg("user", "original"))
        save_session(s, path)
        original_content = path.read_text()

        # Make the directory read-only to simulate write failure.
        # (Only works on POSIX where we can restrict dir perms.)
        if os.name == "posix":
            path.parent.chmod(0o444)
            try:
                s2 = Session()
                s2.push_message(_text_msg("user", "new"))
                with pytest.raises(Exception):
                    save_session(s2, path)
            finally:
                path.parent.chmod(0o755)
            assert path.read_text() == original_content


# --- SessionStore ---------------------------------------------------------


class TestSessionStore:
    def test_save_and_load(self, tmp_path):
        store = SessionStore(tmp_path / "store")
        s = Session()
        s.push_message(_text_msg("user", "hello"))
        s.push_message(
            _text_msg("assistant", "hi", usage=TokenUsage(10, 20, 0, 0))
        )
        store.save(s)
        data = store.load(s.session_id)
        assert data["session_id"] == s.session_id
        assert data["input_tokens"] == 10
        assert data["output_tokens"] == 20
        assert len(data["messages"]) == 2

    def test_list_sessions(self, tmp_path):
        store = SessionStore(tmp_path / "store")
        s1 = Session(session_id="aaa")
        s2 = Session(session_id="bbb")
        store.save(s1)
        store.save(s2)
        assert store.list_sessions() == ["aaa", "bbb"]

    def test_load_missing(self, tmp_path):
        store = SessionStore(tmp_path / "store")
        with pytest.raises(SessionIOError):
            store.load("nonexistent")


# --- Error paths ----------------------------------------------------------


class TestErrorPaths:
    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(SessionIOError):
            load_session(tmp_path / "nope.json")

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        with pytest.raises(SessionFormatError, match="Empty"):
            load_session(path)
