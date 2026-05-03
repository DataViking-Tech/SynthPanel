"""Transcript stamping tests for AC-7 of the v1.0.0 frozen contract.

Every row in a panel-run JSONL transcript must carry the
``decision_being_informed`` field, so any single line is self-describing
and can be tied back to the decision the panel was informing.
"""

from __future__ import annotations

import json

import pytest

from synth_panel.cost import TokenUsage
from synth_panel.persistence import (
    ConversationMessage,
    Session,
    append_message,
    load_session,
    save_session,
)

_DECISION = "Should we ship the new pricing tier next quarter?"


def _text_msg(role: str, text: str, usage: TokenUsage | None = None) -> ConversationMessage:
    return ConversationMessage(
        role=role,
        content=[{"type": "text", "text": text}],
        usage=usage,
    )


def test_every_row_carries_decision(tmp_path):
    """Canonical AC-7 test: every JSONL row is stamped with the decision."""
    s = Session(decision_being_informed=_DECISION)
    s.push_message(_text_msg("user", "What do you think about price?"))
    s.push_message(_text_msg("assistant", "It feels high.", usage=TokenUsage(10, 20, 0, 0)))
    s.push_message(_text_msg("user", "Why?"))
    s.push_message(_text_msg("assistant", "Compared to peers."))
    # Force a compaction record so all three row-types are exercised.
    s.compact("Earlier turns covered price perception.", keep_last=2)

    path = tmp_path / "panel.jsonl"
    save_session(s, path, fmt="jsonl")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "JSONL file should not be empty"

    seen_types = set()
    for i, line in enumerate(lines, start=1):
        record = json.loads(line)
        seen_types.add(record.get("type"))
        assert record.get("decision_being_informed") == _DECISION, (
            f"row {i} (type={record.get('type')!r}) missing or wrong decision_being_informed: {record!r}"
        )

    # All three row types must exist and all must be stamped.
    assert {"session_meta", "message", "compaction"} <= seen_types


def test_append_message_stamps_decision(tmp_path):
    """append_message stamps the row when given decision_being_informed."""
    path = tmp_path / "live.jsonl"
    append_message(
        path,
        _text_msg("user", "first turn"),
        decision_being_informed=_DECISION,
    )
    append_message(
        path,
        _text_msg("assistant", "reply"),
        decision_being_informed=_DECISION,
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["type"] == "message"
        assert record["decision_being_informed"] == _DECISION


def test_no_decision_means_no_stamp(tmp_path):
    """Sessions without a decision (e.g. run_prompt transcripts) emit no stamp.

    decision_being_informed is explicitly NOT used on run_prompt (per the
    v1.0.0 contract in schemas/v1.0.0.json), so its absence must be a clean
    omission rather than an empty-string sentinel.
    """
    s = Session()
    s.push_message(_text_msg("user", "hi"))
    path = tmp_path / "prompt.jsonl"
    save_session(s, path, fmt="jsonl")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        record = json.loads(line)
        assert "decision_being_informed" not in record, (
            f"decision_being_informed must be omitted, not nulled: {record!r}"
        )


def test_decision_round_trips_through_jsonl(tmp_path):
    """Loading a stamped JSONL transcript restores Session.decision_being_informed."""
    s = Session(decision_being_informed=_DECISION)
    s.push_message(_text_msg("user", "q"))
    path = tmp_path / "rt.jsonl"
    save_session(s, path, fmt="jsonl")

    loaded = load_session(path)
    assert loaded.decision_being_informed == _DECISION


def test_decision_round_trips_through_json(tmp_path):
    """Full-JSON format also carries the decision so format choice is lossless."""
    s = Session(decision_being_informed=_DECISION)
    s.push_message(_text_msg("user", "q"))
    path = tmp_path / "rt.json"
    save_session(s, path, fmt="json")

    loaded = load_session(path)
    assert loaded.decision_being_informed == _DECISION


def test_legacy_jsonl_without_stamp_still_loads():
    """Older transcripts (pre-AC-7) parse cleanly with decision = None."""
    legacy = (
        '{"type":"session_meta","session_id":"x","created_at":"t","updated_at":"t"}\n'
        '{"type":"message","role":"user","content":[{"type":"text","text":"hi"}]}\n'
    )
    s = Session.from_jsonl(legacy)
    assert s.session_id == "x"
    assert s.decision_being_informed is None


def test_decision_with_newline_is_rejected_at_persistence_boundary():
    """Persistence refuses to stamp a value that would break JSONL framing.

    A newline in the field would split a single record across multiple lines
    and silently corrupt every downstream reader. The validator (AC-2) is the
    contract authority, but persistence must still be self-protective at its
    boundary so a misuse can't produce malformed transcripts.
    """
    with pytest.raises(ValueError, match="newline"):
        Session(decision_being_informed="bad\nvalue").to_jsonl()
