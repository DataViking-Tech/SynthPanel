"""Tests for :mod:`synth_panel.reporting.loader` (sp-viz-layer T2).

Covers the five core cases from structure.md §6 plus the path-vs-ID
disambiguation edge cases flagged in plan.md Risk #3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_panel.reporting.loader import ReportLoadError, load_panel_json

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def test_load_from_path_happy():
    """A real fixture JSON path returns a dict with the stem as ``id``."""
    path = FIXTURES / "rounds_shape.json"
    data = load_panel_json(str(path))
    assert isinstance(data, dict)
    assert data["id"] == "rounds_shape"
    assert "rounds" in data


def test_load_from_id_delegates_to_mcp_data(monkeypatch):
    """A bare ID that does not refer to a file delegates to MCP data."""
    sentinel = {"id": "result-xyz", "model": "claude-sonnet-4-6"}
    captured: dict[str, str] = {}

    def fake_get(result_id: str) -> dict:
        captured["id"] = result_id
        return sentinel

    monkeypatch.setattr("synth_panel.mcp.data.get_panel_result", fake_get)

    data = load_panel_json("result-xyz")
    assert data is sentinel
    assert captured["id"] == "result-xyz"


def test_load_missing_file_raises_not_found(tmp_path, monkeypatch):
    """A bare ID with no matching pack file surfaces as ``not_found``.

    Also covers plan.md Risk #3's 'path that does not exist but looks
    like an ID' ambiguity — the input falls through to MCP data because
    it has no ``.json`` suffix, and the empty data dir forces a miss.
    """
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    with pytest.raises(ReportLoadError) as exc_info:
        load_panel_json("missing-result-id")
    assert exc_info.value.code == "not_found"


def test_load_invalid_json_raises_invalid_json(tmp_path):
    """A path whose contents are not valid JSON raises ``invalid_json``."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ReportLoadError) as exc_info:
        load_panel_json(str(bad))
    assert exc_info.value.code == "invalid_json"


def test_load_non_object_root_raises_invalid_shape(tmp_path):
    """A JSON root that is not an object raises ``invalid_shape``."""
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ReportLoadError) as exc_info:
        load_panel_json(str(arr))
    assert exc_info.value.code == "invalid_shape"


def test_id_ending_in_json_falls_through_to_mcp_data(monkeypatch):
    """Plan.md Risk #3: an ID that happens to end in ``.json`` but has no
    matching file on disk must delegate to MCP data, not crash."""
    captured: dict[str, str] = {}

    def fake_get(result_id: str) -> dict:
        captured["id"] = result_id
        return {"id": result_id, "rounds": []}

    monkeypatch.setattr("synth_panel.mcp.data.get_panel_result", fake_get)

    data = load_panel_json("looks-like-a-file.json")
    assert captured["id"] == "looks-like-a-file.json"
    assert data["id"] == "looks-like-a-file.json"
