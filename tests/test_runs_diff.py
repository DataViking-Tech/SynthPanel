"""Tests for runs diff subcommand (sy-1b3n / GH-349)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_panel.diff import compute_diff, load_result
from synth_panel.main import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panelist(name: str, responses: list[dict]) -> dict:
    return {"name": name, "responses": responses}


def _text_resp(question: str, text: str) -> dict:
    return {"question": question, "response": text, "follow_up": False}


def _structured_resp(question: str, value: object) -> dict:
    """Structured categorical response (a dict body extract_category can read).

    Uses "rating" as the key — a member of _BOUNDED_KEYS — so extract_category
    returns the raw string value rather than JSON-serializing the whole dict.
    """
    return {"question": question, "response": {"rating": str(value)}, "follow_up": False}


def _make_result(
    rid: str,
    model: str,
    questions: list[dict],
    panelists: list[dict],
    **extra,
) -> dict:
    return {
        "id": rid,
        "model": model,
        "persona_count": len(panelists),
        "question_count": len(questions),
        "created_at": "2026-04-30T00:00:00Z",
        "total_cost": "$0.01",
        "total_usage": {"input": 100, "output": 50},
        "questions": questions,
        "results": panelists,
        **extra,
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

Q_TEXT = "What frustrates you most about the product?"
Q_ENUM = "How satisfied are you with the product?"

_QUESTIONS_TEXT = [{"text": Q_TEXT, "response_schema": {"type": "text"}}]
_QUESTIONS_ENUM = [
    {
        "text": Q_ENUM,
        "response_schema": {"type": "enum", "options": ["satisfied", "neutral", "dissatisfied"]},
    }
]

# Run A: workflow/interface frustrations
RESULT_A_TEXT = _make_result(
    rid="run_a",
    model="claude-sonnet-4-6",
    questions=_QUESTIONS_TEXT,
    panelists=[
        _panelist("Alice", [_text_resp(Q_TEXT, "The workflow interface feels clunky and confusing")]),
        _panelist("Bob", [_text_resp(Q_TEXT, "Interface navigation takes too many clicks to complete tasks")]),
        _panelist("Carol", [_text_resp(Q_TEXT, "Workflow steps are not intuitive or clear enough")]),
    ],
)

# Run B: pricing frustrations (different themes)
RESULT_B_TEXT = _make_result(
    rid="run_b",
    model="claude-haiku-4-5",
    questions=_QUESTIONS_TEXT,
    panelists=[
        _panelist("Dave", [_text_resp(Q_TEXT, "The pricing model feels expensive and unclear")]),
        _panelist("Eve", [_text_resp(Q_TEXT, "Cost structure makes budgeting difficult for teams")]),
        _panelist("Frank", [_text_resp(Q_TEXT, "Pricing tiers are confusing and expensive compared to alternatives")]),
    ],
)

# Run A: mostly satisfied
RESULT_A_ENUM = _make_result(
    rid="run_a_enum",
    model="claude-sonnet-4-6",
    questions=_QUESTIONS_ENUM,
    panelists=[
        _panelist("Alice", [_structured_resp(Q_ENUM, "satisfied")]),
        _panelist("Bob", [_structured_resp(Q_ENUM, "satisfied")]),
        _panelist("Carol", [_structured_resp(Q_ENUM, "neutral")]),
    ],
)

# Run B: shift toward dissatisfied
RESULT_B_ENUM = _make_result(
    rid="run_b_enum",
    model="claude-haiku-4-5",
    questions=_QUESTIONS_ENUM,
    panelists=[
        _panelist("Dave", [_structured_resp(Q_ENUM, "dissatisfied")]),
        _panelist("Eve", [_structured_resp(Q_ENUM, "dissatisfied")]),
        _panelist("Frank", [_structured_resp(Q_ENUM, "neutral")]),
    ],
)

RESULT_EMPTY = _make_result(
    rid="run_empty",
    model="claude-sonnet-4-6",
    questions=[],
    panelists=[],
)


# ---------------------------------------------------------------------------
# compute_diff unit tests
# ---------------------------------------------------------------------------


class TestComputeDiff:
    def test_metadata_fields(self) -> None:
        diff = compute_diff(RESULT_A_TEXT, RESULT_B_TEXT)
        m = diff.metadata
        assert m.result_a_id == "run_a"
        assert m.result_b_id == "run_b"
        assert m.model_a == "claude-sonnet-4-6"
        assert m.model_b == "claude-haiku-4-5"
        assert m.persona_count_a == 3
        assert m.persona_count_b == 3

    def test_text_question_diff(self) -> None:
        diff = compute_diff(RESULT_A_TEXT, RESULT_B_TEXT)
        assert len(diff.text_questions) == 1
        assert len(diff.categorical_questions) == 0
        q = diff.text_questions[0]
        assert q.question_key  # non-empty
        # Run A themes should mention interface/workflow; B should mention pricing
        assert q.top_themes_a  # has themes
        assert q.top_themes_b  # has themes
        # New themes in B should include pricing-related words not in A
        assert any("pric" in t for t in q.new_themes + q.top_themes_b)

    def test_categorical_question_diff(self) -> None:
        diff = compute_diff(RESULT_A_ENUM, RESULT_B_ENUM)
        assert len(diff.categorical_questions) == 1
        assert len(diff.text_questions) == 0
        q = diff.categorical_questions[0]
        assert 0.0 <= q.jsd <= 1.0
        assert "satisfied" in q.distribution_a or "neutral" in q.distribution_a
        assert "dissatisfied" in q.distribution_b or "neutral" in q.distribution_b
        # Distribution shifted — JSD should be non-zero
        assert q.jsd > 0

    def test_no_question_data(self) -> None:
        diff = compute_diff(RESULT_EMPTY, RESULT_EMPTY)
        assert diff.categorical_questions == []
        assert diff.text_questions == []


class TestLoadResult:
    def test_load_from_path(self, tmp_path: Path) -> None:
        result_file = tmp_path / "run_x.json"
        result_file.write_text(json.dumps(RESULT_A_TEXT), encoding="utf-8")
        loaded = load_result(str(result_file))
        assert loaded["id"] == "run_a"

    def test_load_injects_stem_as_id(self, tmp_path: Path) -> None:
        data = {k: v for k, v in RESULT_A_TEXT.items() if k != "id"}
        f = tmp_path / "my_run.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_result(str(f))
        assert loaded["id"] == "my_run"

    def test_load_missing_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_result("/nonexistent/path/result.json")


# ---------------------------------------------------------------------------
# CLI wiring tests
# ---------------------------------------------------------------------------


class TestRunsDiffCLI:
    def test_text_mode(self, tmp_path: Path) -> None:
        a_path = tmp_path / "run_a.json"
        b_path = tmp_path / "run_b.json"
        a_path.write_text(json.dumps(RESULT_A_TEXT), encoding="utf-8")
        b_path.write_text(json.dumps(RESULT_B_TEXT), encoding="utf-8")
        rc = main(["runs", "diff", str(a_path), str(b_path)])
        assert rc == 0

    def test_json_mode(self, tmp_path: Path, capsys) -> None:
        a_path = tmp_path / "run_a_enum.json"
        b_path = tmp_path / "run_b_enum.json"
        a_path.write_text(json.dumps(RESULT_A_ENUM), encoding="utf-8")
        b_path.write_text(json.dumps(RESULT_B_ENUM), encoding="utf-8")
        rc = main(["--output-format", "json", "runs", "diff", str(a_path), str(b_path)])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        # emit() merges the extra dict directly into the top-level payload.
        assert payload["message"] == "runs_diff"
        assert payload["result_a"] == "run_a_enum"
        assert payload["result_b"] == "run_b_enum"
        cat_qs = payload["categorical_questions"]
        assert len(cat_qs) == 1
        assert "jsd" in cat_qs[0]
        assert 0.0 <= cat_qs[0]["jsd"] <= 1.0

    def test_missing_result_a(self, tmp_path: Path) -> None:
        b_path = tmp_path / "run_b.json"
        b_path.write_text(json.dumps(RESULT_B_TEXT), encoding="utf-8")
        rc = main(["runs", "diff", "/no/such/file.json", str(b_path)])
        assert rc == 1

    def test_missing_result_b(self, tmp_path: Path) -> None:
        a_path = tmp_path / "run_a.json"
        a_path.write_text(json.dumps(RESULT_A_TEXT), encoding="utf-8")
        rc = main(["runs", "diff", str(a_path), "/no/such/file.json"])
        assert rc == 1
