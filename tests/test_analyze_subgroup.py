"""Tests for `synthpanel analyze --by` subgroup analysis (GH-341)."""

from __future__ import annotations

import io
import json
import sys

import yaml

from synth_panel.cli.parser import build_parser
from synth_panel.main import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    questions: list[dict],
    panelists: list[dict],
) -> dict:
    """Build a minimal panel result dict."""
    results = []
    for p in panelists:
        responses = []
        for q in questions:
            responses.append(
                {
                    "question": q["text"],
                    "response": p["responses"].get(q["text"], ""),
                    "follow_up": False,
                    "error": False,
                }
            )
        results.append(
            {
                "persona": p["name"],
                "responses": responses,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "cost": "$0.00",
                "error": None,
            }
        )
    return {
        "id": "test-result",
        "model": "claude-test",
        "persona_count": len(panelists),
        "question_count": len(questions),
        "total_usage": {"input_tokens": 10, "output_tokens": 5},
        "total_cost": "$0.00",
        "results": results,
        "questions": [{"text": q["text"], "extraction_schema": q.get("schema", {"type": "text"})} for q in questions],
    }


def _make_personas_yaml(personas: list[dict]) -> str:
    return yaml.dump({"personas": personas})


def _capture(cmd: list[str]) -> tuple[int, str, str]:
    """Run main(cmd) capturing stdout and stderr."""
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        rc = main(cmd)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestAnalyzeSubgroupParser:
    def test_parser_accepts_by_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--by", "occupation"])
        assert args.command == "analyze"
        assert args.result == "result-123"
        assert args.by == "occupation"

    def test_parser_by_with_personas(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--by", "age", "--personas", "my-pack"])
        assert args.personas == "my-pack"

    def test_parser_old_analyze_unaffected(self):
        """Existing `analyze <result>` invocation must still work."""
        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123"])
        assert args.command == "analyze"
        assert args.result == "result-123"
        assert getattr(args, "by", None) is None

    def test_parser_old_analyze_json_output(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--output", "json"])
        assert args.output == "json"

    def test_parser_by_and_output_are_independent(self):
        """--by and --output can coexist (future cross-feature use)."""
        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--by", "role", "--output", "text"])
        assert args.by == "role"
        assert args.output == "text"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeSubgroupCLI:
    def test_scale_by_occupation(self, tmp_path):
        questions = [{"text": "Rate 1-5", "schema": {"type": "scale", "min": 1, "max": 5}}]
        panelists = [
            {"name": "Alice", "responses": {"Rate 1-5": 5}},
            {"name": "Bob", "responses": {"Rate 1-5": 4}},
            {"name": "Carol", "responses": {"Rate 1-5": 2}},
            {"name": "Dave", "responses": {"Rate 1-5": 3}},
        ]
        personas = [
            {"name": "Alice", "occupation": "engineer"},
            {"name": "Bob", "occupation": "engineer"},
            {"name": "Carol", "occupation": "teacher"},
            {"name": "Dave", "occupation": "teacher"},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(["analyze", str(result_file), "--by", "occupation", "--personas", str(personas_file)])
        assert rc == 0
        assert "occupation" in out
        assert "engineer" in out
        assert "teacher" in out
        assert "mean=" in out

    def test_enum_by_group(self, tmp_path):
        questions = [{"text": "Prefer A or B?", "schema": {"type": "enum", "options": ["A", "B"]}}]
        panelists = [
            {"name": "Alice", "responses": {"Prefer A or B?": "A"}},
            {"name": "Bob", "responses": {"Prefer A or B?": "A"}},
            {"name": "Carol", "responses": {"Prefer A or B?": "B"}},
            {"name": "Dave", "responses": {"Prefer A or B?": "B"}},
        ]
        personas = [
            {"name": "Alice", "group": "X"},
            {"name": "Bob", "group": "X"},
            {"name": "Carol", "group": "Y"},
            {"name": "Dave", "group": "Y"},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(["analyze", str(result_file), "--by", "group", "--personas", str(personas_file)])
        assert rc == 0
        assert "group" in out
        assert "X" in out
        assert "Y" in out

    def test_json_format(self, tmp_path):
        questions = [{"text": "Rate it", "schema": {"type": "scale", "min": 1, "max": 5}}]
        panelists = [
            {"name": "Alice", "responses": {"Rate it": 4}},
            {"name": "Bob", "responses": {"Rate it": 2}},
        ]
        personas = [
            {"name": "Alice", "role": "dev"},
            {"name": "Bob", "role": "pm"},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(
            [
                "analyze",
                str(result_file),
                "--by",
                "role",
                "--personas",
                str(personas_file),
                "--output",
                "json",
            ]
        )
        assert rc == 0
        payload = json.loads(out)
        assert "per_question" in payload
        assert "fields" in payload
        assert payload["fields"] == ["role"]

    def test_auto_bin_integer_field(self, tmp_path):
        questions = [{"text": "Rate it", "schema": {"type": "scale", "min": 1, "max": 5}}]
        panelists = [
            {"name": "Alice", "responses": {"Rate it": 5}},
            {"name": "Bob", "responses": {"Rate it": 4}},
            {"name": "Carol", "responses": {"Rate it": 2}},
        ]
        personas = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35},
            {"name": "Carol", "age": 45},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(["analyze", str(result_file), "--by", "age_decade", "--personas", str(personas_file)])
        assert rc == 0
        # Values 25/35/45 → AGE_DECADE_BANDS bins 18-27, 28-37, 38-47
        assert "18-27" in out or "28-37" in out or "38-47" in out

    def test_unknown_field_without_personas_gives_helpful_error(self, tmp_path):
        questions = [{"text": "Q?", "schema": {"type": "text"}}]
        panelists = [{"name": "Alice", "responses": {"Q?": "hello"}}]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))

        rc, _, err = _capture(["analyze", str(result_file), "--by", "occupation"])
        assert rc != 0
        assert "--personas" in err

    def test_missing_result_file(self):
        rc, _, err = _capture(["analyze", "nonexistent-result-id", "--by", "role"])
        assert rc != 0
        assert "not found" in err

    def test_effect_size_scale_present_in_json(self, tmp_path):
        questions = [{"text": "Rate it", "schema": {"type": "scale", "min": 1, "max": 5}}]
        # 3 per group so sparse-suppression doesn't trigger
        panelists = [
            {"name": p["name"], "responses": {"Rate it": p["score"]}}
            for p in [
                {"name": "A", "score": 5},
                {"name": "B", "score": 5},
                {"name": "C", "score": 5},
                {"name": "D", "score": 1},
                {"name": "E", "score": 1},
                {"name": "F", "score": 1},
            ]
        ]
        personas = [
            {"name": "A", "group": "hi"},
            {"name": "B", "group": "hi"},
            {"name": "C", "group": "hi"},
            {"name": "D", "group": "lo"},
            {"name": "E", "group": "lo"},
            {"name": "F", "group": "lo"},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(
            ["analyze", str(result_file), "--by", "group", "--personas", str(personas_file), "--output", "json"]
        )
        assert rc == 0
        payload = json.loads(out)
        bd = payload["per_question"][0]["by_field"][0]
        assert bd["effect_size_type"] == "eta-squared"
        assert bd["effect_size"] is not None
        assert bd["effect_size"] > 0

    def test_cramers_v_enum_present_in_json(self, tmp_path):
        questions = [{"text": "Choice?", "schema": {"type": "enum", "options": ["X", "Y"]}}]
        # 3 per group so sparse-suppression doesn't trigger
        panelists = [
            {"name": "A", "responses": {"Choice?": "X"}},
            {"name": "B", "responses": {"Choice?": "X"}},
            {"name": "C", "responses": {"Choice?": "X"}},
            {"name": "D", "responses": {"Choice?": "Y"}},
            {"name": "E", "responses": {"Choice?": "Y"}},
            {"name": "F", "responses": {"Choice?": "Y"}},
        ]
        personas = [
            {"name": "A", "cohort": "alpha"},
            {"name": "B", "cohort": "alpha"},
            {"name": "C", "cohort": "alpha"},
            {"name": "D", "cohort": "beta"},
            {"name": "E", "cohort": "beta"},
            {"name": "F", "cohort": "beta"},
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_make_result(questions, panelists)))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_make_personas_yaml(personas))

        rc, out, _ = _capture(
            ["analyze", str(result_file), "--by", "cohort", "--personas", str(personas_file), "--output", "json"]
        )
        assert rc == 0
        payload = json.loads(out)
        bd = payload["per_question"][0]["by_field"][0]
        assert bd["effect_size_type"] == "cramers-v"
        assert bd["effect_size"] is not None
