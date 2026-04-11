"""Tests for the analyze CLI subcommand and analysis engine."""

from __future__ import annotations

import json
import tempfile

from synth_panel.analyze import (
    AnalysisResult,
    analysis_to_dict,
    analyze_panel_result,
    format_csv,
    format_text,
)
from synth_panel.main import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel_result(
    *,
    n_personas: int = 5,
    n_questions: int = 2,
    multi_model: bool = False,
) -> dict:
    """Build a synthetic panel result dict for testing."""
    personas = [f"Persona_{i}" for i in range(n_personas)]
    questions = [f"What do you think about topic {q}?" for q in range(n_questions)]
    choices = ["agree", "disagree", "neutral"]

    results = []
    for i, name in enumerate(personas):
        responses = []
        for qi, q_text in enumerate(questions):
            responses.append(
                {
                    "question": q_text,
                    "response": choices[(i + qi) % len(choices)],
                    "follow_up": False,
                    "error": False,
                }
            )
        entry: dict = {
            "persona": name,
            "responses": responses,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "cost": "$0.01",
        }
        if multi_model:
            entry["model"] = "model-a" if i % 2 == 0 else "model-b"
        results.append(entry)

    return {
        "id": "result-test-001",
        "created_at": "2026-04-10T12:00:00Z",
        "model": "haiku",
        "persona_count": n_personas,
        "question_count": n_questions,
        "total_usage": {"input_tokens": 500, "output_tokens": 250},
        "total_cost": "$0.05",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Analysis engine tests
# ---------------------------------------------------------------------------


class TestAnalyzeEngine:
    def test_basic_analysis_produces_4_sections(self):
        data = _make_panel_result()
        result = analyze_panel_result(data)

        assert isinstance(result, AnalysisResult)
        assert result.result_id == "result-test-001"
        assert result.persona_count == 5
        assert result.question_count == 2
        assert len(result.per_question) == 2

        # Section 1+2: descriptive + inferential per question
        for qa in result.per_question:
            assert qa.frequency is not None
            assert qa.frequency.total == 5
            assert len(qa.frequency.rows) > 0

        # Section 3: no convergence for single-model
        assert result.convergence is None
        assert not result.is_multi_model

        # Section 4: no clusters for N < 30
        assert result.clusters is None
        assert any("clustering skipped" in w for w in result.warnings)

    def test_frequency_table_has_bootstrap_cis(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)

        for qa in result.per_question:
            for row in qa.frequency.rows:
                assert 0.0 <= row.ci_lower <= row.ci_upper <= 1.0

    def test_chi_squared_present(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)

        for qa in result.per_question:
            assert qa.chi_squared is not None
            assert qa.chi_squared.statistic >= 0
            assert qa.chi_squared.cramers_v >= 0

    def test_borda_count_present(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)

        for qa in result.per_question:
            assert qa.borda is not None
            assert len(qa.borda) > 0

    def test_kendall_w_present(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)

        for qa in result.per_question:
            if qa.kendall_w is not None:
                assert 0.0 <= qa.kendall_w.w <= 1.0

    def test_multi_model_convergence(self):
        data = _make_panel_result(n_personas=10, multi_model=True)
        result = analyze_panel_result(data)

        assert result.is_multi_model
        assert result.convergence is not None
        assert result.convergence.n_models == 2
        assert "model-a" in result.convergence.model_names
        assert "model-b" in result.convergence.model_names

    def test_cluster_with_large_n(self):
        data = _make_panel_result(n_personas=35)
        result = analyze_panel_result(data)

        # Clustering should be attempted (may or may not produce results
        # depending on response diversity)
        # At minimum, the clustering-skipped warning should NOT be present
        assert not any("clustering skipped" in w for w in result.warnings)

    def test_warnings_for_low_n(self):
        data = _make_panel_result(n_personas=3)
        result = analyze_panel_result(data)

        assert any("unreliable" in w for w in result.warnings)

    def test_empty_results(self):
        data = {
            "id": "empty",
            "model": "haiku",
            "persona_count": 0,
            "question_count": 0,
            "results": [],
        }
        result = analyze_panel_result(data)
        assert result.persona_count == 0
        assert len(result.per_question) == 0

    def test_error_responses_excluded(self):
        data = _make_panel_result(n_personas=5)
        # Mark one response as error
        data["results"][0]["responses"][0]["error"] = True
        result = analyze_panel_result(data)

        # First question should have 4 responses (1 error excluded)
        assert result.per_question[0].n_responses == 4


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_analysis_to_dict_roundtrip(self):
        data = _make_panel_result(n_personas=10, multi_model=True)
        result = analyze_panel_result(data)
        d = analysis_to_dict(result)

        # Must be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["result_id"] == "result-test-001"
        assert len(parsed["per_question"]) == 2
        assert "convergence" in parsed
        assert "warnings" in parsed

    def test_dict_has_frequency_table(self):
        data = _make_panel_result()
        result = analyze_panel_result(data)
        d = analysis_to_dict(result)

        for q in d["per_question"]:
            assert "frequency_table" in q
            for row in q["frequency_table"]:
                assert "category" in row
                assert "count" in row
                assert "proportion" in row

    def test_dict_has_chi_squared(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)
        d = analysis_to_dict(result)

        for q in d["per_question"]:
            if "chi_squared" in q:
                assert "statistic" in q["chi_squared"]
                assert "cramers_v" in q["chi_squared"]


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestTextFormat:
    def test_text_contains_sections(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)
        text = format_text(result)

        assert "DESCRIPTIVE STATISTICS" in text
        assert "INFERENTIAL STATISTICS" in text
        assert "WARNINGS" in text

    def test_text_contains_frequency_table(self):
        data = _make_panel_result()
        result = analyze_panel_result(data)
        text = format_text(result)

        assert "Category" in text
        assert "Count" in text
        assert "Prop" in text
        assert "95% CI" in text

    def test_text_multi_model_shows_convergence(self):
        data = _make_panel_result(n_personas=10, multi_model=True)
        result = analyze_panel_result(data)
        text = format_text(result)

        assert "CROSS-MODEL CONVERGENCE" in text
        assert "model-a" in text

    def test_text_contains_chi_squared(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)
        text = format_text(result)

        assert "Chi-squared" in text
        assert "Cramer's V" in text


# ---------------------------------------------------------------------------
# CSV formatting tests
# ---------------------------------------------------------------------------


class TestCSVFormat:
    def test_csv_has_header(self):
        data = _make_panel_result()
        result = analyze_panel_result(data)
        csv_output = format_csv(result)

        lines = csv_output.strip().split("\n")
        assert lines[0].startswith("section,question_index,")

    def test_csv_has_frequency_rows(self):
        data = _make_panel_result()
        result = analyze_panel_result(data)
        csv_output = format_csv(result)

        freq_lines = [line for line in csv_output.split("\n") if line.startswith("frequency,")]
        assert len(freq_lines) > 0

    def test_csv_has_inferential_rows(self):
        data = _make_panel_result(n_personas=10)
        result = analyze_panel_result(data)
        csv_output = format_csv(result)

        inf_lines = [line for line in csv_output.split("\n") if line.startswith("inferential,")]
        assert len(inf_lines) > 0

    def test_csv_multi_model_has_convergence(self):
        data = _make_panel_result(n_personas=10, multi_model=True)
        result = analyze_panel_result(data)
        csv_output = format_csv(result)

        conv_lines = [line for line in csv_output.split("\n") if line.startswith("convergence,")]
        assert len(conv_lines) > 0


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parser_accepts_analyze(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123"])
        assert args.command == "analyze"
        assert args.result == "result-123"
        assert args.output == "text"

    def test_parser_analyze_json(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--output", "json"])
        assert args.output == "json"

    def test_parser_analyze_csv(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "result-123", "--output", "csv"])
        assert args.output == "csv"

    def test_cli_analyze_file(self, capsys):
        data = _make_panel_result()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(["analyze", f.name])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "DESCRIPTIVE STATISTICS" in captured.out

    def test_cli_analyze_json_output(self, capsys):
        data = _make_panel_result()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(["analyze", f.name, "--output", "json"])

        captured = capsys.readouterr()
        assert exit_code == 0
        parsed = json.loads(captured.out)
        assert "per_question" in parsed
        assert "warnings" in parsed

    def test_cli_analyze_csv_output(self, capsys):
        data = _make_panel_result()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(["analyze", f.name, "--output", "csv"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "section,question_index" in captured.out

    def test_cli_analyze_missing_result(self, capsys):
        exit_code = main(["analyze", "nonexistent-result-id"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "not found" in captured.err
