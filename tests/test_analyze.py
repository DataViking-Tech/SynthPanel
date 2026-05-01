"""Tests for the analyze CLI subcommand and analysis engine."""

from __future__ import annotations

import json
import tempfile

from synth_panel.analyze import (
    DEFAULT_RESPONSE_CSV_COLUMNS,
    AnalysisResult,
    analysis_to_dict,
    analyze_panel_result,
    format_csv,
    format_csv_responses,
    format_text,
    parse_response_csv_columns,
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

    def test_frequency_table_alignment_with_non_ascii_categories(self):
        """Frequency-table rows must align in rendered cells when categories
        contain CJK / accented Latin / emoji.

        Regression: SP#298 — naive ``f"{cat:<30}"`` counted code points,
        so a row with ``"王芳"`` (4 cells but 2 codepoints) shifted right
        of an ASCII row, breaking column alignment.
        """
        import re

        from synth_panel.text_width import display_width

        data = _make_panel_result(n_personas=6)
        non_ascii_choices = ["José", "王芳", "Naoko 🌸", "agree"]
        for i, entry in enumerate(data["results"]):
            for r in entry["responses"]:
                r["response"] = non_ascii_choices[i % len(non_ascii_choices)]

        result = analyze_panel_result(data)
        text = format_text(result)

        # Frequency-table data rows match: two-space indent, then category,
        # then count, prop and CI. We anchor on the CI bracket so Borda
        # lines (4-space indent, "name: score") don't match.
        row_re = re.compile(r"^  .*?(\d+)\s+(\d+\.\d{2})\s+\[\d")
        rows = [ln for ln in text.splitlines() if row_re.match(ln)]
        assert rows, "expected at least one frequency-table row"

        # The count column is right-justified to width 5. Measure the
        # rendered width of the prefix up to and including the count —
        # for every aligned row this is the same (2 indent + 30 cat + 1
        # space + 5 count = 38 cells).
        widths = set()
        for row in rows:
            m = row_re.match(row)
            widths.add(display_width(row[: m.end(1)]))

        assert len(widths) == 1, f"frequency table misaligned, prefix widths: {widths}"
        assert next(iter(widths)) == 38, f"expected 38-cell prefix, got {widths}"


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


# ---------------------------------------------------------------------------
# Responses CSV (raw export, sp-tzavk0 / GH #297)
# ---------------------------------------------------------------------------


def _make_csv_panel_result(
    *,
    n_personas: int = 3,
    response_overrides: list[list[str]] | None = None,
    persona_overrides: list[str] | None = None,
    questions: list[dict] | None = None,
    include_questions_block: bool = True,
) -> dict:
    """Build a panel result tailored for responses-CSV testing."""
    if questions is None:
        questions = [
            {"text": "What frustrates you about your workflow?"},
            {"text": "Pick one option.", "response_schema": {"type": "enum", "options": ["A", "B"]}},
        ]
    n_questions = len(questions)
    persona_names = (
        list(persona_overrides) if persona_overrides is not None else [f"Persona_{i}" for i in range(n_personas)]
    )
    if response_overrides is None:
        response_overrides = [[f"persona {i}, q{q}" for q in range(n_questions)] for i in range(n_personas)]

    results = []
    for i, name in enumerate(persona_names):
        responses = []
        for qi in range(n_questions):
            responses.append(
                {
                    "question": questions[qi].get("text", ""),
                    "response": response_overrides[i][qi],
                    "usage": {"input_tokens": 60, "output_tokens": 24},
                }
            )
        results.append(
            {
                "persona": name,
                "model": "claude-sonnet-4-6",
                "responses": responses,
            }
        )

    data: dict = {
        "id": "result-csv-001",
        "model": "claude-sonnet-4-6",
        "persona_count": n_personas,
        "question_count": n_questions,
        "total_usage": {"input_tokens": 180, "output_tokens": 72},
        "total_cost": "$0.0036",
        "results": results,
    }
    if include_questions_block:
        data["questions"] = questions
    return data


class TestResponsesCSV:
    def test_default_columns(self):
        cols = parse_response_csv_columns(None)
        assert cols == list(DEFAULT_RESPONSE_CSV_COLUMNS)
        assert "persona_id" in cols
        assert "response_type" in cols

    def test_parse_columns_custom_subset(self):
        cols = parse_response_csv_columns("persona_name, response, cost")
        assert cols == ["persona_name", "response", "cost"]

    def test_parse_columns_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown column"):
            parse_response_csv_columns("persona_name,bogus_column")

    def test_parse_columns_blank_returns_default(self):
        cols = parse_response_csv_columns("   ")
        assert cols == list(DEFAULT_RESPONSE_CSV_COLUMNS)

    def test_round_trip_via_dict_reader(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result()
        text = format_csv_responses(data)

        reader = _csv.DictReader(_io.StringIO(text))
        rows = list(reader)

        # 3 personas × 2 questions = 6 rows
        assert len(rows) == 6
        assert reader.fieldnames == list(DEFAULT_RESPONSE_CSV_COLUMNS)

        first = rows[0]
        assert first["persona_id"] == "p0"
        assert first["persona_name"] == "Persona_0"
        assert first["question_id"] == "q0"
        assert first["question_text"] == "What frustrates you about your workflow?"
        assert first["response"] == "persona 0, q0"
        assert first["response_type"] == "free-text"
        assert first["cost"].startswith("$")

        # Per-question response_type honors response_schema (q1 is enum).
        q1_row = rows[1]
        assert q1_row["response_type"] == "multiple-choice"

    def test_response_types_map_correctly(self):
        questions = [
            {"text": "Free text?"},
            {"text": "Pick one.", "response_schema": {"type": "enum", "options": ["A", "B"]}},
            {"text": "Rate.", "response_schema": {"type": "scale", "min": 1, "max": 5}},
            {
                "text": "Tag themes.",
                "response_schema": {"type": "tagged_themes", "taxonomy": ["price", "ux"]},
            },
            {"text": "Default text.", "response_schema": {"type": "text"}},
        ]
        data = _make_csv_panel_result(
            n_personas=1,
            questions=questions,
            response_overrides=[["a", "A", "3", "price", "x"]],
        )
        import csv as _csv
        import io as _io

        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert [r["response_type"] for r in rows] == [
            "free-text",
            "multiple-choice",
            "likert",
            "tagged-themes",
            "free-text",
        ]

    def test_csv_injection_safety_persona_name(self):
        import csv as _csv
        import io as _io

        # Personas whose names look like spreadsheet formulas must be neutralized.
        data = _make_csv_panel_result(
            n_personas=4,
            persona_overrides=["=cmd|'/c calc'!A1", "+evil()", "-bad", "@import"],
        )
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        # Each malicious cell must start with a single quote so spreadsheet
        # apps treat it as plain text rather than a formula.
        names = [r["persona_name"] for r in rows[::2]]  # one row per persona (q0)
        assert names[0] == "'=cmd|'/c calc'!A1"
        assert names[1] == "'+evil()"
        assert names[2] == "'-bad"
        assert names[3] == "'@import"

    def test_csv_injection_safety_response_cell(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(
            n_personas=1,
            response_overrides=[['=HYPERLINK("http://evil","x")', "ok"]],
        )
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["response"].startswith("'=HYPERLINK")
        assert rows[1]["response"] == "ok"

    def test_embedded_newlines_quoted(self):
        import csv as _csv
        import io as _io

        multiline = "line one\nline two, with comma\nline three"
        data = _make_csv_panel_result(
            n_personas=1,
            response_overrides=[[multiline, "ok"]],
        )
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["response"] == multiline

    def test_embedded_double_quotes_round_trip(self):
        import csv as _csv
        import io as _io

        quoted = 'They said "hello, world" — twice.'
        data = _make_csv_panel_result(
            n_personas=1,
            response_overrides=[[quoted, "ok"]],
        )
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["response"] == quoted

    def test_columns_subset_filters_output(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=2)
        text = format_csv_responses(data, columns=["persona_name", "response"])
        reader = _csv.DictReader(_io.StringIO(text))
        assert reader.fieldnames == ["persona_name", "response"]
        rows = list(reader)
        assert len(rows) == 4  # 2 personas × 2 questions
        for r in rows:
            assert set(r.keys()) == {"persona_name", "response"}

    def test_extra_columns(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=1)
        text = format_csv_responses(
            data,
            columns=["persona_name", "model", "input_tokens", "output_tokens"],
        )
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["model"] == "claude-sonnet-4-6"
        assert rows[0]["input_tokens"] == "60"
        assert rows[0]["output_tokens"] == "24"

    def test_skips_panelist_with_no_responses(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=2)
        # Wipe one panelist's responses; output should still parse cleanly.
        data["results"][0]["responses"] = []
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert len(rows) == 2  # only the second panelist contributes 2 rows
        assert all(r["persona_name"] == "Persona_1" for r in rows)

    def test_error_response_renders_blank(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=1)
        data["results"][0]["responses"][0]["error"] = "rate limit"
        data["results"][0]["responses"][0]["response"] = ""
        text = format_csv_responses(
            data,
            columns=["persona_name", "response", "error"],
        )
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["response"] == ""
        assert rows[0]["error"] == "rate limit"

    def test_uses_response_question_text_when_questions_block_absent(self):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=1, include_questions_block=False)
        # Without a top-level questions block we still emit the question_text
        # from each panelist's per-response ``question`` field.
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        assert rows[0]["question_text"] == "What frustrates you about your workflow?"
        # response_type defaults to free-text when no schema is available.
        assert rows[0]["response_type"] == "free-text"

    def test_structured_response_serialized_as_json(self):
        import csv as _csv
        import io as _io
        import json as _json2

        data = _make_csv_panel_result(n_personas=1)
        # Replace the q0 response with a structured payload; the cell should
        # contain a JSON serialization that parses back round-trip.
        data["results"][0]["responses"][0]["response"] = {"choice": "A", "score": 4}
        text = format_csv_responses(data)
        rows = list(_csv.DictReader(_io.StringIO(text)))
        parsed = _json2.loads(rows[0]["response"])
        assert parsed == {"choice": "A", "score": 4}

    def test_uses_crlf_line_terminator(self):
        # RFC 4180 specifies CRLF for CSV records. Round-trip safety checks
        # already cover parsing, but downstream tools (e.g. Excel on Windows)
        # are pickier about strict CRLF — pin the terminator explicitly.
        data = _make_csv_panel_result(n_personas=1)
        text = format_csv_responses(data)
        # Expect every record (header + at least one data row) to end with
        # \r\n. Splitting on \r\n should yield records back; any naked \n in
        # the middle of records is fine (they belong to embedded newlines).
        lines = text.split("\r\n")
        # Trailing CRLF means the last element is empty.
        assert lines[-1] == ""
        # Header is the first element and contains no embedded \n.
        assert "persona_id" in lines[0]
        assert "\n" not in lines[0]


class TestResponsesCSVCli:
    def test_parser_accepts_responses_csv(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "result-123",
                "--output",
                "responses-csv",
                "--columns",
                "persona_name,response",
            ]
        )
        assert args.output == "responses-csv"
        assert args.columns == "persona_name,response"

    def test_cli_responses_csv_output(self, capsys):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=2)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(["analyze", f.name, "--output", "responses-csv"])

        captured = capsys.readouterr()
        assert exit_code == 0
        rows = list(_csv.DictReader(_io.StringIO(captured.out)))
        assert len(rows) == 4  # 2 personas × 2 questions
        assert rows[0]["persona_name"] == "Persona_0"

    def test_cli_responses_csv_columns_filter(self, capsys):
        import csv as _csv
        import io as _io

        data = _make_csv_panel_result(n_personas=1)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(
                [
                    "analyze",
                    f.name,
                    "--output",
                    "responses-csv",
                    "--columns",
                    "persona_name,response",
                ]
            )

        captured = capsys.readouterr()
        assert exit_code == 0
        reader = _csv.DictReader(_io.StringIO(captured.out))
        assert reader.fieldnames == ["persona_name", "response"]

    def test_cli_responses_csv_unknown_column(self, capsys):
        data = _make_csv_panel_result(n_personas=1)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            exit_code = main(
                [
                    "analyze",
                    f.name,
                    "--output",
                    "responses-csv",
                    "--columns",
                    "persona_name,bogus",
                ]
            )

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Unknown column" in captured.err
