"""CLI integration tests for ``synthpanel report`` (sp-viz-layer T4).

Covers the five cases listed in ``specs/sp-viz-layer/structure.md`` §6:
stdout-happy-path, file-output, not-found in TEXT and JSON modes, and
malformed JSON on disk. Each test drives the full :func:`main` entry
point so parser-wiring regressions surface here.
"""

from __future__ import annotations

import json
from pathlib import Path

from synth_panel.main import main
from synth_panel.reporting.markdown import BANNER, FOOTER

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def test_report_to_stdout(capsys):
    """``synthpanel report <fixture.json>`` writes Markdown to stdout."""
    fixture = FIXTURES / "flat_shape.json"

    code = main(["report", str(fixture)])

    assert code == 0
    captured = capsys.readouterr()
    assert "# Panel Report:" in captured.out
    assert BANNER in captured.out
    assert FOOTER in captured.out
    # Nothing on stderr when writing to stdout.
    assert captured.err == ""


def test_report_to_file(tmp_path, capsys):
    """``synthpanel report <fixture.json> -o FILE`` writes Markdown to FILE."""
    fixture = FIXTURES / "flat_shape.json"
    out_file = tmp_path / "report.md"

    code = main(["report", str(fixture), "-o", str(out_file)])

    assert code == 0
    assert out_file.exists()
    contents = out_file.read_text(encoding="utf-8")
    assert "# Panel Report:" in contents
    assert BANNER in contents
    assert FOOTER in contents

    captured = capsys.readouterr()
    # TEXT mode emits a status line to stderr so humans get confirmation.
    assert "Report written" in captured.err
    assert str(out_file) in captured.err
    # stdout stays clean — the file is the output.
    assert captured.out == ""


def test_report_not_found_returns_1_stderr(tmp_path, monkeypatch, capsys):
    """Missing result in TEXT mode: exit 1 with error message on stderr."""
    # Point the MCP data dir at an empty tmp_path so the lookup misses.
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))

    code = main(["report", "no-such-result-id"])

    assert code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "no-such-result-id" in captured.err
    assert captured.out == ""


def test_report_not_found_returns_1_json_error(tmp_path, monkeypatch, capsys):
    """Missing result in JSON mode: exit 1 with a JSON error payload on stdout."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))

    code = main(["--output-format", "json", "report", "no-such-result-id"])

    assert code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["error"] == "not_found"
    assert "no-such-result-id" in payload["message"]


def test_report_invalid_json_returns_1(tmp_path, capsys):
    """A ``.json`` file with malformed contents: exit 1 with invalid_json."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")

    code = main(["--output-format", "json", "report", str(bad)])

    assert code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["error"] == "invalid_json"
