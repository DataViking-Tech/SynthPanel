"""sy-546: CLI integration for the model reachability pre-flight.

A multi-model run with a bad slug aborts before spending, naming the bad
slug; --skip-preflight bypasses the check; --dry-run runs the same check.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.main import main
from synth_panel.preflight import ModelProbe, PreflightReport

BAD = "openrouter/google/gemini-2.0-flash-001"
GOOD_A = "openrouter/openai/gpt-4o-mini"
GOOD_B = "openrouter/anthropic/claude-haiku-4.5"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    personas = tmp_path / "personas.yaml"
    personas.write_text("personas:\n  - name: A\n  - name: B\n")
    survey = tmp_path / "survey.yaml"
    survey.write_text("instrument:\n  version: 1\n  questions:\n    - text: Q?\n")
    return personas, survey


def _report(reachable: list[str], unreachable: list[str]) -> PreflightReport:
    probes = [ModelProbe(model=m, status="reachable") for m in reachable]
    probes += [ModelProbe(model=m, status="unreachable", detail="404 no endpoints") for m in unreachable]
    return PreflightReport(probes=probes)


@patch("synth_panel.preflight.preflight_models")
def test_bad_slug_aborts_naming_it(
    mock_preflight: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    mock_preflight.return_value = _report([GOOD_A], [BAD])
    personas, survey = _write_inputs(tmp_path)

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(survey),
            "--models",
            f"{GOOD_A},{BAD}",
            "--no-synthesis",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "Pre-flight failed" in err
    assert BAD in err
    # The orchestrator never ran — preflight was the gate.
    mock_preflight.assert_called_once()


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.preflight.preflight_models")
def test_skip_preflight_bypasses_check(
    mock_preflight: MagicMock,
    mock_runtime: MagicMock,
    tmp_path: Path,
) -> None:
    mock_preflight.return_value = _report([], [BAD])  # would abort if consulted
    runtime = MagicMock()
    from synth_panel.cost import TokenUsage as CostTokenUsage
    from synth_panel.persistence import ConversationMessage
    from synth_panel.runtime import TurnSummary

    usage = CostTokenUsage(input_tokens=5, output_tokens=2)
    runtime.run_turn.return_value = TurnSummary(
        assistant_messages=[
            ConversationMessage(role="assistant", content=[{"type": "text", "text": "ok"}], usage=usage)
        ],
        iterations=1,
        usage=usage,
    )
    mock_runtime.return_value = runtime
    personas, survey = _write_inputs(tmp_path)

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(survey),
            "--models",
            f"{GOOD_A},{BAD}",
            "--skip-preflight",
            "--no-synthesis",
            "--max-concurrent",
            "1",
        ]
    )

    # With --skip-preflight the bad-slug report is never consulted.
    mock_preflight.assert_not_called()
    assert code == 0


@patch("synth_panel.preflight.preflight_models")
def test_dry_run_performs_preflight(
    mock_preflight: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    mock_preflight.return_value = _report([GOOD_A], [BAD])
    personas, survey = _write_inputs(tmp_path)

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(survey),
            "--models",
            f"{GOOD_A},{BAD}",
            "--dry-run",
            "--no-synthesis",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert BAD in err
    mock_preflight.assert_called_once()


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.preflight.preflight_models")
def test_min_models_allows_degraded_run(
    mock_preflight: MagicMock,
    mock_runtime: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # 2 reachable, 1 bad; --min-models 2 should proceed with a warning.
    mock_preflight.return_value = _report([GOOD_A, GOOD_B], [BAD])
    runtime = MagicMock()
    from synth_panel.cost import TokenUsage as CostTokenUsage
    from synth_panel.persistence import ConversationMessage
    from synth_panel.runtime import TurnSummary

    usage = CostTokenUsage(input_tokens=5, output_tokens=2)
    runtime.run_turn.return_value = TurnSummary(
        assistant_messages=[
            ConversationMessage(role="assistant", content=[{"type": "text", "text": "ok"}], usage=usage)
        ],
        iterations=1,
        usage=usage,
    )
    mock_runtime.return_value = runtime
    personas, survey = _write_inputs(tmp_path)

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(survey),
            "--models",
            f"{GOOD_A},{GOOD_B},{BAD}",
            "--min-models",
            "2",
            "--no-synthesis",
            "--max-concurrent",
            "1",
        ]
    )

    assert code == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert BAD in err
