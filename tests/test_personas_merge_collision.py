"""End-to-end coverage for --personas-merge name-collision handling (sp-g270).

The merge pipeline silently dedupes by persona ``name``; at n>=50 with
bundled packs that silent drop can shrink a declared panel by double
digits. These tests pin the loud-warning behavior, the JSON surface, the
error-on-collision policy, and the clean-path no-op case.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from synth_panel.cost import ZERO_USAGE, TokenUsage
from synth_panel.main import main
from synth_panel.persistence import ConversationMessage
from synth_panel.runtime import TurnSummary


def _mock_turn(text: str = "ok") -> TurnSummary:
    usage = TokenUsage(input_tokens=5, output_tokens=5)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


def _mock_synthesis():
    from synth_panel.cost import CostEstimate
    from synth_panel.synthesis import SynthesisResult

    return SynthesisResult(
        summary="s",
        themes=[],
        agreements=[],
        disagreements=[],
        surprises=[],
        recommendation="",
        usage=ZERO_USAGE,
        cost=CostEstimate(),
        model="sonnet",
    )


def _write_fixtures(tmp_path, *, base_names, merge_names):
    base = tmp_path / "base.yaml"
    base.write_text(
        "personas:\n"
        + "".join(f"  - name: {n}\n    age: 30\n" for n in base_names)
    )
    merge = tmp_path / "merge.yaml"
    merge.write_text(
        "personas:\n"
        + "".join(f"  - name: {n}\n    age: 40\n" for n in merge_names)
    )
    survey = tmp_path / "survey.yaml"
    survey.write_text("instrument:\n  questions:\n    - text: Q?\n")
    return base, merge, survey


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_collision_emits_warning_to_stderr(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _mock_turn()
    mock_runtime_cls.return_value = mock_runtime
    mock_synth.return_value = _mock_synthesis()

    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice", "Bob"], merge_names=["Alice", "Carol"]
    )

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--instrument",
            str(survey),
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "personas-merge name collisions" in err.lower() or "collisions dropped" in err
    assert "Alice" in err
    # Post-dedup panel size is 3 (Alice + Bob + Carol), pre-dedup would be 4.
    assert "3" in err
    assert "4" in err


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_collision_appears_in_json_output(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _mock_turn()
    mock_runtime_cls.return_value = mock_runtime
    mock_synth.return_value = _mock_synthesis()

    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice", "Bob"], merge_names=["Alice", "Carol"]
    )

    code = main(
        [
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--instrument",
            str(survey),
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "personas_merge_warnings" in data
    warnings = data["personas_merge_warnings"]
    assert isinstance(warnings, list)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["type"] == "name_collision"
    assert w["name"] == "Alice"
    assert w["source_path"] == str(merge)
    assert w["post_dedup_count"] == 3
    assert w["pre_dedup_count"] == 4


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_dry_run_surfaces_collision(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    # Dry-run short-circuits before any LLM call fires; the mocks exist
    # only to guarantee we never accidentally hit the provider layer.
    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice"], merge_names=["Alice"]
    )

    code = main(
        [
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--instrument",
            str(survey),
            "--dry-run",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "collisions dropped" in captured.err or "name collisions" in captured.err
    assert "Alice" in captured.err

    data = json.loads(captured.out)
    assert data.get("dry_run") is True
    assert "personas_merge_warnings" in data
    assert len(data["personas_merge_warnings"]) == 1
    assert data["personas_merge_warnings"][0]["name"] == "Alice"

    # No LLM runtime should have been constructed for a dry run.
    mock_runtime_cls.assert_not_called()


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_no_collision_no_warning(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _mock_turn()
    mock_runtime_cls.return_value = mock_runtime
    mock_synth.return_value = _mock_synthesis()

    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice"], merge_names=["Bob"]
    )

    code = main(
        [
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--instrument",
            str(survey),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "personas-merge name collisions" not in captured.err.lower()
    assert "collisions dropped" not in captured.err

    data = json.loads(captured.out)
    # Merge was used but produced no collisions → empty array, not absent.
    # Consumers can rely on the key being present when --personas-merge
    # was passed.
    assert data.get("personas_merge_warnings") == []


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_on_collision_error_fails_run(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice"], merge_names=["Alice"]
    )

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--personas-merge-on-collision",
            "error",
            "--instrument",
            str(survey),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Alice" in err
    assert "error" in err.lower()
    # No LLM calls should have happened.
    mock_runtime_cls.assert_not_called()


def test_on_collision_keep_is_reserved(capsys, tmp_path):
    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice"], merge_names=["Alice"]
    )

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--personas-merge-on-collision",
            "keep",
            "--instrument",
            str(survey),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "keep" in err
    assert "reserved" in err.lower()


@patch("synth_panel.cli.commands.synthesize_panel")
@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_default_policy_is_dedup_with_warning(
    mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
):
    """Default --personas-merge-on-collision is dedup (back-compat).

    The behavioral change in sp-g270 is the warning, not the merge
    semantics; without an explicit flag the run should still succeed
    and the later file should still win in place.
    """
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _mock_turn()
    mock_runtime_cls.return_value = mock_runtime
    mock_synth.return_value = _mock_synthesis()

    base, merge, survey = _write_fixtures(
        tmp_path, base_names=["Alice"], merge_names=["Alice"]
    )

    code = main(
        [
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(base),
            "--personas-merge",
            str(merge),
            "--instrument",
            str(survey),
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["persona_count"] == 1
    assert data["personas_merge_warnings"][0]["name"] == "Alice"
