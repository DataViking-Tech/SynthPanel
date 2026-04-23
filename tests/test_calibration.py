"""Tests for ``--calibrate-against`` parse-time validation (sp-a6jc).

T2 of sp-inline-calibration. These tests cover the CLI surface only:
flag registration, format validation, redistribution-tier gating, and
conflict reconciliation against ``--convergence-baseline``. The hard
contract is that every failure mode short-circuits BEFORE any LLM spend
— matching the sp-yaru "no panelist call before validation" pattern.

JSD computation, schema auto-derivation, and report attachment land in
later tickets (T3, T4); they are intentionally not exercised here.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cli.commands import _INLINE_CALIBRATION_ALLOWED
from synth_panel.cli.parser import build_parser
from synth_panel.main import main

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def panel_files(tmp_path):
    """Minimal valid persona + instrument pair for ``panel run`` invocations."""
    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n")
    survey_file = tmp_path / "survey.yaml"
    # Use a Likert / pick-one bounded question so identify_tracked_questions
    # would (in principle) accept it for convergence — but our validation
    # tests exit before tracker construction either way.
    survey_file.write_text(
        "instrument:\n"
        "  questions:\n"
        "    - text: How happy are you these days?\n"
        "      response_schema:\n"
        "        type: pick_one\n"
        "        options: [Very happy, Pretty happy, Not too happy]\n"
    )
    return str(personas_file), str(survey_file)


def _panel_run_args(personas_path: str, instrument_path: str, *extra: str) -> list[str]:
    return [
        "panel",
        "run",
        "--personas",
        personas_path,
        "--instrument",
        instrument_path,
        *extra,
    ]


# ── Parser-level: flag is registered ──────────────────────────────────


class TestParserRegistration:
    def test_calibrate_against_parses_dataset_question(self):
        parser = build_parser()
        args = parser.parse_args(
            ["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml", "--calibrate-against", "gss:HAPPY"]
        )
        assert args.calibrate_against == "gss:HAPPY"

    def test_calibrate_against_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert getattr(args, "calibrate_against", "MISSING") is None

    def test_help_text_names_gss_and_cadence_pairing(self, capsys):
        """Per S-gate OQ3, help must document the explicit pairing with
        ``--convergence-check-every`` so users know cadence is not implicit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["panel", "run", "--help"])
        help_out = capsys.readouterr().out
        # The flag itself is registered.
        assert "--calibrate-against" in help_out
        # GSS (case-insensitive) is documented as the v1 source.
        assert "gss" in help_out.lower()
        # Cadence pairing is documented per S-gate OQ3.
        assert "convergence-check-every" in help_out


# ── Allowlist constant is what the spec ratified ──────────────────────


class TestAllowlistConstant:
    def test_v1_allowlist_is_gss_and_ntia(self):
        assert set(_INLINE_CALIBRATION_ALLOWED) == {"gss", "ntia"}


# ── Format validation: must be DATASET:QUESTION (both non-empty) ──────


class TestFormatValidation:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_bare_dataset_no_colon_exits_2(self, mock_client_cls, mock_runtime_cls, capsys, panel_files):
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", "gss"))

        assert code == 2
        err = capsys.readouterr().err
        assert "DATASET:QUESTION" in err
        assert "colon-separated" in err
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_empty_question_after_colon_exits_2(self, mock_client_cls, mock_runtime_cls, capsys, panel_files):
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", "gss:"))

        assert code == 2
        assert "DATASET:QUESTION" in capsys.readouterr().err
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_empty_dataset_before_colon_exits_2(self, mock_client_cls, mock_runtime_cls, capsys, panel_files):
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", ":HAPPY"))

        assert code == 2
        assert "DATASET:QUESTION" in capsys.readouterr().err
        mock_runtime.run_turn.assert_not_called()


# ── Redistribution-tier gate ──────────────────────────────────────────


class TestRedistributionGate:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_gated_dataset_wvs_exits_1(self, mock_client_cls, mock_runtime_cls, capsys, panel_files, monkeypatch):
        # Make sure the override env var is NOT set for this test.
        monkeypatch.delenv("SYNTHBENCH_ALLOW_GATED", raising=False)
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", "wvs:FOO"))

        assert code == 1
        err = capsys.readouterr().err
        assert "inline-publishable" in err
        # Allowlist names appear in the message (sorted alphabetically).
        assert "gss" in err
        assert "ntia" in err
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_gated_override_env_lets_through(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        capsys,
        panel_files,
        monkeypatch,
    ):
        """SYNTHBENCH_ALLOW_GATED=1 (internal escape hatch) bypasses the
        allowlist and allows execution to proceed past the validation block."""
        monkeypatch.setenv("SYNTHBENCH_ALLOW_GATED", "1")
        # sp-ttwy: now that --calibrate-against sources the baseline fetch
        # here (single-fetch reuse), mock the loader so the gate-bypass
        # test doesn't try to hit a real WVS baseline that synthbench may
        # not ship.
        mock_loader.return_value = {
            "dataset": "wvs",
            "question_key": "FOO",
            "human_distribution": {"A": 0.5, "B": 0.5},
        }
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = MagicMock(text="answer", usage=None, tool_calls=[])
        mock_runtime_cls.return_value = mock_runtime

        # Downstream pipeline may explode against our half-mocked
        # environment — that's fine. We only assert validation passed.
        with contextlib.suppress(Exception):
            main(_panel_run_args(personas, instrument, "--calibrate-against", "wvs:FOO"))

        err = capsys.readouterr().err
        # The gate-failure message must NOT appear — env override applied.
        assert "inline-publishable" not in err
        # And the run progressed at least to the panelist loop (proof that
        # validation didn't short-circuit with a non-zero exit).
        mock_runtime.run_turn.assert_called()


# ── Conflict reconciliation against --convergence-baseline ────────────


class TestConvergenceBaselineConflict:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_different_specs_exit_2(self, mock_client_cls, mock_runtime_cls, capsys, panel_files):
        personas, instrument = panel_files
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(
            _panel_run_args(
                personas,
                instrument,
                "--calibrate-against",
                "gss:HAPPY",
                "--convergence-baseline",
                "gss:OTHER",
            )
        )

        assert code == 2
        err = capsys.readouterr().err
        assert "same DATASET:QUESTION" in err
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_matching_specs_pass_validation(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        capsys,
        panel_files,
    ):
        """Same spec on both flags is the documented composition path: no
        conflict, single baseline fetch downstream."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "HAPPY",
            "human_distribution": {
                "Very happy": 0.3,
                "Pretty happy": 0.5,
                "Not too happy": 0.2,
            },
        }
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = MagicMock(text="answer", usage=None, tool_calls=[])
        mock_runtime_cls.return_value = mock_runtime

        with contextlib.suppress(Exception):
            main(
                _panel_run_args(
                    personas,
                    instrument,
                    "--calibrate-against",
                    "gss:HAPPY",
                    "--convergence-baseline",
                    "gss:HAPPY",
                )
            )

        # Validation passed → loader was reached.
        assert "same DATASET:QUESTION" not in capsys.readouterr().err
        mock_loader.assert_called_once_with("gss:HAPPY")


# ── Auto-enable convergence tracking ──────────────────────────────────


class TestAutoEnableConvergence:
    @patch("synth_panel.cli.commands.identify_tracked_questions")
    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_calibrate_against_alone_triggers_convergence_path(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        mock_identify,
        capsys,
        panel_files,
    ):
        """``--calibrate-against`` with no other convergence flag must still
        enter the convergence-tracking branch (force-enables ``wants_convergence``).
        Proxy: ``identify_tracked_questions`` is only called inside the
        ``if wants_convergence:`` block — so its invocation proves the toggle
        flipped on."""
        personas, instrument = panel_files
        # Empty list = "no bounded questions found"; the branch exits early with
        # a warning, which is fine for our purpose: we just need to prove the
        # branch was entered without any other --convergence-* flags set.
        mock_identify.return_value = []
        mock_loader.return_value = None
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = MagicMock(text="answer", usage=None, tool_calls=[])
        mock_runtime_cls.return_value = mock_runtime

        with contextlib.suppress(Exception):
            main(_panel_run_args(personas, instrument, "--calibrate-against", "gss:HAPPY"))

        mock_identify.assert_called_once()
        # And the warning that fires when no bounded questions are found
        # confirms wants_convergence was True.
        assert "convergence tracking disabled" in capsys.readouterr().err


# ── T3: auto-derive wiring + single-fetch reuse + hard-fail paths ─────


class TestAutoDeriveWiring:
    """sp-ttwy (T3): schema auto-derivation at run time from the
    SynthBench baseline, single-fetch reuse with ``--convergence-baseline``,
    provenance labels, and the two hard-fail modes from structure.md §7.
    """

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_small_enum_baseline_auto_derives_schema(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        mock_parallel,
        capsys,
        panel_files,
    ):
        """Small-enum baseline + no ``--extract-schema`` → derives a
        pick_one schema, injects it into the extractor pipeline, and
        emits the sp-yaru-style stderr log line."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "HAPPY",
            "human_distribution": {
                "Very happy": 0.3,
                "Pretty happy": 0.5,
                "Not too happy": 0.2,
            },
        }
        mock_parallel.return_value = ([], {}, {})
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        with contextlib.suppress(Exception):
            main(_panel_run_args(personas, instrument, "--calibrate-against", "gss:HAPPY"))

        err = capsys.readouterr().err
        # Stderr log line in sp-yaru style.
        assert "[convergence] auto-derived pick_one schema" in err
        assert "gss:HAPPY" in err
        assert "3 options" in err
        # Schema was injected into the extractor path.
        assert mock_parallel.called, "run_panel_parallel was never invoked"
        injected = mock_parallel.call_args.kwargs.get("extract_schema")
        assert injected is not None
        assert injected["properties"]["choice"]["enum"] == [
            "Not too happy",
            "Pretty happy",
            "Very happy",
        ]

    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_likert_baseline_without_schema_hard_fails(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        capsys,
        panel_files,
    ):
        """Numeric-keyed (Likert-looking) baseline + no ``--extract-schema``
        → exit 1 with the Likert-specific error message per §7."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "LIKERT_Q",
            "human_distribution": {"1": 0.2, "2": 0.3, "3": 0.5},
        }
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", "gss:LIKERT_Q"))

        assert code == 1
        err = capsys.readouterr().err
        assert "cannot auto-derive for Likert" in err
        # Panelist loop never reached — R2 regression guard.
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_wide_enum_baseline_without_schema_hard_fails(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        capsys,
        panel_files,
    ):
        """>5-option baseline + no ``--extract-schema`` → exit 1 with the
        "max 5 for auto-derive" message per §7."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "BIG",
            "human_distribution": {
                "A": 0.1,
                "B": 0.1,
                "C": 0.1,
                "D": 0.1,
                "E": 0.1,
                "F": 0.5,
            },
        }
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        code = main(_panel_run_args(personas, instrument, "--calibrate-against", "gss:BIG"))

        assert code == 1
        err = capsys.readouterr().err
        assert "max 5 for auto-derive" in err
        assert "6 options" in err
        mock_runtime.run_turn.assert_not_called()

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_user_supplied_extract_schema_bypasses_derivation(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        mock_parallel,
        capsys,
        tmp_path,
        panel_files,
    ):
        """User-supplied ``--extract-schema`` skips auto-derivation even
        when the baseline would otherwise trigger the >5-option hard-fail.
        The auto-derive log line must NOT fire; the user schema flows
        through verbatim."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "BIG",
            "human_distribution": {
                "A": 0.1,
                "B": 0.1,
                "C": 0.1,
                "D": 0.1,
                "E": 0.1,
                "F": 0.5,
            },
        }
        mock_parallel.return_value = ([], {}, {})
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        # Minimal user-supplied pick_one schema on disk.
        import json as _json

        user_schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]},
            },
            "required": ["choice"],
        }
        schema_path = tmp_path / "user_schema.json"
        schema_path.write_text(_json.dumps(user_schema))

        with contextlib.suppress(Exception):
            main(
                _panel_run_args(
                    personas,
                    instrument,
                    "--calibrate-against",
                    "gss:BIG",
                    "--extract-schema",
                    str(schema_path),
                )
            )

        err = capsys.readouterr().err
        assert "auto-derived pick_one schema" not in err
        assert "max 5 for auto-derive" not in err
        injected = mock_parallel.call_args.kwargs.get("extract_schema")
        assert injected == user_schema

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.load_synthbench_baseline")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_single_baseline_fetch_when_both_flags_present(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_loader,
        mock_parallel,
        capsys,
        panel_files,
    ):
        """Acceptance: ``Baseline fetched exactly once even when both
        convergence + calibrate flags present``. Also proves the payload
        is reused for auto-derivation (no second round-trip)."""
        personas, instrument = panel_files
        mock_loader.return_value = {
            "dataset": "gss",
            "question_key": "HAPPY",
            "human_distribution": {
                "Very happy": 0.3,
                "Pretty happy": 0.5,
                "Not too happy": 0.2,
            },
        }
        mock_parallel.return_value = ([], {}, {})
        mock_runtime = MagicMock()
        mock_runtime_cls.return_value = mock_runtime

        with contextlib.suppress(Exception):
            main(
                _panel_run_args(
                    personas,
                    instrument,
                    "--calibrate-against",
                    "gss:HAPPY",
                    "--convergence-baseline",
                    "gss:HAPPY",
                )
            )

        mock_loader.assert_called_once_with("gss:HAPPY")
        # And the derivation still fired (proof that the single fetched
        # payload was reused, not bypassed).
        assert "auto-derived pick_one schema" in capsys.readouterr().err
