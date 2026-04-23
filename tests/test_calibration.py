"""Tests for ``--calibrate-against`` parse-time validation (sp-a6jc).

T2 of sp-inline-calibration. These tests cover the CLI surface only:
flag registration, format validation, redistribution-tier gating, and
conflict reconciliation against ``--convergence-baseline``. The hard
contract is that every failure mode short-circuits BEFORE any LLM spend
— matching the sp-yaru "no panelist call before validation" pattern.

JSD computation, schema auto-derivation, and report attachment land in
later tickets (T3, T4); they are intentionally not exercised here.

T7 (sp-idqa) adds an ``@pytest.mark.acceptance`` end-to-end test that
runs a real 20-panelist panel against the live GSS HAPPY baseline. It
is skipped by default (CI opt-in via ``pytest -m acceptance``) and
requires both ``ANTHROPIC_API_KEY`` and a locally provisioned GSS
aggregate (see ``synthbench.datasets.gss`` for setup instructions).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
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


# ── T7: end-to-end acceptance against the live GSS HAPPY baseline ─────

# GSS HAPPY is the v1 demo wedge (design.md D-gate #2). The baseline
# is expected to carry exactly these three option strings verbatim;
# any divergence in casing or coding surfaces as an alignment error —
# that is the S-gate OQ1 contract working as designed.
_GSS_HAPPY_EXPECTED_OPTIONS = frozenset({"Very happy", "Pretty happy", "Not too happy"})


def _gss_happy_baseline_or_skip() -> dict:
    """Fetch the GSS HAPPY baseline or ``pytest.skip`` with a setup hint.

    The synthbench GSS adapter requires a locally provisioned aggregate
    CSV (see ``synthbench.datasets.gss._raise_setup_instructions``). On
    a fresh checkout the data is not present, so this test is skipped
    rather than failing — it is opt-in CI, not a core regression gate.
    """
    try:
        import synthbench  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("synthbench not installed; install synthpanel[convergence]")
    try:
        baseline = synthbench.load_convergence_baseline(dataset="gss", question_key="HAPPY")
    except Exception as exc:  # DatasetDownloadError, BaselineUnavailable, ...
        pytest.skip(f"GSS HAPPY baseline unavailable (live fetch failed): {exc}")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("human_distribution"), dict):
        pytest.skip(f"GSS HAPPY baseline has unexpected shape: {type(baseline).__name__}")
    return baseline


def _generate_personas_yaml(n: int) -> str:
    """Produce a minimal but diverse ``n``-persona YAML document.

    The personas do not need to be psychologically rich for this test —
    we only need enough variation that the panel produces a non-trivial
    distribution over the three happiness options.
    """
    archetypes = [
        ("content professional", 34, "Software engineer at a stable mid-size firm", ["reflective", "steady"]),
        ("burnt-out manager", 45, "Middle manager juggling too many priorities", ["stressed", "cynical"]),
        ("optimistic retiree", 68, "Recently retired teacher with strong community ties", ["cheerful", "grateful"]),
        ("struggling parent", 38, "Single parent of two, works two jobs", ["tired", "protective"]),
        ("enthusiastic student", 21, "Undergrad discovering their passion", ["curious", "energetic"]),
        ("lonely widow", 72, "Lost spouse last year, lives alone", ["withdrawn", "reminiscent"]),
        ("thriving entrepreneur", 41, "Running a profitable small business", ["driven", "confident"]),
        ("anxious job-seeker", 29, "Six months out of work", ["worried", "searching"]),
        ("stable teacher", 52, "Elementary teacher, loves their work", ["patient", "warm"]),
        ("restless artist", 36, "Freelance illustrator chasing bigger clients", ["creative", "moody"]),
    ]
    lines = ["personas:"]
    for i in range(n):
        label, age, bg, traits = archetypes[i % len(archetypes)]
        lines.append(f"  - name: Panelist {i + 1} ({label})")
        lines.append(f"    age: {age}")
        lines.append(f"    occupation: {label}")
        lines.append(f"    background: {bg}")
        lines.append("    personality_traits:")
        for t in traits:
            lines.append(f"      - {t}")
    return "\n".join(lines) + "\n"


# The instrument is "general-survey style" (broad opinion questions on
# work/tech/daily life) but includes one GSS-aligned happiness question
# with a bounded response_schema whose option strings match the GSS
# HAPPY baseline verbatim. This is required because the bundled
# ``general-survey`` pack has no bounded questions, and
# ``identify_tracked_questions`` (pre-derivation) would otherwise return
# empty and disable convergence tracking — see design.md:79-90 and
# docs/convergence.md's worked-example caveat.
_INSTRUMENT_YAML = """\
instrument:
  questions:
    - text: >
        Taken all together, how would you say things are these days — would
        you say that you are very happy, pretty happy, or not too happy?
      response_schema:
        type: pick_one
        options:
          - Very happy
          - Pretty happy
          - Not too happy
    - text: >
        How do you feel about the pace of change in the world around you?
      response_schema:
        type: text
"""


@pytest.mark.acceptance
def test_general_survey_gss_happy_end_to_end(tmp_path: Path) -> None:
    """End-to-end: 20-panelist run against live GSS HAPPY (T7 of sp-0ku0).

    Proves the full pipeline — ``--calibrate-against`` validation,
    baseline fetch, auto-derived pick_one schema, structured extraction,
    convergence tracking, and ``per_question[key].calibration`` wire
    format — works against real data with a real LLM.

    Skipped when ``ANTHROPIC_API_KEY`` is absent or the local GSS
    aggregate is not provisioned.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Fail fast if the baseline is unreachable rather than burning 20
    # panelist calls before we discover it.
    baseline = _gss_happy_baseline_or_skip()
    baseline_keys = set(baseline["human_distribution"].keys())

    personas_path = tmp_path / "personas.yaml"
    instrument_path = tmp_path / "instrument.yaml"
    personas_path.write_text(_generate_personas_yaml(20))
    instrument_path.write_text(_INSTRUMENT_YAML)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "synth_panel",
            "panel",
            "run",
            "--personas",
            str(personas_path),
            "--instrument",
            str(instrument_path),
            "--model",
            "haiku",
            "--calibrate-against",
            "gss:HAPPY",
            "--convergence-check-every",
            "10",
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    # (a) exit 0
    assert result.returncode == 0, (
        f"Panel run failed (rc={result.returncode})\nstderr: {result.stderr[-2000:]}\nstdout: {result.stdout[-500:]}"
    )

    # Auto-derive log line must have fired — pick_one:auto-derived is
    # only emitted when the baseline yielded a derived schema.
    assert "auto-derived pick_one schema" in result.stderr, (
        f"Expected auto-derive stderr log, got:\n{result.stderr[-2000:]}"
    )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Stdout was not valid JSON ({exc}):\n{result.stdout[:2000]}")

    convergence = payload.get("convergence")
    assert isinstance(convergence, dict), f"Expected convergence dict in output, got: {type(convergence).__name__}"

    per_question = convergence.get("per_question") or {}
    assert per_question, (
        "convergence.per_question is empty — the happiness question was not tracked. "
        f"tracked_questions={convergence.get('tracked_questions')!r}"
    )

    # Locate the tracked question that carries the calibration sub-object.
    # There is exactly one bounded question in the instrument, so we
    # take the single entry rather than guessing at the key format.
    calibration_entries = {k: v for k, v in per_question.items() if isinstance(v, dict) and "calibration" in v}
    assert calibration_entries, (
        f"No tracked question carries a calibration sub-object. per_question keys: {list(per_question.keys())!r}"
    )
    _happy_key, entry = next(iter(calibration_entries.items()))
    calibration = entry["calibration"]

    # (e) If alignment_error is present, fail LOUD with the exact diff.
    # This is the S-gate OQ1 contract: verbatim match is the spec; any
    # mismatch between the extractor's category vocabulary and the
    # baseline's option strings must surface immediately so the test
    # catches GSS format drift rather than papering over it.
    if "alignment_error" in calibration:
        pytest.fail(
            "GSS HAPPY extractor/baseline option-string alignment broke "
            "(S-gate OQ1 contract): calibration.alignment_error = "
            f"{calibration['alignment_error']!r}. Expected baseline keys "
            f"{sorted(_GSS_HAPPY_EXPECTED_OPTIONS)!r}; synthbench returned "
            f"{sorted(baseline_keys)!r}. This test is working as designed — "
            "fix the option-string alignment or update the expected set."
        )

    # (b) jsd exists and is a float in [0, 1]
    jsd = calibration.get("jsd")
    assert isinstance(jsd, (int, float)), f"calibration.jsd must be numeric, got {type(jsd).__name__}: {jsd!r}"
    assert 0.0 <= float(jsd) <= 1.0, f"calibration.jsd out of [0, 1]: {jsd!r}"

    # (c) extractor provenance
    assert calibration.get("extractor") == "pick_one:auto-derived", (
        f"Expected extractor='pick_one:auto-derived', got {calibration.get('extractor')!r}"
    )

    # (d) auto_derived flag
    assert calibration.get("auto_derived") is True, (
        f"Expected auto_derived=True, got {calibration.get('auto_derived')!r}"
    )

    # Provenance cross-check: baseline_spec round-trips verbatim.
    assert calibration.get("baseline_spec") == "gss:HAPPY", (
        f"Expected baseline_spec='gss:HAPPY', got {calibration.get('baseline_spec')!r}"
    )

    # Soft confirmation of the fixture expectation recorded in the bead:
    # if GSS ever ships different option strings, the alignment_error
    # branch above would have already fired — this is belt-and-braces.
    assert baseline_keys == _GSS_HAPPY_EXPECTED_OPTIONS, (
        f"GSS HAPPY baseline keys drifted: expected {sorted(_GSS_HAPPY_EXPECTED_OPTIONS)!r}, "
        f"got {sorted(baseline_keys)!r}. Update _GSS_HAPPY_EXPECTED_OPTIONS if intentional."
    )
