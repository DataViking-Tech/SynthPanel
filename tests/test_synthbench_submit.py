"""Tests for SynthBench opt-in submission (sp-ezz).

Covers:

* ``is_submittable`` gating on the convergence + calibration shape
* ``build_submission_payload`` transformation against canonical input
* Consent prompt, recording, and bypass via ``--yes`` / pre-recorded file
* HTTP transport via ``httpx.MockTransport``: success, server rejection,
  network error
* CLI parse-time validation: ``--submit-to-synthbench`` without
  ``--calibrate-against`` and without ``SYNTHBENCH_API_KEY`` both
  short-circuit before any LLM call

The tests deliberately avoid asserting against the *exact* SynthBench
schema fields beyond what the bead's contract pins down (config /
aggregate / per_question keys + the per-question distribution shape) so
a future SynthBench-side schema change does not need to ripple here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from synth_panel.synthbench_submit import (
    API_KEY_ENV,
    CONSENT_VERSION,
    build_submission_payload,
    consent_recorded,
    is_submittable,
    prompt_consent,
    record_consent,
    submit,
    submit_panel_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def isolated_consent(tmp_path, monkeypatch):
    """Redirect consent storage at a tmp path so tests do not touch ``$HOME``."""
    fake_dir = tmp_path / "synthpanel"
    fake_path = fake_dir / "synthbench-consent.json"
    monkeypatch.setattr("synth_panel.synthbench_submit.CONSENT_DIR", fake_dir)
    monkeypatch.setattr("synth_panel.synthbench_submit.CONSENT_PATH", fake_path)
    return fake_path


@pytest.fixture
def calibrated_extra():
    """A panel_extra dict that *is* submittable: convergence + calibration."""
    return {
        "run_invalid": False,
        "convergence": {
            "final_n": 100,
            "tracked_questions": ["q1"],
            "per_question": {
                "q1": {
                    "final_n": 100,
                    "converged_at": 60,
                    "support_size": 3,
                    "calibration": {
                        "jsd": 0.0123,
                        "baseline_spec": "gss:HAPPY",
                        "extractor": "pick_one:auto-derived",
                        "auto_derived": True,
                    },
                }
            },
        },
    }


@pytest.fixture
def baseline_payload():
    return {
        "human_distribution": {
            "very happy": 0.30,
            "pretty happy": 0.55,
            "not too happy": 0.15,
        }
    }


@pytest.fixture
def model_distributions():
    return {
        "q1": {
            "very happy": 0.32,
            "pretty happy": 0.50,
            "not too happy": 0.18,
        }
    }


# ---------------------------------------------------------------------------
# is_submittable
# ---------------------------------------------------------------------------


def test_is_submittable_happy_path(calibrated_extra):
    ok, reason = is_submittable(calibrated_extra)
    assert ok is True
    assert reason is None


def test_is_submittable_rejects_invalid_run(calibrated_extra):
    calibrated_extra["run_invalid"] = True
    ok, reason = is_submittable(calibrated_extra)
    assert ok is False
    assert "invalid" in reason


def test_is_submittable_rejects_missing_convergence():
    ok, reason = is_submittable({"run_invalid": False})
    assert ok is False
    assert "convergence" in reason


def test_is_submittable_rejects_no_calibration():
    extra = {
        "run_invalid": False,
        "convergence": {
            "final_n": 100,
            "per_question": {"q1": {"final_n": 100, "support_size": 0}},
        },
    }
    ok, reason = is_submittable(extra)
    assert ok is False
    assert "calibration" in reason


# ---------------------------------------------------------------------------
# build_submission_payload
# ---------------------------------------------------------------------------


def test_build_submission_payload_shape(calibrated_extra, baseline_payload, model_distributions):
    payload = build_submission_payload(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline_payload,
        model_distributions=model_distributions,
        panelist_model="claude-sonnet-4-6",
        instrument_name="happiness-probe",
        persona_pack_name="general-public",
    )
    assert payload["benchmark"] == "synthbench"
    assert payload["config"]["calibration_spec"] == "gss:HAPPY"
    assert payload["config"]["panelist_model"] == "claude-sonnet-4-6"
    assert payload["config"]["instrument"] == "happiness-probe"
    assert payload["config"]["persona_pack"] == "general-public"
    assert payload["config"]["client"] == "synthpanel"
    assert payload["config"]["n"] == 100

    q1 = payload["per_question"]["q1"]
    assert q1["jsd"] == pytest.approx(0.0123)
    assert q1["n"] == 100
    assert q1["extractor"] == "pick_one:auto-derived"
    assert q1["auto_derived"] is True
    # Distributions must sum to ~1.0 each (the server enforces this).
    assert sum(q1["model_distribution"].values()) == pytest.approx(1.0)
    assert sum(q1["human_distribution"].values()) == pytest.approx(1.0)
    assert payload["aggregate"]["mean_jsd"] == pytest.approx(0.0123)
    assert payload["aggregate"]["n"] == 100


def test_build_submission_payload_omits_questions_with_no_distribution(calibrated_extra, baseline_payload):
    # Model distribution missing for q1 → the question is dropped, not zero-filled.
    payload = build_submission_payload(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline_payload,
        model_distributions={},
        panelist_model=None,
        instrument_name=None,
        persona_pack_name=None,
    )
    assert payload["per_question"] == {}
    # Aggregate still carries n; mean_jsd is omitted when nothing scored.
    assert payload["aggregate"]["n"] == 100
    assert "mean_jsd" not in payload["aggregate"]


def test_build_submission_payload_threads_alignment_error(calibrated_extra, model_distributions):
    calibrated_extra["convergence"]["per_question"]["q1"]["calibration"]["alignment_error"] = "['a'] vs ['x']"
    payload = build_submission_payload(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload={"human_distribution": {"x": 1.0}},
        model_distributions=model_distributions,
        panelist_model=None,
        instrument_name=None,
        persona_pack_name=None,
    )
    assert payload["per_question"]["q1"]["alignment_error"] == "['a'] vs ['x']"


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_consent_not_recorded_by_default(isolated_consent):
    assert consent_recorded() is False


def test_record_consent_persists(isolated_consent):
    record_consent()
    assert consent_recorded() is True
    saved = json.loads(isolated_consent.read_text())
    assert saved["accepted"] is True
    assert saved["version"] == CONSENT_VERSION


def test_consent_corrupted_file_treated_as_absent(isolated_consent):
    isolated_consent.parent.mkdir(parents=True, exist_ok=True)
    isolated_consent.write_text("not json{")
    assert consent_recorded() is False


def test_prompt_consent_yes(isolated_consent, capsys):
    answered = prompt_consent(input_fn=lambda _: "y")
    assert answered is True
    err = capsys.readouterr().err
    assert "SynthBench" in err


def test_prompt_consent_default_is_no(isolated_consent):
    assert prompt_consent(input_fn=lambda _: "") is False
    assert prompt_consent(input_fn=lambda _: "n") is False


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def test_submit_success_returns_id_and_url():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "sub_abc123", "status": "validating"},
        )

    result = submit(
        {"benchmark": "synthbench"},
        api_key="sk-test",
        api_url="https://api.synthbench.test",
        client=_client(handler),
    )
    assert result.accepted is True
    assert result.submission_id == "sub_abc123"
    assert result.status == "validating"
    assert result.leaderboard_url == "https://synthbench.org/submit/sub_abc123"
    assert captured["url"] == "https://api.synthbench.test/submit"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["benchmark"] == "synthbench"


def test_submit_uses_server_provided_leaderboard_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "submission_id": "x",
                "status": "accepted",
                "leaderboard_url": "https://synthbench.org/lb/abc",
            },
        )

    result = submit({}, api_key="k", client=_client(handler))
    assert result.leaderboard_url == "https://synthbench.org/lb/abc"


def test_submit_surfaces_server_validation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": "per_question.q1.model_distribution does not sum to 1.0"},
        )

    result = submit({}, api_key="k", client=_client(handler))
    assert result.accepted is False
    assert result.status == "http_422"
    assert "does not sum to 1.0" in result.error


def test_submit_handles_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = submit({}, api_key="k", client=_client(handler))
    assert result.accepted is False
    assert result.status == "error"
    assert "connection refused" in result.error


def test_submit_handles_non_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    result = submit({}, api_key="k", client=_client(handler))
    assert result.accepted is False
    assert result.status == "http_500"


# ---------------------------------------------------------------------------
# submit_panel_result top-level entry
# ---------------------------------------------------------------------------


def test_submit_panel_result_missing_api_key_short_circuits(
    calibrated_extra, baseline_payload, model_distributions, monkeypatch, isolated_consent
):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    result = submit_panel_result(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline_payload,
        model_distributions=model_distributions,
    )
    assert result.accepted is False
    assert result.status == "missing_api_key"


def test_submit_panel_result_not_submittable_skips_network(
    baseline_payload, model_distributions, monkeypatch, isolated_consent
):
    monkeypatch.setenv(API_KEY_ENV, "sk-test")
    extra = {"run_invalid": True}
    result = submit_panel_result(
        panel_extra=extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline_payload,
        model_distributions=model_distributions,
    )
    assert result.accepted is False
    assert result.status == "not_submittable"


def test_submit_panel_result_skip_consent_bypasses_prompt(
    calibrated_extra, baseline_payload, model_distributions, monkeypatch, isolated_consent
):
    monkeypatch.setenv(API_KEY_ENV, "sk-test")
    posted = {}

    def handler(request: httpx.Request) -> httpx.Response:
        posted["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "sub_y", "status": "accepted"})

    result = submit_panel_result(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline_payload,
        model_distributions=model_distributions,
        panelist_model="sonnet",
        skip_consent=True,
        client=_client(handler),
    )
    assert result.accepted is True
    assert posted["body"]["config"]["calibration_spec"] == "gss:HAPPY"
    # skip_consent=True should NOT also write the consent file — that flag
    # is for CI; the file-based consent is for interactive users.
    assert consent_recorded() is False


def test_submit_panel_result_empty_payload_skips_post(calibrated_extra, monkeypatch, isolated_consent):
    monkeypatch.setenv(API_KEY_ENV, "sk-test")
    # Distributions empty → payload has zero per_question entries → skip POST.

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("submit() must not be called when payload is empty")

    result = submit_panel_result(
        panel_extra=calibrated_extra,
        calibration_spec="gss:HAPPY",
        baseline_payload=None,
        model_distributions={},
        skip_consent=True,
        client=_client(handler),
    )
    assert result.accepted is False
    assert result.status == "empty_payload"


# ---------------------------------------------------------------------------
# CLI parse-time validation
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_files(tmp_path):
    personas = tmp_path / "personas.yaml"
    personas.write_text(
        "personas:\n  - name: Sample\n    age: 30\n    occupation: Tester\n    background: 'A test persona.'\n"
    )
    instrument = tmp_path / "survey.yaml"
    instrument.write_text(
        "instrument:\n  version: 1\n  questions:\n    - text: 'A?'\n      response_schema: { type: text }\n"
    )
    return personas, instrument


def _run_cli(args, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "synth_panel", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )


def test_cli_submit_without_calibrate_against_fails_at_parse_time(panel_files):
    personas, instrument = panel_files
    proc = _run_cli(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(instrument),
            "--submit-to-synthbench",
            "--yes",
        ],
        env_extra={"SYNTHBENCH_API_KEY": "sk-test"},
    )
    assert proc.returncode == 2, proc.stderr
    assert "--submit-to-synthbench requires --calibrate-against" in proc.stderr


def test_cli_submit_without_api_key_fails_at_parse_time(panel_files):
    personas, instrument = panel_files
    proc = _run_cli(
        [
            "panel",
            "run",
            "--personas",
            str(personas),
            "--instrument",
            str(instrument),
            "--calibrate-against",
            "gss:HAPPY",
            "--submit-to-synthbench",
            "--yes",
        ],
    )
    assert proc.returncode == 2, proc.stderr
    assert "SYNTHBENCH_API_KEY" in proc.stderr
    assert "synthbench.org/account" in proc.stderr
