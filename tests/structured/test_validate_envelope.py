"""Tests for the v1.0.0 validator-core (AC-2).

The validator returns ``None`` when input conforms to the frozen contract,
and a typed :class:`~synth_panel.structured.validate.ErrorEnvelope` (whose
shape mirrors ``error_envelope`` in ``schemas/v1.0.0.json``) on failure.
"""

from __future__ import annotations

import pytest

from synth_panel.structured.validate import (
    ErrorEnvelope,
    validate_request,
    validate_response,
)

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"


def _verdict(**overrides):
    base = {
        "headline": "Strong support for the new tier with pricing caveats.",
        "convergence": 0.72,
        "dissent_count": 1,
        "top_3_verbatims": [
            {"persona_id": "p1", "quote": "I'd switch immediately."},
        ],
        "flags": [{"code": "small_n", "severity": "warn"}],
        "extension": [],
        "full_transcript_uri": "file:///tmp/run.jsonl",
        "meta": {"decision_being_informed": _VALID_DECISION},
        "schema_version": "1.0.0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------


def test_missing_decision_returns_typed_error() -> None:
    err = validate_request("run_panel", {})
    assert err is not None
    assert err.error_code == "MISSING_DECISION"
    assert err.field_path == "decision_being_informed"
    assert err.schema_version == "1.0.0"
    assert err.retry_safe is True
    assert "decision_being_informed" in err.message


def test_valid_request_returns_none() -> None:
    assert validate_request("run_panel", {"decision_being_informed": _VALID_DECISION}) is None
    assert validate_request("run_quick_poll", {"decision_being_informed": _VALID_DECISION}) is None
    assert validate_request("extend_panel", {"decision_being_informed": _VALID_DECISION}) is None


def test_run_prompt_rejects_decision_field() -> None:
    err = validate_request("run_prompt", {"decision_being_informed": _VALID_DECISION})
    assert err is not None
    assert err.error_code == "INVALID_TOOL_ARG"
    assert err.field_path == "decision_being_informed"


def test_run_prompt_without_decision_passes() -> None:
    assert validate_request("run_prompt", {}) is None


def test_unknown_tool_rejected() -> None:
    err = validate_request("not_a_tool", {"decision_being_informed": _VALID_DECISION})
    assert err is not None
    assert err.error_code == "INVALID_TOOL_ARG"


def test_decision_too_long_returns_typed_error() -> None:
    err = validate_request("run_panel", {"decision_being_informed": "x" * 281})
    assert err is not None
    assert err.error_code == "DECISION_TOO_LONG"
    assert err.field_path == "decision_being_informed"


def test_decision_too_short_after_trim_returns_invalid_arg() -> None:
    err = validate_request("run_panel", {"decision_being_informed": "   short    "})
    assert err is not None
    assert err.error_code == "INVALID_TOOL_ARG"
    assert err.field_path == "decision_being_informed"


def test_whitespace_only_decision_treated_as_missing() -> None:
    err = validate_request("run_panel", {"decision_being_informed": "          "})
    assert err is not None
    assert err.error_code == "MISSING_DECISION"


def test_decision_with_newline_rejected() -> None:
    err = validate_request("run_panel", {"decision_being_informed": "Decide whether\nto ship."})
    assert err is not None
    assert err.error_code == "INVALID_TOOL_ARG"
    assert "newline" in err.message.lower()


def test_decision_must_be_string() -> None:
    err = validate_request("run_panel", {"decision_being_informed": 42})
    assert err is not None
    assert err.error_code == "INVALID_TOOL_ARG"


def test_envelope_to_dict_roundtrip() -> None:
    err = validate_request("run_panel", {})
    assert err is not None
    payload = err.to_dict()
    assert payload["error_code"] == "MISSING_DECISION"
    assert payload["schema_version"] == "1.0.0"
    assert payload["retry_safe"] is True
    assert payload["field_path"] == "decision_being_informed"


def test_envelope_omits_field_path_when_unset() -> None:
    env = ErrorEnvelope(
        error_code="INTERNAL_ERROR",
        message="boom",
        field_path=None,
        schema_version="1.0.0",
        retry_safe=False,
    )
    assert "field_path" not in env.to_dict()


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


def test_valid_panel_verdict_returns_none() -> None:
    assert validate_response(_verdict()) is None


def test_response_missing_required_field_is_schema_drift() -> None:
    artifact = _verdict()
    del artifact["headline"]
    err = validate_response(artifact)
    assert err is not None
    assert err.error_code == "SCHEMA_DRIFT"
    assert err.field_path == "headline"
    assert err.retry_safe is False


def test_response_wrong_schema_version_is_schema_drift() -> None:
    err = validate_response(_verdict(schema_version="0.12.0"))
    assert err is not None
    assert err.error_code == "SCHEMA_DRIFT"
    assert err.field_path == "schema_version"


@pytest.mark.parametrize("bad_code", ["totally_made_up", "", "Low_Convergence"])
def test_response_invalid_flag_code_returns_invalid_flag(bad_code: str) -> None:
    err = validate_response(_verdict(flags=[{"code": bad_code, "severity": "warn"}]))
    assert err is not None
    assert err.error_code == "INVALID_FLAG"
    assert err.field_path == "flags[0].code"


def test_response_invalid_flag_severity_returns_invalid_flag() -> None:
    err = validate_response(_verdict(flags=[{"code": "small_n", "severity": "critical"}]))
    assert err is not None
    assert err.error_code == "INVALID_FLAG"
    assert err.field_path == "flags[0].severity"


def test_response_meta_missing_decision_is_schema_drift() -> None:
    err = validate_response(_verdict(meta={}))
    assert err is not None
    assert err.error_code == "SCHEMA_DRIFT"
    assert err.field_path == "meta.decision_being_informed"


def test_response_non_dict_artifact_is_schema_drift() -> None:
    err = validate_response("not a dict")  # type: ignore[arg-type]
    assert err is not None
    assert err.error_code == "SCHEMA_DRIFT"
    assert err.retry_safe is False
