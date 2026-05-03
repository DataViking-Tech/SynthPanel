"""AC-4: v1.0.x → v1.1.0 grace-period shim for ``decision_being_informed``.

The shim sits in front of :func:`synth_panel.structured.validate.validate_request`
on the MCP request path. Under the v1.0.x grace window, a panel-running call
that omits ``decision_being_informed`` is allowed through with a synthesized
placeholder and a loud ``W_DECISION_MISSING`` warning. Setting
``SYNTHPANEL_SCHEMA_MIN="1.1.0"`` flips the shim into hard-reject mode (no
synthesis, no warning — the underlying validator will reject).
"""

from __future__ import annotations

import logging

import pytest

from synth_panel.mcp.compat import (
    LEGACY_DECISION_PLACEHOLDER,
    W_DECISION_MISSING,
    apply_legacy_grace,
)
from synth_panel.structured.validate import validate_request


def test_v1_0_synthesizes_legacy_decision(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A panel-running call missing the field gets the placeholder + warning."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    payload: dict = {"stimulus": "ship at $49 or $79?"}

    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_panel", payload)

    assert out["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER
    assert out["decision_being_informed"] == "unspecified-legacy-call"
    assert any(W_DECISION_MISSING in rec.message for rec in caplog.records)
    assert validate_request("run_panel", out) is None


def test_input_payload_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shim must return a new dict — callers may share the input upstream."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    payload: dict = {"stimulus": "x"}
    out = apply_legacy_grace("run_panel", payload)
    assert "decision_being_informed" not in payload
    assert out is not payload


@pytest.mark.parametrize("tool", ["run_panel", "run_quick_poll", "extend_panel"])
def test_all_decision_tools_get_synthesis(tool: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    out = apply_legacy_grace(tool, {})
    assert out["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER


def test_present_decision_passes_through(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A real decision must not be overwritten and must not trigger the warning."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    real = "Should we ship the new pricing tier next quarter?"
    payload = {"decision_being_informed": real}

    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_panel", payload)

    assert out["decision_being_informed"] == real
    assert not any(W_DECISION_MISSING in rec.message for rec in caplog.records)


def test_run_prompt_never_synthesizes(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """``run_prompt`` is sub-decisional; the field must not be injected on it."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_prompt", {"prompt": "hello"})
    assert "decision_being_informed" not in out
    assert not any(W_DECISION_MISSING in rec.message for rec in caplog.records)


def test_unknown_tool_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim is not the validator — unknown tools defer to ``validate_request``."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    out = apply_legacy_grace("not_a_tool", {"x": 1})
    assert out == {"x": 1}


def test_empty_string_decision_is_synthesized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty/whitespace-only field is treated as missing under the grace window."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_panel", {"decision_being_informed": "   "})
    assert out["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER
    assert any(W_DECISION_MISSING in rec.message for rec in caplog.records)


def test_non_string_decision_is_synthesized(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Type-wrong decision is treated as missing — synthesize, let validator
    of the *new* payload pass; the legacy caller's malformed input is masked
    only under grace."""
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_panel", {"decision_being_informed": 42})
    assert out["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER


def test_schema_min_1_1_0_disables_synthesis(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """``SYNTHPANEL_SCHEMA_MIN="1.1.0"`` opts the host into v1.1 hard-reject."""
    monkeypatch.setenv("SYNTHPANEL_SCHEMA_MIN", "1.1.0")
    with caplog.at_level(logging.WARNING, logger="synth_panel.mcp.compat"):
        out = apply_legacy_grace("run_panel", {})
    assert "decision_being_informed" not in out
    assert not any(W_DECISION_MISSING in rec.message for rec in caplog.records)

    err = validate_request("run_panel", out)
    assert err is not None
    assert err.error_code == "MISSING_DECISION"


def test_schema_min_1_0_0_keeps_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``SYNTHPANEL_SCHEMA_MIN="1.0.0"`` is the same as unset."""
    monkeypatch.setenv("SYNTHPANEL_SCHEMA_MIN", "1.0.0")
    out = apply_legacy_grace("run_panel", {})
    assert out["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER


def test_schema_min_future_version_disables_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any min ≥ 1.1.0 disables the shim — including 2.x."""
    monkeypatch.setenv("SYNTHPANEL_SCHEMA_MIN", "2.0.0")
    out = apply_legacy_grace("run_panel", {})
    assert "decision_being_informed" not in out


def test_synthesized_placeholder_satisfies_v1_0_validator() -> None:
    """The placeholder must be a legal value under the v1.0.0 contract -
    12-280 chars, single-line, non-empty after trim - so the synthesized
    payload survives the validator that sits behind this shim."""
    err = validate_request("run_panel", {"decision_being_informed": LEGACY_DECISION_PLACEHOLDER})
    assert err is None
