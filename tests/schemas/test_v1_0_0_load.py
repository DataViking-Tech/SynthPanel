"""Test the v1.0.0 schema loader.

The v1.0.0 contract is FROZEN: ``load(version="1.0.0")`` returns an
immutable mapping that callers cannot mutate. New contract versions ship
as parallel files alongside ``v1.0.0.json``; this file is append-only.
"""

from __future__ import annotations

import pytest

from synth_panel.schemas import load


def test_schema_loads_and_is_frozen() -> None:
    schema = load(version="1.0.0")

    assert schema["schema_version"] == "1.0.0"
    for key in (
        "request_fragments",
        "tools",
        "panel_verdict",
        "error_envelope",
        "flags_enum",
    ):
        assert key in schema, f"v1.0.0 schema missing top-level key: {key}"

    flag_codes = list(schema["flags_enum"]["enum"])
    assert sorted(flag_codes) == sorted(
        [
            "low_convergence",
            "demographic_skew",
            "small_n",
            "persona_collision",
            "out_of_distribution",
            "refusal_or_degenerate",
            "schema_drift",
        ]
    )

    decision = schema["request_fragments"]["decision_being_informed"]
    assert decision["minLength"] == 12
    assert decision["maxLength"] == 280

    with pytest.raises(TypeError):
        schema["panel_verdict"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        schema["request_fragments"]["decision_being_informed"]["minLength"] = 0  # type: ignore[index]
