"""CI gate against Pydantic minor-version drift in `model_json_schema()`.

Synthpanel's wire format is the JSON Schema dict at
``althing.structured.schemas`` — Pydantic adoption (v1.0.3 P1) must
not change what we emit on the wire. ``model_json_schema()`` output has
been observed to drift across Pydantic minors (e.g.
``additionalProperties`` defaults shifted 2.5→2.6), so we lock the
generated schema against the hand-written one structurally.

If this test fails after a Pydantic bump, EITHER update the static
schema in ``structured/schemas.py`` (with a v1.x bump if the change is
observable to MCP callers) OR pin Pydantic harder.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic>=2.7 is a v1.0.3 base dep (P1)")
models = pytest.importorskip(
    "althing.structured.models",
    reason="structured.models lands with v1.0.3 P1 (hq-e25n)",
)

from althing.structured.schemas import (
    ANNOTATED_CHOICE_SCHEMA,
    LIKERT_SCHEMA,
    PICK_ONE_SCHEMA,
    RANKING_SCHEMA,
    YES_NO_SCHEMA,
)

PickOne = models.PickOne
Likert = models.Likert
YesNo = models.YesNo
Ranking = models.Ranking
AnnotatedChoice = models.AnnotatedChoice


@pytest.mark.parametrize(
    "model,schema,name",
    [
        (PickOne, PICK_ONE_SCHEMA, "pick_one"),
        (Likert, LIKERT_SCHEMA, "likert"),
        (YesNo, YES_NO_SCHEMA, "yes_no"),
        (Ranking, RANKING_SCHEMA, "ranking"),
        (AnnotatedChoice, ANNOTATED_CHOICE_SCHEMA, "annotated_choice"),
    ],
)
def test_model_schema_matches_static_wire(model, schema, name):
    """Pydantic-generated schema must match the static wire schema structurally."""
    generated = model.model_json_schema()

    gen_props = set(generated.get("properties", {}).keys())
    wire_props = set(schema.get("properties", {}).keys())
    assert gen_props == wire_props, (
        f"{name}: property name drift — "
        f"generated={sorted(gen_props)} wire={sorted(wire_props)} "
        f"(extra-in-generated={sorted(gen_props - wire_props)}, "
        f"missing-in-generated={sorted(wire_props - gen_props)})"
    )

    gen_required = sorted(generated.get("required", []))
    wire_required = sorted(schema.get("required", []))
    assert gen_required == wire_required, (
        f"{name}: required-field drift — generated={gen_required} wire={wire_required}"
    )

    for prop, spec in schema["properties"].items():
        gen = generated["properties"].get(prop)
        assert gen is not None, f"{name}.{prop}: property missing in generated schema"
        wire_type = spec.get("type")
        gen_type = gen.get("type")
        assert gen_type == wire_type, f"{name}.{prop}: type drift — generated={gen_type!r} wire={wire_type!r}"


def test_pydantic_version_pinned():
    """Sanity: althing requires pydantic>=2.7 in the resolved env."""
    import pydantic

    parts = pydantic.VERSION.split(".", 2)
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 7), f"Synthpanel requires pydantic>=2.7; got {pydantic.VERSION}"
