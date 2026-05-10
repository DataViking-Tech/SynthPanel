"""Tests for the typed Pydantic mirror of the bundled extraction schemas.

Covers v1.0.3 P1 (hq-e25n): the ``models.py`` registry, the
``resolve_extract_schema`` dispatch (str / dict / BaseModel / None),
and that the schemas.py side correctly re-exports MODEL_REGISTRY.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from synth_panel._runners import resolve_extract_schema
from synth_panel.structured.models import (
    MODEL_REGISTRY,
    AnnotatedChoice,
    Likert,
    PickOne,
    Ranking,
    YesNo,
)
from synth_panel.structured.schemas import (
    ANNOTATED_CHOICE_SCHEMA,
    LIKERT_SCHEMA,
    PICK_ONE_SCHEMA,
    YES_NO_SCHEMA,
)
from synth_panel.structured.schemas import MODEL_REGISTRY as SCHEMAS_MODEL_REGISTRY


class TestModelRegistry:
    def test_re_export_is_same_object(self):
        """``schemas.MODEL_REGISTRY`` and ``models.MODEL_REGISTRY`` are the
        same dict — schemas.py re-exports models.py rather than building
        a parallel mapping that could drift."""
        assert SCHEMAS_MODEL_REGISTRY is MODEL_REGISTRY

    def test_registry_covers_all_bundled_schemas(self):
        """The 5 names in the bundled JSON Schema registry must each
        have a typed Pydantic mirror — otherwise dispatch on a string
        name silently produces ``model=None``."""
        assert set(MODEL_REGISTRY) == {
            "ranking",
            "likert",
            "yes_no",
            "pick_one",
            "annotated_choice",
        }

    def test_registry_values_are_basemodel_subclasses(self):
        for name, cls in MODEL_REGISTRY.items():
            assert issubclass(cls, BaseModel), f"{name} → {cls!r}"


class TestPickOne:
    def test_minimal(self):
        m = PickOne(choice="A")
        assert m.choice == "A"
        assert m.reasoning == ""

    def test_empty_choice_rejected(self):
        with pytest.raises(ValidationError):
            PickOne(choice="")


class TestLikert:
    def test_in_range(self):
        m = Likert(rating=3)
        assert m.rating == 3

    def test_out_of_range_surfaces_field_path(self):
        """AC #2: a ``rating: 7`` Likert response surfaces a usable
        field-path error so callers can pinpoint the violation."""
        with pytest.raises(ValidationError) as exc_info:
            Likert(rating=7)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("rating",) for err in errors)

    def test_below_range_rejected(self):
        with pytest.raises(ValidationError):
            Likert(rating=0)


class TestYesNo:
    def test_true(self):
        assert YesNo(answer=True).answer is True

    def test_false(self):
        assert YesNo(answer=False).answer is False


class TestRanking:
    def test_well_formed(self):
        m = Ranking(ranked=[{"name": "A", "rank": 1}, {"name": "B", "rank": 2}])
        assert len(m.ranked) == 2
        assert m.ranked[0].name == "A"
        assert m.ranked[0].rank == 1

    def test_empty_ranked_rejected(self):
        with pytest.raises(ValidationError):
            Ranking(ranked=[])


class TestAnnotatedChoice:
    def test_without_attachment(self):
        """No attachment is represented by the empty string, not None —
        the wire contract types ``attachment_id`` as ``string`` and
        conveys "missing" via absence from the payload."""
        m = AnnotatedChoice(choice="A")
        assert m.attachment_id == ""

    def test_with_attachment(self):
        m = AnnotatedChoice(choice="A", attachment_id="att_123")
        assert m.attachment_id == "att_123"

    def test_validate_from_dict(self):
        """``model_validate`` is the canonical post-extraction parse
        path (the LLM's tool-use forcing returns dicts, not JSON
        strings, so we don't need ``model_validate_json`` here)."""
        m = AnnotatedChoice.model_validate({"choice": "B", "reasoning": "ok"})
        assert m.choice == "B"
        assert m.reasoning == "ok"


class TestModelJsonSchema:
    """``model_json_schema()`` produces a JSON Schema that names the
    same properties as the static ``*_SCHEMA`` dicts. The static
    schemas remain the wire/MCP contract (frozen) — these models may
    add stricter constraints (e.g. Likert 1..5, PickOne min_length=1)
    that don't appear in the static schemas."""

    @pytest.mark.parametrize(
        "model,static_schema",
        [
            (Likert, LIKERT_SCHEMA),
            (PickOne, PICK_ONE_SCHEMA),
            (YesNo, YES_NO_SCHEMA),
            (AnnotatedChoice, ANNOTATED_CHOICE_SCHEMA),
        ],
    )
    def test_object_type_matches(self, model, static_schema):
        gen = model.model_json_schema()
        assert gen["type"] == static_schema["type"]
        assert set(static_schema["properties"]) <= set(gen["properties"])

    def test_required_fields_match(self):
        """Required fields are the structurally-load-bearing claim of
        the wire contract — these *must* line up with the static
        schemas or the LLM tool-use forcing will diverge."""
        for name, model in MODEL_REGISTRY.items():
            if name == "ranking":
                # Ranking has nested item required fields; checked separately.
                continue
            from synth_panel.structured.schemas import get_schema

            static = get_schema(name)
            gen = model.model_json_schema()
            assert set(gen.get("required", [])) == set(static.get("required", [])), (
                f"required mismatch for {name}: gen={gen.get('required')} static={static.get('required')}"
            )


class TestResolveExtractSchema:
    def test_none_returns_none(self):
        assert resolve_extract_schema(None) is None

    def test_raw_dict_wraps_with_no_model(self):
        raw = {"type": "object", "properties": {"x": {"type": "string"}}}
        out = resolve_extract_schema(raw)
        assert out == {"schema": raw, "model": None}

    def test_pydantic_class_attaches_model(self):
        """AC: ``extract_schema=AnnotatedChoice`` yields the typed model
        and a generated wire schema, both self-consistent."""
        out = resolve_extract_schema(AnnotatedChoice)
        assert out is not None
        assert out["model"] is AnnotatedChoice
        assert out["schema"]["type"] == "object"
        assert "choice" in out["schema"]["properties"]

    def test_registered_name_with_pydantic_mirror(self):
        """A name in the bundled schema registry resolves to both the
        wire schema and the Pydantic class."""
        out = resolve_extract_schema("annotated_choice")
        assert out is not None
        assert out["model"] is AnnotatedChoice
        assert out["schema"] is ANNOTATED_CHOICE_SCHEMA

    def test_registered_name_without_pydantic_mirror(self):
        """``sentiment`` is in the analytic registry but not in
        MODEL_REGISTRY — model should resolve to None, schema still
        wired up correctly."""
        out = resolve_extract_schema("sentiment")
        assert out is not None
        assert out["model"] is None
        assert out["schema"]["properties"].keys() == {"sentiment", "confidence"}

    def test_unknown_name_raises_with_known_listed(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_extract_schema("not_a_thing")
        msg = str(exc_info.value)
        assert "not_a_thing" in msg
        # Both registries' names should be advertised in the error.
        assert "sentiment" in msg
        assert "annotated_choice" in msg

    def test_non_basemodel_class_rejected(self):
        class NotABaseModel:
            pass

        with pytest.raises(TypeError):
            resolve_extract_schema(NotABaseModel)  # type: ignore[arg-type]

    def test_int_rejected(self):
        with pytest.raises(TypeError):
            resolve_extract_schema(42)  # type: ignore[arg-type]
