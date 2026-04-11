"""Tests for bundled extraction schema registry."""

from __future__ import annotations

import pytest

from synth_panel.structured.schemas import (
    LIKERT_SCHEMA,
    PICK_ONE_SCHEMA,
    RANKING_SCHEMA,
    YES_NO_SCHEMA,
    SchemaNotFoundError,
    get_schema,
    is_known_schema,
    list_schemas,
)


class TestGetSchema:
    def test_ranking(self):
        assert get_schema("ranking") is RANKING_SCHEMA
        assert "ranked" in get_schema("ranking")["properties"]

    def test_likert(self):
        assert get_schema("likert") is LIKERT_SCHEMA
        assert "rating" in get_schema("likert")["properties"]

    def test_yes_no(self):
        assert get_schema("yes_no") is YES_NO_SCHEMA
        assert "answer" in get_schema("yes_no")["properties"]

    def test_pick_one(self):
        assert get_schema("pick_one") is PICK_ONE_SCHEMA
        assert "choice" in get_schema("pick_one")["properties"]

    def test_unknown_raises(self):
        with pytest.raises(SchemaNotFoundError, match="nonexistent"):
            get_schema("nonexistent")

    def test_error_lists_known_schemas(self):
        with pytest.raises(SchemaNotFoundError, match="likert") as exc_info:
            get_schema("nope")
        assert exc_info.value.name == "nope"


class TestListSchemas:
    def test_returns_all_four(self):
        schemas = list_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"ranking", "likert", "yes_no", "pick_one"}

    def test_entries_have_schema_key(self):
        for entry in list_schemas():
            assert "schema" in entry
            assert isinstance(entry["schema"], dict)


class TestIsKnownSchema:
    def test_known(self):
        assert is_known_schema("ranking") is True
        assert is_known_schema("likert") is True
        assert is_known_schema("yes_no") is True
        assert is_known_schema("pick_one") is True

    def test_unknown(self):
        assert is_known_schema("nonexistent") is False


class TestSchemaStructure:
    """Verify each schema is a valid JSON Schema object with required fields."""

    @pytest.mark.parametrize("name", ["ranking", "likert", "yes_no", "pick_one"])
    def test_is_object_type(self, name: str):
        schema = get_schema(name)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_ranking_items_structure(self):
        items = RANKING_SCHEMA["properties"]["ranked"]["items"]
        assert "name" in items["properties"]
        assert "rank" in items["properties"]

    def test_likert_rating_is_integer(self):
        assert LIKERT_SCHEMA["properties"]["rating"]["type"] == "integer"

    def test_yes_no_answer_is_boolean(self):
        assert YES_NO_SCHEMA["properties"]["answer"]["type"] == "boolean"

    def test_pick_one_choice_is_string(self):
        assert PICK_ONE_SCHEMA["properties"]["choice"]["type"] == "string"
