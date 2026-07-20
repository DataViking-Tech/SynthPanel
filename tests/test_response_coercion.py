"""sy-547: post-hoc coercion of free-text answers to a typed response_schema.

Covers the enum/scale mapping rules in
:mod:`althing.response_coercion`:

* ``"Blue."`` → ``"blue"`` (case + trailing punctuation normalized).
* Unmappable / ambiguous answers report ``mapped=False`` (no fabrication).
* Scale answers coerce to an in-range integer, reject out-of-range.
* Non-typed schemas / non-string raw values return ``None`` (caller skips).
"""

from __future__ import annotations

from althing.response_coercion import (
    CoercionResult,
    coerce_enum,
    coerce_response,
    coerce_scale,
    is_typed_schema,
)


class TestIsTypedSchema:
    def test_enum_and_scale_are_typed(self) -> None:
        assert is_typed_schema({"type": "enum", "options": ["a"]})
        assert is_typed_schema({"type": "scale", "min": 1, "max": 5})

    def test_text_and_tagged_and_legacy_are_not(self) -> None:
        assert not is_typed_schema({"type": "text"})
        assert not is_typed_schema({"type": "tagged_themes", "taxonomy": ["x"]})
        assert not is_typed_schema({"type": "object", "properties": {}})
        assert not is_typed_schema(None)
        assert not is_typed_schema("nope")


class TestCoerceEnum:
    OPTIONS = ["red", "green", "blue"]

    def test_repro_blue_period_maps_to_blue(self) -> None:
        # The exact #547 repro: model answers "Blue." for a lowercase enum.
        result = coerce_enum("Blue.", self.OPTIONS)
        assert result == CoercionResult(kind="enum", raw="Blue.", value="blue", mapped=True)

    def test_exact_case_insensitive(self) -> None:
        assert coerce_enum("GREEN", self.OPTIONS).value == "green"

    def test_substring_in_prose(self) -> None:
        result = coerce_enum("I would definitely pick green here", self.OPTIONS)
        assert result.value == "green"
        assert result.mapped

    def test_word_boundary_avoids_false_substring(self) -> None:
        # "red" must not match inside "predisposed".
        result = coerce_enum("I am predisposed to neither", self.OPTIONS)
        assert not result.mapped
        assert result.value is None

    def test_ambiguous_answer_does_not_map(self) -> None:
        # Mentions two options — refuse to guess.
        result = coerce_enum("between red and blue", self.OPTIONS)
        assert not result.mapped
        assert result.value is None

    def test_unmappable_answer(self) -> None:
        result = coerce_enum("Maybe a nice teal?", self.OPTIONS)
        assert not result.mapped

    def test_multiword_option_matches(self) -> None:
        opts = ["price band a", "price band b"]
        assert coerce_enum("I'd choose Price Band A.", opts).value == "price band a"


class TestCoerceScale:
    def test_in_range_integer(self) -> None:
        assert coerce_scale("7", 1, 10).value == 7

    def test_integer_in_prose(self) -> None:
        result = coerce_scale("I'd say 7 out of 10", 1, 10)
        assert result.value == 7
        assert result.mapped

    def test_out_of_range_does_not_map(self) -> None:
        result = coerce_scale("42", 1, 10)
        assert not result.mapped
        assert result.value is None

    def test_no_number_does_not_map(self) -> None:
        assert not coerce_scale("eleven", 1, 10).mapped


class TestCoerceResponse:
    def test_enum_dispatch(self) -> None:
        schema = {"type": "enum", "options": ["red", "green", "blue"]}
        assert coerce_response(schema, "Blue.").value == "blue"

    def test_scale_dispatch(self) -> None:
        schema = {"type": "scale", "min": 1, "max": 5}
        assert coerce_response(schema, "3").value == 3

    def test_non_typed_schema_returns_none(self) -> None:
        assert coerce_response({"type": "text"}, "anything") is None

    def test_non_string_raw_returns_none(self) -> None:
        schema = {"type": "enum", "options": ["a"]}
        assert coerce_response(schema, None) is None
        assert coerce_response(schema, {"a": 1}) is None
        assert coerce_response(schema, "   ") is None
