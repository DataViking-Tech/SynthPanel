"""Tests for sp-2hpi analysis scaffolding: response_schema validation +
deterministic distributions + subgroup breakdowns.

The analysis pipeline must be deterministic: the same inputs always
produce the same output shape and counts, because downstream layers
(narrative-over-stats, structured-mode output, --analysis-mode auto)
will rely on byte-stable aggregation in tests and CI.
"""

from __future__ import annotations

import pytest

from synth_panel.analysis import (
    InvalidResponseSchemaError,
    UnknownPersonaFieldError,
    coerce_enum_value,
    coerce_scale_value,
    coerce_tagged_themes,
    distribution_for_question,
    subgroup_breakdown,
)
from synth_panel.instrument import InstrumentError, parse_instrument

# ---------------------------------------------------------------------------
# response_schema validation in the instrument parser
# ---------------------------------------------------------------------------


class TestResponseSchemaValidation:
    def test_scale_schema_accepted(self):
        data = {"questions": [{"text": "Rate it", "response_schema": {"type": "scale", "min": 0, "max": 10}}]}
        inst = parse_instrument(data)
        assert inst.questions[0]["response_schema"]["type"] == "scale"

    def test_scale_min_not_int_raises(self):
        data = {"questions": [{"text": "q", "response_schema": {"type": "scale", "min": "0", "max": 10}}]}
        with pytest.raises(InstrumentError, match="'min' must be an integer"):
            parse_instrument(data)

    def test_scale_min_gte_max_raises(self):
        data = {"questions": [{"text": "q", "response_schema": {"type": "scale", "min": 5, "max": 5}}]}
        with pytest.raises(InstrumentError, match="strictly less than"):
            parse_instrument(data)

    def test_enum_schema_accepted(self):
        data = {
            "questions": [
                {
                    "text": "Pick",
                    "response_schema": {"type": "enum", "options": ["a", "b", "c"]},
                }
            ]
        }
        inst = parse_instrument(data)
        assert inst.questions[0]["response_schema"]["options"] == ["a", "b", "c"]

    def test_enum_empty_options_raises(self):
        data = {"questions": [{"text": "q", "response_schema": {"type": "enum", "options": []}}]}
        with pytest.raises(InstrumentError, match="non-empty list"):
            parse_instrument(data)

    def test_enum_duplicate_options_raises(self):
        data = {
            "questions": [
                {
                    "text": "q",
                    "response_schema": {"type": "enum", "options": ["a", "a"]},
                }
            ]
        }
        with pytest.raises(InstrumentError, match="must be unique"):
            parse_instrument(data)

    def test_tagged_themes_accepted(self):
        data = {
            "questions": [
                {
                    "text": "Tag",
                    "response_schema": {
                        "type": "tagged_themes",
                        "taxonomy": ["price", "quality"],
                        "multi": True,
                    },
                }
            ]
        }
        inst = parse_instrument(data)
        assert inst.questions[0]["response_schema"]["multi"] is True

    def test_tagged_themes_missing_taxonomy_raises(self):
        data = {"questions": [{"text": "q", "response_schema": {"type": "tagged_themes"}}]}
        with pytest.raises(InstrumentError, match="'taxonomy'"):
            parse_instrument(data)

    def test_text_max_tokens_zero_raises(self):
        data = {"questions": [{"text": "q", "response_schema": {"type": "text", "max_tokens": 0}}]}
        with pytest.raises(InstrumentError, match="positive integer"):
            parse_instrument(data)

    def test_response_schema_not_dict_raises(self):
        data = {"questions": [{"text": "q", "response_schema": "scale"}]}
        with pytest.raises(InstrumentError, match="must be a mapping"):
            parse_instrument(data)

    def test_legacy_inline_schema_passthrough(self):
        """Dicts without a recognized 'type' are accepted unchanged for
        backward compatibility with inline JSON Schema usage."""
        data = {
            "questions": [
                {
                    "text": "q",
                    "response_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                    },
                }
            ]
        }
        inst = parse_instrument(data)
        assert inst.questions[0]["response_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


class TestCoerceScale:
    def test_in_range_int(self):
        assert coerce_scale_value(7, 0, 10) == 7

    def test_float_rounds(self):
        assert coerce_scale_value(7.4, 0, 10) == 7
        assert coerce_scale_value(7.6, 0, 10) == 8

    def test_numeric_string(self):
        assert coerce_scale_value("8", 0, 10) == 8
        assert coerce_scale_value("  8.2 ", 0, 10) == 8

    def test_out_of_range_returns_none(self):
        assert coerce_scale_value(-1, 0, 10) is None
        assert coerce_scale_value(11, 0, 10) is None

    def test_bool_rejected(self):
        assert coerce_scale_value(True, 0, 10) is None

    def test_unparseable_string(self):
        assert coerce_scale_value("eight", 0, 10) is None
        assert coerce_scale_value("", 0, 10) is None


class TestCoerceEnum:
    def test_exact_match(self):
        assert coerce_enum_value("yes", ["yes", "no"]) == "yes"

    def test_case_insensitive_returns_canonical(self):
        assert coerce_enum_value("YES", ["yes", "no"]) == "yes"
        assert coerce_enum_value("  No ", ["yes", "no"]) == "no"

    def test_unknown_returns_none(self):
        assert coerce_enum_value("maybe", ["yes", "no"]) is None
        assert coerce_enum_value("", ["yes", "no"]) is None

    def test_non_string_returns_none(self):
        assert coerce_enum_value(42, ["yes", "no"]) is None


class TestCoerceTaggedThemes:
    def test_comma_separated_string(self):
        in_tax, off_tax = coerce_tagged_themes("price, quality, support", ["price", "quality"], multi=True)
        assert in_tax == ["price", "quality"]
        assert off_tax == ["support"]

    def test_list_input(self):
        in_tax, off_tax = coerce_tagged_themes(["Price", "Unknown"], ["price", "quality"], multi=True)
        assert in_tax == ["price"]
        assert off_tax == ["Unknown"]

    def test_multi_false_keeps_only_first(self):
        in_tax, off_tax = coerce_tagged_themes("price, quality", ["price", "quality"], multi=False)
        assert in_tax == ["price"]
        assert off_tax == []

    def test_empty_returns_none(self):
        assert coerce_tagged_themes("", ["price"], multi=True) is None
        assert coerce_tagged_themes(None, ["price"], multi=True) is None


# ---------------------------------------------------------------------------
# distribution_for_question — scale
# ---------------------------------------------------------------------------


class TestScaleDistribution:
    def test_scale_distribution_computed_deterministically(self):
        schema = {"type": "scale", "min": 0, "max": 3}
        responses = [0, 1, 1, 2, 3, 3, 3]
        out = distribution_for_question(responses, schema)
        assert out["type"] == "scale"
        assert out["n"] == 7
        assert out["n_valid"] == 7
        assert out["n_invalid"] == 0
        assert out["distribution"] == [
            {"value": 0, "count": 1},
            {"value": 1, "count": 2},
            {"value": 2, "count": 1},
            {"value": 3, "count": 3},
        ]
        assert out["stats"]["mean"] == pytest.approx(13 / 7, abs=1e-4)
        assert out["stats"]["median"] == 2.0
        assert out["stats"]["observed_min"] == 0
        assert out["stats"]["observed_max"] == 3

    def test_scale_deterministic_repeats(self):
        schema = {"type": "scale", "min": 1, "max": 5}
        responses = [5, 4, "3", 2.7, None, "x"]
        first = distribution_for_question(responses, schema)
        second = distribution_for_question(responses, schema)
        assert first == second
        assert first["n_invalid"] == 2  # None and "x"
        assert first["n_valid"] == 4

    def test_scale_all_invalid_omits_stats_body(self):
        schema = {"type": "scale", "min": 0, "max": 10}
        out = distribution_for_question(["nope", None, False], schema)
        assert out["n_valid"] == 0
        assert "mean" not in out["stats"]


# ---------------------------------------------------------------------------
# distribution_for_question — enum
# ---------------------------------------------------------------------------


class TestEnumDistribution:
    def test_enum_frequencies_stable(self):
        schema = {"type": "enum", "options": ["yes", "no", "maybe"]}
        responses = ["yes", "YES", "no", "maybe", "maybe", "unknown"]
        out = distribution_for_question(responses, schema)
        assert out["type"] == "enum"
        assert out["distribution"] == [
            {"option": "yes", "count": 2},
            {"option": "no", "count": 1},
            {"option": "maybe", "count": 2},
        ]
        assert out["n_valid"] == 5
        assert out["n_invalid"] == 1
        assert out["top_option"] in {"yes", "maybe"}  # tie — first in declaration wins
        assert out["top_option"] == "yes"

    def test_enum_zero_filled_when_no_responses_match(self):
        schema = {"type": "enum", "options": ["a", "b"]}
        out = distribution_for_question(["c", "d"], schema)
        assert out["distribution"] == [
            {"option": "a", "count": 0},
            {"option": "b", "count": 0},
        ]
        assert out["top_option"] is None


# ---------------------------------------------------------------------------
# distribution_for_question — tagged_themes
# ---------------------------------------------------------------------------


class TestTaggedThemesAggregation:
    def test_tagged_themes_aggregation(self):
        schema = {
            "type": "tagged_themes",
            "taxonomy": ["price", "quality", "support"],
            "multi": True,
        }
        responses = [
            "price, quality",
            ["support", "other"],
            "QUALITY",
            None,
            "",
        ]
        out = distribution_for_question(responses, schema)
        assert out["type"] == "tagged_themes"
        assert out["distribution"] == [
            {"theme": "price", "count": 1},
            {"theme": "quality", "count": 2},
            {"theme": "support", "count": 1},
        ]
        assert out["off_taxonomy"] == [{"theme": "other", "count": 1}]
        assert out["n_valid"] == 3
        assert out["total_mentions"] == 5

    def test_tagged_themes_single_tag_when_multi_false(self):
        schema = {
            "type": "tagged_themes",
            "taxonomy": ["a", "b"],
            "multi": False,
        }
        out = distribution_for_question(["a, b", "b, a"], schema)
        # Only the first tag is counted per response
        assert out["distribution"] == [
            {"theme": "a", "count": 1},
            {"theme": "b", "count": 1},
        ]


# ---------------------------------------------------------------------------
# distribution_for_question — text & error paths
# ---------------------------------------------------------------------------


class TestTextDistribution:
    def test_text_counts_nonempty(self):
        schema = {"type": "text"}
        out = distribution_for_question(["hi", "", "  ", "there"], schema)
        assert out == {"type": "text", "n": 4, "n_valid": 2, "n_invalid": 2}


class TestInvalidSchema:
    def test_unknown_type_raises(self):
        with pytest.raises(InvalidResponseSchemaError):
            distribution_for_question([], {"type": "matrix"})

    def test_missing_type_raises(self):
        with pytest.raises(InvalidResponseSchemaError):
            distribution_for_question([], {})


# ---------------------------------------------------------------------------
# subgroup_breakdown
# ---------------------------------------------------------------------------


class TestSubgroupBreakdown:
    def test_subgroup_breakdown_by_pack(self):
        schema = {"type": "enum", "options": ["yes", "no"]}
        personas = [
            {"name": "a", "pack": "alpha"},
            {"name": "b", "pack": "alpha"},
            {"name": "c", "pack": "beta"},
        ]
        responses = ["yes", "no", "yes"]
        out = subgroup_breakdown(responses, personas, field="pack", response_schema=schema)
        assert out["field"] == "pack"
        assert out["n_buckets"] == 2
        # Alphabetical ordering of named buckets
        assert [b["label"] for b in out["buckets"]] == ["alpha", "beta"]
        alpha = out["buckets"][0]
        assert alpha["n"] == 2
        assert alpha["distribution"]["distribution"] == [
            {"option": "yes", "count": 1},
            {"option": "no", "count": 1},
        ]

    def test_age_bands(self):
        schema = {"type": "scale", "min": 0, "max": 10}
        personas = [
            {"age": 22},
            {"age": 35},
            {"age": 41},
            {"age": 70},  # out of band — goes to missing
        ]
        responses = [8, 7, 9, 5]
        out = subgroup_breakdown(
            responses,
            personas,
            field="age",
            response_schema=schema,
            age_bands=[(18, 29), (30, 44)],
        )
        labels = [b["label"] for b in out["buckets"]]
        assert labels == ["18-29", "30-44", "unknown"]
        assert out["buckets"][0]["n"] == 1  # 22
        assert out["buckets"][1]["n"] == 2  # 35, 41
        assert out["buckets"][2]["n"] == 1  # 70

    def test_missing_field_bucket(self):
        schema = {"type": "text"}
        personas = [{"pack": "x"}, {}, {"pack": "y"}]
        responses = ["a", "b", "c"]
        out = subgroup_breakdown(responses, personas, field="pack", response_schema=schema)
        labels = [b["label"] for b in out["buckets"]]
        assert "unknown" in labels
        assert labels[-1] == "unknown"  # missing_label always last

    def test_no_persona_has_field_raises(self):
        schema = {"type": "text"}
        with pytest.raises(UnknownPersonaFieldError):
            subgroup_breakdown(["a"], [{"name": "p"}], field="pack", response_schema=schema)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            subgroup_breakdown(
                ["a"],
                [{"pack": "x"}, {"pack": "y"}],
                field="pack",
                response_schema={"type": "text"},
            )
