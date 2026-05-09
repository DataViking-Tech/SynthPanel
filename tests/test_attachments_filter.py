"""Tests for per-persona attachment stratification (hq-iczd)."""

from __future__ import annotations

import pytest

from synth_panel.attachments.filter import count_strata, filter_attachments
from synth_panel.routing import _evaluate_predicate


class TestEvaluatePredicateOpenFields:
    """Open-allowlist mode is what attachments use."""

    def test_missing_field_is_false_not_raise(self):
        # Open mode: persona without the trait → predicate False.
        assert not _evaluate_predicate(
            {"field": "device", "op": "equals", "value": "mobile"},
            {"name": "Sarah"},
            valid_fields=None,
        )

    def test_gte_numeric_coercion(self):
        assert _evaluate_predicate(
            {"field": "age", "op": "gte", "value": 25},
            {"age": 34},
            valid_fields=None,
        )
        assert not _evaluate_predicate(
            {"field": "age", "op": "gte", "value": 25},
            {"age": 22},
            valid_fields=None,
        )

    def test_gte_string_to_float_coerces(self):
        # Persona authored numbers as strings still works.
        assert _evaluate_predicate(
            {"field": "age", "op": "gte", "value": "25"},
            {"age": "34"},
            valid_fields=None,
        )

    def test_gte_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _evaluate_predicate(
                {"field": "occupation", "op": "gte", "value": 25},
                {"occupation": "Product Manager"},
                valid_fields=None,
            )

    def test_lte_numeric(self):
        assert _evaluate_predicate(
            {"field": "age", "op": "lte", "value": 40},
            {"age": 34},
            valid_fields=None,
        )
        assert not _evaluate_predicate(
            {"field": "age", "op": "lte", "value": 30},
            {"age": 34},
            valid_fields=None,
        )

    def test_in_with_list_target(self):
        # personality_traits is list-typed; "any element in value" wins.
        assert _evaluate_predicate(
            {"field": "traits", "op": "in", "value": ["analytical", "creative"]},
            {"traits": ["analytical", "pragmatic"]},
            valid_fields=None,
        )

    def test_equals_list_member(self):
        assert _evaluate_predicate(
            {"field": "traits", "op": "equals", "value": "analytical"},
            {"traits": ["analytical", "pragmatic"]},
            valid_fields=None,
        )

    def test_contains_substring_against_prose_trait(self):
        assert _evaluate_predicate(
            {"field": "occupation", "op": "contains", "value": "Product Manager"},
            {"occupation": "Senior Product Manager at SaaS co."},
            valid_fields=None,
        )


class TestFilterAttachments:
    def test_no_filter_means_pass_through(self):
        atts = [{"id": "ad1", "kind": "image"}, {"id": "ad2", "kind": "image"}]
        out = filter_attachments(atts, {"name": "Sarah", "device": "desktop"})
        assert out == atts

    def test_empty_filter_list_passes(self):
        atts = [{"id": "ad1", "filter": []}]
        assert filter_attachments(atts, {"device": "mobile"}) == atts

    def test_filter_match_includes_attachment(self):
        atts = [
            {
                "id": "mobile_ad",
                "filter": [{"field": "device", "op": "equals", "value": "mobile"}],
            }
        ]
        out = filter_attachments(atts, {"device": "mobile"})
        assert out == atts

    def test_filter_miss_excludes_attachment(self):
        atts = [
            {
                "id": "mobile_ad",
                "filter": [{"field": "device", "op": "equals", "value": "mobile"}],
            }
        ]
        out = filter_attachments(atts, {"device": "desktop"})
        assert out == []

    def test_implicit_and_across_predicates(self):
        atts = [
            {
                "id": "mobile_25plus",
                "filter": [
                    {"field": "device", "op": "equals", "value": "mobile"},
                    {"field": "age", "op": "gte", "value": 25},
                ],
            }
        ]
        assert filter_attachments(atts, {"device": "mobile", "age": 34}) == atts
        # All-or-nothing: failing one predicate drops the attachment.
        assert filter_attachments(atts, {"device": "mobile", "age": 22}) == []
        assert filter_attachments(atts, {"device": "desktop", "age": 34}) == []

    def test_mixed_attachments_keep_only_matching(self):
        ad_mobile = {"id": "m", "filter": [{"field": "device", "op": "equals", "value": "mobile"}]}
        ad_unconditional = {"id": "u"}
        ad_desktop = {"id": "d", "filter": [{"field": "device", "op": "equals", "value": "desktop"}]}
        out = filter_attachments([ad_mobile, ad_unconditional, ad_desktop], {"device": "mobile"})
        assert out == [ad_mobile, ad_unconditional]

    def test_persona_missing_trait_is_excluded(self):
        atts = [{"id": "x", "filter": [{"field": "device", "op": "equals", "value": "mobile"}]}]
        # No "device" key on the persona at all → predicate evaluates False.
        assert filter_attachments(atts, {"name": "Anon"}) == []

    def test_in_filter_multi_device(self):
        atts = [
            {
                "id": "mob_or_tab",
                "filter": [{"field": "device", "op": "in", "value": ["mobile", "tablet"]}],
            }
        ]
        assert filter_attachments(atts, {"device": "tablet"}) == atts
        assert filter_attachments(atts, {"device": "desktop"}) == []


class TestCountStrata:
    def test_no_attachments_one_stratum(self):
        personas = [{"name": "A", "device": "mobile"}, {"name": "B", "device": "desktop"}]
        assert count_strata(personas, []) == 1

    def test_unfiltered_attachments_one_stratum(self):
        # Every persona sees every attachment → single partition.
        personas = [{"name": "A", "device": "mobile"}, {"name": "B", "device": "desktop"}]
        atts = [{"id": "ad1"}, {"id": "ad2"}]
        assert count_strata(personas, atts) == 1

    def test_two_partitions_from_device_filter(self):
        personas = [
            {"name": "A", "device": "mobile"},
            {"name": "B", "device": "mobile"},
            {"name": "C", "device": "desktop"},
        ]
        atts = [
            {"id": "mobile_ad", "filter": [{"field": "device", "op": "equals", "value": "mobile"}]},
        ]
        # Mobile personas see {mobile_ad}; desktop sees {} → 2 strata.
        assert count_strata(personas, atts) == 2

    def test_three_partitions(self):
        personas = [
            {"name": "A", "device": "mobile"},
            {"name": "B", "device": "tablet"},
            {"name": "C", "device": "desktop"},
        ]
        atts = [
            {"id": "m", "filter": [{"field": "device", "op": "equals", "value": "mobile"}]},
            {"id": "t", "filter": [{"field": "device", "op": "equals", "value": "tablet"}]},
        ]
        # A sees {m}, B sees {t}, C sees {} → 3 strata.
        assert count_strata(personas, atts) == 3

    def test_empty_personas_zero_strata(self):
        assert count_strata([], [{"id": "ad1"}]) == 0


class TestInstrumentValidatesAttachmentFilter:
    """The instrument parser must reject typo'd predicate ops at parse time."""

    def test_unknown_op_rejected(self):
        from synth_panel.instrument import InstrumentError, parse_instrument

        instrument = {
            "version": 1,
            "questions": [
                {
                    "text": "Does this resonate?",
                    "attachments": [
                        {
                            "id": "ad1",
                            "filter": [{"field": "device", "op": "equalz", "value": "mobile"}],
                        }
                    ],
                }
            ],
        }
        with pytest.raises(InstrumentError, match="op"):
            parse_instrument(instrument)

    def test_in_requires_list_value_at_parse(self):
        from synth_panel.instrument import InstrumentError, parse_instrument

        instrument = {
            "version": 1,
            "questions": [
                {
                    "text": "Q",
                    "attachments": [{"filter": [{"field": "device", "op": "in", "value": "mobile"}]}],
                }
            ],
        }
        with pytest.raises(InstrumentError, match="'in' op requires a list"):
            parse_instrument(instrument)

    def test_valid_filter_parses(self):
        # Per D-phase data-model (hq-xzsm decision 2): attachments on a
        # question are string IDs that resolve into the top-level
        # ``attachments`` bank; the bank entry carries the per-persona
        # ``filter`` clause. Inline dict-shaped refs go in
        # ``inline_attachments`` instead.
        from synth_panel.instrument import parse_instrument

        instrument = {
            "version": 1,
            "attachments": {
                "ad1": {
                    "type": "html",
                    "text": "<p>ad copy</p>",
                    "filter": [
                        {"field": "device", "op": "equals", "value": "mobile"},
                        {"field": "age", "op": "gte", "value": 25},
                    ],
                },
            },
            "questions": [
                {
                    "text": "Q",
                    "attachments": ["ad1"],
                }
            ],
        }
        parsed = parse_instrument(instrument)
        assert parsed.version == 1
