"""Tests for the predicate engine and round router."""

from __future__ import annotations

import pytest

from althing.routing import (
    END_SENTINEL,
    RoutingError,
    evaluate_predicate,
    route_round,
)


@pytest.fixture
def ctx():
    return {
        "summary": "Users want better pricing transparency.",
        "themes": ["pricing pain", "onboarding friction", "docs gaps"],
        "agreements": ["transparent pricing matters"],
        "disagreements": ["frequency of releases"],
        "surprises": ["nobody mentioned mobile"],
        "recommendation": "Publish a clear pricing page.",
    }


class TestContains:
    def test_list_substring_hit(self, ctx):
        assert evaluate_predicate({"field": "themes", "op": "contains", "value": "pricing"}, ctx)

    def test_list_substring_miss(self, ctx):
        assert not evaluate_predicate({"field": "themes", "op": "contains", "value": "mobile"}, ctx)

    def test_string_field_substring(self, ctx):
        assert evaluate_predicate({"field": "summary", "op": "contains", "value": "transparency"}, ctx)


class TestEquals:
    def test_string_exact(self, ctx):
        assert evaluate_predicate(
            {
                "field": "recommendation",
                "op": "equals",
                "value": "Publish a clear pricing page.",
            },
            ctx,
        )

    def test_string_not_substring(self, ctx):
        assert not evaluate_predicate({"field": "recommendation", "op": "equals", "value": "pricing"}, ctx)

    def test_list_exact_member(self, ctx):
        assert evaluate_predicate({"field": "themes", "op": "equals", "value": "pricing pain"}, ctx)


class TestMatches:
    def test_regex_search_not_match(self, ctx):
        # re.search, not re.match — pattern need not anchor at start
        assert evaluate_predicate({"field": "summary", "op": "matches", "value": r"transparen\w+"}, ctx)

    def test_regex_against_list(self, ctx):
        assert evaluate_predicate({"field": "themes", "op": "matches", "value": r"^onboard"}, ctx)

    def test_regex_miss(self, ctx):
        assert not evaluate_predicate({"field": "summary", "op": "matches", "value": r"^Nothing"}, ctx)


class TestNewOps:
    """hq-iczd: gte/lte/in operators added for attachment stratification.

    These ops are exposed via ``evaluate_predicate`` against the routing
    allowlist as a side-effect of sharing the predicate engine. Routing
    fields are strings, so numeric ops here must coerce or raise.
    """

    def test_in_list_target_any_member_match(self, ctx):
        assert evaluate_predicate({"field": "themes", "op": "in", "value": ["pricing pain", "noise"]}, ctx)

    def test_in_string_target_membership(self, ctx):
        assert evaluate_predicate(
            {"field": "recommendation", "op": "in", "value": ["A", "Publish a clear pricing page."]},
            ctx,
        )

    def test_in_no_match(self, ctx):
        assert not evaluate_predicate({"field": "themes", "op": "in", "value": ["mobile", "tablet"]}, ctx)

    def test_in_requires_list_value(self, ctx):
        with pytest.raises(ValueError, match="'in' op requires a list"):
            evaluate_predicate({"field": "themes", "op": "in", "value": "pricing pain"}, ctx)


class TestErrors:
    def test_unknown_field_raises_keyerror_with_name(self, ctx):
        with pytest.raises(KeyError) as exc:
            evaluate_predicate({"field": "bogus", "op": "contains", "value": "x"}, ctx)
        assert "bogus" in str(exc.value)

    def test_unknown_op_raises(self, ctx):
        with pytest.raises(ValueError, match="unknown predicate op"):
            evaluate_predicate({"field": "themes", "op": "weird", "value": "x"}, ctx)


class TestRouteRound:
    def test_first_match_wins(self, ctx):
        rw = [
            {
                "if": {"field": "themes", "op": "contains", "value": "pricing"},
                "goto": "probe_pricing",
            },
            {
                "if": {"field": "themes", "op": "contains", "value": "docs"},
                "goto": "probe_docs",
            },
            {"else": "wrap_up"},
        ]
        assert route_round(rw, ctx) == "probe_pricing"

    def test_falls_through_to_else(self, ctx):
        rw = [
            {
                "if": {"field": "themes", "op": "contains", "value": "mobile"},
                "goto": "probe_mobile",
            },
            {"else": "wrap_up"},
        ]
        assert route_round(rw, ctx) == "wrap_up"

    def test_else_to_end_sentinel(self, ctx):
        rw = [{"else": END_SENTINEL}]
        assert route_round(rw, ctx) == END_SENTINEL

    def test_no_match_no_else_raises(self, ctx):
        rw = [
            {
                "if": {"field": "themes", "op": "contains", "value": "mobile"},
                "goto": "probe_mobile",
            },
        ]
        with pytest.raises(RoutingError):
            route_round(rw, ctx)
