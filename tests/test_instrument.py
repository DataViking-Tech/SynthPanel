"""Tests for instrument v1/v2 parser and round validation."""

from __future__ import annotations

import pytest

from synth_panel.instrument import Instrument, InstrumentError, Round, parse_instrument


# ---------------------------------------------------------------------------
# v1 (flat questions) → single "default" round
# ---------------------------------------------------------------------------

class TestV1Parsing:
    def test_basic_v1(self):
        data = {"version": 1, "questions": [{"text": "Hello?"}]}
        inst = parse_instrument(data)
        assert inst.version == 1
        assert len(inst.rounds) == 1
        assert inst.rounds[0].name == "default"
        assert inst.rounds[0].questions == [{"text": "Hello?"}]
        assert inst.rounds[0].depends_on is None

    def test_v1_default_version(self):
        data = {"questions": [{"text": "Q1"}]}
        inst = parse_instrument(data)
        assert inst.version == 1

    def test_v1_multiple_questions(self):
        qs = [{"text": "Q1"}, {"text": "Q2"}, {"text": "Q3"}]
        inst = parse_instrument({"questions": qs})
        assert inst.questions == qs

    def test_v1_is_not_multi_round(self):
        inst = parse_instrument({"questions": [{"text": "Q"}]})
        assert not inst.is_multi_round

    def test_v1_empty_questions_raises(self):
        with pytest.raises(InstrumentError, match="non-empty list"):
            parse_instrument({"questions": []})

    def test_v1_questions_not_list_raises(self):
        with pytest.raises(InstrumentError, match="non-empty list"):
            parse_instrument({"questions": "not a list"})


# ---------------------------------------------------------------------------
# v2 (multi-round) parsing
# ---------------------------------------------------------------------------

class TestV2Parsing:
    def test_basic_multi_round(self):
        data = {
            "version": 2,
            "rounds": [
                {"name": "discovery", "questions": [{"text": "Q1"}]},
                {
                    "name": "deep_dive",
                    "depends_on": "discovery",
                    "questions": [{"text": "Q2"}],
                },
            ],
        }
        inst = parse_instrument(data)
        assert inst.version == 2
        assert len(inst.rounds) == 2
        assert inst.rounds[0].name == "discovery"
        assert inst.rounds[0].depends_on is None
        assert inst.rounds[1].name == "deep_dive"
        assert inst.rounds[1].depends_on == "discovery"

    def test_three_round_chain(self):
        data = {
            "version": 2,
            "rounds": [
                {"name": "a", "questions": [{"text": "Q1"}]},
                {"name": "b", "depends_on": "a", "questions": [{"text": "Q2"}]},
                {"name": "c", "depends_on": "b", "questions": [{"text": "Q3"}]},
            ],
        }
        inst = parse_instrument(data)
        assert inst.is_multi_round
        assert [r.name for r in inst.rounds] == ["a", "b", "c"]

    def test_round_without_depends_on(self):
        data = {
            "version": 2,
            "rounds": [
                {"name": "intro", "questions": [{"text": "Q1"}]},
                {"name": "outro", "questions": [{"text": "Q2"}]},
            ],
        }
        inst = parse_instrument(data)
        assert all(r.depends_on is None for r in inst.rounds)

    def test_questions_property_multi_round(self):
        data = {
            "version": 2,
            "rounds": [
                {"name": "a", "questions": [{"text": "Q1"}, {"text": "Q2"}]},
                {"name": "b", "questions": [{"text": "Q3"}]},
            ],
        }
        inst = parse_instrument(data)
        assert inst.questions == [{"text": "Q1"}, {"text": "Q2"}, {"text": "Q3"}]

    def test_single_round_v2(self):
        """v2 format with a single round is valid but not multi-round."""
        data = {
            "version": 2,
            "rounds": [{"name": "only", "questions": [{"text": "Q1"}]}],
        }
        inst = parse_instrument(data)
        assert not inst.is_multi_round
        assert inst.rounds[0].name == "only"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_no_questions_or_rounds_raises(self):
        with pytest.raises(InstrumentError, match="'questions'.*or.*'rounds'"):
            parse_instrument({"version": 1})

    def test_empty_rounds_raises(self):
        with pytest.raises(InstrumentError, match="non-empty list"):
            parse_instrument({"rounds": []})

    def test_rounds_not_list_raises(self):
        with pytest.raises(InstrumentError, match="non-empty list"):
            parse_instrument({"rounds": "not a list"})

    def test_round_missing_name_raises(self):
        with pytest.raises(InstrumentError, match="'name' string"):
            parse_instrument({"rounds": [{"questions": [{"text": "Q"}]}]})

    def test_round_empty_name_raises(self):
        with pytest.raises(InstrumentError, match="'name' string"):
            parse_instrument({"rounds": [{"name": "", "questions": [{"text": "Q"}]}]})

    def test_round_missing_questions_raises(self):
        with pytest.raises(InstrumentError, match="non-empty 'questions'"):
            parse_instrument({"rounds": [{"name": "a"}]})

    def test_round_empty_questions_raises(self):
        with pytest.raises(InstrumentError, match="non-empty 'questions'"):
            parse_instrument({"rounds": [{"name": "a", "questions": []}]})

    def test_round_not_mapping_raises(self):
        with pytest.raises(InstrumentError, match="must be a mapping"):
            parse_instrument({"rounds": ["not a dict"]})

    def test_duplicate_round_name_raises(self):
        with pytest.raises(InstrumentError, match="Duplicate round name"):
            parse_instrument({
                "rounds": [
                    {"name": "a", "questions": [{"text": "Q1"}]},
                    {"name": "a", "questions": [{"text": "Q2"}]},
                ],
            })

    def test_forward_ref_depends_on_allowed(self):
        """v3 relaxes the earlier-only rule; forward refs are valid."""
        inst = parse_instrument({
            "rounds": [
                {
                    "name": "first",
                    "depends_on": "second",
                    "questions": [{"text": "Q1"}],
                },
                {"name": "second", "questions": [{"text": "Q2"}]},
            ],
        })
        assert inst.rounds[0].depends_on == "second"

    def test_self_ref_depends_on_raises(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument({
                "rounds": [
                    {
                        "name": "self_ref",
                        "depends_on": "self_ref",
                        "questions": [{"text": "Q1"}],
                    },
                ],
            })

    def test_nonexistent_depends_on_raises(self):
        with pytest.raises(InstrumentError, match="does not exist"):
            parse_instrument({
                "rounds": [
                    {
                        "name": "a",
                        "depends_on": "nonexistent",
                        "questions": [{"text": "Q1"}],
                    },
                ],
            })

    def test_depends_on_not_string_raises(self):
        with pytest.raises(InstrumentError, match="must be a string"):
            parse_instrument({
                "rounds": [
                    {"name": "a", "questions": [{"text": "Q1"}]},
                    {
                        "name": "b",
                        "depends_on": ["a"],
                        "questions": [{"text": "Q2"}],
                    },
                ],
            })


# ---------------------------------------------------------------------------
# Dataclass behavior
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_round_defaults(self):
        r = Round(name="test", questions=[{"text": "Q"}])
        assert r.depends_on is None

    def test_instrument_defaults(self):
        inst = Instrument(version=1)
        assert inst.rounds == []
        assert inst.questions == []
        assert not inst.is_multi_round


# ---------------------------------------------------------------------------
# Integration: example YAML file
# ---------------------------------------------------------------------------

class TestExampleYAML:
    def test_multi_round_study_parses(self):
        """Verify the shipped example file parses without errors."""
        import yaml
        from pathlib import Path

        example = Path(__file__).parent.parent / "examples" / "multi-round-study.yaml"
        if not example.exists():
            pytest.skip("Example file not found")

        with open(example) as f:
            data = yaml.safe_load(f)

        raw = data["instrument"]
        inst = parse_instrument(raw)
        assert inst.version == 2
        assert len(inst.rounds) == 3
        assert inst.rounds[0].name == "discovery"
        assert inst.rounds[1].depends_on == "discovery"
        assert inst.rounds[2].depends_on == "deep_dive"
        assert inst.is_multi_round


# ---------------------------------------------------------------------------
# v3 (branching, route_when) parsing + DAG validation
# ---------------------------------------------------------------------------

class TestV3Branching:
    def test_simple_route_when(self):
        data = {
            "version": 3,
            "rounds": [
                {
                    "name": "intro",
                    "questions": [{"text": "Q1"}],
                    "route_when": [
                        {"if": "x", "goto": "probe"},
                        {"else": "wrap"},
                    ],
                },
                {"name": "probe", "questions": [{"text": "Q2"}]},
                {"name": "wrap", "questions": [{"text": "Q3"}]},
            ],
        }
        inst = parse_instrument(data)
        assert inst.rounds[0].route_when is not None
        assert len(inst.rounds[0].route_when) == 2
        assert inst.warnings == []

    def test_forward_goto_resolves(self):
        inst = parse_instrument({
            "rounds": [
                {
                    "name": "a",
                    "questions": [{"text": "Q"}],
                    "route_when": [{"else": "probe_pricing"}],
                },
                {"name": "probe_pricing", "questions": [{"text": "Q"}]},
            ],
        })
        assert inst.rounds[1].name == "probe_pricing"

    def test_goto_end_sentinel(self):
        inst = parse_instrument({
            "rounds": [
                {
                    "name": "a",
                    "questions": [{"text": "Q"}],
                    "route_when": [{"else": "__end__"}],
                },
            ],
        })
        assert inst.rounds[0].route_when[0]["else"] == "__end__"

    def test_missing_else_rejected(self):
        with pytest.raises(InstrumentError, match="no else clause"):
            parse_instrument({
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"if": "x", "goto": "b"}],
                    },
                    {"name": "b", "questions": [{"text": "Q"}]},
                ],
            })

    def test_bad_goto_target_rejected(self):
        with pytest.raises(InstrumentError, match="goto 'nope' does not exist"):
            parse_instrument({
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "nope"}],
                    },
                ],
            })

    def test_cycle_detected_with_path(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument({
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "b"}],
                    },
                    {
                        "name": "b",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "a"}],
                    },
                ],
            })

    def test_cycle_via_depends_on(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument({
                "rounds": [
                    {"name": "a", "depends_on": "b", "questions": [{"text": "Q"}]},
                    {"name": "b", "depends_on": "a", "questions": [{"text": "Q"}]},
                ],
            })

    def test_unreachable_round_warns(self):
        inst = parse_instrument({
            "rounds": [
                {
                    "name": "a",
                    "questions": [{"text": "Q"}],
                    "route_when": [{"else": "__end__"}],
                },
                {"name": "orphan", "questions": [{"text": "Q"}]},
            ],
        })
        assert any("orphan" in w for w in inst.warnings)

    def test_v2_linear_no_warnings(self):
        inst = parse_instrument({
            "version": 2,
            "rounds": [
                {"name": "a", "questions": [{"text": "Q1"}]},
                {"name": "b", "depends_on": "a", "questions": [{"text": "Q2"}]},
                {"name": "c", "depends_on": "b", "questions": [{"text": "Q3"}]},
            ],
        })
        assert inst.warnings == []

    def test_route_when_must_be_list(self):
        with pytest.raises(InstrumentError, match="route_when.*non-empty list"):
            parse_instrument({
                "rounds": [
                    {"name": "a", "questions": [{"text": "Q"}], "route_when": "bad"},
                ],
            })
