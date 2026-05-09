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
            parse_instrument(
                {
                    "rounds": [
                        {"name": "a", "questions": [{"text": "Q1"}]},
                        {"name": "a", "questions": [{"text": "Q2"}]},
                    ],
                }
            )

    def test_forward_ref_depends_on_allowed(self):
        """v3 relaxes the earlier-only rule; forward refs are valid."""
        inst = parse_instrument(
            {
                "rounds": [
                    {
                        "name": "first",
                        "depends_on": "second",
                        "questions": [{"text": "Q1"}],
                    },
                    {"name": "second", "questions": [{"text": "Q2"}]},
                ],
            }
        )
        assert inst.rounds[0].depends_on == "second"

    def test_self_ref_depends_on_raises(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument(
                {
                    "rounds": [
                        {
                            "name": "self_ref",
                            "depends_on": "self_ref",
                            "questions": [{"text": "Q1"}],
                        },
                    ],
                }
            )

    def test_nonexistent_depends_on_raises(self):
        with pytest.raises(InstrumentError, match="does not exist"):
            parse_instrument(
                {
                    "rounds": [
                        {
                            "name": "a",
                            "depends_on": "nonexistent",
                            "questions": [{"text": "Q1"}],
                        },
                    ],
                }
            )

    def test_depends_on_not_string_raises(self):
        with pytest.raises(InstrumentError, match="must be a string"):
            parse_instrument(
                {
                    "rounds": [
                        {"name": "a", "questions": [{"text": "Q1"}]},
                        {
                            "name": "b",
                            "depends_on": ["a"],
                            "questions": [{"text": "Q2"}],
                        },
                    ],
                }
            )


# ---------------------------------------------------------------------------
# Follow-up condition validation (sp-t5ok)
# ---------------------------------------------------------------------------


class TestFollowUpConditionValidation:
    def test_unknown_condition_in_v1_raises(self):
        data = {
            "version": 1,
            "questions": [
                {
                    "text": "Q?",
                    "follow_ups": [
                        {"text": "Why?", "condition": "response_contians: yes"},
                    ],
                }
            ],
        }
        with pytest.raises(InstrumentError, match="response_contians"):
            parse_instrument(data)

    def test_known_condition_in_v1_passes(self):
        data = {
            "version": 1,
            "questions": [
                {
                    "text": "Q?",
                    "follow_ups": [
                        {"text": "Why?", "condition": "response_contains: yes"},
                        {"text": "And?", "condition": "always"},
                        "Plain string follow-up",
                    ],
                }
            ],
        }
        inst = parse_instrument(data)
        assert len(inst.rounds[0].questions) == 1

    def test_unknown_condition_in_round_raises_with_context(self):
        data = {
            "version": 2,
            "rounds": [
                {
                    "name": "probe",
                    "questions": [
                        {
                            "text": "Q?",
                            "follow_ups": [{"text": "Why?", "condition": "typo_here"}],
                        }
                    ],
                }
            ],
        }
        with pytest.raises(InstrumentError, match="round 'probe'"):
            parse_instrument(data)

    def test_no_condition_field_is_fine(self):
        """A follow-up dict without a condition key is valid (defaults to always)."""
        data = {
            "version": 1,
            "questions": [
                {"text": "Q?", "follow_ups": [{"text": "Elaborate?"}]},
            ],
        }
        parse_instrument(data)


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
        from pathlib import Path

        import yaml

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
        inst = parse_instrument(
            {
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "probe_pricing"}],
                    },
                    {"name": "probe_pricing", "questions": [{"text": "Q"}]},
                ],
            }
        )
        assert inst.rounds[1].name == "probe_pricing"

    def test_goto_end_sentinel(self):
        inst = parse_instrument(
            {
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "__end__"}],
                    },
                ],
            }
        )
        assert inst.rounds[0].route_when[0]["else"] == "__end__"

    def test_missing_else_rejected(self):
        with pytest.raises(InstrumentError, match="no else clause"):
            parse_instrument(
                {
                    "rounds": [
                        {
                            "name": "a",
                            "questions": [{"text": "Q"}],
                            "route_when": [{"if": "x", "goto": "b"}],
                        },
                        {"name": "b", "questions": [{"text": "Q"}]},
                    ],
                }
            )

    def test_bad_goto_target_rejected(self):
        with pytest.raises(InstrumentError, match="goto 'nope' does not exist"):
            parse_instrument(
                {
                    "rounds": [
                        {
                            "name": "a",
                            "questions": [{"text": "Q"}],
                            "route_when": [{"else": "nope"}],
                        },
                    ],
                }
            )

    def test_cycle_detected_with_path(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument(
                {
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
                }
            )

    def test_cycle_via_depends_on(self):
        with pytest.raises(InstrumentError, match="Cycle detected"):
            parse_instrument(
                {
                    "rounds": [
                        {"name": "a", "depends_on": "b", "questions": [{"text": "Q"}]},
                        {"name": "b", "depends_on": "a", "questions": [{"text": "Q"}]},
                    ],
                }
            )

    def test_unreachable_round_warns(self):
        inst = parse_instrument(
            {
                "rounds": [
                    {
                        "name": "a",
                        "questions": [{"text": "Q"}],
                        "route_when": [{"else": "__end__"}],
                    },
                    {"name": "orphan", "questions": [{"text": "Q"}]},
                ],
            }
        )
        assert any("orphan" in w for w in inst.warnings)

    def test_v2_linear_no_warnings(self):
        inst = parse_instrument(
            {
                "version": 2,
                "rounds": [
                    {"name": "a", "questions": [{"text": "Q1"}]},
                    {"name": "b", "depends_on": "a", "questions": [{"text": "Q2"}]},
                    {"name": "c", "depends_on": "b", "questions": [{"text": "Q3"}]},
                ],
            }
        )
        assert inst.warnings == []

    def test_route_when_must_be_list(self):
        with pytest.raises(InstrumentError, match="route_when.*non-empty list"):
            parse_instrument(
                {
                    "rounds": [
                        {"name": "a", "questions": [{"text": "Q"}], "route_when": "bad"},
                    ],
                }
            )


# ---------------------------------------------------------------------------
# extraction_schema per question
# ---------------------------------------------------------------------------


class TestExtractionSchema:
    def test_v1_string_schema_name_accepted(self):
        """Known schema names pass validation."""
        data = {
            "version": 1,
            "questions": [
                {"text": "Rate this", "extraction_schema": "likert"},
            ],
        }
        inst = parse_instrument(data)
        assert inst.questions[0]["extraction_schema"] == "likert"

    def test_v1_inline_dict_schema_accepted(self):
        """Inline dict schemas pass validation without registry lookup."""
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
        data = {
            "version": 1,
            "questions": [
                {"text": "Score it", "extraction_schema": schema},
            ],
        }
        inst = parse_instrument(data)
        assert inst.questions[0]["extraction_schema"] == schema

    def test_v1_unknown_schema_name_rejected(self):
        """Unknown schema names raise InstrumentError at parse time."""
        data = {
            "version": 1,
            "questions": [
                {"text": "Q?", "extraction_schema": "nonexistent"},
            ],
        }
        with pytest.raises(InstrumentError, match="Unknown extraction schema.*nonexistent"):
            parse_instrument(data)

    def test_v1_invalid_type_rejected(self):
        """Non-string, non-dict extraction_schema raises InstrumentError."""
        data = {
            "version": 1,
            "questions": [
                {"text": "Q?", "extraction_schema": 42},
            ],
        }
        with pytest.raises(InstrumentError, match="extraction_schema must be a string"):
            parse_instrument(data)

    def test_v1_no_extraction_schema_ok(self):
        """Questions without extraction_schema parse fine."""
        data = {"version": 1, "questions": [{"text": "Hello?"}]}
        inst = parse_instrument(data)
        assert "extraction_schema" not in inst.questions[0]

    def test_rounds_schema_name_validated(self):
        """extraction_schema in round questions is validated."""
        data = {
            "version": 3,
            "rounds": [
                {
                    "name": "intro",
                    "questions": [
                        {"text": "Yes or no?", "extraction_schema": "yes_no"},
                    ],
                    "route_when": [{"else": "__end__"}],
                },
            ],
        }
        inst = parse_instrument(data)
        assert inst.rounds[0].questions[0]["extraction_schema"] == "yes_no"

    def test_rounds_unknown_schema_rejected(self):
        """Unknown schema names in round questions raise InstrumentError."""
        data = {
            "version": 3,
            "rounds": [
                {
                    "name": "intro",
                    "questions": [
                        {"text": "Q?", "extraction_schema": "bogus"},
                    ],
                    "route_when": [{"else": "__end__"}],
                },
            ],
        }
        with pytest.raises(InstrumentError, match="Unknown extraction schema.*bogus"):
            parse_instrument(data)

    def test_mixed_questions_some_with_schema(self):
        """Only questions with extraction_schema are validated."""
        data = {
            "version": 1,
            "questions": [
                {"text": "Open ended"},
                {"text": "Pick one", "extraction_schema": "pick_one"},
                {"text": "Rank them", "extraction_schema": "ranking"},
            ],
        }
        inst = parse_instrument(data)
        assert len(inst.questions) == 3
        assert inst.questions[1]["extraction_schema"] == "pick_one"
        assert inst.questions[2]["extraction_schema"] == "ranking"

    def test_all_bundled_schema_names_accepted(self):
        """All four bundled schema names pass validation."""
        for name in ("ranking", "likert", "yes_no", "pick_one"):
            data = {
                "version": 1,
                "questions": [{"text": "Q?", "extraction_schema": name}],
            }
            inst = parse_instrument(data)
            assert inst.questions[0]["extraction_schema"] == name


# ---------------------------------------------------------------------------
# Attachment bank parsing (hq-l0lw)
# ---------------------------------------------------------------------------


class TestAttachments:
    """Top-level ``Instrument.attachments`` bank + per-question references."""

    def _bank(self, **overrides):
        base = {
            "type": "image",
            "media_type": "image/png",
            "source": {"type": "base64", "data": "AAAA"},
        }
        base.update(overrides)
        return base

    def test_default_attachments_is_empty(self):
        inst = parse_instrument({"questions": [{"text": "Q"}]})
        assert inst.attachments == {}

    def test_image_inline_base64(self):
        inst = parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q", "attachments": ["hero"]}],
                "attachments": {"hero": self._bank()},
            }
        )
        assert "hero" in inst.attachments
        assert inst.attachments["hero"]["type"] == "image"

    def test_image_url_source(self):
        inst = parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q"}],
                "attachments": {
                    "remote": {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "source": {"type": "url", "url": "https://example.com/x.jpg"},
                    }
                },
            }
        )
        assert inst.attachments["remote"]["source"]["type"] == "url"

    def test_document_pdf(self):
        parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q"}],
                "attachments": {
                    "spec": {
                        "type": "document",
                        "source": {"type": "file", "file_id": "file_abc"},
                    }
                },
            }
        )

    def test_url_block(self):
        parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q"}],
                "attachments": {"page": {"type": "url", "url": "https://example.com"}},
            }
        )

    def test_html_block(self):
        parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q"}],
                "attachments": {"snippet": {"type": "html", "text": "<b>hi</b>"}},
            }
        )

    def test_unknown_type_rejected(self):
        with pytest.raises(InstrumentError, match="unknown type"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q"}],
                    "attachments": {"bad": {"type": "video"}},
                }
            )

    def test_invalid_id_rejected(self):
        with pytest.raises(InstrumentError, match="must match"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q"}],
                    "attachments": {"BadCase": self._bank()},
                }
            )

    def test_invalid_image_media_type_rejected(self):
        with pytest.raises(InstrumentError, match="media_type"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q"}],
                    "attachments": {
                        "x": {
                            "type": "image",
                            "media_type": "image/bmp",
                            "source": {"type": "base64", "data": "x"},
                        }
                    },
                }
            )

    def test_invalid_url_rejected(self):
        with pytest.raises(InstrumentError, match="syntactically valid URL"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q"}],
                    "attachments": {"x": {"type": "url", "url": "not a url"}},
                }
            )

    def test_unresolved_question_ref_rejected(self):
        with pytest.raises(InstrumentError, match="does not resolve"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q", "attachments": ["nope"]}],
                    "attachments": {},
                }
            )

    def test_question_ref_must_be_string(self):
        with pytest.raises(InstrumentError, match="must be a string"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q", "attachments": [{"id": "x"}]}],
                    "attachments": {"x": self._bank()},
                }
            )

    def test_inline_attachment_text_block_accepted(self):
        parse_instrument(
            {
                "version": 1,
                "questions": [
                    {
                        "text": "Q",
                        "inline_attachments": [{"type": "text", "text": "headline"}],
                    }
                ],
            }
        )

    def test_inline_attachment_unknown_type_rejected(self):
        with pytest.raises(InstrumentError, match="unknown"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [
                        {
                            "text": "Q",
                            "inline_attachments": [{"type": "video", "data": "x"}],
                        }
                    ],
                }
            )

    def test_dual_ephemeral_rejected(self):
        with pytest.raises(InstrumentError, match="at most one attachment"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q", "attachments": ["a", "b"]}],
                    "attachments": {
                        "a": self._bank(cache_control="ephemeral"),
                        "b": self._bank(cache_control="ephemeral"),
                    },
                }
            )

    def test_ephemeral_must_mark_last_shared(self):
        # Marker on the FIRST of two shared blocks violates the contract.
        with pytest.raises(InstrumentError, match="LAST shared"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q", "attachments": ["a", "b"]}],
                    "attachments": {
                        "a": self._bank(cache_control="ephemeral"),
                        "b": self._bank(),
                    },
                }
            )

    def test_ephemeral_last_shared_accepted(self):
        # Marker on the LAST of the shared prefix is the supported shape.
        parse_instrument(
            {
                "version": 1,
                "questions": [{"text": "Q", "attachments": ["a", "b"]}],
                "attachments": {
                    "a": self._bank(),
                    "b": self._bank(cache_control="ephemeral"),
                },
            }
        )

    def test_ephemeral_on_inline_rejected(self):
        # Inline blocks are per-question divergent, so they're never shared
        # cache prefix — marking one ephemeral violates the architectural
        # constraint we're protecting at parse time.
        with pytest.raises(InstrumentError, match="LAST shared"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [
                        {
                            "text": "Q",
                            "attachments": ["a"],
                            "inline_attachments": [
                                {
                                    "type": "html",
                                    "text": "<i>x</i>",
                                    "cache_control": "ephemeral",
                                }
                            ],
                        }
                    ],
                    "attachments": {"a": self._bank()},
                }
            )

    def test_attachments_carries_through_v3_rounds(self):
        inst = parse_instrument(
            {
                "version": 3,
                "rounds": [
                    {
                        "name": "intro",
                        "questions": [{"text": "Q1", "attachments": ["hero"]}],
                        "route_when": [{"else": "wrap"}],
                    },
                    {
                        "name": "wrap",
                        "questions": [{"text": "Q2", "attachments": ["hero"]}],
                    },
                ],
                "attachments": {"hero": self._bank()},
            }
        )
        assert "hero" in inst.attachments
        assert len(inst.rounds) == 2

    def test_attachments_must_be_mapping(self):
        with pytest.raises(InstrumentError, match="must be a mapping"):
            parse_instrument(
                {
                    "version": 1,
                    "questions": [{"text": "Q"}],
                    "attachments": [{"id": "hero"}],
                }
            )
