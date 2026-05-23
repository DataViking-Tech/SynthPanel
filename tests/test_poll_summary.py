"""sy-4yd: deterministic poll summary computation.

Tests cover the contract surface of :mod:`synth_panel.poll_summary`:

* Enum questions roll up to first-choice / second-choice / weighted-score
  tables with the highest-vote winner.
* Scale questions produce a mean + histogram with no false-positive bucket
  collapse (1 vs "1").
* Segment splits join on persona attributes from either the embedded
  result record or the explicit ``personas`` list.
* Objection harvesting reads ``extraction.objections`` and
  ``extraction.concerns`` lists or scalars.
* Free-text questions don't fabricate vote counts.
* Two calls over identical input produce identical output (no
  set-iteration / dict-ordering flake — the summary is the agent-facing
  source of truth so byte-stability matters).
"""

from __future__ import annotations

from synth_panel.poll_summary import (
    PollSummary,
    QuestionSummary,
    build_poll_summary,
)


def _enum_result_envelope() -> dict[str, object]:
    """Three-panelist run with one enum question + one scale question.

    Panelists carry an ``occupation`` attribute so segment splits land.
    Mirrors the persistence shape ``save_panel_result`` writes.
    """
    return {
        "persona_count": 3,
        "question_count": 2,
        "questions": [
            {
                "text": "Which name do you prefer?",
                "response_schema": {"type": "enum", "options": ["Alpha", "Beta", "Gamma"]},
            },
            {
                "text": "Confidence in your pick (1-5)?",
                "response_schema": {"type": "scale", "min": 1, "max": 5},
            },
        ],
        "results": [
            {
                "persona": "Sarah",
                "occupation": "engineer",
                "responses": [
                    {
                        "question": "Which name do you prefer?",
                        "response": "Alpha",
                        "extraction": {
                            "choice": "Alpha",
                            "second_choice": "Beta",
                            "score": 5,
                            "concerns": ["domain availability"],
                        },
                    },
                    {"question": "Confidence in your pick (1-5)?", "response": "5"},
                ],
            },
            {
                "persona": "Marcus",
                "occupation": "designer",
                "responses": [
                    {
                        "question": "Which name do you prefer?",
                        "response": "Alpha",
                        "extraction": {
                            "choice": "Alpha",
                            "second_choice": "Gamma",
                            "score": 4,
                            "concerns": "domain availability",
                        },
                    },
                    {"question": "Confidence in your pick (1-5)?", "response": "4"},
                ],
            },
            {
                "persona": "Priya",
                "occupation": "engineer",
                "responses": [
                    {
                        "question": "Which name do you prefer?",
                        "response": "Beta",
                        "extraction": {
                            "choice": "Beta",
                            "second_choice": "Alpha",
                            "score": 3,
                            "concerns": ["pronunciation"],
                        },
                    },
                    {"question": "Confidence in your pick (1-5)?", "response": "3"},
                ],
            },
        ],
        "synthesis": {
            "recommendation": "Validate 'Alpha' with a 50-person paid survey.",
        },
    }


class TestEnumRollup:
    """First-choice + second-choice + weighted-score + winner."""

    def test_winner_is_highest_first_choice_count(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q0 = summary.questions[0]
        assert q0.kind == "enum"
        assert q0.winner == "Alpha"

    def test_first_choice_counts_in_descending_order(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q0 = summary.questions[0]
        assert list(q0.first_choice_counts.items()) == [("Alpha", 2), ("Beta", 1)]

    def test_second_choice_counts_aggregate_independently(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q0 = summary.questions[0]
        # Sarah → Beta, Marcus → Gamma, Priya → Alpha
        assert q0.second_choice_counts == {"Beta": 1, "Gamma": 1, "Alpha": 1}

    def test_weighted_scores_average_per_option(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q0 = summary.questions[0]
        # Alpha: (5 + 4) / 2 = 4.5; Beta: 3 / 1 = 3.0
        assert q0.weighted_scores == {"Alpha": 4.5, "Beta": 3.0}

    def test_n_counts_total_responses(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        assert summary.questions[0].n == 3
        assert summary.questions[0].n_unparseable == 0


class TestScaleRollup:
    def test_average_is_mean_of_numeric_responses(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q1 = summary.questions[1]
        assert q1.kind == "scale"
        # (5 + 4 + 3) / 3 = 4.0
        assert q1.metric_average == 4.0

    def test_histogram_uses_integer_strings_for_integral_values(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q1 = summary.questions[1]
        # No "5.0" / "4.0" / "3.0" — they round-trip as "5"/"4"/"3"
        assert q1.metric_distribution == {"3": 1, "4": 1, "5": 1}

    def test_histogram_is_sorted_by_numeric_value(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        q1 = summary.questions[1]
        # Insertion would have been 5, 4, 3 (the order panelists answered).
        # The summary sorts numerically so the histogram is scannable.
        assert list(q1.metric_distribution.keys()) == ["3", "4", "5"]

    def test_non_numeric_scale_response_counts_as_unparseable(self) -> None:
        env = _enum_result_envelope()
        # Replace one numeric response with free-text so the unparseable
        # path is exercised.
        env["results"][0]["responses"][1]["response"] = "no idea"
        summary = build_poll_summary(env)
        q1 = summary.questions[1]
        assert q1.n_unparseable == 1
        assert q1.metric_average == 3.5  # (4 + 3) / 2


class TestSegmentSplits:
    def test_split_by_occupation_groups_panelists(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        # Q0 (enum) — split should join on the inline ``occupation`` attr.
        splits = summary.segment_splits["0"]
        assert "occupation" in splits
        # engineer: Sarah → Alpha, Priya → Beta
        # designer: Marcus → Alpha
        assert splits["occupation"]["engineer"] == {"Alpha": 1, "Beta": 1}
        assert splits["occupation"]["designer"] == {"Alpha": 1}

    def test_empty_segment_by_disables_splits(self) -> None:
        summary = build_poll_summary(_enum_result_envelope(), segment_by=[])
        assert summary.segment_splits == {}

    def test_scale_questions_do_not_produce_segment_splits(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        # Q1 is scale; we deliberately skip segment splitting for scale
        # questions because the natural rollup is the mean, not the
        # vote count.
        assert "1" not in summary.segment_splits


class TestObjectionHarvesting:
    def test_collects_from_scalar_and_list_extraction_fields(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        # Sarah + Priya have list-shaped objections; Marcus has a scalar.
        # The most common ("domain availability") appears twice.
        assert summary.top_objections[0] == "domain availability"
        assert "pronunciation" in summary.top_objections

    def test_caps_top_objections_at_five(self) -> None:
        env: dict[str, object] = {
            "persona_count": 1,
            "questions": [{"text": "What's the issue?", "response_schema": {"type": "text"}}],
            "results": [
                {
                    "persona": "Tester",
                    "responses": [
                        {
                            "question": "What's the issue?",
                            "response": "various concerns",
                            "extraction": {
                                "objections": [f"objection-{i}" for i in range(10)],
                            },
                        }
                    ],
                }
            ],
        }
        summary = build_poll_summary(env)
        assert len(summary.top_objections) == 5


class TestRecommendedNextTest:
    def test_echoes_synthesis_recommendation(self) -> None:
        summary = build_poll_summary(_enum_result_envelope())
        assert summary.recommended_next_test == "Validate 'Alpha' with a 50-person paid survey."

    def test_returns_none_when_synthesis_absent(self) -> None:
        env = _enum_result_envelope()
        del env["synthesis"]
        summary = build_poll_summary(env)
        assert summary.recommended_next_test is None

    def test_falls_back_to_terminal_round_synthesis(self) -> None:
        env: dict[str, object] = {
            "persona_count": 1,
            "rounds": [
                {
                    "name": "discovery",
                    "results": [],
                    "synthesis": {"recommendation": "discovery rec"},
                },
                {
                    "name": "wrap_up",
                    "results": [],
                    "synthesis": {"recommendation": "wrap_up rec — this one wins"},
                },
            ],
        }
        summary = build_poll_summary(env)
        # The summary prefers the LAST round's synthesis as the natural
        # "terminal" finding for v3 branching runs.
        assert summary.recommended_next_test == "wrap_up rec — this one wins"


class TestFreeTextQuestions:
    def test_free_text_kind_does_not_fabricate_counts(self) -> None:
        env: dict[str, object] = {
            "persona_count": 2,
            "questions": [{"text": "Why?", "response_schema": {"type": "text"}}],
            "results": [
                {
                    "persona": "A",
                    "responses": [{"question": "Why?", "response": "The pricing felt too high for what we got."}],
                },
                {
                    "persona": "B",
                    "responses": [{"question": "Why?", "response": "Onboarding took too long."}],
                },
            ],
        }
        summary = build_poll_summary(env)
        q0 = summary.questions[0]
        assert q0.kind == "text"
        assert q0.first_choice_counts is None
        assert q0.metric_average is None


class TestDeterminism:
    def test_same_input_produces_byte_identical_dict(self) -> None:
        env = _enum_result_envelope()
        a = build_poll_summary(env).to_dict()
        b = build_poll_summary(env).to_dict()
        assert a == b

    def test_summary_is_pickleable_safe(self) -> None:
        """No exotic objects — everything must round-trip through ``asdict``."""
        env = _enum_result_envelope()
        summary = build_poll_summary(env)
        as_dict = summary.to_dict()
        # The dataclass should preserve every documented top-level key.
        for key in ("persona_count", "questions", "segment_splits", "top_objections", "recommended_next_test"):
            assert key in as_dict


class TestExtractionFallbacks:
    """Schemaless / extraction-free results should still produce a summary
    when responses repeat heavily (inference fallback)."""

    def test_repeated_string_responses_infer_enum(self) -> None:
        env: dict[str, object] = {
            "persona_count": 6,
            "questions": [{"text": "Yes or no?"}],  # no schema
            "results": [
                {"persona": "p1", "responses": [{"question": "Yes or no?", "response": "yes"}]},
                {"persona": "p2", "responses": [{"question": "Yes or no?", "response": "yes"}]},
                {"persona": "p3", "responses": [{"question": "Yes or no?", "response": "yes"}]},
                {"persona": "p4", "responses": [{"question": "Yes or no?", "response": "no"}]},
                {"persona": "p5", "responses": [{"question": "Yes or no?", "response": "no"}]},
                {"persona": "p6", "responses": [{"question": "Yes or no?", "response": "yes"}]},
            ],
        }
        summary = build_poll_summary(env)
        q0 = summary.questions[0]
        assert q0.kind == "enum"
        assert q0.winner == "yes"
        assert q0.first_choice_counts == {"yes": 4, "no": 2}


def test_personas_lookup_supplies_missing_inline_attrs() -> None:
    """When panelist records lack attributes, the personas arg fills them in."""
    env: dict[str, object] = {
        "persona_count": 2,
        "questions": [{"text": "Pick", "response_schema": {"type": "enum", "options": ["X", "Y"]}}],
        "results": [
            {"persona": "Alice", "responses": [{"question": "Pick", "response": "X"}]},
            {"persona": "Bob", "responses": [{"question": "Pick", "response": "Y"}]},
        ],
    }
    personas = [
        {"name": "Alice", "tier": "free"},
        {"name": "Bob", "tier": "paid"},
    ]
    summary = build_poll_summary(env, personas=personas)
    splits = summary.segment_splits["0"]
    assert splits["tier"] == {"free": {"X": 1}, "paid": {"Y": 1}}


def test_empty_result_returns_empty_summary() -> None:
    summary = build_poll_summary({"results": []})
    assert isinstance(summary, PollSummary)
    assert summary.persona_count == 0
    assert summary.questions == []
    assert summary.top_objections == []


def test_question_summary_truncates_long_question_text() -> None:
    long_q = "Q" + "x" * 500
    env: dict[str, object] = {
        "questions": [{"text": long_q, "response_schema": {"type": "enum"}}],
        "results": [
            {"persona": "A", "responses": [{"question": long_q, "response": "Yes"}]},
            {"persona": "B", "responses": [{"question": long_q, "response": "Yes"}]},
            {"persona": "C", "responses": [{"question": long_q, "response": "No"}]},
        ],
    }
    qs: QuestionSummary = build_poll_summary(env).questions[0]
    # 280 chars is the safety cap so a single bad question can't push
    # the serialized summary into MB territory.
    assert len(qs.question) <= 280


# ---------------------------------------------------------------------------
# sy-bn7: structured choice-field detection
# ---------------------------------------------------------------------------


def _structured_schema_envelope() -> dict[str, object]:
    """Three-panelist run with --schema (JSON Schema) enforcing a structured response.

    Mirrors the shape persisted by ``synthpanel panel run --schema=...``:
    ``response`` is a dict, ``extraction`` is absent. Pre-sy-bn7 this
    shape silently fell through to ``kind: "text"`` (the GH #496 bug).
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "choice": {"type": "string", "enum": ["A", "B"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string"},
        },
        "required": ["choice", "confidence", "rationale"],
    }
    return {
        "persona_count": 3,
        "question_count": 1,
        "questions": [{"text": "A or B?", "response_schema": schema}],
        "results": [
            {
                "persona": "p1",
                "responses": [
                    {
                        "question": "A or B?",
                        "response": {"choice": "B", "confidence": 4, "rationale": "..."},
                    }
                ],
            },
            {
                "persona": "p2",
                "responses": [
                    {
                        "question": "A or B?",
                        "response": {"choice": "B", "confidence": 5, "rationale": "..."},
                    }
                ],
            },
            {
                "persona": "p3",
                "responses": [
                    {
                        "question": "A or B?",
                        "response": {"choice": "A", "confidence": 3, "rationale": "..."},
                    }
                ],
            },
        ],
    }


class TestStructuredChoiceFieldDetection:
    """sy-bn7: ``--schema`` runs land structured dicts in ``response`` rather
    than ``extraction``. The kind detector must recognize the JSON Schema
    enum property AND the dict-shaped response, otherwise the poll falls
    through to ``text`` and counts/winner come back as ``None`` (GH #496).
    """

    def test_json_schema_object_with_enum_property_detected_as_enum(self) -> None:
        summary = build_poll_summary(_structured_schema_envelope())
        q0 = summary.questions[0]
        assert q0.kind == "enum"
        assert q0.winner == "B"
        assert q0.first_choice_counts == {"B": 2, "A": 1}

    def test_structured_dict_response_without_schema_still_detected(self) -> None:
        """Even when the persisted ``questions`` block has no response_schema
        (legacy / minimal saves), the response-dict probe should kick in."""
        env = _structured_schema_envelope()
        env["questions"] = [{"text": "A or B?"}]  # strip schema
        summary = build_poll_summary(env)
        q0 = summary.questions[0]
        assert q0.kind == "enum"
        assert q0.first_choice_counts == {"B": 2, "A": 1}

    def test_confidence_field_powers_overall_metric_average(self) -> None:
        """sy-bn7: enum polls with a numeric confidence field expose
        ``metric_average`` (panel-wide confidence) alongside the per-choice
        weighted_scores."""
        summary = build_poll_summary(_structured_schema_envelope())
        q0 = summary.questions[0]
        # (4 + 5 + 3) / 3 = 4.0
        assert q0.metric_average == 4.0
        # weighted_scores stay per-choice: B mean=(4+5)/2=4.5, A=3
        assert q0.weighted_scores == {"B": 4.5, "A": 3.0}

    def test_ambiguous_schema_picks_known_key_and_warns(self) -> None:
        """When the schema has multiple enum properties, the detector
        deterministically prefers a name from ``_FIRST_CHOICE_KEYS`` and
        warns so the operator knows which field won the tiebreak."""
        env = _structured_schema_envelope()
        # Inject a second enum field that isn't in _FIRST_CHOICE_KEYS.
        env["questions"][0]["response_schema"]["properties"]["mood"] = {  # type: ignore[index]
            "type": "string",
            "enum": ["calm", "annoyed"],
        }
        for r in env["results"]:  # type: ignore[union-attr]
            r["responses"][0]["response"]["mood"] = "calm"
        summary = build_poll_summary(env)
        q0 = summary.questions[0]
        # 'choice' is in _FIRST_CHOICE_KEYS so it wins the tiebreak.
        assert q0.kind == "enum"
        assert q0.first_choice_counts == {"B": 2, "A": 1}
        # The warning announces both candidates and the chosen one.
        assert any("multiple enum fields" in w and "'choice'" in w for w in summary.warnings), (
            f"Expected ambiguity warning, got {summary.warnings!r}"
        )

    def test_explicit_choice_field_overrides_auto_detection(self) -> None:
        """``--choice-field`` wins over auto-detection AND silences the
        ambiguity warning — the caller already made the call."""
        env = _structured_schema_envelope()
        # Add a second enum the caller wants to vote on.
        env["questions"][0]["response_schema"]["properties"]["mood"] = {  # type: ignore[index]
            "type": "string",
            "enum": ["calm", "annoyed"],
        }
        for r, mood in zip(env["results"], ["annoyed", "calm", "annoyed"]):  # type: ignore[arg-type]
            r["responses"][0]["response"]["mood"] = mood
        summary = build_poll_summary(env, choice_field="mood")
        q0 = summary.questions[0]
        assert q0.kind == "enum"
        # Counts are now on `mood`, not `choice`.
        assert q0.first_choice_counts == {"annoyed": 2, "calm": 1}
        # No ambiguity warning — caller resolved it.
        assert not any("multiple enum fields" in w for w in summary.warnings)

    def test_explicit_confidence_field_drives_metric_average(self) -> None:
        env = _structured_schema_envelope()
        # Repurpose: add a second numeric field; verify --confidence-field
        # selects which one drives the metric_average.
        for r, urgency in zip(env["results"], [1, 2, 5]):  # type: ignore[arg-type]
            r["responses"][0]["response"]["urgency"] = urgency
        # Default: picks 'confidence' (in _SCORE_KEYS).
        default_summary = build_poll_summary(env)
        assert default_summary.questions[0].metric_average == 4.0
        # Override: pick 'urgency' instead.
        overridden = build_poll_summary(env, confidence_field="urgency")
        # (1 + 2 + 5) / 3 = 2.6667 → rounded to 4dp
        assert overridden.questions[0].metric_average == 2.6667

    def test_segment_splits_respect_choice_field_override(self) -> None:
        env = _structured_schema_envelope()
        # Tag each panelist with an inline occupation so the split joins.
        for r, occ in zip(env["results"], ["pm", "eng", "pm"]):  # type: ignore[arg-type]
            r["occupation"] = occ
        env["questions"][0]["response_schema"]["properties"]["mood"] = {  # type: ignore[index]
            "type": "string",
            "enum": ["calm", "annoyed"],
        }
        for r, mood in zip(env["results"], ["annoyed", "calm", "annoyed"]):  # type: ignore[arg-type]
            r["responses"][0]["response"]["mood"] = mood
        summary = build_poll_summary(env, choice_field="mood")
        # pm: annoyed (p1) + annoyed (p3) = {annoyed: 2}
        # eng: calm  (p2) = {calm: 1}
        assert summary.segment_splits["0"]["occupation"]["pm"] == {"annoyed": 2}
        assert summary.segment_splits["0"]["occupation"]["eng"] == {"calm": 1}
