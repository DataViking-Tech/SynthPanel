"""Tests for synth_panel.templates — template engine for dynamic questions."""

from __future__ import annotations

from synth_panel.synthesis import SynthesisResult
from synth_panel.templates import (
    build_template_context,
    render_questions,
    validate_template,
)


def _make_synthesis(**overrides) -> SynthesisResult:
    defaults = dict(
        summary="Overall positive reception",
        themes=["usability", "pricing", "integration"],
        agreements=["easy to learn"],
        disagreements=["pricing model"],
        surprises=["unexpected mobile usage"],
        recommendation="Focus on mobile experience",
    )
    defaults.update(overrides)
    return SynthesisResult(**defaults)


class TestBuildTemplateContext:
    def test_basic_flattening(self):
        synthesis = _make_synthesis()
        ctx = build_template_context(synthesis)

        assert ctx["summary"] == "Overall positive reception"
        assert ctx["recommendation"] == "Focus on mobile experience"
        assert ctx["theme_0"] == "usability"
        assert ctx["theme_1"] == "pricing"
        assert ctx["theme_2"] == "integration"
        assert ctx["agreement_0"] == "easy to learn"
        assert ctx["disagreement_0"] == "pricing model"
        assert ctx["surprise_0"] == "unexpected mobile usage"

    def test_empty_synthesis(self):
        synthesis = _make_synthesis(
            summary="",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
        )
        ctx = build_template_context(synthesis)

        assert ctx["summary"] == ""
        assert ctx["recommendation"] == ""
        # No indexed keys should exist
        assert "theme_0" not in ctx
        assert "agreement_0" not in ctx
        assert "disagreement_0" not in ctx
        assert "surprise_0" not in ctx

    def test_many_themes(self):
        synthesis = _make_synthesis(themes=[f"theme_{i}" for i in range(10)])
        ctx = build_template_context(synthesis)
        for i in range(10):
            assert ctx[f"theme_{i}"] == f"theme_{i}"


class TestRenderQuestions:
    def test_basic_rendering(self):
        ctx = {"theme_0": "usability", "summary": "Good overall"}
        questions = [{"text": "Tell me about {theme_0}"}]
        result = render_questions(questions, ctx)

        assert result[0]["text"] == "Tell me about usability"

    def test_missing_key_renders_literal(self):
        ctx = {"theme_0": "usability"}
        questions = [{"text": "Tell me about {nonexistent}"}]
        result = render_questions(questions, ctx)

        assert result[0]["text"] == "Tell me about {nonexistent}"

    def test_does_not_mutate_original(self):
        ctx = {"theme_0": "usability"}
        questions = [{"text": "About {theme_0}"}]
        result = render_questions(questions, ctx)

        assert questions[0]["text"] == "About {theme_0}"
        assert result[0]["text"] == "About usability"

    def test_follow_ups_string(self):
        ctx = {"theme_0": "usability"}
        questions = [
            {
                "text": "Main question",
                "follow_ups": ["Tell me more about {theme_0}"],
            }
        ]
        result = render_questions(questions, ctx)
        assert result[0]["follow_ups"][0] == "Tell me more about usability"

    def test_follow_ups_dict(self):
        ctx = {"agreement_0": "easy to learn"}
        questions = [
            {
                "text": "Main question",
                "follow_ups": [{"text": "You agreed on {agreement_0}?"}],
            }
        ]
        result = render_questions(questions, ctx)
        assert result[0]["follow_ups"][0]["text"] == "You agreed on easy to learn?"

    def test_multiple_placeholders(self):
        ctx = {"theme_0": "usability", "theme_1": "pricing"}
        questions = [{"text": "Compare {theme_0} and {theme_1}"}]
        result = render_questions(questions, ctx)

        assert result[0]["text"] == "Compare usability and pricing"

    def test_no_placeholders(self):
        ctx = {"theme_0": "usability"}
        questions = [{"text": "A plain question with no placeholders"}]
        result = render_questions(questions, ctx)

        assert result[0]["text"] == "A plain question with no placeholders"

    def test_empty_questions(self):
        result = render_questions([], {"theme_0": "usability"})
        assert result == []

    def test_non_text_fields_untouched(self):
        ctx = {"theme_0": "usability"}
        questions = [{"text": "About {theme_0}", "response_schema": {"type": "text"}}]
        result = render_questions(questions, ctx)

        assert result[0]["response_schema"] == {"type": "text"}

    def test_injection_attempt_curly_braces(self):
        """Ensure format strings can't trigger code execution."""
        ctx = {"theme_0": "usability"}
        questions = [{"text": "{__class__.__mro__}"}]
        result = render_questions(questions, ctx)
        # Should render as literal — not expose internals
        assert result[0]["text"] == "{__class__.__mro__}"

    def test_injection_attempt_format_spec(self):
        ctx = {"theme_0": "usability"}
        questions = [{"text": "{theme_0!r}"}]
        result = render_questions(questions, ctx)
        assert result[0]["text"] == "'usability'"


class TestValidateTemplate:
    def test_all_keys_valid(self):
        ctx = {"theme_0": "usability", "summary": "Good"}
        result = validate_template("About {theme_0}: {summary}", ctx)
        assert result == []

    def test_missing_keys(self):
        ctx = {"theme_0": "usability"}
        result = validate_template("About {theme_0} and {missing_key}", ctx)
        assert result == ["missing_key"]

    def test_no_placeholders(self):
        result = validate_template("A plain string", {})
        assert result == []

    def test_multiple_missing(self):
        ctx = {}
        result = validate_template("{a} and {b} and {c}", ctx)
        assert set(result) == {"a", "b", "c"}

    def test_empty_context(self):
        result = validate_template("{theme_0}", {})
        assert result == ["theme_0"]

    def test_empty_text(self):
        result = validate_template("", {"theme_0": "usability"})
        assert result == []
