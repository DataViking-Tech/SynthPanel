"""Tests for Jinja2 system prompt template support (GH-315)."""

from __future__ import annotations

import jinja2
import pytest

from synth_panel.instrument import Instrument, InstrumentError, parse_instrument
from synth_panel.prompts import (
    compile_jinja2_template,
    is_jinja2_template,
    persona_system_prompt_from_template,
)

PERSONA = {
    "name": "Alice Chen",
    "age": 34,
    "occupation": "Product Manager",
    "background": "10 years in SaaS",
    "personality_traits": ["analytical", "pragmatic"],
    "custom_field": "enterprise-focused",
}


class TestIsJinja2Template:
    def test_double_brace_variable(self):
        assert is_jinja2_template("Hello {{ name }}") is True

    def test_block_tag(self):
        assert is_jinja2_template("{% if age %}Age: {{ age }}{% endif %}") is True

    def test_comment_tag(self):
        assert is_jinja2_template("{# this is a comment #}") is True

    def test_legacy_format_string(self):
        assert is_jinja2_template("Hello {name}, age {age}") is False

    def test_plain_string(self):
        assert is_jinja2_template("You are a helpful assistant.") is False

    def test_empty_string(self):
        assert is_jinja2_template("") is False


class TestCompileJinja2Template:
    def test_valid_template_returns_template_object(self):
        t = compile_jinja2_template("Hello {{ name }}")
        assert isinstance(t, jinja2.Template)

    def test_syntax_error_raises(self):
        with pytest.raises(jinja2.TemplateSyntaxError):
            compile_jinja2_template("Hello {% if unclosed")

    def test_render_basic_variable(self):
        t = compile_jinja2_template("Hello {{ name }}")
        assert t.render(name="Alice") == "Hello Alice"

    def test_sandboxed_blocks_dunder_attribute_access(self):
        t = compile_jinja2_template("{{ ''.__class__ }}")
        result = t.render()
        assert "str" not in result and "type" not in result

    def test_sandboxed_blocks_nested_dunder_access(self):
        t = compile_jinja2_template("{{ ''.__class__.__mro__ }}")
        with pytest.raises(Exception):
            t.render()


class TestPersonaSystemPromptFromTemplate:
    def test_precompiled_jinja2_template(self):
        t = compile_jinja2_template("You are {{ name }}, age {{ age }}.")
        result = persona_system_prompt_from_template(PERSONA, t)
        assert result == "You are Alice Chen, age 34."

    def test_raw_jinja2_string_auto_detected(self):
        template = "You are {{ name }}, a {{ occupation }}."
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "You are Alice Chen, a Product Manager."

    def test_legacy_format_string(self):
        template = "You are {name}, age {age}."
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "You are Alice Chen, age 34."

    def test_legacy_unknown_key_preserved(self):
        template = "You are {name}. {unknown_key} here."
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "You are Alice Chen. {unknown_key} here."

    def test_jinja2_conditional(self):
        template = "You are {{ name }}.{% if background %} Background: {{ background }}.{% endif %}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "You are Alice Chen. Background: 10 years in SaaS."

    def test_jinja2_conditional_missing_field(self):
        template = "You are {{ name }}.{% if missing_field %} {{ missing_field }}.{% endif %}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "You are Alice Chen."

    def test_jinja2_personality_traits_as_list(self):
        template = "Traits: {% for t in personality_traits %}{{ t }}{% if not loop.last %}, {% endif %}{% endfor %}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Traits: analytical, pragmatic"

    def test_jinja2_personality_traits_join_filter(self):
        template = "Traits: {{ personality_traits | join(', ') }}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Traits: analytical, pragmatic"

    def test_legacy_personality_traits_joined(self):
        template = "Traits: {personality_traits}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Traits: analytical, pragmatic"

    def test_jinja2_custom_persona_field(self):
        template = "Focus: {{ custom_field }}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Focus: enterprise-focused"

    def test_legacy_custom_persona_field(self):
        template = "Focus: {custom_field}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Focus: enterprise-focused"

    def test_jinja2_numeric_fields(self):
        template = "Age: {{ age }}"
        result = persona_system_prompt_from_template(PERSONA, template)
        assert result == "Age: 34"

    def test_precompiled_used_per_render(self):
        t = compile_jinja2_template("{{ name }}")
        p1 = {"name": "Alice"}
        p2 = {"name": "Bob"}
        assert persona_system_prompt_from_template(p1, t) == "Alice"
        assert persona_system_prompt_from_template(p2, t) == "Bob"


class TestInstrumentSystemPromptTemplate:
    def _make_instrument_data(self, spt=None):
        data = {
            "version": 1,
            "questions": [{"text": "What do you think?"}],
        }
        if spt is not None:
            data["system_prompt_template"] = spt
        return data

    def test_no_template_defaults_none(self):
        inst = parse_instrument(self._make_instrument_data())
        assert inst.system_prompt_template is None

    def test_legacy_template_stored(self):
        inst = parse_instrument(self._make_instrument_data(spt="You are {name}."))
        assert inst.system_prompt_template == "You are {name}."

    def test_jinja2_template_stored(self):
        inst = parse_instrument(self._make_instrument_data(spt="You are {{ name }}."))
        assert inst.system_prompt_template == "You are {{ name }}."

    def test_jinja2_syntax_error_raises_instrument_error(self):
        with pytest.raises(InstrumentError, match="system_prompt_template"):
            parse_instrument(self._make_instrument_data(spt="{% if unclosed"))

    def test_non_string_template_raises(self):
        data = self._make_instrument_data()
        data["system_prompt_template"] = 42
        with pytest.raises(InstrumentError, match="must be a string"):
            parse_instrument(data)

    def test_valid_jinja2_with_conditionals_parses_ok(self):
        spt = "You are {{ name }}, age {{ age }}.{% if background %} Background: {{ background }}.{% endif %}"
        inst = parse_instrument(self._make_instrument_data(spt=spt))
        assert inst.system_prompt_template == spt

    def test_v3_instrument_with_template(self):
        data = {
            "version": 3,
            "system_prompt_template": "You are {{ name }}.",
            "rounds": [
                {
                    "name": "intro",
                    "questions": [{"text": "What brings you here?"}],
                    "route_when": [{"else": "__end__"}],
                }
            ],
        }
        inst = parse_instrument(data)
        assert inst.system_prompt_template == "You are {{ name }}."

    def test_template_renders_with_custom_fields(self):
        spt = "You are {{ name }}. Focus: {{ custom_field | default('general') }}."
        inst = parse_instrument(self._make_instrument_data(spt=spt))
        compiled = compile_jinja2_template(inst.system_prompt_template)
        result = compiled.render(name="Alice", custom_field="enterprise")
        assert result == "You are Alice. Focus: enterprise."

    def test_template_default_filter_for_missing_field(self):
        spt = "You are {{ name }}. Focus: {{ custom_field | default('general') }}."
        inst = parse_instrument(self._make_instrument_data(spt=spt))
        compiled = compile_jinja2_template(inst.system_prompt_template)
        result = compiled.render(name="Alice")
        assert result == "You are Alice. Focus: general."

    def test_instrument_dataclass_field_present(self):
        inst = Instrument(version=1)
        assert inst.system_prompt_template is None
        inst.system_prompt_template = "You are {{ name }}."
        assert inst.system_prompt_template == "You are {{ name }}."
