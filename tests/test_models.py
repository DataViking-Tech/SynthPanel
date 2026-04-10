"""Tests for LLM data models."""

from __future__ import annotations

from synth_panel.llm.models import (
    CompletionResponse,
    TextBlock,
    TokenUsage,
    ToolChoice,
    ToolChoiceKind,
    ToolInvocationBlock,
)


class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(input_tokens=10, output_tokens=20, cache_write_tokens=5, cache_read_tokens=3)
        assert u.total_tokens == 38

    def test_addition(self):
        a = TokenUsage(input_tokens=10, output_tokens=20)
        b = TokenUsage(input_tokens=5, output_tokens=15, cache_write_tokens=2)
        c = a + b
        assert c.input_tokens == 15
        assert c.output_tokens == 35
        assert c.cache_write_tokens == 2
        assert c.cache_read_tokens == 0

    def test_default_zeros(self):
        u = TokenUsage()
        assert u.total_tokens == 0


class TestToolChoice:
    def test_auto(self):
        tc = ToolChoice.auto()
        assert tc.kind == ToolChoiceKind.AUTO
        assert tc.name is None

    def test_any(self):
        tc = ToolChoice.any()
        assert tc.kind == ToolChoiceKind.ANY

    def test_specific(self):
        tc = ToolChoice.specific("respond")
        assert tc.kind == ToolChoiceKind.SPECIFIC
        assert tc.name == "respond"


class TestCompletionResponse:
    def test_text_property(self):
        r = CompletionResponse(
            id="r1",
            model="test",
            content=[TextBlock(text="Hello "), TextBlock(text="world")],
        )
        assert r.text == "Hello world"

    def test_tool_calls_property(self):
        tc = ToolInvocationBlock(id="tc1", name="respond", input={"key": "val"})
        r = CompletionResponse(
            id="r1",
            model="test",
            content=[TextBlock(text="preamble"), tc],
        )
        assert r.tool_calls == [tc]
        assert r.text == "preamble"
