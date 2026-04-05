"""Tests for condition evaluation module."""

import json

from synth_panel.conditions import evaluate_condition, normalize_follow_up


# ---------------------------------------------------------------------------
# evaluate_condition — "always"
# ---------------------------------------------------------------------------

class TestAlways:
    def test_always_returns_true(self):
        assert evaluate_condition("always", "anything") is True

    def test_always_case_insensitive(self):
        assert evaluate_condition("Always", "") is True
        assert evaluate_condition("ALWAYS", "text") is True

    def test_always_with_whitespace(self):
        assert evaluate_condition("  always  ", "text") is True


# ---------------------------------------------------------------------------
# evaluate_condition — "never"
# ---------------------------------------------------------------------------

class TestNever:
    def test_never_returns_false(self):
        assert evaluate_condition("never", "anything") is False

    def test_never_case_insensitive(self):
        assert evaluate_condition("Never", "") is False
        assert evaluate_condition("NEVER", "text") is False


# ---------------------------------------------------------------------------
# evaluate_condition — "response_contains"
# ---------------------------------------------------------------------------

class TestResponseContains:
    def test_match(self):
        assert evaluate_condition("response_contains: yes", "I said yes to that") is True

    def test_no_match(self):
        assert evaluate_condition("response_contains: yes", "I said no to that") is False

    def test_case_insensitive(self):
        assert evaluate_condition("response_contains: YES", "yes indeed") is True
        assert evaluate_condition("response_contains: hello", "HELLO world") is True

    def test_keyword_with_spaces(self):
        assert evaluate_condition("response_contains: very much", "I like it very much") is True

    def test_empty_keyword(self):
        # Empty keyword is always contained in any string
        assert evaluate_condition("response_contains:", "any text") is True

    def test_empty_response(self):
        assert evaluate_condition("response_contains: word", "") is False


# ---------------------------------------------------------------------------
# evaluate_condition — unknown conditions (forward-compat)
# ---------------------------------------------------------------------------

class TestUnknownCondition:
    def test_unknown_defaults_to_true(self):
        assert evaluate_condition("response_sentiment: positive", "great!") is True

    def test_completely_unknown_type(self):
        assert evaluate_condition("some_future_condition: arg", "text") is True

    def test_unknown_without_arg(self):
        assert evaluate_condition("future_check", "text") is True


# ---------------------------------------------------------------------------
# evaluate_condition — structured responses
# ---------------------------------------------------------------------------

class TestStructuredResponses:
    def test_dict_response_serialized(self):
        response = {"rating": 5, "comment": "excellent product"}
        assert evaluate_condition("response_contains: excellent", response) is True

    def test_dict_response_no_match(self):
        response = {"rating": 1, "comment": "terrible"}
        assert evaluate_condition("response_contains: excellent", response) is False

    def test_list_response(self):
        response = ["item1", "item2", "target_word"]
        assert evaluate_condition("response_contains: target_word", response) is True

    def test_numeric_response(self):
        assert evaluate_condition("response_contains: 42", 42) is True

    def test_none_response(self):
        assert evaluate_condition("response_contains: null", None) is True

    def test_bool_response(self):
        assert evaluate_condition("response_contains: true", True) is True


# ---------------------------------------------------------------------------
# normalize_follow_up
# ---------------------------------------------------------------------------

class TestNormalizeFollowUp:
    def test_string_to_dict(self):
        result = normalize_follow_up("Can you elaborate?")
        assert result == {"text": "Can you elaborate?", "condition": "always"}

    def test_dict_passthrough_with_condition(self):
        inp = {"text": "Why?", "condition": "response_contains: yes"}
        result = normalize_follow_up(inp)
        assert result == {"text": "Why?", "condition": "response_contains: yes"}

    def test_dict_defaults_condition(self):
        result = normalize_follow_up({"text": "Tell me more"})
        assert result == {"text": "Tell me more", "condition": "always"}

    def test_dict_preserves_extra_keys(self):
        inp = {"text": "Q?", "condition": "always", "weight": 0.5}
        result = normalize_follow_up(inp)
        assert result["weight"] == 0.5

    def test_does_not_mutate_input(self):
        inp = {"text": "Q?"}
        normalize_follow_up(inp)
        assert "condition" not in inp
