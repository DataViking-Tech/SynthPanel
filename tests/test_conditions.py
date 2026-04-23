"""Tests for condition evaluation module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import synth_panel.conditions as conditions_module
from synth_panel.conditions import (
    ConditionError,
    evaluate_condition,
    normalize_follow_up,
    validate_condition_string,
)
from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage


@pytest.fixture(autouse=True)
def _reset_condition_warn_state():
    """Reset the module-level warn dedup tables between tests."""
    conditions_module._warned_unknown_types.clear()
    conditions_module._warned_missing_client = False
    yield
    conditions_module._warned_unknown_types.clear()
    conditions_module._warned_missing_client = False


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
    def test_completely_unknown_type(self):
        assert evaluate_condition("some_future_condition: arg", "text") is True

    def test_unknown_without_arg(self):
        assert evaluate_condition("future_check", "text") is True

    def test_unknown_emits_warning_once(self, caplog):
        """Typos log a warning on first occurrence so instrument authors notice."""
        with caplog.at_level("WARNING", logger="synth_panel.conditions"):
            evaluate_condition("response_contians: yes", "I said yes")
            evaluate_condition("response_contians: yes", "and again")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "response_contians" in warnings[0].getMessage()
        assert "typo" in warnings[0].getMessage().lower()

    def test_distinct_unknowns_warn_separately(self, caplog):
        with caplog.at_level("WARNING", logger="synth_panel.conditions"):
            evaluate_condition("typo_one", "text")
            evaluate_condition("typo_two", "text")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# evaluate_condition — "response_sentiment" (LLM-based)
# ---------------------------------------------------------------------------


def _mock_client(classification: str) -> MagicMock:
    """Create a mock LLMClient that returns *classification* as text."""
    client = MagicMock()
    client.send.return_value = CompletionResponse(
        id="test",
        model="claude-haiku-4-5-20251001",
        content=[TextBlock(text=classification)],
        usage=TokenUsage(),
    )
    return client


class TestResponseSentiment:
    def test_positive_match(self):
        client = _mock_client("positive")
        assert (
            evaluate_condition(
                "response_sentiment: positive",
                "I love this product!",
                client=client,
            )
            is True
        )
        client.send.assert_called_once()

    def test_negative_match(self):
        client = _mock_client("negative")
        assert (
            evaluate_condition(
                "response_sentiment: negative",
                "This is terrible",
                client=client,
            )
            is True
        )

    def test_neutral_match(self):
        client = _mock_client("neutral")
        assert (
            evaluate_condition(
                "response_sentiment: neutral",
                "It's okay I guess",
                client=client,
            )
            is True
        )

    def test_mismatch(self):
        client = _mock_client("negative")
        assert (
            evaluate_condition(
                "response_sentiment: positive",
                "This is terrible",
                client=client,
            )
            is False
        )

    def test_case_insensitive_target(self):
        client = _mock_client("positive")
        assert (
            evaluate_condition(
                "response_sentiment: POSITIVE",
                "great!",
                client=client,
            )
            is True
        )

    def test_case_insensitive_classification(self):
        client = _mock_client("  Positive  ")
        assert (
            evaluate_condition(
                "response_sentiment: positive",
                "great!",
                client=client,
            )
            is True
        )

    def test_no_client_defaults_to_true(self):
        """Graceful degradation: no client means condition passes."""
        assert evaluate_condition("response_sentiment: negative", "text") is True

    def test_no_client_warns_once(self, caplog):
        """Missing client is a loud failure mode — first occurrence warns (sp-t5ok)."""
        with caplog.at_level("WARNING", logger="synth_panel.conditions"):
            evaluate_condition("response_sentiment: positive", "a")
            evaluate_condition("response_sentiment: negative", "b")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "without an LLM client" in warnings[0].getMessage()

    def test_cache_hit_avoids_llm_call(self):
        client = _mock_client("positive")
        cache: dict[str, str] = {"I love it": "positive"}
        result = evaluate_condition(
            "response_sentiment: positive",
            "I love it",
            client=client,
            sentiment_cache=cache,
        )
        assert result is True
        client.send.assert_not_called()

    def test_cache_populated_after_call(self):
        client = _mock_client("negative")
        cache: dict[str, str] = {}
        evaluate_condition(
            "response_sentiment: negative",
            "awful experience",
            client=client,
            sentiment_cache=cache,
        )
        assert cache["awful experience"] == "negative"

    def test_cache_shared_across_calls(self):
        client = _mock_client("positive")
        cache: dict[str, str] = {}
        # First call populates cache
        evaluate_condition(
            "response_sentiment: positive",
            "great product",
            client=client,
            sentiment_cache=cache,
        )
        assert client.send.call_count == 1
        # Second call uses cache
        evaluate_condition(
            "response_sentiment: positive",
            "great product",
            client=client,
            sentiment_cache=cache,
        )
        assert client.send.call_count == 1  # No additional call

    def test_cache_with_lock(self):
        """Cache works correctly when a lock is supplied."""
        client = _mock_client("positive")
        cache: dict[str, str] = {}
        lock = threading.Lock()
        evaluate_condition(
            "response_sentiment: positive",
            "great stuff",
            client=client,
            sentiment_cache=cache,
            sentiment_cache_lock=lock,
        )
        assert cache["great stuff"] == "positive"
        assert client.send.call_count == 1
        # Second call uses cache (via lock)
        evaluate_condition(
            "response_sentiment: positive",
            "great stuff",
            client=client,
            sentiment_cache=cache,
            sentiment_cache_lock=lock,
        )
        assert client.send.call_count == 1

    def test_unexpected_classification_defaults_neutral(self):
        client = _mock_client("ambivalent")  # not a valid sentiment
        assert (
            evaluate_condition(
                "response_sentiment: neutral",
                "mixed feelings",
                client=client,
            )
            is True
        )
        # "ambivalent" → "neutral" (default), matches "neutral" target
        assert (
            evaluate_condition(
                "response_sentiment: positive",
                "mixed feelings",
                client=client,
            )
            is False
        )


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


# ---------------------------------------------------------------------------
# validate_condition_string (parse-time check, sp-t5ok)
# ---------------------------------------------------------------------------


class TestValidateConditionString:
    @pytest.mark.parametrize(
        "condition",
        [
            "always",
            "never",
            "response_contains: yes",
            "response_contains:",
            "response_sentiment: positive",
            "ALWAYS",
            "  response_contains: anything  ",
        ],
    )
    def test_known_conditions_pass(self, condition):
        # Should not raise.
        validate_condition_string(condition)

    @pytest.mark.parametrize(
        "condition",
        [
            "response_contians: yes",  # classic typo from sp-t5ok
            "sentiment: positive",  # missing response_ prefix
            "future_check",
            "",
            "   ",
        ],
    )
    def test_unknown_or_empty_raises(self, condition):
        with pytest.raises(ConditionError):
            validate_condition_string(condition)

    def test_non_string_raises(self):
        with pytest.raises(ConditionError, match="must be a string"):
            validate_condition_string(42)

    def test_error_message_includes_context(self):
        with pytest.raises(ConditionError, match="round 'probe'"):
            validate_condition_string("response_contians: x", context="round 'probe'")
