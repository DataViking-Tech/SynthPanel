"""Tests for LLM error classification."""

from __future__ import annotations

from synth_panel.llm.errors import LLMError, LLMErrorCategory, classify_http_status


def test_retryable_categories():
    assert LLMError("x", LLMErrorCategory.TRANSPORT).retryable is True
    assert LLMError("x", LLMErrorCategory.RATE_LIMIT).retryable is True
    assert LLMError("x", LLMErrorCategory.SERVER_ERROR).retryable is True

    assert LLMError("x", LLMErrorCategory.MISSING_CREDENTIALS).retryable is False
    assert LLMError("x", LLMErrorCategory.AUTHENTICATION).retryable is False
    assert LLMError("x", LLMErrorCategory.BAD_REQUEST).retryable is False
    assert LLMError("x", LLMErrorCategory.DESERIALIZATION).retryable is False
    assert LLMError("x", LLMErrorCategory.RETRIES_EXHAUSTED).retryable is False


def test_classify_http_status():
    assert classify_http_status(401) == LLMErrorCategory.AUTHENTICATION
    assert classify_http_status(403) == LLMErrorCategory.AUTHENTICATION
    assert classify_http_status(429) == LLMErrorCategory.RATE_LIMIT
    assert classify_http_status(400) == LLMErrorCategory.BAD_REQUEST
    assert classify_http_status(422) == LLMErrorCategory.BAD_REQUEST
    assert classify_http_status(500) == LLMErrorCategory.SERVER_ERROR
    assert classify_http_status(503) == LLMErrorCategory.SERVER_ERROR
