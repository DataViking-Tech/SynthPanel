"""Acceptance tests for live API validation (SPEC.md §11).

These tests exercise real LLM API calls. Run with ANTHROPIC_API_KEY set:

    ANTHROPIC_API_KEY=sk-... pytest tests/test_acceptance.py -v

Every test is marked with ``pytest.mark.acceptance`` so the full unit suite
can skip them by default (``-m "not acceptance"``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from synth_panel.cost import (
    BudgetError,
    BudgetGate,
    estimate_cost,
    lookup_pricing,
)
from synth_panel.llm.aliases import resolve_alias
from synth_panel.llm.client import LLMClient
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    StreamEventType,
    TextBlock,
)
from synth_panel.persistence import Session, load_session, save_session
from synth_panel.runtime import AgentRuntime
from synth_panel.structured import StructuredOutputConfig, StructuredOutputEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_REASON = "ANTHROPIC_API_KEY not set"

acceptance = pytest.mark.acceptance


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _require_api_key():
    if not _has_api_key():
        pytest.skip(_SKIP_REASON)


def _hello_request(model: str = "sonnet", stream: bool = False) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=128,
        messages=[InputMessage(role="user", content=[TextBlock(text="Say hello in one sentence.")])],
        stream=stream,
    )


# ===================================================================
# LLM Client Abstraction
# ===================================================================


@acceptance
class TestLLMClientSend:
    """Send a simple prompt and receive a text response."""

    def test_send_hello(self):
        _require_api_key()
        client = LLMClient()
        response = client.send(_hello_request())

        assert isinstance(response, CompletionResponse)
        assert response.text, "Expected non-empty text response"
        assert len(response.text.split()) >= 1

    def test_send_uses_correct_model(self):
        _require_api_key()
        client = LLMClient()
        response = client.send(_hello_request("haiku"))

        # Anthropic returns the full model ID (may include a date suffix)
        assert "haiku" in response.model


@acceptance
class TestLLMClientStream:
    """Stream a prompt and verify event sequence."""

    def test_stream_event_sequence(self):
        _require_api_key()
        client = LLMClient()
        events = list(client.stream(_hello_request(stream=True)))

        assert len(events) > 0, "Expected at least one stream event"

        # Collect event types in order
        types = [e.type for e in events]

        # Spec requires: message_start -> content_block_start ->
        # content_block_delta(s) -> content_block_stop -> message_delta -> message_stop
        assert StreamEventType.MESSAGE_START in types
        assert StreamEventType.CONTENT_BLOCK_START in types
        assert StreamEventType.CONTENT_BLOCK_DELTA in types
        assert StreamEventType.CONTENT_BLOCK_STOP in types
        assert StreamEventType.MESSAGE_DELTA in types
        assert StreamEventType.MESSAGE_STOP in types

        # message_start must be first non-ping event
        non_ping = [t for t in types if t != StreamEventType.PING]
        assert non_ping[0] == StreamEventType.MESSAGE_START

        # message_stop must be last non-ping event
        assert non_ping[-1] == StreamEventType.MESSAGE_STOP


@acceptance
class TestLLMClientErrors:
    """Verify error classification for bad credentials and unreachable hosts."""

    def test_invalid_api_key(self):
        _require_api_key()
        # Temporarily override the env var
        original = os.environ["ANTHROPIC_API_KEY"]
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-invalid-key-for-testing"
            client = LLMClient(max_retries=0)
            with pytest.raises(LLMError) as exc_info:
                client.send(_hello_request())
            assert exc_info.value.category == LLMErrorCategory.AUTHENTICATION
        finally:
            os.environ["ANTHROPIC_API_KEY"] = original

    def test_unreachable_base_url(self):
        original_url = os.environ.get("ANTHROPIC_BASE_URL")
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        try:
            os.environ["ANTHROPIC_BASE_URL"] = "http://192.0.2.1:1"  # RFC 5737 TEST-NET
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-for-test"
            client = LLMClient(max_retries=1, initial_backoff=0.1, max_backoff=0.2)
            with pytest.raises(LLMError) as exc_info:
                client.send(_hello_request())
            assert exc_info.value.category in (
                LLMErrorCategory.TRANSPORT,
                LLMErrorCategory.RETRIES_EXHAUSTED,
            )
        finally:
            if original_url is None:
                os.environ.pop("ANTHROPIC_BASE_URL", None)
            else:
                os.environ["ANTHROPIC_BASE_URL"] = original_url
            if original_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = original_key


@acceptance
class TestModelAliasResolution:
    """Model alias resolution: 'sonnet' resolves to a valid canonical ID."""

    def test_sonnet_alias(self):
        resolved = resolve_alias("sonnet")
        assert resolved.startswith("claude-")
        assert "sonnet" in resolved

    def test_haiku_alias(self):
        resolved = resolve_alias("haiku")
        assert resolved.startswith("claude-")
        assert "haiku" in resolved

    def test_unknown_passthrough(self):
        assert resolve_alias("my-custom-model") == "my-custom-model"

    def test_alias_works_in_live_call(self):
        _require_api_key()
        client = LLMClient()
        # Use alias directly — client should resolve it
        response = client.send(_hello_request("sonnet"))
        assert response.text


# ===================================================================
# Structured Output
# ===================================================================


@acceptance
class TestStructuredOutput:
    """Force the LLM to respond via tool-use with a declared schema."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
        },
        "required": ["name", "sentiment", "confidence"],
    }

    def test_extract_structured(self):
        _require_api_key()
        client = LLMClient()
        engine = StructuredOutputEngine(client)
        config = StructuredOutputConfig(schema=self.SCHEMA)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=[
                InputMessage(
                    role="user",
                    content=[
                        TextBlock(
                            text=(
                                "Analyze the sentiment of this review: "
                                "'I absolutely love this product, it changed my life!'"
                            )
                        )
                    ],
                )
            ],
            config=config,
        )

        assert not result.is_fallback, f"Extraction failed: {result.error}"
        assert isinstance(result.data, dict)
        assert "name" in result.data
        assert result.data["sentiment"] in ("positive", "negative", "neutral")
        assert isinstance(result.data["confidence"], (int, float))

    def test_schema_validation(self):
        _require_api_key()
        client = LLMClient()
        engine = StructuredOutputEngine(client)
        config = StructuredOutputConfig(schema=self.SCHEMA)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=[
                InputMessage(
                    role="user",
                    content=[TextBlock(text="The weather is quite neutral today.")],
                )
            ],
            config=config,
        )

        assert isinstance(result.data, dict)
        # Confidence should be numeric
        if not result.is_fallback:
            assert isinstance(result.data["confidence"], (int, float))


# ===================================================================
# Cost / Budget Tracking
# ===================================================================


@acceptance
class TestCostTracking:
    """Verify token usage and cost estimation after live calls."""

    def test_usage_non_zero(self):
        _require_api_key()
        client = LLMClient()
        response = client.send(_hello_request())

        usage = response.usage
        assert usage.input_tokens > 0, "Expected non-zero input tokens"
        assert usage.output_tokens > 0, "Expected non-zero output tokens"

    def test_cost_estimate_non_zero(self):
        _require_api_key()
        client = LLMClient()
        response = client.send(_hello_request())

        # Convert LLM TokenUsage to cost-layer TokenUsage
        from synth_panel.cost import TokenUsage as CostTokenUsage

        cost_usage = CostTokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=response.usage.cache_write_tokens,
            cache_read_input_tokens=response.usage.cache_read_tokens,
        )

        pricing, _ = lookup_pricing("sonnet")
        cost = estimate_cost(cost_usage, pricing)

        assert cost.total_cost > 0, "Expected non-zero cost estimate"
        assert cost.format_usd().startswith("$")

    def test_budget_enforcement(self):
        _require_api_key()
        client = LLMClient()
        response = client.send(_hello_request())

        from synth_panel.cost import TokenUsage as CostTokenUsage

        cost_usage = CostTokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        # Set a budget smaller than what was just used
        gate = BudgetGate(max_tokens=100)
        gate.record_turn(cost_usage)

        # Should raise on the next check
        with pytest.raises(BudgetError):
            gate.check()


# ===================================================================
# Session Persistence
# ===================================================================


@acceptance
class TestSessionPersistence:
    """Run a turn, save, load, fork — verify roundtrip integrity."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(client=client, session=session, model="sonnet")

        runtime.run_turn("What is 2+2?")
        msg_count = len(session.messages)
        assert msg_count >= 2  # at least user + assistant

        # Save and reload
        path = tmp_path / "session.json"
        save_session(session, path)
        loaded = load_session(path)

        assert len(loaded.messages) == msg_count
        assert loaded.session_id == session.session_id

    def test_jsonl_append(self, tmp_path: Path):
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(client=client, session=session, model="sonnet")

        runtime.run_turn("Say one word.")

        path = tmp_path / "session.jsonl"
        save_session(session, path, fmt="jsonl")

        # Read raw lines
        lines = path.read_text().strip().splitlines()
        # JSONL: 1 meta line + N message lines
        assert len(lines) >= 3  # meta + user msg + assistant msg

        # Each line must be valid JSON
        for line in lines:
            json.loads(line)

    def test_fork_session(self, tmp_path: Path):
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(client=client, session=session, model="sonnet")

        runtime.run_turn("What color is the sky?")

        forked = session.fork_session(branch_name="test-fork")
        assert forked.fork is not None
        assert forked.fork.parent_session_id == session.session_id
        assert len(forked.messages) == len(session.messages)
        assert forked.session_id != session.session_id


# ===================================================================
# Agent Runtime
# ===================================================================


@acceptance
class TestAgentRuntime:
    """Live runtime tests — single turn, usage tracking, compaction."""

    def test_single_turn(self):
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(client=client, session=session, model="sonnet")

        summary = runtime.run_turn("What is 2+2?")

        assert len(summary.assistant_messages) >= 1
        # The response should contain text
        text = ""
        for msg in summary.assistant_messages:
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        assert text, "Expected non-empty assistant response"

    def test_cumulative_usage(self):
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(client=client, session=session, model="sonnet")

        runtime.run_turn("Say hello.")
        usage1 = runtime.usage_tracker.cumulative_usage

        runtime.run_turn("Say goodbye.")
        usage2 = runtime.usage_tracker.cumulative_usage

        assert usage2.input_tokens > usage1.input_tokens
        assert runtime.usage_tracker.turn_count == 2

    def test_auto_compaction(self):
        """Set a very low compaction threshold so it triggers."""
        _require_api_key()
        client = LLMClient()
        session = Session()
        runtime = AgentRuntime(
            client=client,
            session=session,
            model="sonnet",
            compaction_threshold=1,  # triggers after first turn
        )

        summary = runtime.run_turn("Tell me a short joke.")
        assert summary.compacted, "Expected auto-compaction to trigger"
        assert session.compaction is not None


# ===================================================================
# CLI Framework
# ===================================================================


@acceptance
class TestCLI:
    """Test CLI commands via subprocess for true end-to-end validation."""

    def _run(self, *args: str, expect_rc: int = 0) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, "-m", "synth_panel", *args],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if expect_rc is not None:
            assert result.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {result.returncode}\n"
                f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
            )
        return result

    def test_prompt_text(self):
        _require_api_key()
        result = self._run("prompt", "Say hello in one word")
        assert len(result.stdout.strip()) > 0

    def test_prompt_json(self):
        _require_api_key()
        result = self._run("prompt", "Say hello", "--output-format", "json")
        data = json.loads(result.stdout)
        assert "message" in data
        assert "usage" in data

    def test_help(self):
        result = self._run("--help")
        assert "synth" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_invalid_subcommand(self):
        result = self._run("nonexistent", expect_rc=2)
        assert result.returncode != 0


# ===================================================================
# End-to-End: 3-Persona Panel Run
# ===================================================================

_PERSONAS_YAML = """\
personas:
  - name: Skeptical CTO
    age: 48
    occupation: CTO
    background: >
      20 years in enterprise software. Seen many products come and go.
      Values proven technology over hype.
    personality_traits:
      - skeptical
      - analytical
      - risk-averse

  - name: Enthusiastic Intern
    age: 22
    occupation: Software Engineering Intern
    background: >
      Recent CS graduate. Excited about new tools and technology.
      Active on social media and follows tech trends closely.
    personality_traits:
      - enthusiastic
      - optimistic
      - trend-following

  - name: Pragmatic PM
    age: 35
    occupation: Product Manager
    background: >
      10 years managing B2B products. Focused on ROI and user adoption metrics.
      Needs tools that integrate with existing workflows.
    personality_traits:
      - pragmatic
      - data-driven
      - user-focused
"""

_INSTRUMENT_YAML = """\
instrument:
  questions:
    - text: >
        What do you think of the name 'Traitprint' for a career matching app?
      response_schema:
        type: text
"""


@acceptance
class TestEndToEnd:
    """Run a full 3-persona panel and verify structured output."""

    def test_panel_run(self, tmp_path: Path):
        _require_api_key()
        personas_path = tmp_path / "personas.yaml"
        instrument_path = tmp_path / "instrument.yaml"
        personas_path.write_text(_PERSONAS_YAML)
        instrument_path.write_text(_INSTRUMENT_YAML)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "synth_panel",
                "panel",
                "run",
                "--personas",
                str(personas_path),
                "--instrument",
                str(instrument_path),
                "--model",
                "sonnet",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        assert result.returncode == 0, (
            f"Panel run failed (rc={result.returncode})\nstdout: {result.stdout[:1000]}\nstderr: {result.stderr[:1000]}"
        )

        output = result.stdout + result.stderr
        # Should see output from each persona (3 responses)
        assert len(output.strip()) > 0, "Expected non-empty panel output"

    def test_panel_run_json(self, tmp_path: Path):
        _require_api_key()
        personas_path = tmp_path / "personas.yaml"
        instrument_path = tmp_path / "instrument.yaml"
        personas_path.write_text(_PERSONAS_YAML)
        instrument_path.write_text(_INSTRUMENT_YAML)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "synth_panel",
                "panel",
                "run",
                "--personas",
                str(personas_path),
                "--instrument",
                str(instrument_path),
                "--model",
                "sonnet",
                "--output-format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        assert result.returncode == 0, (
            f"Panel run (JSON) failed (rc={result.returncode})\nstderr: {result.stderr[:1000]}"
        )

        # JSON mode should produce parseable output
        stdout = result.stdout.strip()
        if stdout:
            data = json.loads(stdout)
            assert isinstance(data, dict)


# ===================================================================
# CLI runner for standalone execution
# ===================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
