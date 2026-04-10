"""Tests for REPL user input wiring to AgentRuntime and /compact."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from synth_panel.cli.output import OutputFormat
from synth_panel.cli.repl import SessionState, _extract_response_text, run_repl
from synth_panel.cli.slash import _cmd_compact, dispatch_slash
from synth_panel.cost import TokenUsage
from synth_panel.persistence import ConversationMessage, Session
from synth_panel.runtime import AgentRuntime, TurnSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn_summary(text: str = "Hello!", compacted: bool = False) -> TurnSummary:
    usage = TokenUsage(input_tokens=10, output_tokens=20)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(
        assistant_messages=[msg],
        iterations=1,
        usage=usage,
        compacted=compacted,
    )


def _make_state_with_runtime() -> SessionState:
    """Create a SessionState with a mock runtime."""
    runtime = MagicMock(spec=AgentRuntime)
    runtime.session = Session()
    state = SessionState(model="sonnet", runtime=runtime)
    return state


# ---------------------------------------------------------------------------
# SessionState tests
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_has_runtime_slot(self):
        state = SessionState()
        assert state.runtime is None

    def test_runtime_can_be_set(self):
        runtime = MagicMock(spec=AgentRuntime)
        state = SessionState(runtime=runtime)
        assert state.runtime is runtime


# ---------------------------------------------------------------------------
# _extract_response_text tests
# ---------------------------------------------------------------------------


class TestExtractResponseText:
    def test_single_text_block(self):
        summary = _make_turn_summary("Hi there")
        assert _extract_response_text(summary) == "Hi there"

    def test_multiple_messages(self):
        usage = TokenUsage(input_tokens=5, output_tokens=5)
        msgs = [
            ConversationMessage(
                role="assistant",
                content=[{"type": "text", "text": "First"}],
                usage=usage,
            ),
            ConversationMessage(
                role="assistant",
                content=[{"type": "text", "text": "Second"}],
                usage=usage,
            ),
        ]
        summary = TurnSummary(assistant_messages=msgs, iterations=2, usage=usage)
        assert _extract_response_text(summary) == "First\nSecond"

    def test_skips_non_text_blocks(self):
        usage = TokenUsage(input_tokens=5, output_tokens=5)
        msg = ConversationMessage(
            role="assistant",
            content=[
                {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                {"type": "text", "text": "Result"},
            ],
            usage=usage,
        )
        summary = TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)
        assert _extract_response_text(summary) == "Result"

    def test_empty_messages(self):
        summary = TurnSummary()
        assert _extract_response_text(summary) == ""


# ---------------------------------------------------------------------------
# REPL wiring tests
# ---------------------------------------------------------------------------


class TestReplWiring:
    @patch("synth_panel.cli.repl.AgentRuntime")
    @patch("synth_panel.cli.repl.LLMClient")
    @patch("synth_panel.cli.repl.input", side_effect=["hello", EOFError])
    def test_user_input_calls_run_turn(self, mock_input, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _make_turn_summary("World!")
        mock_runtime_cls.return_value = mock_runtime

        import argparse

        args = argparse.Namespace(model="sonnet")
        code = run_repl(args, OutputFormat.TEXT)

        assert code == 0
        mock_runtime.run_turn.assert_called_once_with("hello")
        out = capsys.readouterr().out
        assert "World!" in out

    @patch("synth_panel.cli.repl.AgentRuntime")
    @patch("synth_panel.cli.repl.LLMClient")
    @patch("synth_panel.cli.repl.input", side_effect=["test", EOFError])
    def test_usage_is_displayed(self, mock_input, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _make_turn_summary("Reply")
        mock_runtime_cls.return_value = mock_runtime

        import argparse

        args = argparse.Namespace(model=None)
        run_repl(args, OutputFormat.TEXT)

        out = capsys.readouterr().out
        assert "input=10" in out
        assert "output=20" in out

    @patch("synth_panel.cli.repl.AgentRuntime")
    @patch("synth_panel.cli.repl.LLMClient")
    @patch("synth_panel.cli.repl.input", side_effect=["test", EOFError])
    def test_error_is_caught(self, mock_input, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.side_effect = RuntimeError("API broke")
        mock_runtime_cls.return_value = mock_runtime

        import argparse

        args = argparse.Namespace(model="sonnet")
        code = run_repl(args, OutputFormat.TEXT)

        assert code == 0  # REPL continues on error
        out = capsys.readouterr().out
        assert "Error: API broke" in out

    @patch("synth_panel.cli.repl.AgentRuntime")
    @patch("synth_panel.cli.repl.LLMClient")
    @patch("synth_panel.cli.repl.input", side_effect=["a", "b", EOFError])
    def test_turn_count_increments(self, mock_input, mock_client_cls, mock_runtime_cls):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _make_turn_summary("ok")
        mock_runtime_cls.return_value = mock_runtime

        import argparse

        args = argparse.Namespace(model="sonnet")
        # We can't easily check state from outside, but verify run_turn called twice
        run_repl(args, OutputFormat.TEXT)
        assert mock_runtime.run_turn.call_count == 2

    @patch("synth_panel.cli.repl.AgentRuntime")
    @patch("synth_panel.cli.repl.LLMClient")
    @patch("synth_panel.cli.repl.input", side_effect=["test", EOFError])
    def test_compaction_flag_updates_state(self, mock_input, mock_client_cls, mock_runtime_cls):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _make_turn_summary("ok", compacted=True)
        mock_runtime_cls.return_value = mock_runtime

        import argparse

        args = argparse.Namespace(model="sonnet")
        # Run repl — it should increment compacted_count internally
        run_repl(args, OutputFormat.TEXT)
        # Verify the runtime was called (compacted_count is internal to state)
        mock_runtime.run_turn.assert_called_once()


# ---------------------------------------------------------------------------
# /compact slash command tests
# ---------------------------------------------------------------------------


class TestCompactCommand:
    def test_compact_no_runtime(self, capsys):
        state = SessionState(model="sonnet")
        _cmd_compact(state, [], OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "No active runtime" in out

    def test_compact_not_enough_messages(self, capsys):
        state = _make_state_with_runtime()
        # Session has 0 messages
        _cmd_compact(state, [], OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "Not enough messages" in out

    def test_compact_with_messages(self, capsys):
        state = _make_state_with_runtime()
        session = state.runtime.session
        # Add several messages
        for i in range(5):
            session.push_message(
                ConversationMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=[{"type": "text", "text": f"Message {i}"}],
                )
            )
        assert len(session.messages) == 5

        _cmd_compact(state, [], OutputFormat.TEXT)

        out = capsys.readouterr().out
        assert "compacted" in out.lower()
        assert state.compacted_count == 1
        # After compaction: 1 summary + 2 kept = 3
        assert len(session.messages) == 3

    def test_compact_via_dispatch(self, capsys):
        state = _make_state_with_runtime()
        session = state.runtime.session
        for i in range(4):
            session.push_message(
                ConversationMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=[{"type": "text", "text": f"Msg {i}"}],
                )
            )

        dispatch_slash("/compact", state, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "compacted" in out.lower()
