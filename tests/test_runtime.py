"""Tests for the agent runtime / session loop (SPEC.md §3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.cost import ZERO_USAGE, TokenUsage, UsageTracker
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolInvocationBlock,
    TokenUsage as LLMTokenUsage,
)
from synth_panel.persistence import ConversationMessage, Session
from synth_panel.runtime import (
    AgentRuntime,
    AllowAllPolicy,
    HookResult,
    IterationLimitError,
    NoOpHookRunner,
    PermissionDecision,
    TurnSummary,
    _convert_usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(
    text: str = "Hello!",
    usage: LLMTokenUsage | None = None,
) -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=usage or LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_tool_response(
    tool_name: str = "search",
    tool_input: dict | None = None,
    tool_id: str = "call-1",
    usage: LLMTokenUsage | None = None,
) -> CompletionResponse:
    return CompletionResponse(
        id="resp-2",
        model="claude-sonnet",
        content=[ToolInvocationBlock(
            id=tool_id,
            name=tool_name,
            input=tool_input or {"query": "test"},
        )],
        stop_reason=StopReason.TOOL_USE,
        usage=usage or LLMTokenUsage(input_tokens=15, output_tokens=8),
    )


class FakeToolExecutor:
    """Simple tool executor for tests."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        self.calls.append((tool_name, tool_input))
        if tool_name in self._results:
            return self._results[tool_name]
        return f"Result for {tool_name}"


class ErrorToolExecutor:
    """Tool executor that raises for specific tools."""

    def __init__(self, error_tools: set[str] | None = None) -> None:
        self._error_tools = error_tools or set()

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name in self._error_tools:
            raise ValueError(f"Tool {tool_name} failed")
        return "ok"


class DenyPolicy:
    """Permission policy that denies all tool calls."""

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        return PermissionDecision.DENY


class DenyHookRunner:
    """Hook runner that denies in pre-tool-use."""

    def run_pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> HookResult:
        return HookResult(denied=True, messages=["Blocked by policy"])

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
    ) -> HookResult:
        return HookResult()


class FailHookRunner:
    """Hook runner that fails in pre-tool-use."""

    def run_pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> HookResult:
        return HookResult(failed=True, messages=["Hook crashed"])

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
    ) -> HookResult:
        return HookResult()


def _make_runtime(
    responses: list[CompletionResponse],
    *,
    tool_executor: Any = None,
    tools: list[ToolDefinition] | None = None,
    permission_policy: Any = None,
    hook_runner: Any = None,
    max_iterations: int = 20,
    compaction_threshold: int = 100_000,
    system_prompt: str | None = "You are a test assistant.",
) -> AgentRuntime:
    client = MagicMock()
    client.send = MagicMock(side_effect=responses)
    return AgentRuntime(
        client=client,
        session=Session(),
        system_prompt=system_prompt,
        tool_executor=tool_executor,
        tools=tools,
        model="claude-sonnet",
        max_tokens=1024,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        permission_policy=permission_policy,
        hook_runner=hook_runner,
    )


# ---------------------------------------------------------------------------
# Tests: basic turn
# ---------------------------------------------------------------------------

class TestBasicTurn:
    def test_simple_text_response(self):
        runtime = _make_runtime([_make_text_response("Hello world")])
        summary = runtime.run_turn("Hi")

        assert summary.iterations == 1
        assert len(summary.assistant_messages) == 1
        assert len(summary.tool_results) == 0
        assert summary.compacted is False

    def test_user_message_pushed_to_session(self):
        runtime = _make_runtime([_make_text_response()])
        runtime.run_turn("Hello there")

        messages = runtime.session.messages
        assert messages[0].role == "user"
        assert messages[0].content[0]["text"] == "Hello there"

    def test_assistant_message_pushed_to_session(self):
        runtime = _make_runtime([_make_text_response("Hey!")])
        runtime.run_turn("Hi")

        messages = runtime.session.messages
        assert messages[1].role == "assistant"
        assert messages[1].content[0]["text"] == "Hey!"

    def test_usage_tracked(self):
        usage = LLMTokenUsage(input_tokens=100, output_tokens=50)
        runtime = _make_runtime([_make_text_response(usage=usage)])
        summary = runtime.run_turn("Hi")

        assert summary.usage.input_tokens == 100
        assert summary.usage.output_tokens == 50
        assert runtime.usage_tracker.turn_count == 1
        assert runtime.usage_tracker.cumulative_usage.input_tokens == 100

    def test_multiple_turns_accumulate_usage(self):
        usage1 = LLMTokenUsage(input_tokens=100, output_tokens=50)
        usage2 = LLMTokenUsage(input_tokens=200, output_tokens=75)
        runtime = _make_runtime([
            _make_text_response("r1", usage=usage1),
            _make_text_response("r2", usage=usage2),
        ])
        runtime.run_turn("Turn 1")
        runtime.run_turn("Turn 2")

        assert runtime.usage_tracker.turn_count == 2
        assert runtime.usage_tracker.cumulative_usage.input_tokens == 300
        assert runtime.usage_tracker.cumulative_usage.output_tokens == 125


# ---------------------------------------------------------------------------
# Tests: tool execution
# ---------------------------------------------------------------------------

class TestToolExecution:
    def test_tool_call_and_response_loop(self):
        executor = FakeToolExecutor({"search": "found it"})
        tool_resp = _make_tool_response("search", {"query": "test"})
        text_resp = _make_text_response("Based on the search: found it")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
            tools=[ToolDefinition(name="search", input_schema={"type": "object"})],
        )
        summary = runtime.run_turn("Search for test")

        assert summary.iterations == 2
        assert len(summary.assistant_messages) == 2
        assert len(summary.tool_results) == 1
        assert executor.calls == [("search", {"query": "test"})]

    def test_tool_error_captured_not_raised(self):
        executor = ErrorToolExecutor(error_tools={"fail_tool"})
        tool_resp = _make_tool_response("fail_tool", {})
        text_resp = _make_text_response("I encountered an error")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
            tools=[ToolDefinition(name="fail_tool", input_schema={})],
        )
        summary = runtime.run_turn("Do something")

        # Tool error becomes error-flagged result, LLM continues
        assert summary.iterations == 2
        assert len(summary.tool_results) == 1
        tool_result = summary.tool_results[0]
        assert tool_result.content[0]["is_error"] is True
        assert "Tool execution error" in tool_result.content[0]["text"]

    def test_no_executor_ends_turn_on_tool_call(self):
        tool_resp = _make_tool_response("search", {})
        runtime = _make_runtime([tool_resp], tool_executor=None)
        summary = runtime.run_turn("Search")

        # Without an executor, tool call ends the turn
        assert summary.iterations == 1
        assert len(summary.tool_results) == 0

    def test_multiple_tool_calls_in_one_response(self):
        executor = FakeToolExecutor({"a": "result_a", "b": "result_b"})
        multi_tool_resp = CompletionResponse(
            id="resp-multi",
            model="claude-sonnet",
            content=[
                ToolInvocationBlock(id="c1", name="a", input={}),
                ToolInvocationBlock(id="c2", name="b", input={"x": 1}),
            ],
            stop_reason=StopReason.TOOL_USE,
            usage=LLMTokenUsage(input_tokens=20, output_tokens=10),
        )
        text_resp = _make_text_response("Done")

        runtime = _make_runtime(
            [multi_tool_resp, text_resp],
            tool_executor=executor,
        )
        summary = runtime.run_turn("Do both")

        assert len(summary.tool_results) == 2
        assert executor.calls == [("a", {}), ("b", {"x": 1})]

    def test_usage_accumulated_across_iterations(self):
        executor = FakeToolExecutor()
        u1 = LLMTokenUsage(input_tokens=50, output_tokens=20)
        u2 = LLMTokenUsage(input_tokens=80, output_tokens=30)
        tool_resp = _make_tool_response(usage=u1)
        text_resp = _make_text_response(usage=u2)

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
        )
        summary = runtime.run_turn("Go")

        assert summary.usage.input_tokens == 130
        assert summary.usage.output_tokens == 50


# ---------------------------------------------------------------------------
# Tests: iteration limit
# ---------------------------------------------------------------------------

class TestIterationLimit:
    def test_iteration_limit_exceeded(self):
        executor = FakeToolExecutor()
        # All responses are tool calls — will exceed limit
        responses = [_make_tool_response() for _ in range(5)]
        runtime = _make_runtime(
            responses,
            tool_executor=executor,
            max_iterations=3,
        )
        with pytest.raises(IterationLimitError) as exc_info:
            runtime.run_turn("Loop forever")
        assert exc_info.value.limit == 3


# ---------------------------------------------------------------------------
# Tests: permission policy
# ---------------------------------------------------------------------------

class TestPermissionPolicy:
    def test_deny_policy_blocks_tool(self):
        executor = FakeToolExecutor()
        tool_resp = _make_tool_response("search", {})
        text_resp = _make_text_response("Ok, I can't search")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
            permission_policy=DenyPolicy(),
        )
        summary = runtime.run_turn("Search")

        assert len(summary.tool_results) == 1
        assert summary.tool_results[0].content[0]["is_error"] is True
        assert "Permission denied" in summary.tool_results[0].content[0]["text"]
        assert executor.calls == []  # Tool was never executed

    def test_allow_all_default(self):
        policy = AllowAllPolicy()
        assert policy.check("anything", {}) == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# Tests: hooks
# ---------------------------------------------------------------------------

class TestHooks:
    def test_pre_hook_deny_blocks_tool(self):
        executor = FakeToolExecutor()
        tool_resp = _make_tool_response("search", {})
        text_resp = _make_text_response("Blocked")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
            hook_runner=DenyHookRunner(),
        )
        summary = runtime.run_turn("Search")

        assert summary.tool_results[0].content[0]["is_error"] is True
        assert "Blocked by policy" in summary.tool_results[0].content[0]["text"]
        assert executor.calls == []

    def test_pre_hook_fail_blocks_tool(self):
        executor = FakeToolExecutor()
        tool_resp = _make_tool_response("search", {})
        text_resp = _make_text_response("Hook failed")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
            hook_runner=FailHookRunner(),
        )
        summary = runtime.run_turn("Search")

        assert summary.tool_results[0].content[0]["is_error"] is True
        assert "Hook crashed" in summary.tool_results[0].content[0]["text"]

    def test_noop_hook_runner(self):
        runner = NoOpHookRunner()
        result = runner.run_pre_tool_use("test", {})
        assert not result.denied
        assert not result.failed
        result = runner.run_post_tool_use("test", {}, "output", False)
        assert not result.denied


# ---------------------------------------------------------------------------
# Tests: auto-compaction
# ---------------------------------------------------------------------------

class TestAutoCompaction:
    def test_compaction_triggered(self):
        # Use a very low threshold
        usage = LLMTokenUsage(input_tokens=200, output_tokens=50)
        runtime = _make_runtime(
            [_make_text_response(usage=usage)],
            compaction_threshold=100,
        )
        # Add some messages to the session so there's something to compact
        for i in range(5):
            runtime.session.push_message(ConversationMessage(
                role="user",
                content=[{"type": "text", "text": f"msg {i}"}],
            ))
            runtime.session.push_message(ConversationMessage(
                role="assistant",
                content=[{"type": "text", "text": f"reply {i}"}],
            ))

        summary = runtime.run_turn("Trigger compaction")
        assert summary.compacted is True
        assert runtime.session.compaction is not None
        assert runtime.session.compaction.compaction_count >= 1

    def test_no_compaction_below_threshold(self):
        usage = LLMTokenUsage(input_tokens=10, output_tokens=5)
        runtime = _make_runtime(
            [_make_text_response(usage=usage)],
            compaction_threshold=100_000,
        )
        summary = runtime.run_turn("Small turn")
        assert summary.compacted is False


# ---------------------------------------------------------------------------
# Tests: usage conversion
# ---------------------------------------------------------------------------

class TestUsageConversion:
    def test_convert_usage(self):
        llm_usage = LLMTokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_write_tokens=20,
            cache_read_tokens=10,
        )
        cost_usage = _convert_usage(llm_usage)
        assert cost_usage.input_tokens == 100
        assert cost_usage.output_tokens == 50
        assert cost_usage.cache_creation_input_tokens == 20
        assert cost_usage.cache_read_input_tokens == 10


# ---------------------------------------------------------------------------
# Tests: session state
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_session_grows_with_turns(self):
        runtime = _make_runtime([
            _make_text_response("r1"),
            _make_text_response("r2"),
        ])
        runtime.run_turn("Turn 1")
        assert len(runtime.session.messages) == 2  # user + assistant

        runtime.run_turn("Turn 2")
        assert len(runtime.session.messages) == 4  # 2 more

    def test_tool_results_in_session(self):
        executor = FakeToolExecutor({"calc": "42"})
        tool_resp = _make_tool_response("calc", {"expr": "6*7"})
        text_resp = _make_text_response("The answer is 42")

        runtime = _make_runtime(
            [tool_resp, text_resp],
            tool_executor=executor,
        )
        runtime.run_turn("Calculate 6*7")

        messages = runtime.session.messages
        # user, assistant (tool_use), tool_result, assistant (text)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "tool"
        assert messages[3].role == "assistant"

    def test_system_prompt_not_in_session(self):
        runtime = _make_runtime(
            [_make_text_response()],
            system_prompt="You are a test bot.",
        )
        runtime.run_turn("Hi")

        # System prompt goes via the request, not as a session message
        roles = [m.role for m in runtime.session.messages]
        assert "system" not in roles
