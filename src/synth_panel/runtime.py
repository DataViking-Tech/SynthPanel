"""Agent Runtime / Session Loop (SPEC.md §3).

Manages the lifecycle of a single agent conversation: accept user input,
send it to the LLM, execute any tool calls the LLM requests, and loop
until the LLM produces a final text response or a limit is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from synth_panel.cost import ZERO_USAGE, TokenUsage, UsageTracker
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    StopReason,
    TextBlock,
    ToolChoice,
    ToolDefinition,
    ToolInvocationBlock,
    ToolResultBlock,
    TokenUsage as LLMTokenUsage,
)
from synth_panel.persistence import ConversationMessage, Session


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RuntimeError_(Exception):
    """Base error for the agent runtime."""


class IterationLimitError(RuntimeError_):
    """Raised when the iteration limit within a turn is exceeded."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"Iteration limit exceeded: {limit}")


# ---------------------------------------------------------------------------
# Tool executor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolExecutor(Protocol):
    """Any component that can execute a tool by name."""

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str: ...


# ---------------------------------------------------------------------------
# Permission policy
# ---------------------------------------------------------------------------

class PermissionDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"


@runtime_checkable
class PermissionPolicy(Protocol):
    """Controls which tools are allowed or denied."""

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision: ...


class AllowAllPolicy:
    """Default policy: allow everything."""

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        return PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# Hook runner
# ---------------------------------------------------------------------------

@dataclass
class HookResult:
    """Result of running a hook chain."""

    denied: bool = False
    failed: bool = False
    messages: list[str] = field(default_factory=list)


@runtime_checkable
class HookRunner(Protocol):
    """Pre- and post-tool-use hook interceptor."""

    def run_pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> HookResult: ...

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
    ) -> HookResult: ...


class NoOpHookRunner:
    """Default hook runner that does nothing."""

    def run_pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> HookResult:
        return HookResult()

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
    ) -> HookResult:
        return HookResult()


# ---------------------------------------------------------------------------
# Turn summary
# ---------------------------------------------------------------------------

@dataclass
class TurnSummary:
    """Result of a complete conversational turn."""

    assistant_messages: list[ConversationMessage] = field(default_factory=list)
    tool_results: list[ConversationMessage] = field(default_factory=list)
    iterations: int = 0
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)
    compacted: bool = False


# ---------------------------------------------------------------------------
# Usage conversion
# ---------------------------------------------------------------------------

def _convert_usage(llm_usage: LLMTokenUsage) -> TokenUsage:
    """Convert LLM-layer TokenUsage to cost-layer TokenUsage."""
    return TokenUsage(
        input_tokens=llm_usage.input_tokens,
        output_tokens=llm_usage.output_tokens,
        cache_creation_input_tokens=llm_usage.cache_write_tokens,
        cache_read_input_tokens=llm_usage.cache_read_tokens,
    )


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _session_messages_to_input(messages: list[ConversationMessage]) -> list[InputMessage]:
    """Convert persistence messages to LLM input messages.

    System messages are skipped (they go via the system prompt parameter).
    Tool-result messages become user messages with tool_result content blocks
    (matching the Anthropic API convention).
    """
    result: list[InputMessage] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "tool":
            # Tool results are sent as user-role messages per API convention
            result.append(InputMessage(role="user", content=[
                ToolResultBlock(
                    tool_use_id=block.get("tool_use_id", ""),
                    content=[TextBlock(text=block.get("text", ""))],
                    is_error=block.get("is_error", False),
                )
                if block.get("type") == "tool_result"
                else TextBlock(text=block.get("text", ""))
                for block in msg.content
            ]))
        elif msg.role in ("user", "assistant"):
            content_blocks = []
            for block in msg.content:
                btype = block.get("type", "text")
                if btype == "text":
                    content_blocks.append(TextBlock(text=block.get("text", "")))
                elif btype == "tool_use":
                    content_blocks.append(ToolInvocationBlock(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
                elif btype == "tool_result":
                    content_blocks.append(ToolResultBlock(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=[TextBlock(text=block.get("text", ""))],
                        is_error=block.get("is_error", False),
                    ))
            result.append(InputMessage(role=msg.role, content=content_blocks))
    return result


def _response_to_conversation_message(
    response: CompletionResponse,
    usage: TokenUsage,
) -> ConversationMessage:
    """Convert an LLM response to a persistence ConversationMessage."""
    content: list[dict[str, Any]] = []
    for block in response.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolInvocationBlock):
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return ConversationMessage(role="assistant", content=content, usage=usage)


def _tool_result_message(
    tool_use_id: str,
    output: str,
    is_error: bool = False,
) -> ConversationMessage:
    """Create a tool result conversation message."""
    return ConversationMessage(
        role="tool",
        content=[{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "text": output,
            "is_error": is_error,
        }],
    )


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_COMPACTION_THRESHOLD = 100_000  # tokens


class AgentRuntime:
    """Core agent loop with tool execution, hooks, and auto-compaction.

    Usage::

        runtime = AgentRuntime(
            client=LLMClient(),
            session=Session(),
            system_prompt="You are a helpful assistant.",
            tool_executor=my_executor,
            tools=[ToolDefinition(name="search", input_schema={...})],
        )
        summary = runtime.run_turn("What is 2+2?")
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        session: Session,
        system_prompt: str | None = None,
        tool_executor: ToolExecutor | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        model: str = "sonnet",
        max_tokens: int = 4096,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        permission_policy: PermissionPolicy | None = None,
        hook_runner: HookRunner | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self._client = client
        self._session = session
        self._system_prompt = system_prompt
        self._tool_executor = tool_executor
        self._tools = tools
        self._tool_choice = tool_choice
        self._model = model
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._compaction_threshold = compaction_threshold
        self._permission_policy = permission_policy or AllowAllPolicy()
        self._hook_runner = hook_runner or NoOpHookRunner()
        self._usage_tracker = usage_tracker or UsageTracker()

    @property
    def session(self) -> Session:
        return self._session

    @property
    def usage_tracker(self) -> UsageTracker:
        return self._usage_tracker

    def run_turn(self, user_input: str) -> TurnSummary:
        """Execute a complete conversational turn.

        A turn may involve multiple iterations if the LLM requests tool use.
        """
        # 1. Push user message
        user_msg = ConversationMessage(
            role="user",
            content=[{"type": "text", "text": user_input}],
        )
        self._session.push_message(user_msg)

        summary = TurnSummary()
        turn_usage = ZERO_USAGE

        for iteration in range(self._max_iterations):
            summary.iterations = iteration + 1

            # 2. Build API request
            input_messages = _session_messages_to_input(self._session.messages)
            request = CompletionRequest(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=input_messages,
                system=self._system_prompt,
                tools=self._tools,
                tool_choice=self._tool_choice,
            )

            # 3. Call LLM
            response = self._client.send(request)

            # 4-5. Process response and record usage
            resp_usage = _convert_usage(response.usage)
            turn_usage = turn_usage + resp_usage

            assistant_msg = _response_to_conversation_message(response, resp_usage)
            self._session.push_message(assistant_msg)
            summary.assistant_messages.append(assistant_msg)

            # 6. Check for tool calls
            tool_calls = response.tool_calls
            if not tool_calls:
                break

            if self._tool_executor is None:
                # No executor registered — treat as end of turn
                break

            # 7. Execute each tool call
            for tool_call in tool_calls:
                tool_result_msg = self._execute_tool(tool_call)
                self._session.push_message(tool_result_msg)
                summary.tool_results.append(tool_result_msg)

            # 8. Check iteration limit (loop continues to step 2)
        else:
            # Loop exhausted without a non-tool-use response
            raise IterationLimitError(self._max_iterations)

        # Record cumulative turn usage
        self._usage_tracker.record_turn(turn_usage)
        summary.usage = turn_usage

        # 9. Check auto-compaction
        cumulative_input = self._usage_tracker.cumulative_usage.input_tokens
        if cumulative_input > self._compaction_threshold:
            self._auto_compact()
            summary.compacted = True

        return summary

    def _execute_tool(self, tool_call: ToolInvocationBlock) -> ConversationMessage:
        """Execute a single tool call with hooks and permission checks."""
        assert self._tool_executor is not None

        # 7a. Pre-tool-use hooks
        pre_result = self._hook_runner.run_pre_tool_use(
            tool_call.name, tool_call.input
        )
        if pre_result.denied:
            reason = "; ".join(pre_result.messages) if pre_result.messages else "Denied by hook"
            return _tool_result_message(tool_call.id, reason, is_error=True)
        if pre_result.failed:
            reason = "; ".join(pre_result.messages) if pre_result.messages else "Hook failed"
            return _tool_result_message(tool_call.id, reason, is_error=True)

        # 7b. Permission check
        decision = self._permission_policy.check(tool_call.name, tool_call.input)
        if decision == PermissionDecision.DENY:
            return _tool_result_message(
                tool_call.id,
                f"Permission denied for tool '{tool_call.name}'",
                is_error=True,
            )

        # 7c. Execute tool
        is_error = False
        try:
            output = self._tool_executor.execute(tool_call.name, tool_call.input)
        except Exception as exc:
            output = f"Tool execution error: {exc}"
            is_error = True

        # 7d. Post-tool-use hooks
        self._hook_runner.run_post_tool_use(
            tool_call.name, tool_call.input, output, is_error
        )

        # 7e. Push tool result
        return _tool_result_message(tool_call.id, output, is_error=is_error)

    def _auto_compact(self) -> None:
        """Compact older messages into a summary."""
        if len(self._session.messages) <= 4:
            return
        # Build a summary from the older messages
        older = self._session.messages[:-2]
        summary_parts: list[str] = []
        for msg in older:
            role = msg.role
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        summary_parts.append(f"[{role}]: {text[:200]}")
        summary_text = (
            "Compacted conversation summary:\n"
            + "\n".join(summary_parts[:20])
        )
        self._session.compact(summary_text, keep_last=2)
