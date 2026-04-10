"""Data models for the LLM client abstraction (SPEC.md §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    """Plain text content."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolInvocationBlock:
    """A tool call requested by the assistant."""

    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    """Result of a tool execution, fed back to the assistant."""

    tool_use_id: str
    content: list[TextBlock]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class ThinkingBlock:
    """Internal reasoning (extended thinking)."""

    thinking: str
    signature: str | None = None
    type: Literal["thinking"] = "thinking"


ContentBlock = TextBlock | ToolInvocationBlock | ToolResultBlock | ThinkingBlock


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class InputMessage:
    """A single message in the conversation."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption counters for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_write_tokens + self.cache_read_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


# ---------------------------------------------------------------------------
# Tool definitions & choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """Schema for a tool the LLM may invoke."""

    name: str
    input_schema: dict[str, Any]
    description: str | None = None


class ToolChoiceKind(Enum):
    AUTO = "auto"
    ANY = "any"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class ToolChoice:
    """Constraint on which tool the LLM should use."""

    kind: ToolChoiceKind
    name: str | None = None  # only for SPECIFIC

    @classmethod
    def auto(cls) -> ToolChoice:
        return cls(kind=ToolChoiceKind.AUTO)

    @classmethod
    def any(cls) -> ToolChoice:
        return cls(kind=ToolChoiceKind.ANY)

    @classmethod
    def specific(cls, name: str) -> ToolChoice:
        return cls(kind=ToolChoiceKind.SPECIFIC, name=name)


# ---------------------------------------------------------------------------
# Completion request
# ---------------------------------------------------------------------------


@dataclass
class CompletionRequest:
    """Everything needed to send one request to an LLM provider."""

    model: str
    max_tokens: int
    messages: list[InputMessage]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    stream: bool = False


# ---------------------------------------------------------------------------
# Completion response
# ---------------------------------------------------------------------------


class StopReason(Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"


@dataclass
class CompletionResponse:
    """Parsed response from an LLM provider."""

    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def text(self) -> str:
        """Convenience: concatenated text from all TextBlocks."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolInvocationBlock]:
        """Convenience: all tool invocation blocks."""
        return [b for b in self.content if isinstance(b, ToolInvocationBlock)]


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


class StreamEventType(Enum):
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    PING = "ping"


@dataclass
class StreamEvent:
    """A single event from an SSE stream."""

    type: StreamEventType
    index: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
