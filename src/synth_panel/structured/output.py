"""Structured output via tool-use forcing (SPEC.md §5).

The primary pattern:
1. Define a "respond" tool whose input schema matches the desired response format.
2. Set tool_choice to "specific" with the respond tool's name.
3. The LLM is forced to produce a tool invocation with valid JSON.
4. Extract the JSON input from the tool invocation as the structured response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synth_panel.llm.client import LLMClient
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    TextBlock,
    ToolChoice,
    ToolDefinition,
    ToolInvocationBlock,
)

_DEFAULT_TOOL_NAME = "respond"
_DEFAULT_RETRY_LIMIT = 2


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output extraction."""
    schema: dict[str, Any]
    tool_name: str = _DEFAULT_TOOL_NAME
    tool_description: str | None = None
    retry_limit: int = _DEFAULT_RETRY_LIMIT
    enabled: bool = True


@dataclass
class StructuredResult:
    """Result of a structured output extraction."""
    data: dict[str, Any]
    response: CompletionResponse
    retries_used: int = 0
    is_fallback: bool = False
    error: str | None = None


class StructuredOutputEngine:
    """Wraps an LLMClient to extract structured responses via tool-use forcing.

    Usage::

        engine = StructuredOutputEngine(client)
        config = StructuredOutputConfig(schema={
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "summary": {"type": "string"},
            },
            "required": ["sentiment", "summary"],
        })
        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=[InputMessage(role="user", content=[TextBlock(text="Analyze this.")])],
            config=config,
        )
        print(result.data)  # {"sentiment": "positive", "summary": "..."}
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[InputMessage],
        config: StructuredOutputConfig,
        system: str | None = None,
    ) -> StructuredResult:
        """Run a completion with tool-use forcing and extract structured data."""
        if not config.enabled:
            # Structured output disabled — do a plain completion
            response = self._client.send(CompletionRequest(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
            ))
            return StructuredResult(data={}, response=response)

        tool_def = ToolDefinition(
            name=config.tool_name,
            description=config.tool_description or "Respond with structured data.",
            input_schema=config.schema,
        )

        last_error: str | None = None
        for attempt in range(1 + config.retry_limit):
            request = CompletionRequest(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
                tools=[tool_def],
                tool_choice=ToolChoice.specific(config.tool_name),
            )

            try:
                response = self._client.send(request)
            except LLMError:
                raise  # LLM errors propagate — retries are at the client level

            # Extract the tool invocation
            extracted = _extract_tool_data(response, config.tool_name)
            if extracted is not None:
                return StructuredResult(
                    data=extracted,
                    response=response,
                    retries_used=attempt,
                )

            # No valid tool call found — retry
            last_error = (
                f"Attempt {attempt + 1}: LLM did not produce a valid "
                f"'{config.tool_name}' tool call"
            )

        # All retries exhausted — return fallback
        return StructuredResult(
            data={"_error": last_error, "_fallback": True},
            response=response,  # type: ignore[possibly-undefined]
            retries_used=config.retry_limit,
            is_fallback=True,
            error=last_error,
        )


def _extract_tool_data(
    response: CompletionResponse,
    tool_name: str,
) -> dict[str, Any] | None:
    """Extract structured data from the first matching tool invocation."""
    for block in response.content:
        if isinstance(block, ToolInvocationBlock) and block.name == tool_name:
            if isinstance(block.input, dict):
                return block.input
    return None
