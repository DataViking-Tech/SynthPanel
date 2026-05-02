"""Structured output via tool-use forcing (SPEC.md §5).

The primary pattern:
1. Define a "respond" tool whose input schema matches the desired response format.
2. Set tool_choice to "specific" with the respond tool's name.
3. The LLM is forced to produce a tool invocation with valid JSON.
4. Extract the JSON input from the tool invocation as the structured response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from synth_panel.llm.client import LLMClient
from synth_panel.llm.errors import LLMError
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    TextBlock,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
    ToolInvocationBlock,
    ToolResultBlock,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_NAME = "respond"
_DEFAULT_RETRY_LIMIT = 2

# Patterns that identify cheap/flash-tier models warranting escalation on
# the final strike (sp-d1x0 retry policy).
_CHEAP_MODEL_PATTERNS = ("flash", "haiku", "lite", "mini", "nano", "small")
_ESCALATION_MODEL = "sonnet"

# sp-d1x0: terminal-failure warning, mirrors the sp-g59o synthesis warning
# surface so operators see consistent signal across extraction failures.
_TERMINAL_FAILURE_WARNING = (
    "structured output extraction failed after all retries — model may have "
    "ignored schema. sp-g59o: consider using a higher-quality model or "
    "simplifying the schema."
)


def _is_cheap_model(model: str) -> bool:
    """Return True when *model* resolves to a flash/cheap tier."""
    from synth_panel.llm.aliases import resolve_alias

    canonical = resolve_alias(model).lower()
    return any(p in canonical for p in _CHEAP_MODEL_PATTERNS)


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
    # sp-d1x0: cumulative usage across all retry attempts; callers should
    # prefer this over response.usage for accurate cost accounting.
    total_usage: TokenUsage = field(default_factory=TokenUsage)


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
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> StructuredResult:
        """Run a completion with tool-use forcing and extract structured data.

        Implements a 3-strike retry policy (sp-d1x0):
        - Strike 1: normal prompt
        - Strike 2: corrective prompt appended (same model)
        - Strike 3: corrective prompt + escalated model when original is cheap/flash
        """
        if not config.enabled:
            response = self._client.send(
                CompletionRequest(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            return StructuredResult(data={}, response=response, total_usage=response.usage)

        tool_def = ToolDefinition(
            name=config.tool_name,
            description=config.tool_description or "Respond with structured data.",
            input_schema=config.schema,
        )

        last_error: str | None = None
        last_response: CompletionResponse | None = None
        cumulative_usage = TokenUsage()

        for attempt in range(1 + config.retry_limit):
            # Strike 3: escalate to a higher-quality model on the final attempt
            # when the original model is in the cheap/flash tier.
            effective_model = model
            if attempt == config.retry_limit and _is_cheap_model(model):
                effective_model = _ESCALATION_MODEL
                logger.debug(
                    "structured output: escalating from %s to %s on final attempt",
                    model,
                    effective_model,
                )

            # Build messages: on retries, append the failed response + correction.
            effective_messages = (
                _build_retry_messages(messages, last_response, config.tool_name, last_error)
                if attempt > 0 and last_response is not None
                else list(messages)
            )

            request = CompletionRequest(
                model=effective_model,
                max_tokens=max_tokens,
                messages=effective_messages,
                system=system,
                tools=[tool_def],
                tool_choice=ToolChoice.specific(config.tool_name),
                temperature=temperature,
                top_p=top_p,
            )

            try:
                response = self._client.send(request)
            except LLMError:
                raise

            cumulative_usage = cumulative_usage + response.usage
            last_response = response

            extracted = _extract_tool_data(response, config.tool_name)
            if extracted is None:
                last_error = f"Attempt {attempt + 1}: LLM did not produce a valid '{config.tool_name}' tool call"
                continue

            # Validate required schema fields (sp-d1x0: schema non-conformance retry)
            missing = _missing_required(extracted, config.schema)
            if missing:
                last_error = f"Attempt {attempt + 1}: tool call missing required fields {missing}"
                continue

            return StructuredResult(
                data=extracted,
                response=response,
                retries_used=attempt,
                total_usage=cumulative_usage,
            )

        # All strikes exhausted — emit warning and return partial/fallback.
        logger.warning(
            "structured output extraction exhausted all %d retries (model=%s): %s",
            config.retry_limit,
            model,
            _TERMINAL_FAILURE_WARNING,
        )
        return StructuredResult(
            data={"_error": last_error, "_fallback": True},
            response=last_response,  # type: ignore[arg-type]
            retries_used=config.retry_limit,
            is_fallback=True,
            error=last_error,
            total_usage=cumulative_usage,
        )


def _extract_tool_data(
    response: CompletionResponse,
    tool_name: str,
) -> dict[str, Any] | None:
    """Extract structured data from the first matching tool invocation."""
    for block in response.content:
        if isinstance(block, ToolInvocationBlock) and block.name == tool_name and isinstance(block.input, dict):
            return block.input
    return None


def _missing_required(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Return names of required schema fields absent from *data*."""
    return [k for k in schema.get("required", []) if k not in data]


def _build_retry_messages(
    original_messages: list[InputMessage],
    failed_response: CompletionResponse,
    tool_name: str,
    error: str | None,
) -> list[InputMessage]:
    """Append the failed response + corrective turn to *original_messages*."""
    messages = list(original_messages)

    # Append the failed assistant turn so the model sees its mistake.
    messages.append(InputMessage(role="assistant", content=list(failed_response.content)))

    # Build a corrective user turn.
    correction: list[Any] = []
    tool_calls = [b for b in failed_response.content if isinstance(b, ToolInvocationBlock)]
    if tool_calls:
        # Provide tool_result blocks per API spec before appending text.
        for tc in tool_calls:
            err_text = f"Schema validation failed: {error}. Please call '{tool_name}' again with all required fields."
            correction.append(
                ToolResultBlock(
                    tool_use_id=tc.id,
                    content=[TextBlock(text=err_text)],
                    is_error=True,
                )
            )
    else:
        correction.append(
            TextBlock(
                text=(
                    f"You did not call the '{tool_name}' tool. "
                    f"You MUST use the '{tool_name}' tool with all required fields. "
                    "Please try again."
                )
            )
        )

    messages.append(InputMessage(role="user", content=correction))
    return messages
