"""Output formatting for synth-panel CLI.

Supports text, json, and ndjson formats per SPEC.md §8.
"""

from __future__ import annotations

import enum
import json
import sys
from typing import Any


class OutputFormat(enum.Enum):
    TEXT = "text"
    JSON = "json"
    NDJSON = "ndjson"


def emit(
    fmt: OutputFormat,
    *,
    message: str,
    usage: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
    file: Any = None,
) -> None:
    """Emit a response in the requested format.

    Args:
        fmt: Output format.
        message: The response text.
        usage: Optional token usage dict (input_tokens, output_tokens).
        extra: Optional additional fields for structured output.
        file: Output stream (defaults to sys.stdout).
    """
    out = file or sys.stdout

    if fmt is OutputFormat.TEXT:
        print(message, file=out)
        if usage:
            print(
                f"  tokens: input={usage.get('input_tokens', 0)} output={usage.get('output_tokens', 0)}",
                file=out,
            )
    elif fmt is OutputFormat.JSON:
        payload: dict[str, Any] = {"message": message}
        if usage:
            payload["usage"] = usage
        if extra:
            payload.update(extra)
        print(json.dumps(payload), file=out)
    elif fmt is OutputFormat.NDJSON:
        payload = {"type": "message", "text": message}
        if usage:
            payload["usage"] = usage
        if extra:
            payload.update(extra)
        print(json.dumps(payload), file=out)
