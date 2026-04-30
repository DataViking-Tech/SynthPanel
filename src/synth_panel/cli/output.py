"""Output formatting for synthpanel CLI.

Supports text, json, and ndjson formats per SPEC.md §8.
"""

from __future__ import annotations

import enum
import json
import shutil
import sys
import textwrap
from typing import Any


def terminal_columns(*, fallback: int = 80) -> int:
    """Best-effort terminal width for CLI wrapping."""

    try:
        return max(20, shutil.get_terminal_size().columns)
    except OSError:
        return fallback


def format_prose_for_inspect(
    text: str,
    *,
    wrap_width: int,
    full: bool,
    indent: str = "    ",
) -> list[str]:
    """Split long persona / pack prose for ``pack inspect`` text output.

    Default (*full* = False): collapse internal whitespace and word-wrap to
    *wrap_width* using *indent* on every line. With *full* = True: preserve
    newlines and do not wrap — every character in *text* appears in output.
    """

    if not str(text).strip():
        return [f"{indent}(empty)"]

    if full:
        lines = str(text).splitlines()
        out: list[str] = []
        for line in lines:
            out.append(f"{indent}{line}")
        return out if out else [f"{indent}"]

    collapsed = " ".join(str(text).split())
    filled = textwrap.fill(
        collapsed,
        width=max(wrap_width, 20),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=True,
        break_on_hyphens=True,
    )
    return filled.splitlines() or [indent]


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
