"""Interactive REPL for synthpanel.

Entered when no subcommand is given. Supports slash commands per SPEC.md §8.
"""

from __future__ import annotations

import argparse

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.slash import dispatch_slash
from synth_panel.llm.client import LLMClient
from synth_panel.persistence import Session
from synth_panel.runtime import AgentRuntime


class SessionState:
    """Mutable state maintained during an interactive session."""

    __slots__ = ("compacted_count", "last_usage", "model", "runtime", "turn_count")

    def __init__(self, model: str | None = None, runtime: AgentRuntime | None = None) -> None:
        self.turn_count: int = 0
        self.compacted_count: int = 0
        self.model: str | None = model
        self.last_usage: dict[str, int] | None = None
        self.runtime: AgentRuntime | None = runtime


PROMPT_CHAR = "\u276f "  # ❯


def _extract_response_text(summary) -> str:
    """Extract plain text from a TurnSummary's assistant messages."""
    parts: list[str] = []
    for msg in summary.assistant_messages:
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def run_repl(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run the interactive REPL loop. Returns exit code."""
    model = args.model or "sonnet"
    client = LLMClient()
    session = Session()
    runtime = AgentRuntime(client=client, session=session, model=model)
    state = SessionState(model=model, runtime=runtime)

    print("synthpanel interactive mode. Type /help for commands, Ctrl-D to exit.")

    while True:
        try:
            line = input(PROMPT_CHAR)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            dispatch_slash(line, state, fmt)
            continue

        # Regular input → send to agent runtime
        state.turn_count += 1
        try:
            summary = runtime.run_turn(line)
        except Exception as exc:
            emit(fmt, message=f"Error: {exc}")
            continue

        state.last_usage = summary.usage.to_dict() if summary.usage else None

        if summary.compacted:
            state.compacted_count += 1

        response_text = _extract_response_text(summary)
        usage_dict = summary.usage.to_dict() if summary.usage else None
        emit(fmt, message=response_text, usage=usage_dict)

    return 0
