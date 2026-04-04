"""Interactive REPL for synth-panel.

Entered when no subcommand is given. Supports slash commands per SPEC.md §8.
"""

from __future__ import annotations

import argparse
import sys

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.slash import dispatch_slash, SLASH_COMMANDS
from synth_panel.llm.client import LLMClient
from synth_panel.persistence import Session
from synth_panel.runtime import AgentRuntime


class SessionState:
    """Mutable state maintained during an interactive session."""

    __slots__ = ("turn_count", "compacted_count", "model", "last_usage",
                 "runtime", "client")

    def __init__(self, model: str | None = None) -> None:
        self.turn_count: int = 0
        self.compacted_count: int = 0
        self.model: str | None = model
        self.last_usage: dict[str, int] | None = None
        self.client: LLMClient = LLMClient()
        self.runtime: AgentRuntime = self._build_runtime()

    def _build_runtime(self) -> AgentRuntime:
        return AgentRuntime(
            client=self.client,
            session=Session(),
            model=self.model or "sonnet",
        )

    def rebuild_runtime(self) -> None:
        """Rebuild the runtime (e.g. after model change or /clear)."""
        self.runtime = self._build_runtime()


PROMPT_CHAR = "\u276f "  # ❯


def run_repl(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run the interactive REPL loop. Returns exit code."""
    state = SessionState(model=args.model)

    print("synth-panel interactive mode. Type /help for commands, Ctrl-D to exit.")

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

        # Regular input → send to LLM via agent runtime
        state.turn_count += 1
        try:
            summary = state.runtime.run_turn(line)
        except Exception as exc:
            emit(fmt, message=f"Error: {exc}")
            continue

        # Extract assistant response text
        response_text = ""
        for msg in summary.assistant_messages:
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    response_text += block.get("text", "")

        state.last_usage = summary.usage.to_dict() if summary.usage else None
        if summary.compacted:
            state.compacted_count += 1

        emit(fmt, message=response_text, usage=state.last_usage if fmt is not OutputFormat.TEXT else None)

    return 0
