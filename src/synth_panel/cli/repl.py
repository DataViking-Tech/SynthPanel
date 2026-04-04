"""Interactive REPL for synth-panel.

Entered when no subcommand is given. Supports slash commands per SPEC.md §8.
"""

from __future__ import annotations

import argparse
import sys

from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.slash import dispatch_slash, SLASH_COMMANDS


class SessionState:
    """Mutable state maintained during an interactive session."""

    __slots__ = ("turn_count", "compacted_count", "model", "last_usage")

    def __init__(self, model: str | None = None) -> None:
        self.turn_count: int = 0
        self.compacted_count: int = 0
        self.model: str | None = model
        self.last_usage: dict[str, int] | None = None


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

        # Regular input → would be sent to agent runtime
        state.turn_count += 1
        # TODO: wire to agent runtime
        emit(fmt, message=f"[stub] Turn {state.turn_count}: {line}")

    return 0
