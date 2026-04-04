"""synth-panel CLI entry point.

Implements the CLI framework from SPEC.md §8: argparse CLI, REPL loop,
slash commands, and output formatting.
"""

from __future__ import annotations

import argparse
import sys

from synth_panel.cli.parser import build_parser
from synth_panel.cli.repl import run_repl
from synth_panel.cli.commands import handle_prompt, handle_login, handle_logout
from synth_panel.cli.output import OutputFormat


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_format = OutputFormat(args.output_format)

    if args.command == "prompt":
        return handle_prompt(args, output_format)
    elif args.command == "login":
        return handle_login(args, output_format)
    elif args.command == "logout":
        return handle_logout(args, output_format)
    else:
        # No subcommand → interactive REPL
        return run_repl(args, output_format)


if __name__ == "__main__":
    sys.exit(main())
