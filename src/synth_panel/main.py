"""synthpanel CLI entry point.

Implements the CLI framework from SPEC.md §8: argparse CLI, REPL loop,
slash commands, and output formatting.
"""

from __future__ import annotations

import sys

from synth_panel.cli.commands import (
    handle_analyze,
    handle_instruments_graph,
    handle_instruments_install,
    handle_instruments_list,
    handle_instruments_show,
    handle_login,
    handle_logout,
    handle_mcp_serve,
    handle_pack_export,
    handle_pack_generate,
    handle_pack_import,
    handle_pack_list,
    handle_pack_show,
    handle_panel_inspect,
    handle_panel_run,
    handle_panel_synthesize,
    handle_prompt,
    handle_whoami,
)
from synth_panel.cli.output import OutputFormat
from synth_panel.cli.parser import build_parser
from synth_panel.cli.repl import run_repl
from synth_panel.logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging from --verbose / --quiet / env var.
    if args.verbose:
        setup_logging("debug")
    elif args.quiet:
        setup_logging("warning")
    else:
        setup_logging()

    output_format = OutputFormat(args.output_format)

    if args.command == "prompt":
        return handle_prompt(args, output_format)
    elif args.command == "panel":
        if getattr(args, "panel_command", None) == "run":
            return handle_panel_run(args, output_format)
        elif getattr(args, "panel_command", None) == "synthesize":
            return handle_panel_synthesize(args, output_format)
        elif getattr(args, "panel_command", None) == "inspect":
            return handle_panel_inspect(args, output_format)
        else:
            parser.parse_args(["panel", "--help"])
            return 1
    elif args.command == "pack":
        sub = getattr(args, "pack_command", None)
        if sub == "list":
            return handle_pack_list(args, output_format)
        elif sub == "import":
            return handle_pack_import(args, output_format)
        elif sub == "export":
            return handle_pack_export(args, output_format)
        elif sub == "show":
            return handle_pack_show(args, output_format)
        elif sub == "generate":
            return handle_pack_generate(args, output_format)
        else:
            parser.parse_args(["pack", "--help"])
            return 1
    elif args.command == "instruments":
        sub = getattr(args, "instruments_command", None)
        if sub == "list":
            return handle_instruments_list(args, output_format)
        elif sub == "install":
            return handle_instruments_install(args, output_format)
        elif sub == "show":
            return handle_instruments_show(args, output_format)
        elif sub == "graph":
            return handle_instruments_graph(args, output_format)
        else:
            parser.parse_args(["instruments", "--help"])
            return 1
    elif args.command == "analyze":
        return handle_analyze(args, output_format)
    elif args.command == "mcp-serve":
        return handle_mcp_serve(args, output_format)
    elif args.command == "login":
        return handle_login(args, output_format)
    elif args.command == "logout":
        return handle_logout(args, output_format)
    elif args.command == "whoami":
        return handle_whoami(args, output_format)
    else:
        # No subcommand → interactive REPL
        return run_repl(args, output_format)


if __name__ == "__main__":
    sys.exit(main())
