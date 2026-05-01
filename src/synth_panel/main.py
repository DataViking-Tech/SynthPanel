"""synthpanel CLI entry point.

Implements the CLI framework from SPEC.md §8: argparse CLI, REPL loop,
slash commands, and output formatting.
"""

from __future__ import annotations

import sys

from synth_panel.cli.commands import (
    handle_analyze,
    handle_analyze_subgroup,
    handle_cost_summary,
    handle_doctor,
    handle_instruments_graph,
    handle_instruments_install,
    handle_instruments_list,
    handle_instruments_show,
    handle_login,
    handle_logout,
    handle_mcp_serve,
    handle_pack_calibrate,
    handle_pack_diff,
    handle_pack_export,
    handle_pack_generate,
    handle_pack_import,
    handle_pack_inspect,
    handle_pack_list,
    handle_pack_search,
    handle_pack_show,
    handle_panel_inspect,
    handle_panel_run,
    handle_panel_synthesize,
    handle_prompt,
    handle_report,
    handle_runs_diff,
    handle_runs_list,
    handle_runs_prune,
    handle_whoami,
)
from synth_panel.cli.output import OutputFormat
from synth_panel.cli.parser import build_parser
from synth_panel.cli.repl import run_repl
from synth_panel.logging_config import setup_logging


# Subcommand names recognised under ``synthpanel analyze``. Used by
# :func:`_rewrite_legacy_analyze` to distinguish modern subcommand
# invocations from the legacy flat form ``analyze RESULT_ID``.
_ANALYZE_SUBCOMMANDS = frozenset({"summary", "subgroup"})


def _rewrite_legacy_analyze(argv: list[str]) -> list[str]:
    """Rewrite ``analyze RESULT_ID …`` into ``analyze summary RESULT_ID …``.

    Adding subparsers under ``analyze`` would otherwise break the
    pre-existing flat invocation that scripts and docs already rely on.
    Inject a virtual ``summary`` subcommand when:

    * ``analyze`` is the top-level command, AND
    * the next token is a positional (does not start with ``-``), AND
    * that token is not one of the recognised subcommands.

    The rewrite leaves help (``analyze --help``), bare ``analyze``,
    and modern invocations (``analyze subgroup …``) untouched.
    """
    if not argv or argv[0] != "analyze":
        return argv
    if len(argv) < 2:
        return argv
    nxt = argv[1]
    if nxt.startswith("-"):
        return argv
    if nxt in _ANALYZE_SUBCOMMANDS:
        return argv
    return [argv[0], "summary", *argv[1:]]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    if argv is None:
        argv = sys.argv[1:]
    argv = _rewrite_legacy_analyze(list(argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging from --debug-all / --quiet / --verbose / env var.
    if getattr(args, "debug_all", False):
        setup_logging(debug_all=True)
    elif args.quiet:
        setup_logging("warning")
    elif args.verbose:
        setup_logging("debug")
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
        elif sub == "inspect":
            return handle_pack_inspect(args, output_format)
        elif sub == "generate":
            return handle_pack_generate(args, output_format)
        elif sub == "search":
            return handle_pack_search(args, output_format)
        elif sub == "calibrate":
            return handle_pack_calibrate(args, output_format)
        elif sub == "diff":
            return handle_pack_diff(args, output_format)
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
    elif args.command == "cost":
        sub = getattr(args, "cost_command", None)
        if sub == "summary":
            return handle_cost_summary(args, output_format)
        else:
            parser.parse_args(["cost", "--help"])
            return 1
    elif args.command == "analyze":
        sub = getattr(args, "analyze_command", None)
        if sub == "subgroup":
            return handle_analyze_subgroup(args, output_format)
        if sub == "summary" or sub is None:
            # ``sub is None`` only when the user typed bare ``analyze``;
            # _rewrite_legacy_analyze rewrites the flat-form positional
            # case before argparse sees it.
            if sub is None:
                parser.parse_args(["analyze", "--help"])
                return 1
            return handle_analyze(args, output_format)
        # Future-proofing: a registered subcommand without a dispatch
        # branch should fail loudly rather than silently fall through.
        parser.parse_args(["analyze", "--help"])
        return 1
    elif args.command == "report":
        return handle_report(args, output_format)
    elif args.command == "mcp-serve":
        return handle_mcp_serve(args, output_format)
    elif args.command == "login":
        return handle_login(args, output_format)
    elif args.command == "logout":
        return handle_logout(args, output_format)
    elif args.command == "whoami":
        return handle_whoami(args, output_format)
    elif args.command == "doctor":
        return handle_doctor(args, output_format)
    elif args.command == "runs":
        sub = getattr(args, "runs_command", None)
        if sub == "prune":
            return handle_runs_prune(args, output_format)
        elif sub == "list":
            return handle_runs_list(args, output_format)
        elif sub == "diff":
            return handle_runs_diff(args, output_format)
        else:
            parser.parse_args(["runs", "--help"])
            return 1
    else:
        # No subcommand → interactive REPL
        return run_repl(args, output_format)


if __name__ == "__main__":
    sys.exit(main())
