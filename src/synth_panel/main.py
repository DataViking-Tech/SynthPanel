"""synthpanel CLI entry point.

Implements the CLI framework from SPEC.md §8: argparse CLI, REPL loop,
slash commands, and output formatting.
"""

from __future__ import annotations

import os
import signal
import sys

from synth_panel.cli.commands import (
    handle_analyze,
    handle_analyze_subgroup,
    handle_cost_summary,
    handle_doctor,
    handle_domains_inspect,
    handle_domains_list,
    handle_install_skills,
    handle_instruments_graph,
    handle_instruments_install,
    handle_instruments_list,
    handle_instruments_show,
    handle_login,
    handle_logout,
    handle_mcp_install,
    handle_mcp_serve,
    handle_pack_calibrate,
    handle_pack_diff,
    handle_pack_export,
    handle_pack_generate,
    handle_pack_import,
    handle_pack_inspect,
    handle_pack_list,
    handle_pack_save,
    handle_pack_search,
    handle_pack_show,
    handle_pack_uninstall,
    handle_panel_inspect,
    handle_panel_run,
    handle_panel_synthesize,
    handle_plugin_lint,
    handle_prompt,
    handle_report,
    handle_results_list,
    handle_results_show,
    handle_runs_diff,
    handle_runs_list,
    handle_runs_prune,
    handle_whoami,
)
from synth_panel.cli.output import OutputFormat
from synth_panel.cli.parser import build_parser
from synth_panel.cli.repl import run_repl
from synth_panel.logging_config import setup_logging


def _quiet_broken_pipe() -> None:
    """Make `synthpanel … | head` exit silently like a regular Unix tool.

    Two failure modes overlap here:

    * A mid-stream write to a closed pipe raises ``BrokenPipeError`` because
      Python overrides the default ``SIGPIPE`` disposition. Restoring
      ``SIG_DFL`` lets the kernel just kill the process on pipe close, the
      way ``cat`` / ``ls`` / ``--help`` from any other tool behaves.

    * Even after handling the mid-stream write, the interpreter shutdown
      flush still finds the broken pipe and prints
      ``Exception ignored while flushing sys.stdout: BrokenPipeError``.
      Redirecting fd 1 to ``/dev/null`` before shutdown swallows that
      final flush. The exit-time flush hits ``/dev/null`` instead of the
      closed pipe, so no warning escapes to stderr.
    """
    if hasattr(signal, "SIGPIPE"):
        # Windows has no SIGPIPE — that's why the hasattr guard.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            # If stdout is detached or already closed there's nothing
            # more we can do; the silence we wanted is already achieved.
            pass


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    # SIGPIPE disposition is process-global. ``main`` is the console-script
    # entry point, but it is *also* imported and called directly by the test
    # suite and library callers — so installing SIG_DFL unconditionally and
    # never undoing it leaks the disposition into the calling process. A later,
    # unrelated broken-pipe write (pytest output capture, coverage teardown, an
    # xdist worker pipe) then takes a SIGPIPE and the whole process dies
    # silently with exit 141. That is the non-deterministic CI killer tracked
    # under sy-6zq / sy-1n1. Capture the prior handler so we can hand the
    # disposition back when ``main`` returns. (Gating on ``isatty()`` would be
    # wrong: stdout is *not* a tty exactly when piped to ``head``, which is the
    # case the SIG_DFL restore exists to handle.)
    prev_sigpipe = None
    have_sigpipe = hasattr(signal, "SIGPIPE")
    if have_sigpipe:
        # Windows has no SIGPIPE — that's why the hasattr guard.
        prev_sigpipe = signal.getsignal(signal.SIGPIPE)
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        return _main(argv)
    except BrokenPipeError:
        _quiet_broken_pipe()
        # 0 — Unix convention treats SIGPIPE-killed pipelines as expected
        # for short-circuiting consumers like `head`. argparse's `--help`
        # already returned its work; signalling success matches `man`,
        # `pydoc`, and other paginated tools.
        return 0
    finally:
        # Even on normal exit, a buffered write may still be pending when
        # the pipe is already closed (argparse `--help` ends in
        # ``parser.exit(0)`` after buffered prints). Flushing here surfaces
        # any deferred BrokenPipeError so we can swap stdout for devnull
        # before the interpreter's own shutdown flush triggers the
        # "Exception ignored" warning on stderr.
        _quiet_broken_pipe()
        # Restore the caller's SIGPIPE handler. For the real CLI process this
        # is harmless cleanup right before exit (stdout is already pointed at
        # /dev/null by _quiet_broken_pipe); for in-process callers it stops
        # SIG_DFL from outliving this invocation. ``getsignal`` returns None
        # when the prior handler was installed from non-Python code, which
        # ``signal.signal`` cannot reinstall — skip that rare case.
        if have_sigpipe and prev_sigpipe is not None:
            signal.signal(signal.SIGPIPE, prev_sigpipe)


def _main(argv: list[str] | None) -> int:
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
        elif sub == "save":
            return handle_pack_save(args, output_format)
        elif sub == "uninstall":
            return handle_pack_uninstall(args, output_format)
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
    elif args.command == "domains":
        sub = getattr(args, "domains_command", None)
        if sub == "list":
            return handle_domains_list(args, output_format)
        elif sub == "inspect":
            return handle_domains_inspect(args, output_format)
        else:
            parser.parse_args(["domains", "--help"])
            return 1
    elif args.command == "cost":
        sub = getattr(args, "cost_command", None)
        if sub == "summary":
            return handle_cost_summary(args, output_format)
        else:
            parser.parse_args(["cost", "--help"])
            return 1
    elif args.command == "analyze":
        if getattr(args, "by", None):
            return handle_analyze_subgroup(args, output_format)
        return handle_analyze(args, output_format)
    elif args.command == "report":
        return handle_report(args, output_format)
    elif args.command == "poll-summary":
        from synth_panel.cli.commands import handle_poll_summary

        return handle_poll_summary(args, output_format)
    elif args.command == "mcp-serve":
        return handle_mcp_serve(args, output_format)
    elif args.command == "mcp":
        sub = getattr(args, "mcp_command", None)
        if sub == "install":
            return handle_mcp_install(args, output_format)
        elif sub == "uninstall":
            # synthbench#262: first-class mirror of `mcp install --uninstall`.
            # Same handler; the flag selects the removal path.
            args.uninstall = True
            return handle_mcp_install(args, output_format)
        else:
            parser.parse_args(["mcp", "--help"])
            return 1
    elif args.command == "plugin":
        sub = getattr(args, "plugin_command", None)
        if sub == "lint":
            return handle_plugin_lint(args, output_format)
        else:
            parser.parse_args(["plugin", "--help"])
            return 1
    elif args.command == "install-skills":
        return handle_install_skills(args, output_format)
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
    elif args.command == "results":
        sub = getattr(args, "results_command", None)
        if sub == "list":
            return handle_results_list(args, output_format)
        elif sub == "show":
            return handle_results_show(args, output_format)
        else:
            parser.parse_args(["results", "--help"])
            return 1
    else:
        # No subcommand → interactive REPL
        return run_repl(args, output_format)


if __name__ == "__main__":
    sys.exit(main())
