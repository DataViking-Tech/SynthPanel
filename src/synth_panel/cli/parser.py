"""Argument parser for synth-panel CLI.

Defines global arguments and subcommands per SPEC.md §8.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with global args and subcommands."""
    parser = argparse.ArgumentParser(
        prog="synth-panel",
        description="Synthetic focus group CLI — orchestrate LLM-powered personas for structured qualitative feedback.",
    )

    # Global arguments (apply to all commands)
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use (e.g. sonnet, opus, grok). Default: best available.",
    )
    parser.add_argument(
        "--permission-mode",
        choices=["read-only", "workspace-write", "full-access"],
        default="full-access",
        help="Control what the agent is allowed to do (default: full-access).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "ndjson"],
        default="text",
        help="Output format (default: text).",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # prompt
    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Run a single non-interactive prompt and exit.",
    )
    prompt_parser.add_argument(
        "text",
        nargs="+",
        help="Prompt text (all remaining arguments are joined).",
    )

    # panel
    panel_parser = subparsers.add_parser(
        "panel",
        help="Panel operations: run synthetic focus groups.",
    )
    panel_subparsers = panel_parser.add_subparsers(dest="panel_command")

    # panel run
    panel_run_parser = panel_subparsers.add_parser(
        "run",
        help="Run a panel with personas and an instrument/survey.",
    )
    panel_run_parser.add_argument(
        "--personas",
        required=True,
        metavar="PATH",
        help="Path to a YAML file defining personas.",
    )
    panel_run_parser.add_argument(
        "--instrument",
        required=True,
        metavar="PATH",
        help="Path to a YAML file defining the survey/instrument.",
    )

    # mcp-serve
    subparsers.add_parser(
        "mcp-serve",
        help="Start the MCP server (stdio transport).",
    )

    # login
    subparsers.add_parser(
        "login",
        help="Start an authentication flow.",
    )

    # logout
    subparsers.add_parser(
        "logout",
        help="Clear saved authentication credentials.",
    )

    return parser
