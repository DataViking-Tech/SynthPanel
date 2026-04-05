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

    # pack
    pack_parser = subparsers.add_parser(
        "pack",
        help="Persona pack management: list, import, export.",
    )
    pack_subparsers = pack_parser.add_subparsers(dest="pack_command")

    # pack list
    pack_subparsers.add_parser(
        "list",
        help="List all saved persona packs.",
    )

    # pack import
    pack_import_parser = pack_subparsers.add_parser(
        "import",
        help="Import a persona pack from a YAML file.",
    )
    pack_import_parser.add_argument(
        "file",
        metavar="FILE",
        help="Path to a YAML persona pack file.",
    )
    pack_import_parser.add_argument(
        "--name",
        default=None,
        help="Name for the pack (default: derived from file or pack content).",
    )
    pack_import_parser.add_argument(
        "--id",
        default=None,
        dest="pack_id",
        help="Custom pack ID (default: auto-generated).",
    )

    # pack export
    pack_export_parser = pack_subparsers.add_parser(
        "export",
        help="Export a saved persona pack to stdout or a file.",
    )
    pack_export_parser.add_argument(
        "pack_id",
        metavar="PACK_ID",
        help="ID of the persona pack to export.",
    )
    pack_export_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write to file instead of stdout.",
    )

    # mcp-serve
    subparsers.add_parser(
        "mcp-serve",
        help="Start the MCP server (stdio transport).",
    )

    return parser
