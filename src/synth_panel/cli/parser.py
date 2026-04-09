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
        help=(
            "LLM model to use (e.g. sonnet, opus, grok, gemini-2.5-flash). "
            "Default: first provider with credentials in the environment "
            "(ANTHROPIC_API_KEY > OPENAI_API_KEY > GEMINI_API_KEY > "
            "XAI_API_KEY). The chosen model is printed to stderr before "
            "each run so you can cancel and override."
        ),
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
    panel_run_parser.add_argument(
        "--schema",
        default=None,
        metavar="SCHEMA",
        help=(
            "JSON Schema for structured output. Can be a file path "
            "(e.g. schema.json) or inline JSON string. When provided, "
            "panelist responses are extracted as structured JSON matching "
            "this schema instead of free text."
        ),
    )
    panel_run_parser.add_argument(
        "--no-synthesis",
        action="store_true",
        default=False,
        help="Skip the synthesis step after panelist responses are collected.",
    )
    panel_run_parser.add_argument(
        "--synthesis-model",
        default=None,
        metavar="MODEL",
        help="Model to use for synthesis (default: sonnet). Overrides --model for the synthesis step only.",
    )
    panel_run_parser.add_argument(
        "--synthesis-prompt",
        default=None,
        metavar="PROMPT",
        help="Custom synthesis prompt. Replaces the default synthesis prompt entirely.",
    )
    panel_run_parser.add_argument(
        "--legacy-output",
        action="store_true",
        default=False,
        help=(
            "Emit the deprecated flat single-round output shape instead "
            "of the rounds-shaped payload. Prints a DeprecationWarning to "
            "stderr; will be removed in 0.6.0."
        ),
    )
    panel_run_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Treat any panelist-question error as fatal. When any error "
            "occurs, synthesis is skipped and the run exits non-zero. "
            "Use this to avoid spending retry budget on a broken run."
        ),
    )
    panel_run_parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        metavar="RATIO",
        help=(
            "Fraction of panelist-question pairs that may error before the "
            "run is declared invalid (default: 0.5). When exceeded, "
            "synthesis is auto-disabled and the run exits non-zero."
        ),
    )
    panel_run_parser.add_argument(
        "--var",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        dest="vars",
        help=(
            "Substitute template placeholders in the instrument's "
            "question text. Repeatable. Example: "
            "--var 'candidates=Name A, Name B' --var theme=pricing. "
            "The value replaces any literal {KEY} token in a question."
        ),
    )
    panel_run_parser.add_argument(
        "--vars-file",
        default=None,
        metavar="PATH",
        help=(
            "YAML file of key: value pairs to substitute into instrument "
            "templates. Values merge with --var (CLI flag wins on "
            "conflicts)."
        ),
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

    # instruments
    instruments_parser = subparsers.add_parser(
        "instruments",
        help="Instrument pack management: list, install, show, graph.",
    )
    instruments_subparsers = instruments_parser.add_subparsers(
        dest="instruments_command"
    )

    instruments_subparsers.add_parser(
        "list",
        help="List installed instrument packs.",
    )

    install_parser = instruments_subparsers.add_parser(
        "install",
        help="Install an instrument pack from a YAML file (or by name if bundled).",
    )
    install_parser.add_argument(
        "source",
        metavar="SOURCE",
        help="Path to a YAML file or the name of a bundled pack.",
    )
    install_parser.add_argument(
        "--name",
        default=None,
        help="Override the installed pack's name (default: from file/manifest).",
    )

    show_parser = instruments_subparsers.add_parser(
        "show",
        help="Print an installed instrument pack's contents.",
    )
    show_parser.add_argument(
        "name",
        metavar="NAME",
        help="Pack name (as listed by 'instruments list').",
    )

    graph_parser = instruments_subparsers.add_parser(
        "graph",
        help="Render the round DAG of an instrument file or pack name.",
    )
    graph_parser.add_argument(
        "source",
        metavar="SOURCE",
        help="YAML file path or installed pack name.",
    )
    graph_parser.add_argument(
        "--format",
        choices=["text", "mermaid"],
        default="text",
        help="DAG output format (default: text).",
    )

    # mcp-serve
    subparsers.add_parser(
        "mcp-serve",
        help="Start the MCP server (stdio transport).",
    )

    return parser
