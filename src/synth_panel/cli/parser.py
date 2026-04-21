"""Argument parser for synthpanel CLI.

Defines global arguments and subcommands per SPEC.md §8.
"""

from __future__ import annotations

import argparse

from synth_panel import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with global args and subcommands."""
    parser = argparse.ArgumentParser(
        prog="synthpanel",
        description="Synthetic focus group CLI — orchestrate LLM-powered personas for structured qualitative feedback.",
    )
    parser.add_argument("--version", action="version", version=f"synthpanel {__version__}")

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
        help="Path to a YAML profile/configuration file.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help=(
            "Named profile to load defaults from. Searches bundled profiles, "
            "./profiles/, and ~/.synthpanel/profiles/. CLI flags override "
            "profile values."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "ndjson"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress most output; only show warnings and errors.",
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
        "--models",
        default=None,
        metavar="SPEC",
        help=(
            "Multi-model ensemble spec: comma-separated model:weight pairs. "
            "E.g. 'haiku:0.5,gemini-2.5-flash:0.5'. Personas are assigned "
            "models proportionally by weight. Mutually exclusive with --model."
        ),
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
        "--extract-schema",
        default=None,
        metavar="SCHEMA",
        help=(
            "JSON Schema for post-hoc extraction from free-text responses. "
            "Can be a file path (e.g. extract.json) or inline JSON string. "
            "Each panelist responds in free text, then a second LLM call "
            "extracts structured data matching this schema. The result is "
            "stored under an 'extraction' key alongside the raw 'response'. "
            "Unlike --schema (which forces structured-only output), this "
            "flag preserves the full text response."
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
        "--temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Sampling temperature for panelist LLM calls (0.0-2.0). "
            "Higher values produce more diverse responses. "
            "Default: provider default (typically 1.0)."
        ),
    )
    panel_run_parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        metavar="FLOAT",
        help=("Nucleus sampling cutoff for panelist LLM calls (0.0-1.0). Default: provider default."),
    )
    panel_run_parser.add_argument(
        "--synthesis-temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Sampling temperature for the synthesis LLM call only. "
            "Overrides --temperature for the synthesis step. "
            "Default: same as --temperature (or provider default)."
        ),
    )
    panel_run_parser.add_argument(
        "--prompt-template",
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom persona system prompt template file. "
            "Uses Python format-string syntax with placeholders like "
            "{name}, {age}, {occupation}, {background}, {personality_traits}. "
            "Default: built-in persona prompt."
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
    panel_run_parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        default=False,
        help=(
            "Proceed even if instrument questions contain unsubstituted "
            "{placeholder} tokens after --var / --vars-file are applied. "
            "By default the run aborts with a descriptive error so that "
            "empty panels are not launched against raw templates. Use "
            "this flag when the literal braces are intentional."
        ),
    )
    panel_run_parser.add_argument(
        "--variants",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate N LLM-perturbed variants per persona before running "
            "the panel. The original personas are replaced by N*M variants "
            "(M = number of personas). Each variant perturbs one axis "
            "(trait swap, mood context, demographic shift, background "
            "rephrase). Requires an LLM call per variant."
        ),
    )
    panel_run_parser.add_argument(
        "--blend",
        action="store_true",
        default=False,
        help=(
            "Enable distribution blending when running with multiple models "
            "via --models. Runs each model on the full panel, then computes "
            "weighted-average response distributions across models for each "
            "question. Weights are taken from the --models spec (e.g. "
            "'haiku:0.5,gemini:0.5'). Requires --models."
        ),
    )
    panel_run_parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help=(
            "Save panel results to ~/.synthpanel/results/ with a generated "
            "result ID. The result ID is printed to stderr on completion and "
            "can be passed to 'synthpanel analyze <RESULT_ID>'."
        ),
    )

    # panel synthesize (sp-5on.5: post-hoc re-synthesis of a saved result)
    panel_synth_parser = panel_subparsers.add_parser(
        "synthesize",
        help="Re-synthesize a saved panel result with a different model or prompt.",
    )
    panel_synth_parser.add_argument(
        "result",
        metavar="RESULT_ID",
        help="Panel result ID (from 'panel run --save') or path to a result JSON file.",
    )
    panel_synth_parser.add_argument(
        "--synthesis-model",
        default=None,
        metavar="MODEL",
        help="Model for re-synthesis (default: original panelist model).",
    )
    panel_synth_parser.add_argument(
        "--synthesis-prompt",
        default=None,
        metavar="PROMPT",
        help="Custom synthesis prompt (replaces the default entirely).",
    )

    # pack
    pack_parser = subparsers.add_parser(
        "pack",
        help="Persona pack management: list, import, export, show.",
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

    # pack show (sp-oem: API parity with `instruments show`)
    pack_show_parser = pack_subparsers.add_parser(
        "show",
        help="Print a saved persona pack's YAML to stdout.",
    )
    pack_show_parser.add_argument(
        "pack_id",
        metavar="PACK_ID",
        help="ID of the persona pack to show.",
    )

    # pack generate (sp-5on.18: LLM-based persona generation)
    pack_generate_parser = pack_subparsers.add_parser(
        "generate",
        help="Generate a persona pack using an LLM.",
    )
    pack_generate_parser.add_argument(
        "--product",
        required=True,
        help="Description of the product or service being researched.",
    )
    pack_generate_parser.add_argument(
        "--audience",
        required=True,
        help="Target audience for the personas.",
    )
    pack_generate_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of personas to generate (default: 5).",
    )
    pack_generate_parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use (default: auto-detect from available API keys).",
    )
    pack_generate_parser.add_argument(
        "--name",
        default=None,
        help="Name for the generated pack (default: derived from product).",
    )
    pack_generate_parser.add_argument(
        "--id",
        default=None,
        dest="pack_id",
        help="Custom pack ID (default: auto-generated).",
    )

    # instruments
    instruments_parser = subparsers.add_parser(
        "instruments",
        help="Instrument pack management: list, install, show, graph.",
    )
    instruments_subparsers = instruments_parser.add_subparsers(dest="instruments_command")

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

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a saved panel result (descriptive, inferential, cross-model, clusters).",
    )
    analyze_parser.add_argument(
        "result",
        metavar="RESULT_ID",
        help="Panel result ID (from 'synthpanel panel run') or path to a result JSON file.",
    )
    analyze_parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text).",
    )

    # mcp-serve
    subparsers.add_parser(
        "mcp-serve",
        help="Start the MCP server (stdio transport).",
    )

    # login (sp-lve: credential UX — persist an API key to disk so CLI
    # use without exported env vars isn't a dead end).
    login_parser = subparsers.add_parser(
        "login",
        help="Store an LLM provider API key at ~/.config/synthpanel/credentials.json (mode 0600).",
    )
    login_parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai", "gemini", "google", "xai", "openrouter"],
        help="Provider to log in to (default: anthropic).",
    )
    login_parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key. If omitted, reads from stdin (hidden when attached to a TTY).",
    )

    # logout
    logout_parser = subparsers.add_parser(
        "logout",
        help="Remove a stored API key from the credential store.",
    )
    logout_parser.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai", "gemini", "google", "xai", "openrouter", "all"],
        help="Provider to log out (default: anthropic). Use 'all' to remove every stored key.",
    )

    # whoami — show which providers have usable creds
    subparsers.add_parser(
        "whoami",
        help="Show which providers have credentials available (env or stored).",
    )

    return parser
