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
        "--personas-merge",
        dest="personas_merge",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional YAML file(s) whose personas are appended to --personas. "
            "Repeatable. Files are merged in order; duplicate names later in the "
            "list override earlier ones."
        ),
    )
    panel_run_parser.add_argument(
        "--personas-merge-on-collision",
        dest="personas_merge_on_collision",
        choices=["dedup", "error", "keep"],
        default="dedup",
        help=(
            "How to handle --personas-merge name collisions. "
            "'dedup' (default): the later file wins and a warning is "
            "emitted naming every dropped persona plus the post-dedup "
            "panel size. 'error': abort the run if any collision occurs. "
            "'keep' is reserved and currently rejected — use dedup."
        ),
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
            "Multi-model spec. Two shapes: "
            "(1) **Weighted per-persona** — 'haiku:0.5,gemini:0.5' splits the "
            "panel across models in list order; 6 personas at 0.5/0.5 → 3/3, "
            "7 personas at 0.5/0.5 → 3/4 (the last model absorbs the "
            "remainder). Weights are normalized, so 'a:2,b:3' and "
            "'a:0.4,b:0.6' behave the same. Weights summing far from 1.0 "
            "emit a warning. Assignment is fully deterministic (same "
            "personas+spec → same split) and printed before the run. "
            "Per-persona YAML 'model' overrides always win. "
            "(2) **Ensemble** — 'haiku,sonnet' (no ':') runs the full panel "
            "once per model. Combine with --blend to weight-average "
            "distributions. Mutually exclusive with --model. "
            "See docs/ensemble.md for the full algorithm."
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
        help="Model to use for synthesis (default: matches --model). Overrides --model for the synthesis step only.",
    )
    panel_run_parser.add_argument(
        "--synthesis-prompt",
        default=None,
        metavar="PROMPT",
        help="Custom synthesis prompt. Replaces the default synthesis prompt entirely.",
    )
    panel_run_parser.add_argument(
        "--synthesis-strategy",
        choices=["single", "map-reduce", "auto"],
        default="auto",
        help=(
            "How to aggregate panelist responses into the final synthesis "
            "(sp-kkzz). 'single' concatenates every response into one LLM "
            "call (cheapest for small panels). 'map-reduce' runs one "
            "summary call per question in parallel then one reduce call "
            "across the summaries (required once responses overflow the "
            "synthesis model's context window, typically n>=50). 'auto' "
            "(default) picks based on a pre-flight token estimate — falls "
            "back to 'single' whenever the estimate fits the synthesis "
            "model's context. Custom --synthesis-prompt forces 'single'."
        ),
    )
    panel_run_parser.add_argument(
        "--synthesis-auto-escalate",
        action="store_true",
        default=False,
        help=(
            "sp-4g6a: when a single question's responses overflow the "
            "synthesis model's context in --synthesis-strategy=map-reduce, "
            "auto-retry that question's map call on a larger-context model "
            "(gemini-2.5-flash-lite, 1M ctx) and emit a warning. Default "
            "behaviour (off) partitions panelists into sub-batches and "
            "performs an inner reduce, preserving single-model semantics."
        ),
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
    panel_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Preview the panel without making any LLM calls. Loads personas "
            "and the instrument, applies --var substitutions, then prints "
            "each question as it would be sent to the LLM along with "
            "persona/question counts and a rough input-token estimate. "
            "Exits without calling any provider."
        ),
    )
    # sp-i2ub: scaled-orchestration knobs
    panel_run_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap concurrent in-flight LLM requests across the panel. "
            "Applies at the client layer, so it throttles all providers "
            "on the same client. Defaults to unbounded (one worker per "
            "panelist). Use this to keep provider rate limits happy on "
            "large n runs."
        ),
    )
    panel_run_parser.add_argument(
        "--rate-limit-rps",
        type=float,
        default=None,
        metavar="RPS",
        help=(
            "Cap requests-per-second across the panel via a token bucket. "
            "Smooths bursts on top of --max-concurrent. Accepts fractional "
            "values (e.g. 0.5 for one request every two seconds)."
        ),
    )
    # sp-hsk3: panelist-level checkpointing + resume
    panel_run_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        metavar="PATH",
        dest="checkpoint_dir",
        help=(
            "Directory under which to persist per-run checkpoints. Each run "
            "gets its own subdirectory (<PATH>/<run-id>/state.json). "
            "Defaults to $SYNTHPANEL_CHECKPOINT_ROOT or "
            "~/.synthpanel/checkpoints. Setting this flag opts into "
            "checkpointing; omit it to run without on-disk snapshots."
        ),
    )
    panel_run_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        dest="checkpoint_every",
        help=(
            "Flush a checkpoint every N completed panelists (default: 25). "
            "Lower values mean more frequent disk writes; higher values "
            "risk losing more progress on abrupt termination. Only takes "
            "effect with --checkpoint-dir or --resume."
        ),
    )
    panel_run_parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        dest="resume",
        help=(
            "Resume a previously-checkpointed run by id. Skips panelists "
            "that already completed, replays the rest, and merges results "
            "into one final output. Refuses to start if the current config "
            "does not match the checkpointed config — rerun without "
            "--resume or restore the original config to continue."
        ),
    )
    # sp-utnk: mid-run cost gate (projected-total ceiling)
    panel_run_parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="USD",
        dest="max_cost",
        help=(
            "Hard ceiling on total panel spend, in USD. After each "
            "panelist completes, running_cost / completed_n * total_n is "
            "compared against this ceiling; if the projected total "
            "exceeds it, the run halts gracefully and produces a valid "
            "partial JSON result with run_invalid: true and "
            "cost_exceeded: true. Exit code 2. Default: off."
        ),
    )
    # sp-yaru: convergence telemetry for large panels
    panel_run_parser.add_argument(
        "--convergence-check-every",
        type=int,
        default=None,
        metavar="N",
        dest="convergence_check_every",
        help=(
            "Compute a running Jensen-Shannon divergence every N completing "
            "panelists for every bounded (Likert / yes-no / pick-one / enum) "
            "question. Enables the post-run convergence report. Default: off; "
            "setting this flag opts in. See docs/convergence.md."
        ),
    )
    panel_run_parser.add_argument(
        "--convergence-log",
        default=None,
        metavar="PATH",
        dest="convergence_log",
        help=(
            "Write each convergence check as a JSON line to PATH instead of "
            "stderr. Useful for streaming into dashboards during long runs."
        ),
    )
    panel_run_parser.add_argument(
        "--auto-stop",
        action="store_true",
        default=False,
        dest="auto_stop",
        help=(
            "Halt the panel once every tracked question's rolling-average JSD "
            "stays below --convergence-eps for --convergence-m consecutive "
            "checks (and n >= --convergence-min-n). Default: off. Requires "
            "--convergence-check-every."
        ),
    )
    panel_run_parser.add_argument(
        "--convergence-eps",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="convergence_eps",
        help="JSD threshold below which a question is treated as converged (default: 0.02).",
    )
    panel_run_parser.add_argument(
        "--convergence-min-n",
        type=int,
        default=None,
        metavar="N",
        dest="convergence_min_n",
        help="Minimum panelist count before --auto-stop is allowed to fire (default: 50).",
    )
    panel_run_parser.add_argument(
        "--convergence-m",
        type=int,
        default=None,
        metavar="N",
        dest="convergence_m",
        help="Consecutive checks below epsilon required to declare convergence (default: 3).",
    )
    panel_run_parser.add_argument(
        "--convergence-baseline",
        default=None,
        metavar="DATASET:QUESTION",
        dest="convergence_baseline",
        help=(
            "Load a human baseline convergence curve from synthbench and "
            "include it in the report. Requires the optional dependency: "
            "pip install 'synthpanel[convergence]'. Format: "
            "'dataset:question_key' (e.g. 'gss:happiness') or just 'dataset' "
            "when the dataset has a single default question."
        ),
    )
    # sp-zq3: SynthBench best-model picker
    panel_run_parser.add_argument(
        "--best-model-for",
        default=None,
        metavar="TOPIC[:DATASET]",
        dest="best_model_for",
        help=(
            "Consult the SynthBench leaderboard (synthbench.org) and use "
            "the top-ranked model for the given topic. Format: 'TOPIC' "
            "(ranked by topic score against the default dataset "
            "'globalopinionqa') or 'TOPIC:DATASET' (topic within a specific "
            "dataset). Pass ':DATASET' with empty topic to rank by SPS "
            "across a dataset instead. A recommendation line is printed to "
            "stderr before the run. Overrides --model when present; "
            "mutually exclusive with --models. Respects "
            "SYNTHPANEL_SYNTHBENCH_OFFLINE / _REFRESH / _URL env vars. The "
            "leaderboard is cached for 24h at "
            "~/.synthpanel/synthbench-cache.json."
        ),
    )
    panel_run_parser.add_argument(
        "--calibrate-against",
        default=None,
        metavar="DATASET:QUESTION",
        dest="calibrate_against",
        help=(
            "Inline calibration against a published human baseline. v1 "
            "supports GSS only (e.g. 'gss:HAPPY'); the redistribution-tier "
            "allowlist is {gss, ntia}. Force-enables convergence tracking, "
            "but cadence is NOT implicit — pair explicitly with "
            "--convergence-check-every to control sampling. Auto-derives a "
            "pick-one extractor schema from the baseline when option count "
            "<= 5; otherwise pass --extract-schema. Requires the optional "
            "dependency: pip install 'synthpanel[convergence]'. Format: "
            "'dataset:question_key' (colon-separated, both non-empty)."
        ),
    )
    # sp-ezz: opt-in submission of calibrated runs to SynthBench. Only
    # meaningful with --calibrate-against (parse-time hard-fail otherwise);
    # requires SYNTHBENCH_API_KEY in env. See docs/synthbench-integration.md.
    panel_run_parser.add_argument(
        "--submit-to-synthbench",
        action="store_true",
        default=False,
        dest="submit_to_synthbench",
        help=(
            "After the panel run, upload the per-question calibration JSD "
            "and distributions to SynthBench's public leaderboard. Requires "
            "--calibrate-against and SYNTHBENCH_API_KEY (mint at "
            "synthbench.org/account). First use prompts for consent and "
            "records it at ~/.synthpanel/synthbench-consent.json. "
            "See docs/synthbench-integration.md for the privacy model."
        ),
    )
    panel_run_parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        dest="yes",
        help=(
            "Bypass the SynthBench consent prompt for non-interactive use. "
            "Equivalent to recording consent ahead of time; intended for CI."
        ),
    )

    # panel inspect (sp-76gm: stats + schema walker for a saved result)
    panel_inspect_parser = panel_subparsers.add_parser(
        "inspect",
        help="Summarize a saved panel result: models, personas, failures, synthesis.",
    )
    panel_inspect_parser.add_argument(
        "result",
        metavar="RESULT",
        help="Panel result ID (from 'panel run --save') or path to a result JSON file.",
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
    pack_list_parser = pack_subparsers.add_parser(
        "list",
        help="List all saved persona packs.",
    )
    pack_list_parser.add_argument(
        "--registry",
        action="store_true",
        help=(
            "List packs from the synthpanel registry instead of local installs. Prints id, name, ref, version columns."
        ),
    )

    # pack search
    pack_search_parser = pack_subparsers.add_parser(
        "search",
        help="Search registry packs by substring (id, name, description, tags).",
    )
    pack_search_parser.add_argument(
        "term",
        metavar="TERM",
        help="Substring to match (case-insensitive) against id/name/description/tags.",
    )

    # pack import
    pack_import_parser = pack_subparsers.add_parser(
        "import",
        help="Import a persona pack from a YAML file or GitHub source.",
    )
    pack_import_parser.add_argument(
        "source",
        metavar="SOURCE",
        help=(
            "Local YAML path, gh:user/repo[@ref][:path] URI, or https URL "
            "(raw.githubusercontent.com or github.com/.../blob/...)."
        ),
    )
    pack_import_parser.add_argument(
        "--name",
        default=None,
        help="Name for the pack (default: derived from source or pack content).",
    )
    pack_import_parser.add_argument(
        "--id",
        default=None,
        dest="pack_id",
        help="Custom pack ID (default: auto-generated).",
    )
    pack_import_parser.add_argument(
        "--unverified",
        action="store_true",
        help=(
            "Allow importing a remote pack that is not listed in the "
            "synthpanel registry. Prints a one-time warning block with "
            "source URL, sha256 checksum, and imported id."
        ),
    )
    pack_import_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing user-saved pack with the same id.",
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

    # pack calibrate (sp-sghl): calibrate a pack against a SynthBench baseline
    # and write the resulting JSD into the pack manifest.
    pack_calibrate_parser = pack_subparsers.add_parser(
        "calibrate",
        help="Calibrate a persona pack against a SynthBench human baseline.",
    )
    pack_calibrate_parser.add_argument(
        "pack_yaml",
        metavar="PACK_YAML",
        help="Path to the persona pack YAML file to calibrate.",
    )
    pack_calibrate_parser.add_argument(
        "--against",
        required=True,
        metavar="DATASET:QUESTION",
        dest="against",
        help=("SynthBench baseline to calibrate against, e.g. gss:HAPPY. v1 inline-publishable allowlist: gss, ntia."),
    )
    pack_calibrate_parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Panel size (default: 50).",
    )
    pack_calibrate_parser.add_argument(
        "--models",
        default=None,
        metavar="MODELS",
        help=(
            "Models to use for the calibration panel (same format as 'panel run --models'). Default: panel run default."
        ),
    )
    pack_calibrate_parser.add_argument(
        "--samples-per-question",
        type=int,
        default=15,
        dest="samples_per_question",
        help="Samples per question for stable JSD (default: 15).",
    )
    pack_calibrate_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Write the updated pack to PATH (default: rewrite PACK_YAML in place).",
    )
    pack_calibrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print what would be written to the YAML without touching the file.",
    )
    pack_calibrate_parser.add_argument(
        "--yes",
        action="store_true",
        dest="yes",
        help="Non-interactive overwrite confirm (skip the y/N prompt).",
    )
    pack_calibrate_parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="Re-raise unexpected errors with full traceback (default: print a clean message).",
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

    # report (sp-viz-layer T4): render a saved panel result as Markdown.
    report_parser = subparsers.add_parser(
        "report",
        help="Render a saved panel result as a shareable Markdown report.",
    )
    report_parser.add_argument(
        "result",
        metavar="RESULT",
        help="Panel result ID or path to a result JSON file.",
    )
    report_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Write to PATH instead of stdout ('-' = stdout, default).",
    )
    report_parser.add_argument(
        "--format",
        default="markdown",
        choices=["markdown"],
        help="Report format (default: markdown). HTML reserved for v2.",
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
