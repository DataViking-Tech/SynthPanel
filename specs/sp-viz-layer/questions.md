# Q-phase: Panel-result consumption paths

*Ticket hidden per QRSPI discipline. These questions drive objective code-mapping only; do not invent design direction from them.*

## Problem (shared with R polecats)

Panel results land as JSON files today. Non-terminal humans — PMs, stakeholders, execs — aren't the primary consumers of those files. Map what paths panel results currently travel from creation to consumption, and what the surrounding ecosystem already provides for transforming structured output into human-facing artifacts.

## Research dimensions

### 1. Panel-result JSON shape and schema

- Where is the panel-result JSON schema defined? Is it documented, versioned, or inferred from callers?
- What are the distinct top-level and nested sections a consumer can read? What shape does each take (scalar, list, dict, tree)?
- Which fields are canonical (always present) vs optional vs strategy-dependent (e.g. single vs map-reduce)?
- Is the schema stable? Are there deprecations, renames, or versioned variants in flight?

### 2. Existing downstream consumers of panel results

- What code reads `panel-result.json` today — inside `synthpanel`, inside SynthBench, inside any mayor/agent workflow?
- What subsets do those consumers use? (e.g. just `total_cost`, just `per_model_results`, just `synthesis`)
- Are there any existing transform/extract utilities (e.g. `analyze`, `inspect`, `synthesize`) that already produce human-readable summaries, and what format do they emit?
- What command-line flags, API functions, or MCP tools already accept a saved panel result as input?

### 3. Existing rendering / templating infrastructure in-repo

- Does the repo already pull in Jinja, any HTML templating, Markdown renderers, or static-site generators? Where, for what?
- How are site/* artifacts generated today (`render_site.py` post sp-ssrw)? Is that pipeline reusable for non-site outputs?
- What styling / layout conventions exist (tailwind, CSS files, theme colors)? Are they in the package or only in the marketing site?
- What's the rough size budget of the package on PyPI today, and which optional extras already fence off heavy dependencies?

### 4. Adjacent OSS tooling in the same ICP

- How do comparable OSS tools (Great Expectations, Prefect, Dagster, W&B local, MLflow, evidently, lm-eval-harness, HELM, Jupyter, nbconvert, quarto, papermill) present run results to non-engineers?
- Which of those tools ship a CLI-only path, a report-generator subcommand, a rendered HTML artifact, a notebook integration, or a live server?
- What packaging patterns do they use for the report-generation bits — core dep, optional extra, separate package?
- What explicit "no web UI" or "no dashboard" positioning exists elsewhere, and how is the substitute surfaced?

### 5. Stakeholder-output primitives already in the codebase

- Does any existing subcommand emit formatted text (tables, markdown, boxed-text CLI output, progress bars, summaries)? List them with their output shape.
- Is there a CLI print-helper abstraction? A structured "emit" layer? A reporter class? A formatter registry?
- Are there any existing `--format` / `--output-format` flags across subcommands? What do they accept (text / json / ndjson / other)? Are they consistent?
- What fields in panel output are intentionally human-readable vs machine-readable (e.g. pre-formatted dollar strings vs raw numbers)?

### 6. Non-CLI invocation surfaces

- The MCP server exposes 12 tools — do any of them return a rendered representation, or only structured data?
- Does the Python SDK provide a `render()` / `report()` / `summarize()` method anywhere? What shapes are returned?
- Are there existing consumers in mayor/ or related repos that build their own rendering on top of panel JSON? Describe the patterns they use (pandas DataFrame, jinja, plain string-formatting, other).
- What are stakeholder handoff patterns founders / PMs / researchers use in practice today (per README, docs, blog posts, community channels)?

### 7. Boundary conditions for any future rendering path

- What's the largest panel-result JSON size observed in the self-audit artifacts (`mayor/results/synthpanel-self-audit/`)? At what n does it cross 1 MB, 10 MB?
- Which sections of the output are safe to render verbatim vs need summarization (e.g. raw panelist text vs per-model cost)?
- Are there any privacy / redaction concerns with rendering panel results — PII in personas, BYOK credentials anywhere in the output?
- What's the current provenance / audit trail per panel (config_hash, timestamps, model versions) and how would a rendered report preserve it?

## Deliverable

`specs/sp-viz-layer/research/<dim>.md` per dimension, written objectively — what exists, not what *should* exist. Research phase synthesized into `research-map.md` for the human D-phase gate.
