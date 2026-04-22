# Research Map — sp-viz-layer

Synthesis of three research dimensions for the D-phase gate. Ticket now revealed: **sp-nj58 — viz/reporting layer for non-CLI stakeholders.**

## What the codebase currently provides

**Panel-result JSON is richly structured but semi-versioned.** Output schema is inferred from dataclasses + docs prose at `sdk.py:224-286` + `synthesis.py:64-97` + `mcp/data.py:482-530`. No explicit `schema_version`; optional-fields-default pattern preserves back-compat. Strategy-dependent fields (`synthesis.strategy`, `synthesis.per_question_synthesis`, `rounds[i].usage`) are only present on map-reduce runs. Largest observed artifact: 6.7 MB (n=100, 5 questions, full synthesis). Linear growth ~40 KB/persona; 1 MB at n≈25, extrapolates to 10 MB at n≈250.

**Zero heavyweight templating infrastructure in-repo.** Core deps `httpx` + `pyyaml` (~150 KB wheel). Jinja is not imported anywhere. Two minimal internal substitution engines exist: `scripts/render_site.py` (regex `{{ key }}` for site generation, not reusable) and `src/synth_panel/templates.py::_SafeFormatter` (question-text `{placeholder}` substitution with security-guard dotted-access block).

**Unified emit layer at `cli/output.py:20-60`** — `emit()` routes through `OutputFormat` enum (`text`/`json`/`ndjson`). All subcommands use it. Per-subcommand `--format` and `--output` overrides exist (`instruments graph`: `text`/`mermaid`; `analyze`: `text`/`json`/`csv`) but are legacy, predating the global flag.

**Pre-formatting at transformation seams:**
- `cost` is always a pre-formatted `$0.0000` USD string (via `format_usd()`) rather than a raw float
- Error banners are pure-function strings that go to stderr regardless of `--output-format`
- Synthesis result → flat template context `{theme_0, agreement_0, ...}` for question templating (synthesis structure not exposed)

**No progress bars, no live status, no dashboards.** Convergence telemetry (sp-yaru) writes JSON lines to stderr or `--convergence-log` file only.

## How comparable OSS tools solve this

Three observable clusters:

1. **Raw structured data + consumer choice** (lm-eval-harness, papermill): emit clean JSON/notebook, defer viz to ecosystem (W&B, Zeno, HF Hub). Explicit "no built-in dashboard" positioning. CLI-first.
2. **Opinionated default reports** (Great Expectations, HELM, Quarto, evidently): ship structured results plus one or more rendered default formats (HTML, text summary, Jupyter notebook). No live UI hosting; static artifacts.
3. **Interactive dashboard-first** (Prefect, Dagster, MLflow, W&B): live web UI is primary path; real-time observability + collaborative features; structured data accessible via API but secondary. Often cloud-backed.

Tools emphasizing structured-data-first (lm-eval-harness, papermill) tend to **explicitly avoid hosting**. Tools without that positioning tend to ship dashboards. Visualization is a strategic choice, not an afterthought.

## Observable seams (natural extension points)

Seven places in the current codebase where a rendering layer could plug in without architectural changes:

1. **`OutputFormat` routing** — add enum variant + `emit()` branch. Zero refactor of panel runners.
2. **Cost display** — `format_summary()` is a pure function; swap to produce HTML tables, Markdown, or structured objects without touching the panel runner.
3. **Error banners** — `_build_*_banner()` pure functions; swap for HTML/Markdown templates.
4. **Analysis output** — `format_text()`/`format_csv()` in `analyze.py:505-544` are pure transformations; adding `format_html()` or `format_markdown()` is additive.
5. **`analysis/inspect.py::InspectReport`** — walks the full schema today, used only by `panel inspect`. Ready for a non-CLI consumer.
6. **Synthesis strategy metadata** — map-reduce `per_question_synthesis` is serialized but no consumer renders it; ready for step-wise disclosure.
7. **Site rendering pipeline** (`scripts/render_site.py`) — the regex substitution engine is minimal but the pattern (template + placeholder dict + render function) generalizes.

## Design-space tensions (for D-phase discussion)

**Output-medium spectrum:** static markdown ↔ HTML report ↔ Jupyter notebook ↔ Voilà-style mini-dashboard ↔ full web UI. Each has different implications for package size, positioning ("CLI-first" vs adding-a-dashboard), and privacy (BYOK credentials shouldn't leak into shareable artifacts).

**Input-source tension:** rendered output can derive from (a) live panel run via CLI subcommand (`synthpanel panel run --report HTML`), (b) post-hoc from saved JSON (`synthpanel report RESULT.json`), (c) MCP server returning rendered representation, (d) Python SDK emitting via a new `render()` method. These aren't mutually exclusive but have different consumer audiences.

**Package-size tension:** current core wheel is ~150 KB. A report-generation extra that pulls in Jinja + HTML sanitizer + possibly a charting library would grow the install footprint. `synthpanel[report]` extra is the obvious fencing pattern; precedent already exists with `[mcp]`, `[composio]`, `[convergence]`.

**Audience-authoring tension:** the rendered output needs an audience definition (exec brief? PM handoff? stakeholder PDF?) before the output shape can settle. Different audiences want different information density, different visual conventions, different length.

**Provenance preservation:** a rendered report must carry `config_hash`, timestamps, model versions, pricing-snapshot-date for reproducibility — otherwise the report is a trust-laundering surface. The metadata fields exist in panel JSON (sp-ui40); the rendering layer needs to surface them.

**Privacy constraints:** panel responses may include PII in personas; BYOK credentials live in the venv's environment, not in output JSON, but worth audit before rendering is shareable.

## What the research does not answer

- What's the primary user journey — someone running a panel and wanting a handoff artifact, or someone receiving a shared panel and wanting to consume it?
- How much does the founder want "we don't ship a UI" to remain load-bearing vs evolvable?
- Whether synthpanel's ICP actually expects charts / tables, or whether the existing structured JSON + external tooling (Jupyter, Quarto) would satisfy the signal we heard from panel audits.
- Whether an MCP-surface rendered representation (returned from an MCP tool) is the clearest fit given the "zero-config inside Claude Desktop" positioning.

## Files

- `research/panel-json-schema.md`
- `research/cli-rendering-primitives.md`
- `research/adjacent-oss-landscape.md`

## Ready for D-phase gate

No design direction proposed. Research complete per the seven dimensions of `questions.md`. Human "brain-surgery" on architectural direction is the next gate.
