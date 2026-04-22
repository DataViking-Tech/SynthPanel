# CLI Rendering Primitives — Objective Codebase Map

## 1. Rendering & Templating Dependencies

**No heavyweight templating engines.** Core deps are `httpx>=0.27` + `pyyaml>=6.0` (`pyproject.toml:25-28`). Jinja is NOT imported anywhere.

**Two internal substitution engines:**
- `scripts/render_site.py:45-52` — regex-based `{{ key }}` placeholder substitution for site HTML. No dotted access, filters, or inheritance.
- `src/synth_panel/templates.py:38-84` — `_SafeFormatter` subclass of `string.Formatter` for question-text `{placeholder}` substitution. Blocks dotted/bracket access; returns literal placeholder on missing keys.

**No Markdown renderer, no HTML template lib, no static-site generator** in deps or code.

## 2. Site HTML Pipeline

- Entry: `scripts/render_site.py` (CLI script; also `render()` importable function).
- Placeholders: `{{ version }}` ← `src/synth_panel/__version__.py` (regex at 25-32); `{{ release_date }}` ← `CHANGELOG.md` `## [X.Y.Z] - YYYY-MM-DD` (35-42).
- Template: `site/index.html.j2` (515 lines); Output: `site/index.html`. Writes via `OUTPUT_PATH.write_text()` (55-62).
- **Not reusable for panel-result rendering** — substitution engine supports only simple key lookups, no schema/loops/conditionals; tailored to static site metadata only.

## 3. Styling & Theme

- Tailwind CSS via CDN (`<script src="https://cdn.tailwindcss.com">` line 73 of index.html.j2). Marketing-site only, not shipped with package.
- Colors: emerald 300/400/500 primary; slate 200/400/900 background; amber 300 accents; dark mode default (line 95).
- No CLI theme/color conventions. No table/banner/progress-bar color scheme.

## 4. Existing Subcommands Emitting Formatted Text

| Subcommand | Formats | Shape | Ref |
|---|---|---|---|
| `prompt` | TEXT | raw response + cost line | cli/commands.py:419-439 |
| `panel run` | TEXT | cost banners + synthesis | 1468-1583 |
| `panel run` (errors) | TEXT | `!` × 70 boxed banner | 1917-2026 |
| `panel inspect` | TEXT | key=value + tables | analysis/inspect.py:452+ |
| `analyze` | TEXT, CSV | stats tables, CSV rows | analyze.py:396-544 |
| `instruments graph` | TEXT, MERMAID | plain DAG or Mermaid flowchart | commands.py:2422-2475 |
| `pack list` | TEXT | name+description list | implicit |

### Cost summary format (`cost.py:561-591` `format_summary()`)

Two lines: `Total: total_tokens=... input=... output=... estimated_cost=$0.0000 model=... pricing=estimated-default` + `  cost breakdown: input=$... output=$... cache_write=$... cache_read=$...`.

### Error banners

- Total failure (commands.py:1917-1950) — `!` × 70 border, "PANEL RUN INVALID", failing models + sample errors
- Failure threshold exceeded (1953-2026) — error rate vs threshold + affected personas (first 4)
- Missing input (1996-2000) — refusal rate + affected personas

### Progress / status

**No progress bars or live updates.** Convergence telemetry (sp-yaru) writes JSON lines to stderr or `--convergence-log` file but not human-formatted.

## 5. Emit Layer

**Central `emit()` at `cli/output.py:20-60`** — single unified formatter.

```python
def emit(fmt: OutputFormat, *, message: str, usage: dict|None, extra: dict|None, file=None) -> None
```

| Format | Output |
|---|---|
| TEXT | message + optional `  tokens: input=N output=M` line |
| JSON | single-line `{"message":..., "usage":..., ...extra}` |
| NDJSON | `{"type":"message","text":..., "usage":..., ...extra}` |

`OutputFormat` enum at `cli/output.py:14-17`: `TEXT="text"`, `JSON="json"`, `NDJSON="ndjson"`.

**No reporter class or formatter registry.** Each subcommand directly calls `emit()` with hardcoded messages. Banner/summary/DAG functions return strings then get printed via `emit()` or `print()`.

**Cost formatting:** `CostEstimate.format_usd()` always returns `f"${total:.4f}"` (cost.py:458-459). Per-panelist cost uses same method in `format_panelist_result()` (_runners.py:516-535). Pre-formatted in panel output; machine consumers see the string, not the float — preserves billing precision across local-table drift.

## 6. `--format` / `--output-format` Flags

**Global `--output-format`:** parser.py:56-60. Choices: `text`, `json`, `ndjson`. Default: `text`. Parsed to `args.output_format` → `OutputFormat` enum → passed to all handlers.

**Per-subcommand overrides (legacy, less consistent):**
| Subcommand | Flag | Choices | Ref |
|---|---|---|---|
| `instruments graph` | `--format` | text, mermaid | parser.py:656-660 |
| `analyze` | `--output` | text, json, csv | parser.py:673-677 |

**MCP tools return only JSON** (always `json.dumps(..., indent=2)`). No `--format` equivalent.

## 7. Human-Readable vs Machine-Readable Fields

| Field | Readability | Source → Seam |
|---|---|---|
| `responses` | Machine | Raw text from LLM, `_runners.py:526-535` |
| `cost` | **Human** | Pre-formatted `$0.0000` via `format_usd()` |
| `usage` | Machine | Raw token counts |
| `synthesis.summary` | Hybrid | Free-text + structured lists |
| `verdict`, `confidence` | Human (verdict), Machine (confidence) | From structured-output schema; enum string vs 0-1 float |

**Transformation seams:**
1. Panelist response → structured extraction (`structured/output.py:52-148`) — LLM text → JSON via tool-use; extraction layer opaque to CLI
2. Panelist usage → cost string (`_runners.py:516-535`) — raw tokens exist in `usage`; formatted `cost` added as separate field; caller uses either
3. Synthesis result → flat template context (`templates.py:17-35`) — nested → `{theme_0, theme_1, agreement_0, ...}` dict; question templating sees flat keys only
4. Error details → banner text (commands.py:1917-2026) — structured dict → multiline boxed string; first sample truncated to 240 chars

**Intentionally pre-formatted:** cost (`$0.0000`), verdicts (enum strings), timestamps (ISO 8601 when present).

## 8. Package Size & Optional Dependencies

**Core:** ~150 KB wheel. `httpx` + `pyyaml` only.

**Optional extras (pyproject.toml:38-57):**
| Extra | Deps | Purpose |
|---|---|---|
| `[mcp]` | `mcp>=1.0` | MCP server, stdio transport (heavy) |
| `[composio]` | `composio>=0.5`, `pydantic>=2.0` | Vendor tool integrations (heavy) |
| `[convergence]` | `synthbench>=0.1` | Convergence + human-baseline lookup |
| `[dev]` | pytest, ruff, mypy, pip-audit | test/lint |

Heavy optional extras correctly fenced. Users get ~150 KB core unless they opt into MCP/Composio/convergence.

## Seams Observed

Seven extension points at transform boundaries:
1. **`OutputFormat` routing** — add enum variant + `emit()` branch to support a new format
2. **Cost display** — `format_summary()` is pure function; swap without touching panel runner
3. **Error banners** — `_build_*_banner()` pure functions; swap for HTML/Markdown templates
4. **Structured schema** — fully user-controlled via `--schema`/`--extract-schema`
5. **Question templating** — extend `_SafeFormatter` to support conditionals/filters/loops
6. **Analysis output** — `format_text()`/`format_csv()` pure transformations; add `format_html()`/`format_markdown()` without touching analysis engine
7. **Site rendering** — swap regex substitution for real templating engine

**No architectural blocker for HTML reports / Markdown docs / interactive dashboards** — seams exist; only implementation is missing.
