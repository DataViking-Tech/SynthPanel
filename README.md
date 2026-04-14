# synthpanel

Open-source synthetic focus groups. Any LLM. Your terminal or your agent's tool call.

Define personas in YAML. Define your research instrument in YAML. Run against any LLM — from your terminal, from a pipeline, or from an AI agent's MCP tool call. Get structured, reproducible output with full cost transparency.

```bash
pip install synthpanel
synthpanel panel run --personas personas.yaml --instrument survey.yaml

# For MCP server support (Claude Code, Cursor, Windsurf, etc.)
pip install synthpanel[mcp]
```

## Why

Traditional focus groups cost $5,000-$15,000 and take weeks. Synthetic panels cost pennies and take seconds. They don't replace real user research, but they're excellent for:

- **Pre-screening** survey instruments before spending budget on real participants
- **Rapid iteration** on product names, copy, and positioning
- **Hypothesis generation** across demographic segments
- **Concept testing** at the speed of thought

## Quick Start

```bash
# Install from PyPI
pip install synthpanel

# For MCP server support (agent integration)
pip install synthpanel[mcp]

# Or install from source for the latest unreleased changes
pip install git+https://github.com/DataViking-Tech/synthpanel.git@main

# Set your API key (Claude, OpenAI, Gemini, xAI, or any OpenAI-compatible provider)
export ANTHROPIC_API_KEY="sk-..."

# Run a single prompt
synthpanel prompt "What do you think of the name Traitprint for a career app?"

# Run a full panel
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml
```

## What You Get

```
============================================================
Persona: Sarah Chen (Product Manager, 34)
============================================================
  Q: What is the most frustrating part of your workflow?
  A: Version control on documents that aren't in a proper system...

  Cost: $0.0779

============================================================
Persona: Marcus Johnson (Small Business Owner, 52)
============================================================
  Q: What is the most frustrating part of your workflow?
  A: I'll send my manager a menu update in an email, she makes
     her changes, sends it back...

  Cost: $0.0761

============================================================
Total: estimated_cost=$0.2360
```

Each persona responds in character with distinct voice, concerns, and perspective. Cost is tracked and printed per-panelist and in aggregate.

## Defining Personas

```yaml
# personas.yaml
personas:
  - name: Sarah Chen
    age: 34
    occupation: Product Manager
    background: >
      Works at a mid-size SaaS company. 8 years in tech,
      previously a software engineer. Manages a team of 5.
    personality_traits:
      - analytical
      - pragmatic
      - detail-oriented

  - name: Marcus Johnson
    age: 52
    occupation: Small Business Owner
    background: >
      Runs a family-owned restaurant chain with 3 locations.
      Not tech-savvy but recognizes the need for digital tools.
    personality_traits:
      - practical
      - skeptical of technology
      - values personal relationships
```

## Defining Instruments

```yaml
# survey.yaml
instrument:
  questions:
    - text: >
        What is the most frustrating part of your current
        workflow when collaborating with others?
      response_schema:
        type: text
      follow_ups:
        - "Can you describe a specific recent example?"

    - text: >
        If you could fix one thing about how you work with
        technology daily, what would it be?
      response_schema:
        type: text
```

## Adaptive Research (0.5.0): Branching Instruments

A v3 instrument is a small DAG of *rounds*. After each round, a routing
predicate decides which round runs next based on the synthesizer's themes
and recommendation. The panel chooses its own probe path — no human in the
loop, no hand-coded conditional flows.

```bash
# The Show HN demo: ~$0.20, one command, the panel decides
# whether to dig into pain, pricing, or alternatives.
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument pricing-discovery
```

`pricing-discovery` is one of five bundled v3 packs (`pricing-discovery`,
`name-test`, `feature-prioritization`, `landing-page-comprehension`,
`churn-diagnosis`). List them with `synthpanel instruments list`.

The output now carries a `path` array recording the routing decisions
that actually fired:

```
discovery -> probe[themes contains price] -> probe_pricing -> validation
```

Render the DAG of any instrument:

```bash
synthpanel instruments graph pricing-discovery --format mermaid
```

### Predicate Reference

`route_when` is a list of clauses evaluated in order. The first matching
clause wins; an `else` clause is **mandatory** as the last entry.

```yaml
route_when:
  - if: { field: themes, op: contains, value: price }
    goto: probe_pricing
  - if: { field: recommendation, op: matches, value: "(?i)wait|delay" }
    goto: probe_objections
  - else: __end__
```

| Field | Source |
|-------|--------|
| `themes` | `SynthesisResult.themes` (list, substring match) |
| `recommendation` | `SynthesisResult.recommendation` (string) |
| `disagreements`, `agreements`, `surprises` | `SynthesisResult` (lists) |
| `summary` | `SynthesisResult.summary` (string) |

| Op | Meaning |
|----|---------|
| `contains` | Substring match against any list entry or the string |
| `equals` | Exact string match |
| `matches` | Python regex match (use `(?i)` for case-insensitive) |

The reserved target `__end__` terminates the run; the path so far feeds
final synthesis.

### Theme Matching: The R3 Caveat

> **Predicates match against the synthesizer's *exact* theme strings.**

`themes contains price` only fires if the synthesizer actually emitted a
theme containing the substring `price`. LLM synthesizers paraphrase —
"cost concerns" or "sticker shock" will not match. The bundled packs
mitigate this with a comment block at the top of the instrument that
hints at the canonical theme tags the synthesizer should prefer:

```yaml
# Synthesizer guidance: when emitting `themes`, prefer the short
# canonical tags below so route_when predicates match reliably:
#   - "pain"   (workflow pain, frustration, broken status quo)
#   - "price"  (cost concerns, perceived value, sticker shock)
#   - "alternative" (existing tools, workarounds, competitors)
```

When you author your own v3 packs, **always** add a similar tag-hint
block. The synthesizer reads it and tends to use the canonical tags;
your `contains` predicates then route reliably. If you skip this step,
expect routes to silently fall through to `else` because the
synthesizer's prose theme labels won't match your predicate values.

Prefer short, lowercase, single-token tags (`price`, `pain`, `confusion`)
over long phrases. `contains` does substring matching, so `price` will
also match `pricing`, `priced`, etc.

### `instruments` Subcommand

```bash
synthpanel instruments list                       # bundled + installed packs
synthpanel instruments show pricing-discovery     # full YAML body
synthpanel instruments install ./my-pack.yaml     # add a local pack
synthpanel instruments graph pricing-discovery    # text DAG
synthpanel instruments graph pricing-discovery \
  --format mermaid                                 # mermaid flowchart
```

The unified instrument resolver (used by `panel run --instrument`) accepts
either a YAML path *or* an installed pack name, so you can iterate on a
local file and then `install` it once it's stable.

## Examples

The [`examples/`](examples/) directory ships a persona pack plus one
instrument per format (v1 flat, v2 linear, v3 branching). Start from
[`examples/README.md`](examples/README.md) for the full index and
annotated walkthroughs — including two v3 branching patterns
(demographic segmentation and A/B concept testing) you can adapt to
your own studies.

## LLM Provider Support

synthpanel works with any LLM provider. Set the appropriate environment variable:

| Provider | Environment Variable | Model Flag |
|----------|---------------------|------------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `--model sonnet` |
| Google (Gemini) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `--model gemini` |
| OpenAI | `OPENAI_API_KEY` | `--model gpt-4o` |
| OpenRouter | `OPENROUTER_API_KEY` | `--model openrouter/anthropic/claude-haiku-4-5` |
| xAI (Grok) | `XAI_API_KEY` | `--model grok` |
| Any OpenAI-compatible | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | `--model <model-id>` |

```bash
# Use Claude (default)
synthpanel panel run --personas p.yaml --instrument s.yaml

# Use GPT-4o
synthpanel panel run --personas p.yaml --instrument s.yaml --model gpt-4o

# Use a local model via Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 \
synthpanel panel run --personas p.yaml --instrument s.yaml --model llama3
```

### Model Aliases

synthpanel ships with short aliases (`sonnet`, `opus`, `haiku`, `grok`,
`gemini`, `gemini-pro`) that map to canonical model identifiers. You can
override or extend these without changing code:

**Resolution order (highest priority wins):**

1. **`SYNTHPANEL_MODEL_ALIASES` env var** — JSON string of alias→model pairs
2. **`~/.synthpanel/aliases.yaml`** — YAML file
3. **Hardcoded defaults** — built into the package

```bash
# Override via env var (JSON)
export SYNTHPANEL_MODEL_ALIASES='{"sonnet": "claude-sonnet-4-6-20250414", "fast": "claude-haiku-4-5-20251001"}'
synthpanel prompt "Hello" --model fast
```

```yaml
# ~/.synthpanel/aliases.yaml
aliases:
  fast: claude-haiku-4-5-20251001
  smart: claude-opus-4-6
  sonnet: claude-sonnet-4-6-20250414
```

Env var entries override file entries, which override hardcoded defaults.
Aliases from all tiers are merged, so you only need to specify the ones you
want to add or change.

## Architecture

synthpanel is a research harness, not an LLM wrapper. It orchestrates the research workflow:

```
personas.yaml ──┐
                 ├──> Orchestrator ──> Panelist 1 ──> LLM ──> Response
instrument.yaml ─┘                 ├──> Panelist 2 ──> LLM ──> Response
                                   └──> Panelist N ──> LLM ──> Response
                                                                  │
                                              Aggregated Report <──┘
```

### Components

| Module | Purpose |
|--------|---------|
| `llm/` | Provider-agnostic LLM client (Anthropic, Google, OpenAI, xAI) |
| `runtime.py` | Agent session loop (turns, tool calls, compaction) |
| `orchestrator.py` | Parallel panelist execution with worker state tracking |
| `structured/` | Schema-validated responses via tool-use forcing |
| `cost.py` | Token tracking, model-specific pricing, budget enforcement |
| `persistence.py` | Session save/load/fork (JSON + JSONL) |
| `plugins/` | Manifest-based extension system with lifecycle hooks |
| `mcp/` | MCP server for agent-native invocation (stdio transport) |
| `cli/` | CLI framework with slash commands, output formatting |

### Design Principles

- **Minimal dependencies** — Python 3.10+ with `httpx` for HTTP and `pyyaml` for YAML parsing. Optional: `mcp` for the MCP server
- **Agent-native** — invoke from your terminal or from an AI agent's MCP tool call
- **Provider agnostic** — swap LLMs without changing research definitions
- **Cost transparent** — every API call is tracked and priced
- **Reproducible** — same personas + same instrument = comparable output
- **Structured by default** — responses conform to declared schemas

## MCP Server (Agent Integration)

synthpanel includes an MCP server so AI agents can run panels as tool calls:

```bash
synthpanel mcp-serve
```

Add to your editor's MCP config (Claude Code, Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "synth_panel": {
      "command": "synthpanel",
      "args": ["mcp-serve"],
      "env": { "ANTHROPIC_API_KEY": "sk-..." }
    }
  }
}
```

Tools exposed (12): `run_prompt`, `run_panel`, `run_quick_poll`, `extend_panel`, `list_persona_packs`, `get_persona_pack`, `save_persona_pack`, `list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack`, `list_panel_results`, `get_panel_result`.

`run_panel` accepts an inline `instrument` dict or an `instrument_pack`
name, so an agent can offload research-design judgment in a single tool
call. v3 responses include `rounds`, `path`, `terminal_round`, and
`warnings` alongside the back-compat `results` array.

`extend_panel` appends a single ad-hoc round to a saved panel result —
it is **not** a re-entry into the authored DAG. Use it for follow-up
probes that the original instrument didn't anticipate.

## Output Formats

```bash
# Human-readable (default)
synthpanel panel run --personas p.yaml --instrument s.yaml

# JSON (pipe to jq, store in database)
synthpanel panel run --personas p.yaml --instrument s.yaml --output-format json

# NDJSON (streaming, one event per line)
synthpanel panel run --personas p.yaml --instrument s.yaml --output-format ndjson
```

## Budget Control

```bash
# Set a dollar budget for the panel
synthpanel panel run --personas p.yaml --instrument s.yaml --config budget.yaml
```

The cost tracker enforces soft budget limits — the current panelist completes, but no new panelists start if the budget is exceeded.

## Persona Prompt Template Variants

The `templates/` directory contains four prompt template variants for benchmarking how persona prompt construction affects response quality:

| Template | File | Fields | Purpose |
|----------|------|--------|---------|
| **Current** | `templates/current.txt` | name, age, occupation, background, personality_traits | Control — documents the default prompt style |
| **Demo** | `templates/demo.txt` | name, age, occupation, education_level, income_bracket, urban_rural, political_leaning, background | Demographic-enriched — adds SubPOP/OpinionsQA stratification axes |
| **Values** | `templates/values.txt` | name, age, occupation, background, core_values, decision_style | Values-enriched — adds belief and decision-making context |
| **Minimal** | `templates/minimal.txt` | name, age, occupation | Ablation control — tests how much narrative matters |

Usage:

```bash
synthpanel panel run --personas personas.yaml --instrument survey.yaml --prompt-template templates/demo.txt
```

Templates use Python format-string syntax (`{field_name}`). Missing persona fields are left as literal `{field_name}` in the output.

## Methodology Notes

Synthetic research is useful for exploration, hypothesis generation, and rapid iteration. It is **not** a replacement for talking to real humans.

Known limitations:
- Synthetic responses tend to cluster around means
- LLMs exhibit sycophancy (tendency to please)
- Cultural and demographic representation has blind spots
- Higher-order correlations between variables are poorly replicated

Use synthpanel to pre-screen and iterate, then validate with real participants.

## Multi-Model Ensemble (0.7.0)

Run the same panel through multiple models and blend their response distributions for higher-fidelity results. [SynthBench experiments](https://github.com/DataViking-Tech/synthbench/blob/main/FINDINGS.md) show 3-model ensembles improve human-parity scores by +5-7 points over any single model.

```bash
# Run 3 models with equal weights and blend distributions
synthpanel panel run \
  --models haiku:0.33,gemini:0.33,gpt-4o-mini:0.34 \
  --blend \
  --personas personas.yaml \
  --instrument survey.yaml

# Each persona is interviewed by all 3 models independently.
# The --blend flag averages response distributions across models,
# producing more representative synthetic survey data.
```

The blended output includes per-model distributions and the weighted ensemble distribution, letting you inspect both individual model perspectives and the consensus view.

## Versions

| Version | Highlights |
|---------|-----------|
| 0.7.0 | Multi-model ensemble blending (`--blend`), OpenRouter provider support, temperature/top_p controls, prompt template customization |
| 0.6.0 | `--models` weighted model spec, `--temperature`/`--top_p` flags, persona prompt templates, pack generation, domain templates, MCP improvements |
| 0.5.0 | v3 branching instruments, router predicates, 5 bundled instrument packs, `instruments` subcommand (list/show/install/graph), MCP `*_instrument_pack` tools, rounds-shaped panel output, `extend_panel` ad-hoc round tool |
| 0.4.0 | `--var KEY=VALUE` and `--vars-file` for instrument templates, fail-loud on all-provider errors, default `--model` respects available credentials, `pack show <id>` alias, publish workflow fix |
| 0.3.0 | Structured output via tool-use forcing, cost tracking, MCP server (stdio), persona-pack persistence |

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and how to submit changes.

## MCP Server Documentation

For detailed MCP server documentation (all 12 tools, 4 resources, 3 prompt templates), see [docs/mcp.md](docs/mcp.md).

## Benchmarked on SynthBench

synth-panel's ability to produce representative synthetic respondents is independently measured by [SynthBench](https://dataviking-tech.github.io/synthbench/), an open benchmark for synthetic survey quality.

- **Want proof it works?** Browse the [leaderboard](https://dataviking-tech.github.io/synthbench/leaderboard/) — ensemble blending of 3 models hits SPS 0.90 (90% human parity).
- **Got a great configuration?** [Submit your scores](https://dataviking-tech.github.io/synthbench/submit/) and compare against baselines.
- **Contributing an adapter?** Heavy PRs with substantial behavior changes benefit from SynthBench results — reviewers can evaluate empirical quality, not just code. See [docs/adapter-guide.md](docs/adapter-guide.md) for the full adapter workflow.

## License

MIT
