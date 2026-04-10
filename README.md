# synth-panel

Open-source synthetic focus groups. Any LLM. Your terminal or your agent's tool call.

Define personas in YAML. Define your research instrument in YAML. Run against any LLM — from your terminal, from a pipeline, or from an AI agent's MCP tool call. Get structured, reproducible output with full cost transparency.

```bash
pip install synth-panel
synth-panel panel run --personas personas.yaml --instrument survey.yaml
```

## Why

Traditional focus groups cost $5,000-$15,000 and take weeks. Synthetic panels cost pennies and take seconds. They don't replace real user research, but they're excellent for:

- **Pre-screening** survey instruments before spending budget on real participants
- **Rapid iteration** on product names, copy, and positioning
- **Hypothesis generation** across demographic segments
- **Concept testing** at the speed of thought

## Quick Start

```bash
# Install from PyPI (v0.4.0+)
pip install synth-panel

# Or install from source for the latest unreleased changes
pip install git+https://github.com/DataViking-Tech/synth-panel.git@main

# Set your API key (Claude, OpenAI, Gemini, xAI, or any OpenAI-compatible provider)
export ANTHROPIC_API_KEY="sk-..."

# Run a single prompt
synth-panel prompt "What do you think of the name Traitprint for a career app?"

# Run a full panel
synth-panel panel run \
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
synth-panel panel run \
  --personas examples/personas.yaml \
  --instrument pricing-discovery
```

`pricing-discovery` is one of five bundled v3 packs (`pricing-discovery`,
`name-test`, `feature-prioritization`, `landing-page-comprehension`,
`churn-diagnosis`). List them with `synth-panel instruments list`.

The output now carries a `path` array recording the routing decisions
that actually fired:

```
discovery -> probe[themes contains price] -> probe_pricing -> validation
```

Render the DAG of any instrument:

```bash
synth-panel instruments graph pricing-discovery --format mermaid
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
synth-panel instruments list                       # bundled + installed packs
synth-panel instruments show pricing-discovery     # full YAML body
synth-panel instruments install ./my-pack.yaml     # add a local pack
synth-panel instruments graph pricing-discovery    # text DAG
synth-panel instruments graph pricing-discovery \
  --format mermaid                                 # mermaid flowchart
```

The unified instrument resolver (used by `panel run --instrument`) accepts
either a YAML path *or* an installed pack name, so you can iterate on a
local file and then `install` it once it's stable.

## LLM Provider Support

synth-panel works with any LLM provider. Set the appropriate environment variable:

| Provider | Environment Variable | Model Flag |
|----------|---------------------|------------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `--model sonnet` |
| Google (Gemini) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `--model gemini` |
| OpenAI | `OPENAI_API_KEY` | `--model gpt-4o` |
| xAI (Grok) | `XAI_API_KEY` | `--model grok` |
| Any OpenAI-compatible | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | `--model <model-id>` |

```bash
# Use Claude (default)
synth-panel panel run --personas p.yaml --instrument s.yaml

# Use GPT-4o
synth-panel panel run --personas p.yaml --instrument s.yaml --model gpt-4o

# Use a local model via Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 \
synth-panel panel run --personas p.yaml --instrument s.yaml --model llama3
```

## Architecture

synth-panel is a research harness, not an LLM wrapper. It orchestrates the research workflow:

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

synth-panel includes an MCP server so AI agents can run panels as tool calls:

```bash
synth-panel mcp-serve
```

Add to your editor's MCP config (Claude Code, Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "synth_panel": {
      "command": "synth-panel",
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
synth-panel panel run --personas p.yaml --instrument s.yaml

# JSON (pipe to jq, store in database)
synth-panel panel run --personas p.yaml --instrument s.yaml --output-format json

# NDJSON (streaming, one event per line)
synth-panel panel run --personas p.yaml --instrument s.yaml --output-format ndjson
```

## Budget Control

```bash
# Set a dollar budget for the panel
synth-panel panel run --personas p.yaml --instrument s.yaml --config budget.yaml
```

The cost tracker enforces soft budget limits — the current panelist completes, but no new panelists start if the budget is exceeded.

## Methodology Notes

Synthetic research is useful for exploration, hypothesis generation, and rapid iteration. It is **not** a replacement for talking to real humans.

Known limitations:
- Synthetic responses tend to cluster around means
- LLMs exhibit sycophancy (tendency to please)
- Cultural and demographic representation has blind spots
- Higher-order correlations between variables are poorly replicated

Use synth-panel to pre-screen and iterate, then validate with real participants.

## Versions

| Version | Highlights |
|---------|-----------|
| 0.5.0 | v3 branching instruments, router predicates, 5 bundled instrument packs, `instruments` subcommand (list/show/install/graph), MCP `*_instrument_pack` tools, rounds-shaped panel output, `extend_panel` ad-hoc round tool |
| 0.4.0 | `--var KEY=VALUE` and `--vars-file` for instrument templates, fail-loud on all-provider errors, default `--model` respects available credentials, `pack show <id>` alias, publish workflow fix |
| 0.3.0 | Structured output via tool-use forcing, cost tracking, MCP server (stdio), persona-pack persistence |

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and how to submit changes.

## MCP Server Documentation

For detailed MCP server documentation (all 12 tools, 4 resources, 3 prompt templates), see [docs/mcp.md](docs/mcp.md).

## License

MIT
