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
# Install
pip install synth-panel

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

## Branching Instruments (0.5.0)

Instruments can branch. Instead of asking every panelist the same fixed list of
questions, a v3 instrument lets the panel decide its own probe path based on
what surfaced in the previous round. The orchestrator runs each round in
parallel, synthesizes the results, then routes to the next round using a
predicate over the synthesizer's output.

The "$0.20 adaptive research" demo uses the bundled `pricing-discovery` pack:

```bash
# List bundled instrument packs
synth-panel instruments list

# Run the branching pricing-discovery instrument against your personas
synth-panel panel run \
  --personas examples/personas.yaml \
  --instrument pricing-discovery
```

`pricing-discovery` opens with a discovery round, then routes into one of
three probes — `probe_pain`, `probe_pricing`, or `probe_alternatives` —
based on whether the panel's themes mention pain, price, or alternatives.
Every path converges through `probe_pricing` and ends at `wrap_up`.

The output adds two keys on top of the rounds-shaped payload:

- `path` — one entry per executed round: `{round, branch, next}`. Lets you
  see which way the panel actually went.
- `warnings` — parser warnings (e.g. unreachable rounds) plus any runtime
  routing issues. Empty when everything is clean.

### Predicate reference

Predicates live in YAML as **dict form** (no string parser, no eval):

```yaml
route_when:
  - if:
      field: themes          # dotted path into the round synthesis
      op: contains           # one of: contains | equals | matches
      value: pricing
    goto: probe_pricing
  - if:
      field: themes
      op: contains
      value: pain
    goto: probe_pain
  - else: probe_alternatives  # mandatory — no silent fall-through
```

Locked rules:

| Rule | Why |
|------|-----|
| Three operators only: `contains`, `equals`, `matches` | No grammar, no eval, ~50 LOC predicate engine |
| `else` is **mandatory** in every `route_when` block | Silent fall-through is a footgun. Use `else: __end__` to terminate explicitly |
| `__end__` is the reserved terminal sentinel | Any `goto: __end__` ends the run and triggers final synthesis on the executed path |
| DAG only — no loops, no back-edges | Validated at parse time; cycles are an error, not a warning |
| Final synthesis sees **executed rounds only** | The report describes what happened, not what could have happened |

### Theme matching expectations (R3)

> ⚠️ **`themes contains 'pricing'` matches against the synthesizer's exact
> tag output.** If the synthesizer emits `"price sensitivity"` instead of
> `"pricing"`, the predicate will not match and the router will fall through
> to `else`. This is the highest-residual UX risk in 0.5.0.

The mitigation is to **tell the synthesizer which tags you want** in your
round's synthesis prompt. The bundled packs follow this pattern — see the
`pricing-discovery.yaml` header:

```yaml
# Synthesizer guidance: when emitting `themes`, prefer the short canonical
# tags below so route_when predicates match reliably:
#   - "pain"        (workflow pain, frustration, broken status quo)
#   - "price"       (cost concerns, perceived value, sticker shock)
#   - "alternative" (existing tools, workarounds, competitors)
```

When you author your own branching instrument:

1. Pick canonical short tags up front (one or two words, lowercase).
2. List them as a comment near the top of the YAML.
3. Reference them verbatim in `route_when` predicates.
4. If you override `synthesis_prompt`, repeat the tag list in the prompt
   text so the model emits the form your predicates expect.
5. Run the instrument once against a small persona pack and inspect the
   `path` field — if the branch you expected was skipped, the synthesizer
   probably emitted a near-miss tag.

`contains` is substring-based, so `"pricing"` will match `"pricing"`,
`"pricing-sensitive"`, and `"product pricing concerns"`, but not
`"price"` on its own. Pick the shorter form when in doubt.

## `instruments` subcommand reference

```bash
# List installed instrument packs (bundled + user)
synth-panel instruments list

# Show the full YAML body of an installed pack
synth-panel instruments show pricing-discovery

# Install a pack from a YAML file (parses + validates before saving)
synth-panel instruments install path/to/my-instrument.yaml

# Render the instrument's branching graph as Mermaid (F3-D)
synth-panel instruments graph pricing-discovery
```

Installed packs live as single-file YAMLs under
`$SYNTH_PANEL_DATA_DIR/packs/instruments/<name>.yaml` (default
`~/.synth-panel/packs/instruments/`). The same name resolver that backs
`--instrument <path>` also accepts `--instrument <pack-name>`, so you can
develop instruments as files and promote them to installed packs without
changing how they're invoked.

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

Tools exposed: `run_prompt`, `run_panel`, `run_quick_poll`, `extend_panel`,
`list_persona_packs`, `get_persona_pack`, `save_persona_pack`,
`list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack`,
`list_panel_results`, `get_panel_result`.

`run_panel` accepts either a flat `questions` list (legacy v1) or an
`instrument` dict / `instrument_pack` name for v2/v3 multi-round and
branching runs. The response gains `path` and `warnings` keys for
branching visibility.

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

## Releases

| Version | Status | Highlights |
|---------|--------|------------|
| 0.5.0 | current | Branching v3 instruments, instrument packs (5 bundled), `instruments` subcommand, rounds-shaped output, `path` + `warnings` in MCP/CLI responses |
| 0.4.0 | shipped | Multi-round v2 instruments, structured response schemas, persona-pack ecosystem |
| 0.3.0 | shipped | MCP server, plugin system, multi-provider LLM client |

## License

MIT
