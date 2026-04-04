# synth-panel

A lightweight, LLM-agnostic research harness for running synthetic focus groups and user research panels using AI personas.

Define personas in YAML. Define your research instrument in YAML. Run against any LLM. Get structured, reproducible output with full cost transparency.

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

# Set your API key (Claude, OpenAI, xAI, or any OpenAI-compatible provider)
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

## LLM Provider Support

synth-panel works with any LLM provider. Set the appropriate environment variable:

| Provider | Environment Variable | Model Flag |
|----------|---------------------|------------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `--model sonnet` |
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
| `llm/` | Provider-agnostic LLM client (Anthropic, OpenAI, xAI) |
| `runtime.py` | Agent session loop (turns, tool calls, compaction) |
| `orchestrator.py` | Parallel panelist execution with worker state tracking |
| `structured/` | Schema-validated responses via tool-use forcing |
| `cost.py` | Token tracking, model-specific pricing, budget enforcement |
| `persistence.py` | Session save/load/fork (JSON + JSONL) |
| `plugins/` | Manifest-based extension system with lifecycle hooks |
| `cli/` | CLI framework with REPL, slash commands, output formatting |

### Design Principles

- **Minimal dependencies** — pure Python 3.10+ with only `httpx` and `pyyaml` as required deps
- **Provider agnostic** — swap LLMs without changing research definitions
- **Cost transparent** — every API call is tracked and priced
- **Reproducible** — same personas + same instrument = comparable output
- **Structured by default** — responses conform to declared schemas

## Output Formats

```bash
# Human-readable (default)
synth-panel panel run --personas p.yaml --instrument s.yaml

# JSON (pipe to jq, store in database)
synth-panel panel run --personas p.yaml --instrument s.yaml --output-format json

# NDJSON (streaming, one event per line)
synth-panel panel run --personas p.yaml --instrument s.yaml --output-format ndjson
```

## Interactive Mode

```bash
synth-panel  # launches REPL

> /help              # list commands
> /model opus        # switch model
> /status            # show session state
> Tell me about yourself
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

## License

MIT
