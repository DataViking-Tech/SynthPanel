# synthpanel

[![PyPI](https://img.shields.io/pypi/v/synthpanel.svg)](https://pypi.org/project/synthpanel/)
[![CI](https://github.com/DataViking-Tech/SynthPanel/actions/workflows/ci.yml/badge.svg)](https://github.com/DataViking-Tech/SynthPanel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/synthpanel.svg)](https://pypi.org/project/synthpanel/)
[![MCP](https://img.shields.io/badge/MCP-enabled-brightgreen.svg)](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md)
[![GHCR](https://img.shields.io/badge/ghcr.io-dataviking--tech%2Fsynthpanel-2496ED?logo=docker&logoColor=white)](https://github.com/DataViking-Tech/SynthPanel/pkgs/container/synthpanel)

Site: <https://synthpanel.dev> · Benchmark: <https://synthbench.org>

Open-source synthetic focus groups. Any LLM. Your terminal or your agent's tool call.

**Zero-config inside any MCP host that speaks sampling** (Claude Desktop, Claude Code, Cursor, Windsurf) — drop the config in and run a panel with **no API key set**. The host runs the model on your behalf, using its own subscription. Bring your own provider key (Claude, GPT, Gemini, Grok, local) when you want reproducibility, ensembles, or larger panels. Personas and instruments are plain YAML; every response is schema-validated with per-turn cost telemetry. Runs from your terminal, a pipeline, or an AI agent's MCP tool call.

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
pip install git+https://github.com/DataViking-Tech/SynthPanel.git@main

# Set your API key (Claude, OpenAI, Gemini, xAI, or any OpenAI-compatible provider)
export ANTHROPIC_API_KEY="sk-..."

# Run a single prompt
synthpanel prompt "What do you think of the name Traitprint for a career app?"

# Run a full panel
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml
```

## Works with

SynthPanel is **MCP-native** — it ships an MCP server, and every major
agent framework now supports MCP as a first-class tool source. That
means SynthPanel works out of the box with any framework that speaks
MCP, with zero framework-specific wrapper packages to install. Runnable
examples for each framework live in
[`examples/integrations/`](examples/integrations/README.md).

| Framework | Example | Bridge | One-line install |
|-----------|---------|--------|------------------|
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/) | [openai_agents.py](examples/integrations/openai_agents.py) | Built-in `MCPServerStdio` | `pip install openai-agents synthpanel[mcp]` |
| [LlamaIndex](https://docs.llamaindex.ai/) | [llamaindex_tool.py](examples/integrations/llamaindex_tool.py) | `llama-index-tools-mcp` | `pip install llama-index-tools-mcp llama-index-llms-anthropic synthpanel[mcp]` |
| [CrewAI](https://docs.crewai.com/) | [crewai_tool.py](examples/integrations/crewai_tool.py) | `crewai-tools[mcp]` | `pip install "crewai-tools[mcp]" crewai synthpanel[mcp]` |
| [LangChain](https://python.langchain.com/) | [langchain_tool.py](examples/integrations/langchain_tool.py) | `langchain-mcp-adapters` | `pip install langchain-mcp-adapters langchain-anthropic synthpanel[mcp]` |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | [langchain_tool.py](examples/integrations/langchain_tool.py) | `langchain-mcp-adapters` | `pip install langchain-mcp-adapters langgraph langchain-anthropic synthpanel[mcp]` |
| [Microsoft Agent Framework 1.0](https://learn.microsoft.com/en-us/agent-framework/) | [microsoft_agent.py](examples/integrations/microsoft_agent.py) | Built-in `MCPStdioTool` | `pip install agent-framework synthpanel[mcp]` |
| [n8n](https://n8n.io/) | [n8n_workflow.json](examples/integrations/n8n_workflow.json) | Built-in MCP Client tool | `pip install synthpanel[mcp]` on the n8n runner |
| [LangChain via Composio](https://composio.dev/) | [composio_langchain.py](examples/integrations/composio_langchain.py) | `synth_panel.integrations.composio` (in-process, non-MCP) | `pip install composio composio_langchain langchain langchain-anthropic synthpanel` |
| [CrewAI via Composio](https://composio.dev/) | [composio_crewai.py](examples/integrations/composio_crewai.py) | `synth_panel.integrations.composio` (in-process, non-MCP) | `pip install composio composio_crewai crewai synthpanel` |

Also reaches [Zapier MCP](https://zapier.com/mcp) (30K+ actions), the
[VS Code AI Toolkit](https://code.visualstudio.com/api/extension-guides/ai/mcp),
Windsurf, Cursor, Zed, Claude Code, and Claude Desktop via the same MCP
server — all clients in that list install SynthPanel with
`pip install synthpanel[mcp]` and a one-line MCP config entry (see
[Use with Claude Code / Cursor / Windsurf / Zed](#use-with-claude-code--cursor--windsurf--zed)).

> **Don't see your framework?** MCP bridges are available for nearly
> every major agent framework. Start from
> [`examples/integrations/README.md`](examples/integrations/README.md)
> — the pattern is identical in each case (point the client at
> `synthpanel mcp-serve` over stdio) — or
> [file an issue](https://github.com/DataViking-Tech/SynthPanel/issues)
> so we can add a sibling example.

## Run via Docker

A pre-built image is published to both
[GitHub Container Registry](https://github.com/DataViking-Tech/SynthPanel/pkgs/container/synthpanel)
and Docker Hub on every tagged release. Use it for ephemeral or serverless
invocation (Lambda, Cloud Run, GitHub Actions, n8n) where you'd rather
spin up a container than pip-install.

```bash
# Pull (either registry works — same image, multi-arch: amd64 + arm64)
docker pull ghcr.io/dataviking-tech/synthpanel:latest
docker pull synthpanel/synthpanel:latest

# One-off prompt
docker run --rm \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  synthpanel/synthpanel \
  prompt "What makes a name feel trustworthy?"

# MCP server on stdio (default CMD — wire this into an agent's MCP config)
docker run --rm -i \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  synthpanel/synthpanel

# Panel run with a mounted instrument file
docker run --rm \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -v "$PWD":/work -w /work \
  synthpanel/synthpanel \
  panel run --personas personas.yaml --instrument survey.yaml
```

The image's default `CMD` is `mcp-serve`, so omitting the command starts
the MCP stdio server. Any `synthpanel` subcommand can be passed as
arguments to override. Provider keys are read from environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`,
`XAI_API_KEY`) — pass whichever your model requires.

Pin to a specific version (`:0.9.1`) in production rather than `:latest`.

## Use as a Python Library

Everything the CLI and MCP server can do is also callable from Python.
No subprocess, no extra install — just import and go.

```python
from synth_panel import quick_poll, run_panel, run_prompt

# One-shot LLM call
reply = run_prompt("What makes a name feel trustworthy?")
print(reply.response, reply.cost)

# Ask a bundled persona pack a single question
poll = quick_poll(
    "Which pricing tier name feels most premium: Core, Plus, or Pro?",
    pack_id="general-consumer",
)
print(poll.synthesis["recommendation"])

# Run a full branching instrument against a bundled pack
panel = run_panel(
    pack_id="general-consumer",
    instrument_pack="pricing-discovery",
)
print(panel.path)         # e.g. ["discovery", "probe_pricing", "validation"]
print(panel.total_cost)
```

The package root exposes eight functions plus three typed return
dataclasses — `PromptResult`, `PollResult`, `PanelResult`. Every
result is dict-compatible (`result["model"]`) so code that used to
consume the MCP JSON payload works unchanged.

| Function | What it does |
|----------|--------------|
| `run_prompt(prompt, *, model=...)` | Single LLM call — no personas |
| `quick_poll(question, pack_id=...)` | One question across a panel + synthesis |
| `run_panel(pack_id=..., instrument_pack=...)` | Full branching panel run |
| `extend_panel(result_id, questions)` | Append an ad-hoc follow-up round |
| `list_personas()` / `list_instruments()` | Discover installed packs |
| `list_panel_results()` / `get_panel_result(id)` | Reload saved results |

Use this path when subprocess overhead hurts (Jupyter, serverless, CI)
or when you want to wrap SynthPanel in a LangChain / LlamaIndex tool
in three lines. See [`examples/sdk_usage.py`](examples/sdk_usage.py)
for a runnable end-to-end walkthrough.

## MCP Server (Agent Integration)

synthpanel ships an MCP server so AI agents can run synthetic focus groups as tool calls.

```bash
pip install synthpanel[mcp]
synthpanel mcp-serve
```

Add to your editor's MCP config (Claude Code, Cursor, Windsurf):

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

### Zero-config first run via sampling

No API key? No problem. When the invoking MCP client (Claude Desktop,
Claude Code, Cursor, Windsurf) advertises the `sampling` capability,
synthpanel falls back to asking the client to run the LLM completion on
its behalf — using the client's own subscription. That means `run_prompt`
and small `run_quick_poll` calls (up to 3 personas) work with zero env
setup:

```json
{
  "mcpServers": {
    "synth_panel": {
      "command": "synthpanel",
      "args": ["mcp-serve"]
    }
  }
}
```

Sampling mode is great for first-touch UX and quick exploratory polls.
For cross-provider ensembles, larger panels, and reproducible model
versioning, set a provider key in `env` to graduate to BYOK. See
[docs/mcp.md#sampling-mode](docs/mcp.md#sampling-mode) for the full
matrix of when sampling kicks in and what it costs.

### Tools (12)

| Tool | Description |
|------|-------------|
| `run_prompt` | Send a single prompt to an LLM — no personas required |
| `run_panel` | Run a full synthetic focus group panel with parallel panelists and synthesis |
| `run_quick_poll` | Quick single-question poll across personas with synthesis |
| `extend_panel` | Append an ad-hoc follow-up round to a saved panel result |
| `list_persona_packs` | List all saved persona packs (bundled + user-saved) |
| `get_persona_pack` | Get a specific persona pack by ID |
| `save_persona_pack` | Save a persona pack for reuse |
| `list_instrument_packs` | List installed instrument packs (bundled + user-saved) |
| `get_instrument_pack` | Load an installed instrument pack by name |
| `save_instrument_pack` | Install an instrument pack with validation |
| `list_panel_results` | List all saved panel results |
| `get_panel_result` | Get a specific panel result with all rounds and synthesis |

`run_panel` accepts an inline `instrument` dict or an `instrument_pack` name for v3 branching runs. `extend_panel` appends one ad-hoc round — it is **not** a re-entry into the v3 DAG. See [docs/mcp.md](docs/mcp.md) for full tool schemas, resources, and prompt templates.

## Use with Claude Code / Cursor / Windsurf / Zed

Copy the JSON snippet for your editor into the listed config file, set
your API key, and restart the editor. `synthpanel mcp-serve` is launched
on demand over stdio — no long-running process to manage.

<details>
<summary><b>Claude Code</b></summary>

Add to `.mcp.json` at your project root (or `~/.claude.json` for all projects):

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

Or install the bundled plugin (adds a `/focus-group` skill):

```
/plugin install synthpanel
```

</details>

<details>
<summary><b>Cursor</b></summary>

Add to `.cursor/mcp.json` at your project root (or `~/.cursor/mcp.json` for all projects):

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

</details>

<details>
<summary><b>Windsurf</b></summary>

Add to `~/.codeium/windsurf/mcp_config.json` (or open
**Settings → Windsurf Settings → MCP Servers → View Raw Config**):

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

</details>

<details>
<summary><b>Zed</b></summary>

Zed uses `context_servers` (not `mcpServers`). Add to `~/.config/zed/settings.json`:

```json
{
  "context_servers": {
    "synth_panel": {
      "source": "custom",
      "command": "synthpanel",
      "args": ["mcp-serve"],
      "env": { "ANTHROPIC_API_KEY": "sk-..." }
    }
  }
}
```

</details>

<details>
<summary><b>Claude Desktop</b></summary>

Open **Settings → Developer → Edit Config** (or edit the file directly):

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

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

Restart Claude Desktop after editing.

</details>

> **Using a non-Anthropic provider?** Swap `ANTHROPIC_API_KEY` for
> `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, or
> `OPENROUTER_API_KEY` — see [LLM Provider Support](#llm-provider-support).
> The `synthpanel` binary must be on the editor's `PATH`; if you installed
> into a virtualenv, point `command` at its absolute path
> (e.g. `/path/to/.venv/bin/synthpanel`).

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

## Benchmarked on SynthBench

synthpanel's ability to produce representative synthetic respondents is independently measured by [SynthBench](https://dataviking-tech.github.io/synthbench/), an open benchmark for synthetic survey quality.

- **Want proof it works?** Browse the [leaderboard](https://dataviking-tech.github.io/synthbench/leaderboard/) — ensemble blending of 3 models hits SPS 0.90 (90% human parity).
- **Got a great configuration?** [Submit your scores](https://dataviking-tech.github.io/synthbench/submit/) and compare against baselines.
- **Contributing an adapter?** Heavy PRs with substantial behavior changes benefit from SynthBench results — reviewers can evaluate empirical quality, not just code. See [docs/adapter-guide.md](docs/adapter-guide.md) for the full adapter workflow.

## License

MIT
