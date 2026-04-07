# synth-panel

A lightweight, LLM-agnostic research harness for running synthetic focus groups using AI personas.

## Architecture

Pure Python 3.10+ with minimal dependencies (`httpx` for HTTP, `pyyaml` for YAML parsing). Optional deps: `mcp` (MCP server).

```
src/synth_panel/
├── llm/                  # Provider-agnostic LLM client
│   ├── client.py         # Unified send/stream interface
│   ├── aliases.py        # Model alias resolution (sonnet → claude-sonnet-4-6)
│   ├── errors.py         # Error types and retry logic
│   ├── models.py         # Data models (CompletionRequest, CompletionResponse, etc.)
│   └── providers/        # Provider implementations
│       ├── anthropic.py  # Claude
│       ├── openai_compat.py  # OpenAI, local models
│       ├── xai.py        # Grok
│       └── gemini.py     # Google Gemini
├── runtime.py            # Agent session loop (turns, tool calls, compaction)
├── orchestrator.py       # Parallel panelist execution (ThreadPoolExecutor)
├── structured/           # Schema-validated responses via tool-use forcing
├── cost.py               # Token tracking, model pricing, budget enforcement
├── persistence.py        # Session save/load/fork (JSON + JSONL)
├── plugins/              # Manifest-based extension system with hooks
├── mcp/                  # MCP server (7 tools, stdio transport)
│   ├── server.py         # MCP server entry point
│   └── data.py           # Persona pack and result persistence
├── cli/                  # CLI framework
│   ├── parser.py         # argparse setup
│   ├── commands.py       # Subcommand handlers (prompt, panel run)
│   ├── repl.py           # Interactive REPL
│   ├── slash.py          # Slash command registry
│   └── output.py         # Output formatting (text, json, ndjson)
└── main.py               # Entry point
```

## Key Commands

```bash
# Single prompt
synth-panel prompt "Say hello"

# Full panel run
synth-panel panel run --personas examples/personas.yaml --instrument examples/survey.yaml

# MCP server (stdio, for Claude Code / Cursor / Windsurf)
synth-panel mcp-serve

# With specific model
synth-panel prompt "Hello" --model haiku
synth-panel prompt "Hello" --model gemini

# Output formats
synth-panel prompt "Hello" --output-format json
synth-panel prompt "Hello" --output-format ndjson
```

## Development

```bash
# Setup (uses uv)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,mcp]"

# Or without install
PYTHONPATH=src python3 -m synth_panel prompt "Hello"

# Run tests
pytest tests/

# Run acceptance tests (requires API key)
ANTHROPIC_API_KEY=sk-... pytest tests/test_acceptance.py -m acceptance
```

## Provider Configuration

Set the appropriate environment variable:

| Provider | Variable | Default Model |
|----------|----------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 |
| OpenAI | `OPENAI_API_KEY` | (must specify --model) |
| xAI | `XAI_API_KEY` | grok-3 |
| Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | gemini-2.5-flash |

## YAML Formats

### Personas
```yaml
personas:
  - name: Sarah Chen
    age: 34
    occupation: Product Manager
    background: "Works at a mid-size SaaS company..."
    personality_traits: [analytical, pragmatic, detail-oriented]
```

### Instruments
```yaml
instrument:
  version: 1              # Schema version (default: 1 if omitted)
  questions:
    - text: "What frustrates you about your workflow?"
      response_schema:
        type: text
      follow_ups:
        - "Can you describe a specific example?"
```

### Branching instruments (v3, 0.5.0)

v3 instruments add `rounds` with `route_when` predicates. Predicates are
**dict form on disk** (no string parser): three operators (`contains`,
`equals`, `matches`), and `else` is mandatory in every block. `__end__`
is the reserved terminal sentinel.

```yaml
instrument:
  version: 3
  rounds:
    - name: discovery
      questions: [{text: "Walk me through ..."}]
      route_when:
        - if: {field: themes, op: contains, value: pricing}
          goto: probe_pricing
        - else: __end__
    - name: probe_pricing
      questions: [{text: "What would feel fair to pay ..."}]
```

**Theme-matching gotcha (R3):** `themes contains 'pricing'` matches against
the synthesizer's exact tag output. If the synthesizer emits
`"price sensitivity"` instead of `"pricing"`, the predicate fails and the
router falls through to `else`. Mitigation: list the canonical tag
vocabulary in a comment near the top of your instrument YAML and (if you
override `synthesis_prompt`) repeat the list in the prompt text. Bundled
packs follow this pattern.

### `extend_panel` vs branching

`extend_panel` always **appends a single ad-hoc round** on top of an
existing panel result. It is *not* a re-entry into the authored DAG —
routing is not re-evaluated, no branches are replayed, and the original
instrument's `route_when` clauses are ignored. Use branching when you
want the panel to choose its own probe path; use `extend_panel` when you
have a saved result and want to ask one more thing.

## MCP Integration

The MCP server exposes 12 tools: `run_prompt`, `run_panel`, `run_quick_poll`,
`extend_panel`, `list_persona_packs`, `get_persona_pack`, `save_persona_pack`,
`list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack`,
`list_panel_results`, `get_panel_result`.

`run_panel` accepts either a flat `questions` list (legacy v1) or an
`instrument` dict / `instrument_pack` name for v2/v3 branching runs. The
response includes `rounds`, `path` (one entry per executed round) and
`warnings` (parser + runtime). For v1 single-round runs `path` has length 1.

Claude Code plugin: install via `/plugin install synth-panel`. Adds `/focus-group` skill.

Standalone MCP config for any editor:
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

## Clean-Room Implementation

This codebase was built via clean-room methodology from a functional specification (SPEC.md). No code was copied from any reference implementation. The spec describes behavioral contracts only — all naming, structure, and implementation are original.

## Conventions

- All modules use dataclasses for data models
- Errors use a category enum (retryable vs non-retryable)
- Provider detection is prefix-based (claude-* → Anthropic, gemini-* → Google, grok-* → xAI)
- Cost tracking is per-turn with 4 token buckets (input, output, cache_write, cache_read)
- MCP mode defaults to haiku (cheap/fast for iterative use); CLI defaults to sonnet
