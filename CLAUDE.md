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

## MCP Integration

The MCP server exposes 12 tools: `run_prompt`, `run_panel`, `run_quick_poll`, `extend_panel`, `list_persona_packs`, `get_persona_pack`, `save_persona_pack`, `list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack`, `list_panel_results`, `get_panel_result`.

### `extend_panel` vs branching (v3 instruments)

`extend_panel` and v3 branching instruments are **different mechanisms** and are easy to confuse. The contract:

- **`extend_panel`** always appends a single, ad-hoc round on top of an existing saved panel result. It reuses each panelist's saved session so the follow-up sees full conversational context, but it does **not** consult the original instrument's `route_when` clauses, makes no routing decision, and is **not** a re-entry into the authored DAG. Use it when you've read the results and want to ask one improvised follow-up.
- **v3 branching instruments** (`route_when` + `goto`) describe an authored DAG that `run_panel` walks at execution time. The router picks the next round based on the just-completed round's synthesis. Use this when the branching logic is part of the research design itself.

Rule of thumb: if the next question is decided **by the research design**, author it as a v3 round; if it's decided **by the human reading results**, use `extend_panel`. They are not interchangeable, and `extend_panel` is intentionally not a back-door into the DAG.

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
