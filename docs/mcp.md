# MCP Server Reference

synth-panel ships an [MCP](https://modelcontextprotocol.io/) server so AI agents can run synthetic focus groups as tool calls. The server uses stdio transport and defaults to the `haiku` model for cheap, fast iterative use.

## Starting the Server

```bash
synth-panel mcp-serve
```

The server communicates over stdin/stdout using JSON-RPC (the MCP protocol). It is designed to be launched by an MCP-aware editor or agent framework.

## Editor Configuration

### Claude Code / Cursor / Windsurf

Add to your MCP config (e.g., `.claude/mcp.json`, `.cursor/mcp.json`):

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

Set the environment variable for whichever LLM provider you want to use. See the main [README](../README.md#llm-provider-support) for the full provider table.

### Claude Code Plugin

```
/plugin install synth-panel
```

This adds the `/focus-group` skill to your Claude Code session.

## Tools (12)

### Research Tools

| Tool | Description |
|------|-------------|
| `run_prompt` | Send a single prompt to an LLM. No personas required. The simplest tool — ask a quick research question. |
| `run_panel` | Run a full synthetic focus group panel. Each persona answers all questions independently in parallel, followed by synthesis. Accepts inline `questions`, an inline `instrument` dict (v1/v2/v3), or an `instrument_pack` name. |
| `run_quick_poll` | Quick single-question poll across personas. A simplified `run_panel` for one question with synthesis. |
| `extend_panel` | Append a single ad-hoc round to a saved panel result. Reuses each panelist's saved session for conversational context. **Not** a re-entry into the v3 DAG — use for human-in-the-loop follow-ups. |

### Persona Pack Management

| Tool | Description |
|------|-------------|
| `list_persona_packs` | List all saved persona packs (bundled + user-saved). Returns ID, name, persona count. |
| `get_persona_pack` | Get a specific persona pack by ID. Returns the full persona definitions. |
| `save_persona_pack` | Save a persona pack for reuse. Validates persona data before saving. |

### Instrument Pack Management

| Tool | Description |
|------|-------------|
| `list_instrument_packs` | List installed instrument packs (bundled + user-saved). Returns manifest metadata. |
| `get_instrument_pack` | Load an installed instrument pack by name. Returns the full YAML body. |
| `save_instrument_pack` | Install an instrument pack. Validates the instrument via the parser before writing to disk. |

### Result Management

| Tool | Description |
|------|-------------|
| `list_panel_results` | List all saved panel results. Returns ID, date, model, and counts. |
| `get_panel_result` | Get a specific panel result by ID. Returns the full result with all rounds and synthesis. |

## Resources (4 URI Patterns)

MCP resources allow agents to read data without invoking a tool.

| URI Pattern | Description |
|-------------|-------------|
| `persona-pack://{pack_id}` | A specific persona pack |
| `persona-pack://` | List all persona packs |
| `panel-result://{result_id}` | A specific panel result |
| `panel-result://` | List all panel results |

## Prompt Templates (3)

Prompt templates provide pre-built research workflows that agents can use as starting points.

| Prompt | Parameters | Description |
|--------|------------|-------------|
| `focus_group` | `topic` (required), `num_personas` (default: 5), `follow_up` (default: true) | Generate a structured focus group discussion prompt for a given topic. |
| `name_test` | `names` (required, comma-separated), `context` (optional) | Test product or feature name options with diverse perspectives. |
| `concept_test` | `concept` (required), `target_audience` (optional) | Test a concept or idea with targeted personas. |

## Response Shape

All panel runs (`run_panel`, `run_quick_poll`, `extend_panel`) return a uniform response shape:

```json
{
  "result_id": "result-20260410-...",
  "model": "haiku",
  "persona_count": 5,
  "question_count": 3,
  "rounds": [
    {
      "name": "discovery",
      "results": [
        {
          "persona": "Sarah Chen",
          "responses": ["..."],
          "usage": { "input_tokens": 450, "output_tokens": 120 },
          "cost": "$0.0012",
          "error": null
        }
      ],
      "synthesis": { "themes": [...], "summary": "..." }
    }
  ],
  "path": [
    { "round": "discovery", "branch": "themes contains price", "next": "probe_pricing" }
  ],
  "terminal_round": "probe_pricing",
  "warnings": [],
  "synthesis": { "themes": [...], "summary": "...", "recommendation": "..." },
  "total_cost": "$0.0234",
  "total_usage": { "input_tokens": 2250, "output_tokens": 600 },
  "results": [...]
}
```

- `rounds` — per-round results with panelist responses and per-round synthesis
- `path` — the routing decisions that fired (v3 branching instruments)
- `terminal_round` — the round whose synthesis fed final synthesis
- `warnings` — parser or runtime warnings
- `results` — back-compat flat array mirroring the terminal round's panelist results

For v1/v2 instruments and raw `questions` input, `path` is empty or linear and `warnings` is typically empty — the shape is uniform across versions.

## Data Storage

Panel results, persona packs, and instrument packs are stored under `~/.synth-panel/` (configurable via `SYNTH_PANEL_DATA_DIR`):

```
~/.synth-panel/
├── persona_packs/          # Saved persona packs (YAML)
├── packs/instruments/      # Installed instrument packs (YAML)
└── results/                # Panel results (JSON) + session data
```
