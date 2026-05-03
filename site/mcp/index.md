# MCP Server — synthpanel · Run synthetic focus groups from your AI editor

 MCP Server · stdio

# synthpanel MCP server

Give your AI coding assistant tool-call access to synthetic focus groups. Run panels, manage persona packs, and query saved results — straight from chat.

Uses the [Model Context Protocol](https://modelcontextprotocol.io/) over stdio transport. Defaults to the `haiku` model for cheap, fast iterative use.

$ pip install "synthpanel[mcp]"

The `[mcp]` extra pulls in the MCP Python SDK required to launch the server.

## Starting the server

The server communicates over stdin/stdout using JSON-RPC. It is launched by your editor, not by you — but you can sanity-check the binary:

$ synthpanel mcp-serve

## Editor configuration

Every editor uses the same underlying config shape — command, args, and environment. Pick yours:

Claude Code recommended

Easiest: add via the CLI. Works at user scope (all projects) or project scope (this repo only).

```
# user scope — available everywhere
claude mcp add --scope user synth_panel -- synthpanel mcp-serve

# or project scope — checked-in .mcp.json for this repo
claude mcp add --scope project synth_panel -- synthpanel mcp-serve
```

Or edit `.mcp.json` in your project root directly:

```
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

Plugin alternative: `/plugin install synthpanel` adds the `/focus-group` skill alongside the MCP server.

Cursor

Project scope: `.cursor/mcp.json`. User scope: `~/.cursor/mcp.json`.

```
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

Windsurf

Edit `~/.codeium/windsurf/mcp_config.json` (or use the Cascade → MCP settings panel).

```
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

Zed

Zed calls them *context servers*. Open `settings.json` (cmd-,) and merge this block:

```
{
  "context_servers": {
    "synth_panel": {
      "command": {
        "path": "synthpanel",
        "args": ["mcp-serve"],
        "env": { "ANTHROPIC_API_KEY": "sk-..." }
      }
    }
  }
}
```

Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or `%APPDATA%\Claude\claude_desktop_config.json` on Windows, then restart the app.

```
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

If `synthpanel` is not on the desktop app's PATH, replace the `command` value with an absolute path (e.g. `/Users/you/.venv/bin/synthpanel`).

Set whichever provider env var you want to use — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, or `GOOGLE_API_KEY`. Multiple keys can be set simultaneously; the model string picks the provider.

## Tools (12)

### Research tools

| Tool | Description |
|---|---|
| `run_prompt` | Send a single prompt to an LLM. No personas required — the simplest tool for a quick research question. |
| `run_panel` | Run a full synthetic focus group. Each persona answers all questions independently in parallel, followed by synthesis. Accepts inline `questions`, an inline `instrument` dict (v1/v2/v3), or an `instrument_pack` name. |
| `run_quick_poll` | Single-question poll across personas. A simplified `run_panel` for one question with synthesis. |
| `extend_panel` | Append a single ad-hoc round to a saved panel result. Reuses each panelist's saved session for conversational context. *Not* a re-entry into the v3 DAG — use for human-in-the-loop follow-ups. |

### Persona pack management

| Tool | Description |
|---|---|
| `list_persona_packs` | List all saved persona packs (bundled + user-saved). Returns ID, name, and persona count. |
| `get_persona_pack` | Get a specific persona pack by ID — full definitions. |
| `save_persona_pack` | Save a persona pack for reuse. Validates persona data before writing to disk. |

### Instrument pack management

| Tool | Description |
|---|---|
| `list_instrument_packs` | List installed instrument packs (bundled + user-saved) with manifest metadata. |
| `get_instrument_pack` | Load an installed instrument pack by name — full YAML body. |
| `save_instrument_pack` | Install an instrument pack. Validates via the parser before writing to disk. |

### Result management

| Tool | Description |
|---|---|
| `list_panel_results` | List saved panel results — ID, date, model, counts. |
| `get_panel_result` | Get a specific panel result by ID — full result with all rounds and synthesis. |

## Resources (4 URI patterns)

MCP resources let agents read data without invoking a tool.

| URI pattern | Description |
|---|---|
| `persona-pack://{pack_id}` | A specific persona pack. |
| `persona-pack://` | List all persona packs. |
| `panel-result://{result_id}` | A specific panel result. |
| `panel-result://` | List all panel results. |

## Prompt templates (3)

Pre-built research workflows agents can use as starting points.

| Prompt | Parameters | Description |
|---|---|---|
| `focus_group` | `topic` (required), `num_personas` (default 5), `follow_up` (default true) | Structured focus group discussion prompt for a given topic. |
| `name_test` | `names` (required, comma-separated), `context` (optional) | Test product or feature name options with diverse perspectives. |
| `concept_test` | `concept` (required), `target_audience` (optional) | Test a concept or idea with targeted personas. |

## Response shape

All panel runs (`run_panel`, `run_quick_poll`, `extend_panel`) return a uniform response:

```
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

- `rounds` — per-round results with panelist responses and per-round synthesis.

- `path` — routing decisions that fired (v3 branching instruments).

- `terminal_round` — the round whose synthesis fed final synthesis.

- `warnings` — parser or runtime warnings.

- `results` — back-compat flat array mirroring the terminal round's panelist results.

For v1/v2 instruments and raw `questions` input, `path` is empty or linear and `warnings` is typically empty — the shape is uniform across versions.

## Data storage

Panel results, persona packs, and instrument packs are stored under `~/.synthpanel/` (configurable via `SYNTH_PANEL_DATA_DIR`):

```
~/.synthpanel/
├── persona_packs/          # Saved persona packs (YAML)
├── packs/instruments/      # Installed instrument packs (YAML)
└── results/                # Panel results (JSON) + session data
```

## Next steps

### [Source docs](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md)

→

The canonical `docs/mcp.md` in the GitHub repo.

### [Report an issue](https://github.com/DataViking-Tech/SynthPanel)

→

Open an issue on GitHub if a tool misbehaves or an editor config needs a tweak.
