# MCP Server Reference

synthpanel ships an [MCP](https://modelcontextprotocol.io/) server so AI agents can run synthetic focus groups as tool calls. The server uses stdio transport and defaults to the `haiku` model for cheap, fast iterative use.

## Starting the Server

```bash
synthpanel mcp-serve
```

The server communicates over stdin/stdout using JSON-RPC (the MCP protocol). It is designed to be launched by an MCP-aware editor or agent framework.

## Editor Configuration

### Claude Code / Cursor / Windsurf

Add to your MCP config (e.g., `.claude/mcp.json`, `.cursor/mcp.json`):

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

Set the environment variable for whichever LLM provider you want to use. See the main [README](../README.md#llm-provider-support) for the full provider table.

Prefer a zero-config first run? Skip the `env` block entirely — see
[Sampling Mode](#sampling-mode) below.

### Claude Code Plugin

```
/plugin install synthpanel
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

## Sampling Mode

MCP has a spec-level feature called
[**sampling**](https://modelcontextprotocol.io/specification/2025-03-26/client/sampling)
where the server can ask the invoking client to run an LLM completion
on its behalf. synthpanel uses this to deliver a zero-configuration
first-run UX: if you haven't set a provider API key and your client
advertises `sampling`, the `run_prompt` and `run_quick_poll` tools
borrow the client's own LLM access instead of failing.

### When sampling kicks in

| Client supports sampling? | Provider key set? | `use_sampling` flag | Mode |
|---------------------------|-------------------|---------------------|------|
| yes                       | no                | (auto)              | **sampling** |
| yes                       | yes               | (auto)              | BYOK |
| yes                       | (any)             | `true`              | **sampling** |
| yes                       | (any)             | `false`             | BYOK |
| no                        | no                | (auto)              | **friendly error** |
| no                        | yes               | (auto)              | BYOK |
| no                        | (any)             | `true`              | **error** — client lacks sampling |

The auto logic only routes to sampling when it's *necessary* for a
first-run experience (no keys set) — existing BYOK users see no change
in behaviour.

### Tradeoffs

Sampling mode is intentionally less capable than BYOK:

- **One provider.** The host agent picks the model (Claude Desktop →
  Claude; other clients may route through whichever provider they have
  configured). Cross-provider ensembles require BYOK.
- **No cost accounting.** Token usage is charged to the host agent's
  subscription; synthpanel returns `"usage": null` and `"cost": null`.
- **Capped panel size.** `run_quick_poll` is limited to 3 personas in
  sampling mode to protect the host agent's context window. Larger
  runs require BYOK.
- **No structured output extraction.** Free-text only.

These limits keep sampling mode focused on what it's for: a frictionless
first invocation that produces real results, not a replacement for the
research-grade BYOK path.

### Response fields

Sampling responses include two extra fields:

- `"mode"` — `"sampling"` or `"byok"` so downstream tooling can
  condition on the execution mode.
- `"hint"` — a one-line hint on the first sampling run explaining how
  to upgrade to BYOK. Safe to surface to end users.

### Opting in explicitly

Pass `use_sampling=True` (or `use_sampling=False`) to either tool to
override the automatic decision — useful when you have keys configured
but want a quick sampling-mode preview, or when you want to force BYOK
inside a sampling-capable client for reproducibility.

### Tools that support sampling

- `run_prompt` — no persona/question caps, fully supported.
- `run_quick_poll` — up to `SAMPLING_MAX_PERSONAS` (3) personas.

The remaining tools (`run_panel`, `extend_panel`, pack/result
management) always use BYOK — heavier workflows benefit from direct
provider access and structured outputs.

## Data Storage

Panel results, persona packs, and instrument packs are stored under `~/.synthpanel/` (configurable via `SYNTH_PANEL_DATA_DIR`):

```
~/.synthpanel/
├── persona_packs/          # Saved persona packs (YAML)
├── packs/instruments/      # Installed instrument packs (YAML)
└── results/                # Panel results (JSON) + session data
```
