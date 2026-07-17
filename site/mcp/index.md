# MCP Server — synthpanel · Run synthetic focus groups from your AI editor

[DataViking](https://dataviking.tech)

MCP Server · stdio

# synthpanel MCP server

Give your AI coding assistant tool-call access to synthetic focus groups. Run panels, manage persona packs, and query saved results — straight from chat.

Uses the [Model Context Protocol](https://modelcontextprotocol.io/) over stdio transport. Defaults to the `haiku` model for cheap, fast iterative use.

Running this unattended? The full operational contract — typed errors and retry semantics, cost gates, checkpoint/resume, determinism per provider, credential handling, logs — is one page with code citations: [docs/production-operations.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/production-operations.md). Condensed on this page: [agent patterns](#agent-patterns), [ops runbook](#production), [determinism](#determinism), [observability](#observability).

$ pip install "synthpanel[mcp]"

The `[mcp]` extra pulls in the MCP Python SDK required to launch the server.

## Starting the server

The server communicates over stdin/stdout using JSON-RPC. It is launched by your editor, not by you — but you can sanity-check the binary:

$ synthpanel mcp-serve

## Editor configuration

Every editor uses the same underlying config shape — command, args, and environment. Pick yours:

Claude Code recommended

Easiest: let synthpanel write the entry for you. Works at user scope (all projects) or project scope (this repo only) and is idempotent — safe to re-run after upgrades.

```
# user scope — writes ~/.claude.json (mode 0600)
synthpanel mcp install

# project scope — writes ./.mcp.json (checked into the repo)
synthpanel mcp install --scope project

# bake credentials into the entry instead of relying on the host env
synthpanel mcp install --env ANTHROPIC_API_KEY=sk-...

# preview without writing, or remove the entry later
synthpanel mcp install --dry-run
synthpanel mcp install --uninstall
```

Or use Claude Code's native CLI, which writes to the same `~/.claude.json` file:

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

Project scope: `.cursor/mcp.json`. User scope: `~/.cursor/mcp.json`. The `synthpanel mcp install` CLI works here too — pass `--target` to point at the right file:

```
synthpanel mcp install --target ~/.cursor/mcp.json
```

Or hand-edit the JSON directly:

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

Hermes

Hermes uses a YAML config with an `mcp_servers` map and explicit timeout fields. Add this block to your Hermes config and restart the host:

```
mcp_servers:
  synthpanel:
    command: "synthpanel"
    args: ["mcp-serve"]
    timeout: 180
    connect_timeout: 60
    env:
      ANTHROPIC_API_KEY: "sk-..."
```

Or run it on demand via `uvx` without a global install:

```
mcp_servers:
  synthpanel:
    command: "uvx"
    args: ["--from", "synthpanel[mcp]", "synthpanel", "mcp-serve"]
    timeout: 180
    connect_timeout: 60
    env:
      ANTHROPIC_API_KEY: "sk-..."
```

The 180s `timeout` covers a full panel run; the 60s `connect_timeout` gives the subprocess room to import the MCP SDK on first launch.

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

## Agent integration patterns

Three patterns that hold up when an agent — not a human — is driving the tools:

Summary first, transcripts on demand `run_panel` and `run_quick_poll` default to `detail: "summary"`: synthesis, verdict, costs and warnings come back, but per-panelist transcripts are dropped (a full 100-persona panel can serialize to megabytes — that would flood the agent's context window). The envelope marks the omission (`results_omitted: true`) and points at the full copy via `transcript_uri`. When the agent actually needs quotes, it calls `get_panel_result` (defaults to `detail: "full"`) or reads the `panel-result://{result_id}` resource.

Structured polling — no prose parsing  Both run tools accept a `response_schema` argument (JSON Schema) that forces structured output at generation time — forced-choice, Likert, tagged themes, ranking — so downstream code branches on fields, not on regexes over prose. Pattern catalogue: [docs/structured-polling.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/structured-polling.md).

Budget-gated iteration  MCP mode defaults to `haiku` and caps panels at 100 personas × 50 questions. Every response carries in-band `total_cost` and per-panelist `usage` / `cost`, so the loop is: pilot with `run_quick_poll`, gate on the returned cost, then scale to a full `run_panel`. Note the hard mid-run spend ceiling (`--max-cost`) is a CLI flag — there is no `max_cost` tool argument on the MCP surface; unattended hard ceilings mean driving the CLI.

Full tool semantics: [docs/mcp.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md) · operational contract: [docs/production-operations.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/production-operations.md).

## Ops runbook

What actually happens when a panel run goes sideways. Failures return typed envelopes (`error_code`, `retry_safe`, `schema_version`) — retry only on `MODEL_TIMEOUT` / `PANEL_TIMEOUT` and pre-exhaustion `SCHEMA_DRIFT`; everything else is terminal by contract. Requests are validated before any model call, so bad arguments cost zero tokens.

| Failure | What happens |
|---|---|
| K of N panelists fail | Errors are recorded per persona and per question (`error` field per row), the panel continues, and `failure_stats` reports the rate. The run is invalid only past `--failure-threshold` (default 0.5) — then synthesis is auto-disabled and the CLI exits `2`. `--strict` = zero tolerance. |
| Synthesis fails | Never shaped like success: the envelope carries `synthesis_error` and the run is marked invalid — but panelist data is saved. Recover post-hoc without re-running the panel: `synthpanel panel synthesize <result-id> --synthesis-model sonnet`. A synthesis-model outage costs one cheap retry, not the panel spend. |
| Rate limits | `--max-concurrent` caps in-flight requests; `--rate-limit-rps` adds a token-bucket cap (fractional values accepted). Rate-limit exhaustion is classified distinctly in `abort_reason` — the signal is "raise the throttle," not "chase a model bug." |
| Cost ceiling hit | `--max-cost` halts gracefully mid-run: pending panelists cancelled, valid partial JSON with `run_invalid: true`, `cost_exceeded: true`, `halted_at_panelist`. Exit `2`. |
| SIGINT / SIGTERM | With checkpointing active, a final checkpoint is flushed and the run marked aborted; stdout still emits a complete, parseable envelope with `abort_reason: "sigint"`. Resume picks up where it stopped. |
| Every panelist fails | Loud, not shaped like success: a structured total-failure diagnostic names the upstream cause (bad model id, provider 400). MCP serializes `{"error", "run_invalid": true, "total_failure"}`; the CLI exits `2`. |

### CLI exit codes

| Exit | Meaning |
|---|---|
| `0` | Run completed and is valid. |
| `1` | Startup/config error (bad flags, missing files, refused flag combos). |
| `2` | Run completed but **invalid** (`run_invalid: true`): failure rate over threshold, total failure, cost gate tripped, SIGINT, or synthesis failure. The envelope is still valid JSON — CI gates on `abort_reason`, not stderr. |
| `3` | `--strict` violation: any panelist-question error at all. |

### Checkpoint + resume

```
# snapshot every 25 panelists (atomic, per-run lock)
synthpanel panel run ... --checkpoint-dir .synthpanel-ckpt

# interrupted? resume skips completed panelists, merges into one result
synthpanel panel run --resume <run-id> --checkpoint-dir .synthpanel-ckpt
```

Resume refuses to continue if the config hash drifted from the checkpointed one (`--allow-drift` downgrades to a warning). Credentials stay out of configs: `synthpanel mcp install` writes no API key unless you pass `--env`; keys come from env vars or the `synthpanel login` store (mode `0600`).

Full contract with code citations: [docs/production-operations.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/production-operations.md) — error and retry semantics, partial-failure behavior, cost gates, timeout budgets, convergence auto-stop, and credential handling.

## Determinism contract

What is and isn’t guaranteed when you re-run a panel:

- `--seed` is forwarded to providers that support it: OpenAI, Gemini, xAI, OpenRouter. **Anthropic has no seed parameter** — the client warns once per provider and proceeds; use `--temperature 0` for closer-to-deterministic Claude output.

- `--resume` replays a previously-cached run instead of re-sampling — the reproducibility tool for “give me the same panel again.”

-  Otherwise, same-config reruns are **equivalent, not identical**: treat seeds as best-effort bias reduction, not a bit-exactness guarantee. There is no cross-provider bit-exact replay.

-  Every saved result stamps provenance in metadata: resolved models, generation params, synthpanel + Python versions, and a SHA-256 `config_hash` — template `--var` values are folded in as one-way hashes, so substitution-only differences don’t collide and raw values are never persisted.

Full contract: [production-operations.md#determinism](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/production-operations.md#determinism-and-reproducibility) · [docs/reproducibility.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/reproducibility.md).

## Observability contract

- **stderr** — structured logs (`%(asctime)s %(name)s %(levelname)s %(message)s`) on the `synth_panel` logger namespace; level via `--verbose`, `SYNTHPANEL_LOG_LEVEL`, or `--debug-all`.

- **stdout stays JSON-pure** — with `--output-format json`, stdout carries exactly one JSON document; progress, hints, and warnings go to stderr. Pipe to `jq` without filtering.

- **Handles in-band** — `--save` puts `result_id` and `saved_path` in the envelope; checkpointed runs add `run_id`. Agents get follow-up handles from stdout, never by scraping stderr.

- **Telemetry in-band** — `failure_stats`, `run_invalid`, `abort_reason`, `cost_gate`, `convergence`, and `warnings[]` live in the result envelope; token usage is tracked per turn in four buckets (input / output / cache-write / cache-read) with cost on every panelist row and the envelope total.

- **Post-hoc spend** — `synthpanel cost summary` reports spend across saved runs, grouped by model or run, filterable by date.

Full contract: [production-operations.md#observability](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/production-operations.md#observability).

## Data storage

Panel results, persona packs, and instrument packs are stored under `~/.synthpanel/` (configurable via `SYNTH_PANEL_DATA_DIR`):

```
~/.synthpanel/
├── persona_packs/          # Saved persona packs (YAML)
├── packs/instruments/      # Installed instrument packs (YAML)
└── results/                # Panel results (JSON) + session data
```

## Troubleshooting

`command not found: synthpanel`  The MCP host can't find the binary on its `PATH`. MCP subprocesses inherit the host's environment, not your shell's. Fix by installing globally (`pip install "synthpanel[mcp]"`), pointing `command` at an absolute path (e.g. `/Users/you/.venv/bin/synthpanel`), or running through `uvx` (`"command": "uvx", "args": ["--from", "synthpanel[mcp]", "synthpanel", "mcp-serve"]`). Claude Desktop on macOS is especially strict — use an absolute path or `uvx`.

Server starts but no tools appear  The MCP server needs the `[mcp]` extra: `pip install "synthpanel[mcp]"`. After config edits or upgrades, fully quit and relaunch the host — tool lists are cached per server entry. Sanity-check by running `synthpanel mcp-serve` in a terminal; it should boot silently.

Missing API key  The subprocess only sees env vars from the MCP `env` block. A shell-profile export does not propagate. Set the key in the `env` block, run `synthpanel login` to seed the on-disk credential store, or omit the key and let MCP *sampling* borrow the host's LLM access (Claude Desktop, Claude Code, Cursor, Windsurf).

Timeouts on long panels  A 5-persona × 3-question BYOK panel takes 30–90 seconds; cross-provider ensembles can take 2–5 minutes. Raise the host's per-tool timeout (Hermes: `timeout: 300`). For exploratory work, prefer `run_quick_poll` over `run_panel`.

`MISSING_DECISION` errors `run_panel`, `run_quick_poll`, and `extend_panel` require a `decision_being_informed` string (12–280 chars). This is a frozen v1.0.0 contract. `run_prompt` does not require it.

Full troubleshooting reference and links into the response contract: [docs/mcp.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md#troubleshooting).

## Next steps

### [Source docs](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md)

→

The canonical `docs/mcp.md` in the GitHub repo.

### [Report an issue](https://github.com/DataViking-Tech/SynthPanel)

→

Open an issue on GitHub if a tool misbehaves or an editor config needs a tweak.
