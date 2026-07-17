# Production Operations

How synthpanel behaves when things go wrong, cost money, or need to be
reproduced — the operational contract for running panels in CI, agent
pipelines, and other unattended environments.

Every claim on this page cites the mechanism that implements it. If the code
and this page ever disagree, the code wins — file an issue.

> **Audience:** operators wiring `synthpanel panel run` or the MCP server
> into automation. For the response *schema* (envelope fields, `flags[]`,
> the typed error object) see [docs/response-contract.md](response-contract.md);
> this page covers runtime behavior: failure, money, scale, determinism,
> secrets, and logs.

## Error contracts

### Typed error envelopes (MCP)

MCP tool failures return a typed error envelope, not free text
(`src/synth_panel/mcp/server.py`, mirrored from the v1.0.0 schema in
`src/synth_panel/schemas/v1.0.0.json`):

```json
{
  "error_code": "INVALID_TOOL_ARG",
  "message": "...",
  "field_path": "...",
  "schema_version": "1.0.0",
  "retry_safe": false
}
```

The closed `error_code` set and `retry_safe` semantics are documented in
[docs/response-contract.md](response-contract.md#typed-error-envelope). The
operationally load-bearing rule: **`retry_safe: true` only for
`MODEL_TIMEOUT` / `PANEL_TIMEOUT` and pre-exhaustion `SCHEMA_DRIFT`.**
Everything else is terminal — fix the request instead of retrying.

Request validation runs *before* any model call (cheap reject, zero token
spend); response validation runs before the artifact leaves the server
(closed `flags[]` enum, `additionalProperties: false`).

### Timeouts

The MCP server enforces a per-panelist time budget of 30 seconds
(`PANELIST_TIMEOUT` in `src/synth_panel/_runners.py`), scaled by panel
shape: `30s x personas x rounds` for multi-round runs, `30s x personas x
(1 + variants)` for single-round (`_panel_timeout_envelope` in
`src/synth_panel/mcp/server.py`). Exceeding it returns a `PANEL_TIMEOUT`
envelope carrying the computed `timeout_seconds` so the caller can see the
budget it blew. Single-completion provider timeouts surface as
`MODEL_TIMEOUT`. Both are `retry_safe: true`.

### Total failure is loud, not shaped like success

When *every* panelist fails — wholesale exception, all responses errored, or
zero tokens with nothing usable — the run raises `PanelTotalFailureError`
(`detect_total_failure` / `run_panel_sync` in `src/synth_panel/_runners.py`)
instead of returning a normally-shaped "panel complete" result. The error
carries a structured diagnostic: the models exercised and up to 3 sample
per-persona error strings, so the banner names the actual upstream failure
(e.g. the bad model id and the provider's HTTP 400).

- **CLI:** the run is marked invalid, the JSON envelope carries
  `total_failure` (the diagnostic) plus a classified `abort_reason`
  (`_classify_total_failure_abort_reason` in
  `src/synth_panel/cli/commands.py` — e.g. rate-limit exhaustion is tagged
  distinctly so you know to raise `--rate-limit-rps` / retries instead of
  chasing a model bug), and the process exits `2`.
- **MCP:** `run_panel`, `run_quick_poll`, and `extend_panel` all serialize
  the same shape (`_total_failure_envelope` in
  `src/synth_panel/mcp/server.py`): `{"error": ..., "run_invalid": true,
  "total_failure": {...}}`.

### Exit codes and abort discipline (CLI)

`synthpanel panel run` exit codes (end of `handle_panel_run`,
`src/synth_panel/cli/commands.py`):

| Exit | Meaning |
|---|---|
| `0` | Run completed and is valid. |
| `1` | Startup/config error (bad flags, missing files, refused flag combos). |
| `2` | Run completed but **invalid** (`run_invalid: true`): failure rate over threshold, total failure, cost gate tripped, SIGINT, missing-input refusals, or synthesis failure. |
| `3` | `--strict` violation: any panelist-question error at all. |

**Every abort path still emits valid partial JSON.** With
`--output-format json`, a cost-gate halt, SIGINT, or total failure does not
produce truncated garbage — it produces a complete, parseable envelope with
`run_invalid: true` and a machine-readable `abort_reason` (`cost_exceeded`,
`sigint`, or a total-failure classification) plus `halted_at_panelist`
where applicable. CI can gate on these fields without scraping stderr.

## Partial failure

### K of N panelists fail

A panel does not fail because one panelist did. Panelists run in parallel
(`run_panel_parallel`, `src/synth_panel/orchestrator.py`); per-panelist and
per-question errors are caught and recorded, not raised:

- Each panelist row in the output carries an `error` field (`null` when
  clean); individual responses that errored are flagged `error: true`
  in-place, so you can see exactly which persona/question pairs are missing.
- The envelope's `failure_stats` reports total pairs, errored pairs, and the
  failure rate.
- The run is declared invalid only when the errored fraction of
  panelist-question pairs exceeds `--failure-threshold` (default `0.5`;
  `src/synth_panel/cli/parser.py`). When exceeded, synthesis is
  auto-disabled (don't synthesize a majority-broken panel) and the run
  exits `2`. `--strict` tightens this to zero tolerance (exit `3`).
- `--question-failure-budget N|0.X` contains a *single bad question*
  (broken schema, format the model rejects): once that question's failures
  cross the threshold it is disabled mid-run, later panelists skip it, and
  the envelope reports `disabled_questions` with per-question counts
  (`src/synth_panel/question_budget.py`).

### Synthesis failure and recovery

Synthesis failures never masquerade as a synthesized result: the envelope
carries a top-level `synthesis_error` payload and the run is marked
invalid (`src/synth_panel/_runners.py`, `src/synth_panel/cli/commands.py`).
The panelist data is still saved — synthesis is recoverable **post-hoc
without re-running the panel**:

```bash
synthpanel panel synthesize <result-id> --synthesis-model sonnet
```

(`panel synthesize`, `src/synth_panel/cli/parser.py`) re-synthesizes a
saved result with a different model or prompt, so a synthesis-model outage
costs you one cheap retry, not the whole panel spend.

Two overflow safeguards on the synthesis step itself:

- Context overflow is detected *before* the call
  (`detect_synthesis_context_overflow`, `src/synth_panel/synthesis.py`);
  `--synthesis-auto-escalate` opts into escalating an overflowing
  synthesis to a large-context model (with a visible warning), with
  map-reduce sub-chunking as the fallback plan.
- The structured-output engine's 3-strike retry escalates its final strike
  **within the run's provider family** (`_escalation_model_for`,
  `src/synth_panel/structured/output.py`): `gemini-*` → `gemini-2.5-pro`,
  `grok-*` → `grok-4`, OpenAI-compat cheap tiers → the non-cheap sibling,
  Anthropic → `sonnet`. A Gemini-only environment never gets asked for an
  Anthropic key mid-run; when no stronger same-family model is known, the
  final strike reuses the original model and terminal failure flows to the
  loud `synthesis_error` path.

## Cost controls

- **`--max-cost USD`** — hard ceiling on projected total spend. After each
  panelist completes, `running_cost / completed_n * total_n` is compared
  against the ceiling; if the projection exceeds it the run halts
  gracefully, cancels pending panelists, and emits a valid partial JSON
  envelope with `run_invalid: true`, `cost_exceeded: true`,
  `halted_at_panelist`, and a `cost_gate` snapshot. Exit code `2`.
  (`CostGate` in `src/synth_panel/cost.py`; wiring in
  `src/synth_panel/cli/commands.py`.)
- **MCP `max_cost` (GH#576)** — the same gate on the MCP surface: the
  `run_panel`, `run_quick_poll`, and `extend_panel` tools accept a
  `max_cost` (USD) argument wired into the identical `CostGate` machinery
  with the same soft-halt semantics. On a trip, the tool response is a
  valid partial envelope with `run_invalid: true`, `cost_exceeded: true`,
  `abort_reason: "cost_exceeded"`, `halted_at_panelist`, the `cost_gate`
  snapshot, and an agent-legible `resume` block (persisted partial
  `result_id`, completed panelists, remaining personas). Synthesis is
  skipped on the partial. BYOK inline-`questions` runs only — sampling
  mode, `models` ensembles, `variants`, and instrument inputs refuse
  `max_cost` with a typed `INVALID_TOOL_ARG` (parity with the CLI's
  multi-round refusal). See [docs/mcp.md](mcp.md#max_cost-hard-spend-ceiling).
- **Per-turn telemetry** — token usage is tracked per turn in four buckets
  (input / output / cache-write / cache-read; `TokenUsage` and
  `UsageTracker` in `src/synth_panel/cost.py`). Every panelist row and the
  envelope total carry `usage` and `cost`; multi-model runs are priced
  per-model at each provider's actual rate (`aggregate_per_model`), and
  models missing from the pricing table produce explicit fallback warnings
  instead of silently wrong totals (`build_cost_fallback_warnings`).
- **`--dry-run`** — prints the fully substituted prompts, persona/question
  counts, LLM call count, a token estimate, and an **estimated cost** from
  the local pricing table, then exits without any provider call
  (`_emit_dry_run_preview`, `src/synth_panel/cli/commands.py`). It also
  fast-fails config errors a real run would hit (e.g. image attachments
  routed to a text-only model).
- **`synthpanel cost summary`** — post-hoc spend reporting across saved
  runs, grouped by model or run, filterable by date
  (`src/synth_panel/cost_summary.py`).
- MCP mode defaults to `haiku` specifically to keep iterative agent use
  cheap; CLI defaults to `sonnet`.

## Resilience and scale

- **Checkpointing + resume** — `--checkpoint-dir` opts into per-run on-disk
  snapshots (`<dir>/<run-id>/state.json`, atomic writes, per-directory lock
  file), flushed every `--checkpoint-every` panelists (default 25;
  `DEFAULT_CHECKPOINT_EVERY` in `src/synth_panel/checkpoint.py`).
  `--resume <run-id>` skips completed panelists, replays the rest, and
  merges into one result; `--personas`/`--instrument` may be omitted (they
  are recovered from the checkpoint's saved CLI args).
- **Config drift refusal** — resume refuses to continue when the current
  config hash does not match the checkpointed one
  (`fingerprint_config`, `src/synth_panel/checkpoint.py`). `--allow-drift`
  downgrades that to a warning and is explicitly documented as
  statistically inconsistent. `--force-overwrite` is required to clobber an
  existing run id's state.
- **Signal-safe aborts** — when checkpointing is active, SIGINT/SIGTERM
  handlers flush a final checkpoint and mark the run aborted
  (`install_signal_handlers` / `mark_aborted`,
  `src/synth_panel/checkpoint.py`); the CLI still emits the valid partial
  envelope with `abort_reason: "sigint"`. An interrupted 500-persona run
  resumes where it stopped instead of restarting from zero.
- **Concurrency and rate limiting** — by default the orchestrator runs one
  worker per panelist (`run_panel_parallel`,
  `src/synth_panel/orchestrator.py`). `--max-concurrent N` caps in-flight
  LLM requests at the client layer (all providers on the client);
  `--rate-limit-rps` adds a token-bucket requests-per-second cap on top
  (fractional values accepted, e.g. `0.5` = one request per two seconds).
- **Convergence auto-stop** — `--convergence-check-every N` computes a
  rolling Jensen-Shannon divergence for every bounded question;
  `--auto-stop` halts the panel once every tracked question's rolling JSD
  stays below `--convergence-eps` (default 0.02) for `--convergence-m`
  consecutive checks (default 3) with at least `--convergence-min-n`
  panelists (default 50) — stop paying for panelists after the
  distribution has stabilized. See [docs/convergence.md](convergence.md).
- **Caps** — panels are bounded at 100 personas / 50 questions
  (`MAX_PERSONAS` / `MAX_QUESTIONS`, `src/synth_panel/_runners.py`),
  enforced on the MCP and SDK surfaces.

## Determinism and reproducibility

- **`--seed`** is forwarded to providers that support it (OpenAI, Gemini,
  xAI, OpenRouter). **Anthropic has no seed parameter** — the client warns
  once per provider and proceeds without determinism
  (`src/synth_panel/llm/client.py`,
  `src/synth_panel/llm/providers/anthropic.py`); use `--temperature 0` for
  closer-to-deterministic Claude output. Treat seeds as best-effort
  bias-reduction, not a bit-exactness guarantee.
- **`--resume`** replays a previously-cached run rather than re-sampling —
  the reproducibility tool for "give me the same panel again."
- **Saved-result provenance** — every saved result's metadata embeds the
  resolved models, generation params, synthpanel + Python versions, timing,
  cost, and a SHA-256 `config_hash` over the resolved config
  (`build_config_hash`, `src/synth_panel/metadata.py`). Template `--var`
  values are folded in as one-way hashes (`template_vars_fingerprint`) so
  runs differing only in substitutions don't collide, without persisting
  potentially sensitive raw values.

See [docs/reproducibility.md](reproducibility.md) for the full statement.

## Credentials

- **Precedence: environment variable first, then the on-disk store**
  (`get_credential`, `src/synth_panel/credentials.py`). Nothing else is
  consulted.
- `synthpanel login` persists a key to
  `~/.config/synthpanel/credentials.json` (override via
  `SYNTHPANEL_CREDENTIALS_PATH` / `XDG_CONFIG_HOME`), written atomically
  with file mode `0600`, parent directory `0700`, and a SHA-256 integrity
  sidecar (`save_credential`, `src/synth_panel/credentials.py`).
- **MCP configs contain no keys by default**: `synthpanel mcp install`
  writes an entry with no `env` block unless you explicitly pass
  `--env KEY=...`, and writes user-scoped config files with mode `0600`.
  Sampling-capable MCP hosts need no key at all (the host's own model does
  the work). See [docs/mcp.md](mcp.md).

## Observability

- **Structured stderr logs** — `%(asctime)s %(name)s %(levelname)s
  %(message)s` on the `synth_panel` logger namespace
  (`src/synth_panel/logging_config.py`). Level via `--verbose`,
  `SYNTHPANEL_LOG_LEVEL`, or `--debug-all` (which also un-caps chatty HTTP/
  SDK loggers that are otherwise pinned to WARNING).
- **Clean stream separation** — with `--output-format json`, stdout carries
  exactly one JSON document; human-facing progress, hints, and warnings go
  to stderr. Pipe stdout to `jq` without filtering.
- **Stable handles in the envelope** — `--save` runs put `result_id` and
  `saved_path` in the JSON output; checkpointed runs add `run_id`, so
  agents get follow-up handles from stdout instead of scraping stderr.
- **Telemetry in-band** — `failure_stats`, `missing_input_stats`,
  `run_invalid`, `abort_reason`, `cost_gate`, `question_failure_budget`,
  `convergence`, and `warnings[]` all live in the result envelope
  (`src/synth_panel/cli/commands.py`), and the convergence stream can be
  tee'd as JSON lines via `--convergence-log PATH` for live dashboards.

## What does not exist yet

Kept here so this page stays trustworthy:

- No automatic retry queue for failed panelists within a run — failures are
  recorded per-row; re-running failed panelists is `--resume`'s job only
  for interrupted (not errored) panelists.
- No server-side/API deployment mode — the MCP server is stdio, spawned by
  the host; there is no hosted multi-tenant endpoint, auth layer, or SLA.
- Seeds are provider-best-effort (see above); there is no cross-provider
  bit-exact replay.
- `--max-cost`, checkpoint/`--resume`, convergence `--auto-stop`, and
  `--question-failure-budget` apply to single-round runs; multi-round
  (branching) instruments refuse these flags loudly up front rather than
  degrading silently (`_multi_round_flag_errors`,
  `src/synth_panel/cli/commands.py`). The MCP `max_cost` argument mirrors
  the same matrix: instrument inputs (which always dispatch through the
  multi-round engine on the MCP surface), ensembles, variants, and
  sampling mode refuse it with a typed `INVALID_TOOL_ARG`.
