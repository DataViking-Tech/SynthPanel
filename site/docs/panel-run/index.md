# panel run Reference — synthpanel · Run synthetic focus groups with any LLM

 Docs · panel run

# panel run reference

Full reference for `synthpanel panel run`. Covers the advanced flags for multi-model panels, persona variants, structured extraction, synthesis tuning, convergence auto-stop, checkpointing, and SynthBench integration.

## Multi-model — `--models`, `--blend`

`--models` has two distinct shapes, selected by whether the spec contains a colon (`:`). It is mutually exclusive with `--model`.

### Weighted per-persona assignment

A spec with colons splits the panel across models in the given ratio. Weights are normalized — `a:2,b:3` and `a:0.4,b:0.6` behave identically. Per-persona `model:` fields in the YAML always win over the `--models` assignment.

```
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml \
  --models 'haiku:0.5,gemini-2.5-flash:0.5'
```

With 6 personas and the spec above, 3 answer on haiku and 3 on Gemini. 7 personas → 3 + 4 (the last model absorbs the remainder). The assignment is fully deterministic and printed to stderr before the run:

```
Model assignment:
  Maya Chen        → haiku
  Derek Washington → haiku
  Priya Patel      → haiku
  Sam Torres       → gemini-2.5-flash
  Julia Hoffman    → gemini-2.5-flash
  Omar Rashid      → gemini-2.5-flash
Totals: haiku=3, gemini-2.5-flash=3
```

### Ensemble mode

A spec without colons runs the full panel once per model. Combine with `--blend` to weight-average the response distributions across models for each question.

```
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument pricing-discovery \
  --models 'haiku,sonnet,gemini-2.5-flash' \
  --blend
```

With 6 personas and 3 models, this runs 18 sessions total. `--blend` then computes weighted-average distributions using the weights in the spec (equal weights when no `:` is given). Use `haiku:0.5,sonnet:0.3,gemini:0.2` (with colons, in ensemble mode with `--blend`) to apply custom blend weights.

| Flag | Default | Description |
|---|---|---|
| `--models SPEC` | — | Multi-model spec. `a:w,b:w` = weighted split; `a,b,c` = ensemble (full panel per model). |
| `--blend` | off | Weight-average distributions across ensemble models. Requires `--models`. |

## Persona variants — `--variants`, `--personas-merge`

### --variants N

Generate N LLM-perturbed variants per persona before running the panel. The original personas are replaced by `N × M` variants (M = number of original personas). Each variant perturbs one axis — trait swap, mood context, demographic shift, or background rephrase — via a single LLM call per variant.

```
synthpanel panel run \
  --personas small-panel.yaml \    # 5 personas
  --instrument survey.yaml \
  --variants 4                     # → 20 total panelists
```

Useful for stress-testing whether results are stable across plausible persona perturbations, or for expanding a small hand-crafted panel into a larger synthetic one.

### --personas-merge

Append additional YAML files to `--personas`. Repeatable. Files are merged in order; later entries override earlier ones on name collision (controlled by `--personas-merge-on-collision`).

```
synthpanel panel run \
  --personas base-panel.yaml \
  --personas-merge extra-personas.yaml \
  --personas-merge regional-overrides.yaml \
  --instrument survey.yaml
```

| Flag | Default | Description |
|---|---|---|
| `--variants N` | — | Generate N LLM-perturbed variants per persona. Panel size becomes N × original count. |
| `--personas-merge PATH` | — | Merge additional persona YAML into the panel. Repeatable. |
| `--personas-merge-on-collision` | `dedup` | `dedup` (later file wins, warning emitted), `error` (abort on any collision). |

## Structured extraction — `--extract-schema`

Unlike `--schema` (which forces structured-only output and replaces free text), `--extract-schema` preserves the full free-text response and adds a second LLM call that extracts structured data into an `extraction` key alongside the raw `response`.

```
synthpanel panel run \
  --personas panel.yaml \
  --instrument feedback-survey.yaml \
  --extract-schema '{"type":"object","properties":{"sentiment":{"type":"string","enum":["positive","neutral","negative"]},"themes":{"type":"array","items":{"type":"string"}}}}'
```

The value can be a JSON file path (`extract.json`) or an inline JSON string. The extraction runs after the panelist answers and is stored under `extraction` in the result JSON. Use this when you need both the qualitative narrative and a structured signal you can aggregate (e.g. sentiment counts, theme frequency).

| Flag | Description |
|---|---|
| `--extract-schema SCHEMA` | JSON Schema for post-hoc extraction. Preserves full free-text; adds structured `extraction` key. File path or inline JSON. |
| `--schema SCHEMA` | JSON Schema for structured-only output. Replaces free-text responses. Use when you want only structured data. |

## Synthesis strategy — `--synthesis-strategy`, `--synthesis-auto-escalate`

The synthesis step aggregates all panelist responses into a final summary. For small panels the default `auto` strategy concatenates everything into one LLM call. For large panels (n ≥ 50) it automatically switches to `map-reduce`.

| Strategy | How it works | When to use |
|---|---|---|
| `single` | All responses concatenated into one call. | Small panels (n<50). Cheapest and most coherent. |
| `map-reduce` | One summary call per question in parallel, then one reduce call across summaries. | Large panels where responses overflow the synthesis model's context. |
| `auto` (default) | Pre-flight token estimate picks `single` or `map-reduce`. | Most runs. Falls back to `single` whenever estimate fits context. |

### --synthesis-auto-escalate

In `map-reduce` mode, a single question whose responses overflow context normally partitions panelists into sub-batches for an inner reduce. With `--synthesis-auto-escalate`, instead of sub-batching, that question's map call is retried on a larger-context model (gemini-2.5-flash-lite, 1 M ctx) and a warning is emitted. Use this to preserve single-model semantics when sub-batch results are unacceptable.

| Flag | Default | Description |
|---|---|---|
| `--synthesis-strategy` | `auto` | Aggregation strategy: `single`, `map-reduce`, or `auto`. |
| `--synthesis-auto-escalate` | off | In map-reduce, retry overflowing question maps on a large-context model instead of sub-batching. |

## Rate limiting — `--rate-limit-rps`

By default the orchestrator fires one concurrent request per panelist (bounded by `--max-concurrent`). `--rate-limit-rps` adds a token-bucket that smooths bursts — useful when a provider enforces a requests-per-second limit on top of a concurrency cap.

```
synthpanel panel run \
  --personas large-panel.yaml \
  --instrument survey.yaml \
  --max-concurrent 20 \
  --rate-limit-rps 5.0          # at most 5 new requests/sec
```

Accepts fractional values: `0.5` means one request every two seconds. Works across all providers on the same client.

| Flag | Default | Description |
|---|---|---|
| `--max-concurrent N` | unbounded | Cap concurrent in-flight LLM requests across the panel. |
| `--rate-limit-rps RPS` | — | Token-bucket rate cap in requests per second. Accepts fractional values. |

## Checkpointing & resume — `--checkpoint-dir`, `--resume`

For long or expensive runs, checkpointing writes per-panelist progress to disk so you can resume after an interruption without re-running completed panelists.

### Starting a checkpointed run

```
synthpanel panel run \
  --personas panel.yaml \
  --instrument survey.yaml \
  --checkpoint-dir /tmp/runs \    # opts in; default: ~/.synthpanel/checkpoints
  --checkpoint-every 10           # flush every 10 completed panelists
```

The run id is printed to stderr. Each checkpoint is written to `<checkpoint-dir>/<run-id>/state.json`. Omitting `--checkpoint-dir` runs without snapshots.

### Resuming

```
synthpanel panel run --resume <run-id>
```

When `--personas` and `--instrument` are omitted they are recovered from the checkpoint's saved CLI args. The resume refuses to start if the current config (model, temperature, questions) does not match the checkpointed config — pass `--allow-drift` to downgrade this to a warning and continue (statistically inconsistent results).

| Flag | Default | Description |
|---|---|---|
| `--checkpoint-dir PATH` | `~/.synthpanel/checkpoints` | Directory for per-run snapshots. Setting this opts in to checkpointing. |
| `--checkpoint-every N` | 25 | Flush a checkpoint every N completed panelists. |
| `--resume RUN_ID` | — | Resume a checkpointed run. Skips already-completed panelists. |
| `--allow-drift` | off | With `--resume`: downgrade config-mismatch errors to warnings. |
| `--force-overwrite` | off | Replace existing checkpoint state for the same run id instead of refusing. |

## Convergence & auto-stop — `--convergence-*`

At large-n scales most of the token budget goes toward diminishing returns — once a response distribution has stabilized, additional panelists add <2% signal. The convergence feature surfaces that signal live so you can see when distributions stabilize and optionally halt early.

Convergence tracking applies only to **bounded** question types (Likert, yes/no, pick-one, any question with a JSON Schema `enum`). Free-text questions are not tracked.

```
synthpanel panel run \
  --personas panel.yaml \
  --instrument pricing-discovery \
  --convergence-check-every 20 \  # compute JSD every 20 panelists
  --auto-stop \                    # halt once all questions converge
  --convergence-eps 0.02 \         # JSD threshold (default: 0.02)
  --convergence-min-n 50 \         # don't stop before n=50
  --convergence-m 3                # 3 consecutive checks below eps
```

When the run finishes the JSON output contains a `convergence` key:

```
{
  "convergence": {
    "final_n": 487,
    "auto_stopped": true,
    "overall_converged_at": 473,  // run this many next time
    "per_question": {
      "pricing": { "converged_at": 473, "curve": [...] },
      "tier_preference": { "converged_at": 410, "curve": [...] }
    }
  }
}
```

`overall_converged_at` answers *"how many panelists did you actually need?"* — use it to right-size future runs.

| Flag | Default | Description |
|---|---|---|
| `--convergence-check-every N` | off | Compute running JSD every N completing panelists. Setting this opts in. |
| `--auto-stop` | off | Halt once all tracked questions converge. Requires `--convergence-check-every`. |
| `--convergence-eps FLOAT` | 0.02 | JSD threshold below which a question is treated as converged. |
| `--convergence-min-n N` | 50 | Minimum panelists before `--auto-stop` is allowed to fire. |
| `--convergence-m N` | 3 | Consecutive checks below epsilon required to declare convergence. |
| `--convergence-log PATH` | stderr | Write each convergence check as a JSON line to PATH (for dashboards). |
| `--convergence-baseline DATASET:Q` | — | Load a human baseline convergence curve from SynthBench and include in the report. Requires `pip install 'synthpanel[convergence]'`. |

## SynthBench — `--calibrate-against`, `--best-model-for`, `--submit-to-synthbench`

### --calibrate-against

Attaches inline calibration to a panel run by comparing the extracted response distribution against a published human baseline. Forces convergence tracking; pair with `--convergence-check-every` to control cadence (it is never implicit). v1 supports `gss` and `ntia`.

Auto-derives a pick-one extractor schema from the baseline when option count ≤ 5; otherwise pass `--extract-schema` explicitly. Requires `pip install 'synthpanel[convergence]'`.

```
synthpanel panel run \
  --personas panel.yaml \
  --instrument happiness-probe \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 20
```

### --best-model-for

Consult the SynthBench leaderboard and use the top-ranked model for the given topic instead of the default. A recommendation line is printed to stderr before the run. Overrides `--model`; mutually exclusive with `--models`.

```
synthpanel panel run \
  --personas panel.yaml \
  --instrument survey.yaml \
  --best-model-for 'political-opinion:globalopinionqa'
```

Format: `TOPIC` (ranked against the default dataset `globalopinionqa`), `TOPIC:DATASET` (specific dataset), or `:DATASET` (rank by SPS across the full dataset). The leaderboard is cached for 24 h at `~/.synthpanel/synthbench-cache.json`.

### --submit-to-synthbench

After a calibrated run, upload the per-question JSD and distributions to the SynthBench public leaderboard. Requires `--calibrate-against` and `SYNTHBENCH_API_KEY` in your environment (mint one at synthbench.org/account). First use prompts for consent, which is recorded locally and not re-prompted. Pass `--yes` to bypass the consent prompt for CI.

```
export SYNTHBENCH_API_KEY=sk_synthbench_...

synthpanel panel run \
  --personas panel.yaml \
  --instrument happiness-probe \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 20 \
  --submit-to-synthbench
```

The submission step is non-fatal — a slow or rejecting SynthBench emits a warning but does not affect the panel run's exit code or output. See [docs/synthbench-integration.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/synthbench-integration.md) for the privacy model and what gets uploaded.

| Flag | Default | Description |
|---|---|---|
| `--calibrate-against DATASET:Q` | — | Inline calibration vs a human baseline. v1 allowlist: `gss`, `ntia`. Requires `synthpanel[convergence]`. |
| `--best-model-for TOPIC[:DATASET]` | — | Use the SynthBench leaderboard top model for the topic. Overrides `--model`. |
| `--submit-to-synthbench` | off | Upload calibration results to SynthBench after the run. Requires `--calibrate-against` and `SYNTHBENCH_API_KEY`. |
| `--yes` | off | Bypass the SynthBench consent prompt (for CI/non-interactive use). |

## See also

### [Pack Calibration](/docs/calibration)

→

Calibrate a persona pack against a human baseline and embed the JSD into the pack YAML.

### [MCP Server](/mcp)

→

Run panels from your AI editor via the MCP `run_panel` tool.

### [Ensemble deep-dive](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/ensemble.md)

→

Full algorithm, edge cases, and MCP equivalents for `--models` and `--blend`.

### [Convergence deep-dive](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/convergence.md)

→

Full algorithm, JSON output schema, and baseline curve comparison for the convergence tracker.
