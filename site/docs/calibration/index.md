# Pack Calibration — synthpanel · Run synthetic focus groups with any LLM

 Docs · Pack Calibration

# Pack calibration

Calibration is how a persona pack proves its fit to a known human baseline. Run `synthpanel pack calibrate` and the result is written back into the pack YAML so the manifest is self-describing.

## CLI reference

```
synthpanel pack calibrate <PACK_YAML>
  --against DATASET:QUESTION   # e.g. gss:HAPPY (v1 allowlist: gss, ntia)
  [--n 50]                     # panel size
  [--models MODELS]            # default: panel run default
  [--samples-per-question 15]  # for stable JSD
  [--output PATH]              # default: rewrite PACK_YAML in place
  [--dry-run]                  # print what would be written
  [--yes]                      # skip overwrite confirm
```

After a successful run the pack YAML gains a top-level `calibration:` list:

```
calibration:
  - dataset: gss
    question: HAPPY
    jsd: 0.18
    n: 100
    samples_per_question: 15
    models: [haiku:0.30, gemini-flash-lite:0.30, qwen3-plus:0.20, deepseek-v3:0.20]
    extractor: pick_one:auto-derived
    panelist_cost_usd: 0.6451
    calibrated_at: 2026-04-26T14:23:00Z
    synthpanel_version: 0.11.1
    methodology_url: https://synthpanel.dev/docs/calibration
```

## What JSD means

Calibration reports **Jensen-Shannon divergence** between the pack's extracted answer distribution and the dataset's published human distribution. JSD is bounded in `[0, 1]` (using log₂):

| JSD | Interpretation |
|---|---|
| `< 0.05` | Tightly aligned — pack mirrors the human distribution. |
| `0.05–0.15` | Strong alignment — small but real distributional drift. |
| `0.15–0.30` | Moderate — usable for directional research, not census. |
| `0.30–0.50` | Weak — major disagreement on at least one mode. |
| `> 0.50` | Effectively uncalibrated. |

JSD is computed locally via the same metric the convergence tracker uses (see `src/synth_panel/convergence.py`); the pack YAML is the **only** durable artifact — calibration does not phone home and is distinct from `--submit-to-synthbench`.

## When to calibrate

Calibrate a pack when **any** of the following is true:

-  The pack will be cited in a research artifact and you need a defensible fit-to-baseline number.

-  You have changed the pack (added/removed personas, edited `personality_traits`, regenerated descriptions) and want to know whether your edits drifted the distribution.

-  You are comparing two packs that target the same audience and need a shared yardstick.

You do **not** need to calibrate before every panel run. The `calibration:` list is a snapshot, not a runtime guard.

## Choosing a baseline

The `--against DATASET:QUESTION` flag accepts the v1 inline-publishable allowlist:

| Dataset | Notes |
|---|---|
| `gss` | General Social Survey aggregates. `HAPPY` is the canonical demo question (3-option Likert); `HEALTH` and `LIFE` also work. |
| `ntia` | NTIA Internet Use Survey. Useful for technology-adoption packs. |

Other datasets (OpinionsQA, PewTech, GlobalOpinionQA, WVS, …) are **gated** for redistribution and require post-hoc calibration via `synthbench` directly. They are intentionally not exposed here so a calibrated pack YAML can be checked into a public repo without relicensing concerns.

To override the allowlist for internal use:

$  SYNTHBENCH_ALLOW_GATED=1  synthpanel pack calibrate ... --against wvs:Q1

## Methodology

For each calibration run:

- 1 The SynthBench baseline payload is fetched (same loader as `panel run --calibrate-against`).

- 2 A `pick_one` extraction schema is auto-derived from the baseline's small-enum distribution. If the baseline is Likert/ranking or too-wide, the command hard-fails with a hint to supply `--extract-schema` via `panel run` instead.

- 3 A panel of `--n` panelists is run against the dataset's question text. Each panelist answers `--samples-per-question` times.

- 4 The model distribution is compared against the baseline's `human_distribution` via Jensen-Shannon divergence.

- 5 The resulting JSD, plus provenance (extractor, models, cost, timestamp, synthpanel version), is merged into the pack YAML's `calibration:` list. Re-running against the same `(dataset, question)` pair **replaces** the prior entry rather than appending, so the list is always a clean record of one-result-per-baseline.

## Idempotence and re-runs

-  Re-running calibration against the same `dataset:question` replaces the prior entry — newest wins.

-  Calibration against a different `dataset:question` appends a new entry alongside any existing ones.

- `--dry-run` prints the rewritten YAML to stdout without touching the file. Use this to preview what would change before committing.

## Limitations

-  Auto-discovering "which question should this pack calibrate against?" is **out of scope** for the v1 command. You pick the baseline.

-  Calibration measures distributional fit on **one** answer distribution. A low JSD on `gss:HAPPY` does not imply low JSD on a different question — calibrate against multiple baselines if you need broad coverage.

-  The command requires the `synthpanel[convergence]` extra (for the `synthbench` baseline loader). Install with `pip install 'synthpanel[convergence]'` if not already present.

## See also

### [MCP Server](/mcp)

→

Run calibrated panel runs via the MCP tools from your AI editor.

### [Source docs](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/calibration.md)

→

The canonical `docs/calibration.md` in the GitHub repo.

### [Report an issue](https://github.com/DataViking-Tech/SynthPanel)

→

Open an issue on GitHub if something looks wrong.
