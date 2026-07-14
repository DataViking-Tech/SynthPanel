# Recommended Models — synthpanel · SynthBench-validated model picks

[DataViking](https://dataviking.tech)

Docs · Recommended Models

# Recommended models

SynthPanel can consult the [SynthBench](https://synthbench.org) public leaderboard to pick the best-ranked model for the kind of research you're running. This closes the credibility loop: scores measured on the bench drive defaults in the harness.

## Quick start

```
# Use the top-ranked model for a specific topic
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument pricing-discovery \
  --best-model-for "Economy & Work"

# Top-ranked model across a whole dataset (by SPS)
synthpanel panel run ... --best-model-for ":globalopinionqa"

# Topic within a non-default dataset
synthpanel panel run ... --best-model-for "Technology & Digital Life:globalopinionqa"
```

Before the run, SynthPanel prints a recommendation line to stderr so you can cancel and override:

```
synthbench: best model for globalopinionqa/Economy & Work → claude-haiku-4-5-20251001 · SPS 0.850 · JSD 0.091 · n=100 · $0.032/100q · cached 0h ago · source=live
```

## How it works

- 1 On first use, SynthPanel fetches `https://synthbench.org/data/leaderboard.json` and caches it at `~/.synthpanel/synthbench-cache.json` for 24 hours.

- 2 Entries are filtered to the requested `dataset` (default `globalopinionqa`), then ranked — by the named topic's score when a topic is given, otherwise by overall SPS.

- 3 The top entry's `model` field is resolved through SynthPanel's alias table (so `"haiku"` becomes `claude-haiku-4-5-20251001`) and stamped onto `--model` for the rest of the pipeline.

## Environment knobs

| Variable | Effect |
|---|---|
| `SYNTHPANEL_SYNTHBENCH_URL` | Override the fetch URL (useful for forks or air-gapped environments). |
| `SYNTHPANEL_SYNTHBENCH_OFFLINE=1` | Never hit the network; use the cache if present, otherwise skip the recommendation. |
| `SYNTHPANEL_SYNTHBENCH_REFRESH=1` | Bypass the 24h TTL and force a fresh fetch (ignores the cached ETag). |
| `SYNTH_PANEL_DATA_DIR` | Override the data dir where the cache lives. |

## Graceful offline behaviour

- **Stale cache + network error** → stderr warning, use stale cache.

- **No cache + network error** → stderr "synthbench unavailable", fall through to whatever `--model` or default was already in effect.

- **Empty entries after filter** → same fall-through.

No recommendation is ever fatal. `--best-model-for` is advisory: a bad network day won't take the panel down.

## Use-case → top-ranked model

Regenerated from the live `leaderboard.json` (`generated_at` 2026-07-13). Picks are the top-ranked *runnable single model* per filter (the 3-model ensemble leads several rows but isn't a plain `--model` value). The live data updates continuously — consult the CLI flag or [synthbench.org](https://synthbench.org) for current picks. Several topics are thinly sampled (small `n`); treat those as suggestive.

| Use case | Dataset | Topic / filter | Top SynthBench pick |
|---|---|---|---|
| General attitudes research | `globalopinionqa` | (overall SPS) | `openai/gpt-4o-mini` |
| Economic / workplace surveys | `globalopinionqa` | "Economy & Work" (n=5) | `google/gemini-2.5-flash` |
| Tech product discovery | `globalopinionqa` | "Technology & Digital Life" (n=3) | `google/gemini-2.5-flash-lite` |
| Health & science messaging | `globalopinionqa` | "Health & Science" (n=2) | `anthropic/claude-sonnet-4.6` |
| International affairs / policy | `globalopinionqa` | "International Relations & Security" (n=50) | `openai/gpt-4o-mini` |
| Trust & wellbeing | `globalopinionqa` | "Trust & Wellbeing" (n=7) | `anthropic/claude-sonnet-4.6` |

## Caveats

- **Display labels, ensembles & product configs (gh-519).** Some leaderboard entries are SynthPanel product configs (`framework=product`, `is_ensemble=true`) or carry a human-readable display label (e.g. `SynthPanel (Gemini Flash Lite)`) in their `model` field. SynthPanel never stamps such a label onto `--model`. Instead it substitutes a runnable id, preferring **(1)** the row's runnable `model_id` published by SynthBench (e.g. `google/gemini-2.5-flash-lite`, joined with `provider_id` when `model_id` is a bare slug), then **(2)** for product/ensemble rows without a `model_id`, a base model inferred from the entry's `config_id` (adopted only when it resolves to a recognized provider id or alias). If neither yields a runnable id the recommendation is **refused** with an actionable stderr message and your existing `--model`/default is kept. A stderr note records any substitution.

- **Sparse topics.** When the top entry's `run_count < 3`, a low-confidence warning is emitted. Treat those recommendations as suggestive rather than authoritative.

- **Provider/model strings vary.** The leaderboard publishes the raw `model` string the run used — sometimes a canonical id, sometimes a short alias. SynthPanel passes the string through the alias resolver so either shape works, but the raw value is preserved in the recommendation line as `raw_model`.

## Scoping

`--best-model-for` picks a single model for the whole panel. It is mutually exclusive with `--models` (which splits the panel across multiple models) — mixing the two is rejected at parse time.

## See also

### [SynthBench leaderboard](https://synthbench.org)

→

Live model rankings at synthbench.org — updated continuously.

### [MCP Server](/mcp)

→

Use `run_panel` from your AI editor with the top-ranked model.

### [Source docs](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/recommended-models.md)

→

The canonical `docs/recommended-models.md` in the GitHub repo.

### [Report an issue](https://github.com/DataViking-Tech/SynthPanel)

→

Open an issue on GitHub if something looks wrong.
