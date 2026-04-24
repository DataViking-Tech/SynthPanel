# Recommended Models (SynthBench-driven)

SynthPanel can consult the [SynthBench](https://synthbench.org) public
leaderboard to pick the best-ranked model for the kind of research you're
running. This closes the credibility loop: scores measured on the bench
drive defaults in the harness.

## TL;DR

```bash
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

Before the run, SynthPanel prints a recommendation line to stderr so you
can cancel and override:

```
synthbench: best model for globalopinionqa/Economy & Work → claude-haiku-4-5-20251001 · SPS 0.850 · JSD 0.091 · n=100 · $0.032/100q · cached 0h ago · source=synthbench.org
```

## How it works

1. On first use, SynthPanel fetches
   `https://synthbench.org/data/leaderboard.json` and caches it at
   `~/.synthpanel/synthbench-cache.json` for 24 hours.
2. Entries are filtered to the requested `dataset` (default
   `globalopinionqa`), then ranked — by the named topic's score when a
   topic is given, otherwise by overall SPS.
3. The top entry's `model` field is resolved through SynthPanel's alias
   table (so `"haiku"` becomes `claude-haiku-4-5-20251001`) and stamped
   onto `--model` for the rest of the pipeline.

### Environment knobs

- `SYNTHPANEL_SYNTHBENCH_URL` — override the fetch URL (useful for forks
  or air-gapped environments).
- `SYNTHPANEL_SYNTHBENCH_OFFLINE=1` — never hit the network; use the
  cache if present, otherwise skip the recommendation.
- `SYNTHPANEL_SYNTHBENCH_REFRESH=1` — bypass the 24h TTL and force a
  fresh fetch (ignores the cached ETag).
- `SYNTH_PANEL_DATA_DIR` — override the data dir where the cache lives.

### Graceful offline behaviour

- **Stale cache + network error** → stderr warning, use stale cache.
- **No cache + network error** → stderr "synthbench unavailable", fall
  through to whatever `--model` or default was already in effect.
- **Empty entries after filter** → same fall-through.

No recommendation is ever fatal. `--best-model-for` is advisory: a bad
network day won't take the panel down.

## Use-case → top-ranked model

Snapshot taken from `leaderboard.json` on 2026-04-24. The live data
updates continuously — this table is for quick orientation; consult the
CLI flag or [synthbench.org](https://synthbench.org) for current picks.

| Use case                                   | Dataset          | Topic / filter                    | Top SynthBench pick        |
|--------------------------------------------|------------------|-----------------------------------|----------------------------|
| General attitudes research                 | globalopinionqa  | (overall SPS)                     | `claude-haiku-4-5-20251001`|
| Economic / workplace surveys               | globalopinionqa  | "Economy & Work"                  | `claude-haiku-4-5-20251001`|
| Tech product discovery                     | globalopinionqa  | "Technology & Digital Life"       | `gemini-2.5-flash`         |
| Health & science messaging                 | globalopinionqa  | "Health & Science"                | see CLI (`--best-model-for "Health & Science"`) |
| International affairs / policy             | globalopinionqa  | "International Relations & Security" | see CLI                 |
| Trust & wellbeing                          | globalopinionqa  | "Trust & Wellbeing"               | see CLI                    |

## Caveats

- **Ensembles & product configs.** Some leaderboard entries are
  SynthPanel product configs (`framework=product`, `is_ensemble=true`).
  These aren't runnable as a plain `--model` value, so SynthPanel falls
  back to the underlying base model inferred from the entry's
  `config_id`. A stderr note records the substitution.
- **Sparse topics.** When the top entry's `run_count < 3`, a
  low-confidence warning is emitted. Treat those recommendations as
  suggestive rather than authoritative.
- **Provider/model strings vary.** The leaderboard publishes the raw
  `model` string the run used — sometimes a canonical id, sometimes a
  short alias. SynthPanel passes the string through the alias resolver
  so either shape works, but the raw value is preserved in the
  recommendation line as `raw_model`.

## Scoping

`--best-model-for` picks a single model for the whole panel. It is
mutually exclusive with `--models` (which splits the panel across
multiple models) — mixing the two is rejected at parse time.
