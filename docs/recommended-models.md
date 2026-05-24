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
synthbench: best model for globalopinionqa/Economy & Work → claude-haiku-4-5-20251001 · SPS 0.850 · JSD 0.091 · n=100 · $0.032/100q · cached 0h ago · source=live
```

The trailing `source=` field is the **provenance discriminator** (sy-klp).
It tells you — and any agent parsing this line — whether the
recommendation reflects the live leaderboard, your local cache, or a
package-bundled fallback:

| `source=` value     | What it means                                                                                       | How current is the recommendation? |
|---------------------|-----------------------------------------------------------------------------------------------------|------------------------------------|
| `live`              | Fetched from the leaderboard URL this run — or upstream returned 304 confirming the cache is current | As of right now                    |
| `cache`             | User's on-disk cache was fresh (< 24h), no network call made                                         | Within the last 24h                |
| `stale-cache`       | Cache exceeded the 24h TTL and the network fetch failed; stale cache used                            | Whenever the cache was last refreshed (see `cached <N>h ago` field) |
| `bundled-snapshot`  | No user cache and the live fetch failed (or offline mode requested without a cache) — package fallback used | The package release date (currently 2026-04-24) — *not* current |

Agents should treat `live` and `cache` as authoritative, `stale-cache`
as user-dated (caller's last successful sync), and `bundled-snapshot`
as package-dated — same age for everyone on the same release, so the
recommendation will drift from leaderboard reality after the snapshot.

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

> **Note (GH #494, v1.5.1+):** The canonical leaderboard URL is currently
> 404'ing upstream. SynthPanel ships a **bundled snapshot** (taken
> 2026-04-24) at `synth_panel/data/synthbench-snapshot.json` and falls
> back to it when the live URL is unreachable AND no user cache exists.
> The fallback produces a real recommendation on stderr and tags the
> entry as `source=bundled-snapshot`. To override with your own mirror,
> set `SYNTHPANEL_SYNTHBENCH_URL=...`.

### Environment knobs

- `SYNTHPANEL_SYNTHBENCH_URL` — override the fetch URL (useful for
  internal mirrors, forks, or air-gapped environments — and the
  documented workaround for the GH #494 outage).
- `SYNTHPANEL_SYNTHBENCH_OFFLINE=1` — never hit the network; use the
  cache if present, otherwise the bundled snapshot, otherwise skip the
  recommendation.
- `SYNTHPANEL_SYNTHBENCH_REFRESH=1` — bypass the 24h TTL and force a
  fresh fetch (ignores the cached ETag).
- `SYNTH_PANEL_DATA_DIR` — override the data dir where the cache lives.

### Graceful offline behaviour

Every degraded path tags the recommendation with a non-`live` `source=`
discriminator (see the table above) and prints a one-line explanation
on stderr so agents and humans can both see how degraded the answer is.

- **Fresh cache hit (< 24h)** → no network. Recommendation tagged
  `source=cache`. No stderr noise.
- **Stale cache + 304** → conditional GET confirms cache is current.
  Recommendation tagged `source=live` and the cache timestamp is
  refreshed. No stderr noise.
- **Stale cache + network error** → stderr `synthpanel: synthbench
  fetch failed (…); using stale cache from <ISO> (source=stale-cache)`.
  Recommendation tagged `source=stale-cache`.
- **No cache + network error** → bundled snapshot fallback (sy-nkh):
  stderr `synthpanel: synthbench unavailable (…); using bundled
  snapshot from YYYY-MM-DD (source=bundled-snapshot) — override the URL
  via $SYNTHPANEL_SYNTHBENCH_URL if you have a mirror …`. Recommendation
  tagged `source=bundled-snapshot` and the CLI also emits a follow-up
  note explaining the implication and the override path.
- **Offline mode + fresh cache** → recommendation tagged
  `source=cache`. No stderr noise.
- **Offline mode + stale cache** → recommendation tagged
  `source=stale-cache`. No stderr noise (offline mode is opt-in; the
  user already knows network is disabled).
- **Offline mode + no cache** → bundled snapshot fallback with
  `source=bundled-snapshot`. Stderr explains the fallback once.
- **No cache + no bundled snapshot** → stderr "synthbench unavailable",
  fall through to whatever `--model` or default was already in effect.
- **Empty entries after filter** → same fall-through.

No recommendation is ever fatal. `--best-model-for` is advisory: a bad
network day won't take the panel down, and a 404'ing upstream URL still
yields a sensible default via the bundled snapshot — at the cost of
freshness, which the `source=bundled-snapshot` discriminator makes
explicit instead of pretending the data is current.

### Reading the source field from code

The wire field is part of the public `synth_panel.synthbench`
surface — `Recommendation.source` and `LoadedLeaderboard.source` both
expose the same closed enum. Allowed values are exported as
`synth_panel.synthbench.RECOMMENDATION_SOURCES`:

```python
from synth_panel import synthbench

rec = synthbench.recommend("Economy & Work")
if rec is None:
    ...  # leaderboard unavailable; honour the existing --model
elif rec.source in ("live", "cache"):
    use_with_confidence(rec.model)
elif rec.source == "stale-cache":
    log_warning(f"using model from cache aged {rec.cache_age_hours:.0f}h")
    use_with_confidence(rec.model)
elif rec.source == "bundled-snapshot":
    log_warning(
        f"recommendation from package snapshot dated {rec.fetched_at.date()}; "
        "current leaderboard may have moved."
    )
    use_with_caveat(rec.model)
```

The same `source=` field appears at the tail of every stderr
recommendation line (`format_line()` output), so log-scraping agents
can extract it with one regex.

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

- **Display labels & runnable ids (gh-519).** Some leaderboard rows carry a
  human-readable display label (e.g. `SynthPanel (Gemini Flash Lite)`) in
  their `model` field rather than a runnable provider model id. SynthPanel
  never stamps such a label onto `--model`. Instead it substitutes a runnable
  id in this order:
  1. The row's runnable `model_id` (e.g. `google/gemini-2.5-flash-lite`)
     published by SynthBench — joined with `provider_id` as
     `<provider_id>/<model_id>` when `model_id` is a bare slug.
  2. For product/ensemble rows (`framework=product`, `is_ensemble=true`)
     without a `model_id`, a base model inferred from the entry's `config_id`,
     adopted only when it resolves to a recognized provider id or alias.
  3. If neither yields a runnable id, the recommendation is **refused** with
     an actionable stderr message and SynthPanel keeps your existing
     `--model`/default. A stderr note records any substitution.
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

For a *model mix* picker — i.e. when the answer to "which models?" is more
than one and you want it calibrated to the decision's stake rather than a
single SynthBench topic — see [docs/model-packs.md](model-packs.md). Packs
like `balanced-research-ensemble` and `high-stakes-validation` exist
precisely because the leaderboard shows ensembles outperforming single
top-ranked models on most contested topics.
