# Convergence telemetry

At competitive-parity scales (n=500..10k), synthpanel panels spend most of
their token budget producing diminishing returns. Once a response
distribution has stabilized for a given question, running more panelists
adds <2% signal for 10-20× the cost. The convergence feature exposes that
signal live so you can (a) see when the distribution stabilized, (b)
optionally auto-stop the run at that point, and (c) walk back to a
smaller representative n next time.

## TL;DR

```bash
synthpanel panel run \
  --personas my-panel.yaml \
  --instrument pricing-discovery \
  --convergence-check-every 20 \
  --auto-stop \
  --output-format json > result.json

# result.json now contains a `convergence` section.
jq '.convergence.overall_converged_at, .convergence.auto_stopped' result.json
```

## How it works

1. At startup synthpanel inspects the instrument and flags every question
   with a bounded response space — Likert (`rating`), yes/no (`answer`),
   pick-one (`choice`), or any JSON Schema with an `enum` field. Free-text
   questions are ignored; JSD on paraphrase noise is ill-defined.
2. As panelists complete, their categorical responses feed a per-question
   running `Counter`. Every `--convergence-check-every N` panelists the
   tracker computes the Jensen-Shannon divergence between the
   distribution-so-far and the last-batch distribution, and a rolling
   average over the last `--convergence-m` checks.
3. A question is declared "converged at n" the first time its rolling
   JSD stays below `--convergence-eps` for `--convergence-m` consecutive
   checks, provided `n >= --convergence-min-n`.
4. With `--auto-stop`, when *every* tracked question has converged the
   run cancels pending futures and returns the panelists finished so far.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--convergence-check-every N` | off (required to opt in) | Compute a convergence check every N completing panelists. |
| `--convergence-log PATH` | stderr | Write each check as a JSON line to PATH instead of stderr. |
| `--auto-stop` | off | Halt the run once every tracked question has converged. |
| `--convergence-eps` | 0.02 | JSD threshold below which a question is treated as converged. |
| `--convergence-min-n` | 50 | Minimum panelists before `--auto-stop` is allowed to fire. |
| `--convergence-m` | 3 | Consecutive checks below epsilon required to declare convergence. |
| `--convergence-baseline DATASET:Q` | off | Load a human baseline curve from synthbench and include in the report. Requires `pip install 'synthpanel[convergence]'`. |

## Interpreting the report

The JSON output grows a top-level `convergence` key:

```jsonc
{
  "convergence": {
    "final_n": 487,
    "check_every": 20,
    "epsilon": 0.02,
    "min_n": 50,
    "m_consecutive": 3,
    "auto_stopped": true,
    "overall_converged_at": 473,       // max converged_at across tracked questions
    "tracked_questions": ["pricing", "tier_preference"],
    "per_question": {
      "pricing": {
        "final_n": 487,
        "converged_at": 473,
        "support_size": 5,
        "curve": [                      // downsampled to ~20 points
          {"n": 20, "jsd": 0.11},
          {"n": 40, "jsd": 0.045},
          ...
        ]
      },
      "tier_preference": {
        "final_n": 487,
        "converged_at": 410,
        "support_size": 3,
        "curve": [...]
      }
    },
    "human_baseline": {                  // present only with --convergence-baseline
      "dataset": "gss",
      "question_key": "happiness",
      "converged_at": 410,
      "curve": [...]
    }
  }
}
```

**Reading it:**
- `overall_converged_at` is the answer to *"how many panelists did you
  actually need?"* — run this many next time and trust the shape of the
  distribution.
- `auto_stopped: true` means the run halted itself; `false` means it ran
  to completion but the report still shows where it converged.
- The per-question `curve` is downsampled to ~20 points for plotting;
  consult the `--convergence-log` JSONL stream for the full sequence if
  you need to re-plot.
- When `human_baseline` is present, compare `overall_converged_at`
  against `human_baseline.converged_at` to measure the sampling-efficiency
  ratio vs real humans. A ratio close to 1.0 means your synthetic panel
  is as "efficient" as the human baseline at stabilizing — higher means
  you need more synthetic respondents to match the same information
  content.

## When is auto-stop safe?

Safe:
- You want a point estimate ("the distribution is roughly X/Y/Z") rather
  than a precise confidence interval.
- Your bounded questions carry the signal you actually care about.
- You have set `--convergence-min-n` high enough that the distribution
  can't trivially look stable just because you've seen 20 panelists.

Unsafe:
- You are planning to slice the responses by a demographic subgroup.
  JSD on the full population can look stable long before any given
  subgroup has enough samples — inspect per-slice n before trusting the
  halt.
- Your key questions are free-text (not tracked) and the synthesis is
  the deliverable. In that case leave `--auto-stop` off and just use
  the telemetry to calibrate future runs.

## Tuning

- **`--convergence-check-every`**: lower values (10-20) give smoother
  curves; higher values (50-100) reduce per-check noise at large n. For
  panels under ~500 the default of 20 is fine.
- **`--convergence-eps`**: 0.02 is tight — distributions that differ by
  at most a couple of percentage points count as equal. Loosen to 0.05
  if your support is small (e.g., yes/no) and you want faster halts.
- **`--convergence-m`**: 3 is the minimum useful "stay below eps"
  bar; lower values (1-2) are noise-prone, higher (5+) only help on
  extremely heavy-tailed distributions.

## Methodology notes

Jensen-Shannon divergence is computed in base-2 so the output is bounded
in `[0, 1]`. Empty or zero-support distributions return `0` (no signal —
waiting on more data). Follow-up questions are excluded from tracking
because they contain conditional text that would pollute the running
distribution. Panelist errors silently drop out of the running counter
— the distribution reflects only panelists that actually answered.

## Related beads

- `sp-yaru` — this feature
- `sb-ygp7` — synthbench convergence bootstrap (required for
  `--convergence-baseline`)
- `sb-gh1n` — real-human microdata baseline (optional upgrade)
