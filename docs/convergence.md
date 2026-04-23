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
| `--calibrate-against DATASET:Q` | off | Attach a `calibration` sub-object to each tracked question with Jensen-Shannon divergence vs the published human distribution. v1 supports GSS only. **Pair explicitly with `--convergence-check-every`** — cadence is never implicit. See [Inline calibration vs a human baseline](#inline-calibration-vs-a-human-baseline). |

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

## Inline calibration vs a human baseline

`--calibrate-against DATASET:QUESTION` computes, inline during the run,
the Jensen-Shannon divergence between your synthetic panel's
distribution and a published human baseline, and attaches it to each
tracked question's convergence record as a `calibration` sub-object. No
second invocation, no post-hoc step — the number lands next to the
panel result.

### Scope (v1)

- **GSS only.** The redistribution-tier allowlist is `{gss, ntia}`;
  other datasets (OpinionsQA, PewTech, GlobalOpinionQA) are gated and
  deferred. Passing a gated dataset hard-fails at CLI parse time.
- **Single question per run.** One `DATASET:QUESTION` spec per
  invocation. Multi-question calibration is v2.
- **Bounded support required.** The target question must produce a
  bounded distribution (enum / pick-one / likert). Free-text responses
  cannot be calibrated because JSD on paraphrase noise is ill-defined.
- **Inline only.** There is no `synthpanel calibrate RESULT.json`
  subcommand; calibration runs during the panel, not against a saved
  result.

### Cadence is explicit — pair with `--convergence-check-every`

`--calibrate-against` force-enables convergence tracking, but it does
**not** imply a check cadence. You must pair it with
`--convergence-check-every N` to control sampling. Running calibration
without a check cadence is an explicit design choice — we don't
surprise you with cost on small-n runs.

```bash
# Correct: explicit cadence
synthpanel panel run \
  --personas my-panel.yaml \
  --instrument general-survey \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 10
```

### The three extractor paths

Calibration needs a bounded distribution over the *same option strings*
as the baseline. There are three ways synthpanel gets one:

1. **Auto-derive (default for small-enum baselines).** If the baseline
   has ≤5 enum options and you don't pass `--extract-schema`,
   synthpanel derives a `pick_one` extractor schema from the baseline's
   option strings and logs the derivation to stderr in sp-yaru style:

   ```
   [convergence] auto-derived pick_one schema from gss:HAPPY → 3 options: ['Not too happy', 'Pretty happy', 'Very happy']
   ```

   Provenance: `calibration.extractor == "pick_one:auto-derived"` and
   `calibration.auto_derived == true`.

2. **`--extract-schema` manual.** Pass your own schema (pick-one or
   Likert). Use this when the baseline has >5 options, is Likert /
   numeric, or you want custom option strings. Provenance:
   `calibration.extractor` is `"pick_one:manual"` or `"likert:manual"`
   (detected by the `rating` property on the schema). Auto-derivation
   is skipped.

3. **Bounded instrument (authoring path).** Author the instrument
   question with a bounded `response_schema` whose option strings match
   the baseline verbatim. No flag needed beyond `--calibrate-against`.
   Recorded as a manual extractor.

Auto-derivation hard-fails at CLI parse — **before any LLM spend** —
when the baseline has >5 options or looks Likert. The two failure
messages are distinct so you know which remedy applies (pick a smaller
question vs. pass `--extract-schema likert`).

### Worked example: `general-survey × gss:HAPPY`

GSS HAPPY is a three-option item ("Very happy", "Pretty happy", "Not
too happy") with decades of trend data — tight support, easy to
explain, ideal first wedge.

```bash
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument general-survey \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 10 \
  --output-format json > result.json
```

`result.json` gains a `calibration` sub-object on every tracked
question (general-survey has no question whose extractor key matches
HAPPY verbatim, so the example below uses a hypothetical
`happiness_question` key to show the attached shape):

```jsonc
{
  "convergence": {
    "tracked_questions": ["happiness_question"],
    "per_question": {
      "happiness_question": {
        "final_n": 20,
        "converged_at": null,
        "support_size": 3,
        "curve": [ ... ],
        "calibration": {
          "jsd": 0.041237,
          "baseline_spec": "gss:HAPPY",
          "extractor": "pick_one:auto-derived",
          "auto_derived": true
        }
      }
    }
  }
}
```

Read `calibration.jsd` as: *how far is this panel's distribution from
the published human distribution, on a [0, 1] scale?* `0.0` = identical;
`1.0` = maximum divergence **or** broken alignment (see next section).

### Reading `calibration.jsd == 1.0` — LOUD disambiguation

`calibration.jsd == 1.0` has **two distinct meanings** and you must
check which one applies before drawing any conclusion. Synthpanel
surfaces the difference explicitly via a provenance field.

**Case A — option-string alignment broke.** The extractor's categories
and the baseline's keys are disjoint (e.g. extractor emits `"VERY
HAPPY"` while the baseline keys on `"Very happy"`, or extractor emits
coded `"1"/"2"/"3"` while the baseline uses prose). When this happens,
the `calibration` sub-object carries an `alignment_error` field:

```jsonc
"calibration": {
  "jsd": 1.0,
  "baseline_spec": "gss:HAPPY",
  "extractor": "pick_one:auto-derived",
  "auto_derived": true,
  "alignment_error": "['NOT TOO HAPPY', 'PRETTY HAPPY', 'VERY HAPPY'] vs ['Not too happy', 'Pretty happy', 'Very happy']"
}
```

**If `alignment_error` is present, the JSD value is not a signal about
your panel's divergence from humans — it means the two distributions
are not even talking about the same categories.** Fix the option
strings (adjust your instrument, pass `--extract-schema` with matching
enum values, or pick a different baseline question) and rerun. We
surface this rather than silently normalizing because verbatim-match is
the contract — a silent lowercase/strip would hide real data-quality
problems in production pipelines.

**Case B — genuine distributional divergence.** The extractor and
baseline agree on the category vocabulary (shared keys exist) but the
mass is concentrated in different categories. `jsd` approaches `1.0`
without an `alignment_error` field:

```jsonc
"calibration": {
  "jsd": 0.987,
  "baseline_spec": "gss:HAPPY",
  "extractor": "pick_one:auto-derived",
  "auto_derived": true
}
```

This is a real finding: your personas answer the question
*substantively* differently from the human reference. Investigate the
persona pack, prompt, or model — the calibration did its job.

**Rule of thumb:** `jsd == 1.0` + `alignment_error` present → fix your
setup. `jsd` high + no `alignment_error` → trust the number, dig into
the persona/model.

### Provenance fields reference

Every `calibration` sub-object carries these fields (the last is
conditional):

| Field | Type | Meaning |
|---|---|---|
| `jsd` | float in [0, 1] | Jensen-Shannon divergence (base-2) between the panel distribution and the baseline's `human_distribution`. `0.0` = identical; `1.0` = maximum divergence *or* broken alignment (check `alignment_error`). |
| `baseline_spec` | string | The `DATASET:QUESTION` spec passed to `--calibrate-against` (e.g. `"gss:HAPPY"`). |
| `extractor` | string | Which extractor path produced the bounded distribution: `"pick_one:auto-derived"`, `"pick_one:manual"`, or `"likert:manual"`. |
| `auto_derived` | bool | `true` when synthpanel derived the extractor schema from the baseline automatically (≤5-option enum, no `--extract-schema` supplied); `false` when you supplied the schema or authored a bounded instrument. |
| `alignment_error` | string (conditional) | Present **only** when extractor keys and baseline keys are disjoint. Value is `"<sorted-model-keys> vs <sorted-baseline-keys>"`. **If this field is present, do not interpret `jsd` as a divergence signal.** |

The `calibration` sub-object is the committed wire format. Downstream
consumers (sp-pack-registry in particular) fingerprint on these exact
field names — additions are permitted but renaming or flattening is a
breaking change.

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
- `sp-0ku0` — inline calibration (`--calibrate-against`, this
  document's calibration section)
